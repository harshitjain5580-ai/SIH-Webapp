# Prescription dataset training

## How the model was trained (Class 10 explanation)

This section explains the project from the beginning, without assuming that you
already know machine learning.

### 1. What problem are we solving?

The app helps a patient answer health-history questions. It can ask the next
safe question in:

- English
- Hindi (Devanagari script)
- Hinglish (Hindi written using English letters)

It is important to understand what the model **does not** do. It does not
diagnose a disease, choose treatment, or prescribe medicine. It only helps
continue a conversation by asking one question at a time.

Think of the model like a student learning how a teacher phrases questions.
It learns examples of good questions; it does not become a doctor.

### 2. What are the basic things used?

| Thing | Simple meaning | What we used |
| --- | --- | --- |
| Dataset | Many examples used for learning | 83 English-Hindi question pairs in the Excel workbook |
| Base model | A model that already understands language | `Qwen/Qwen2.5-1.5B-Instruct` |
| Tokenizer | Breaks text into small pieces the computer can number | Qwen tokenizer |
| GPU | Special processor that makes training faster | NVIDIA RTX 4050, when available |
| Training script | Instructions for preparing data and learning | `train_qwen_qlora.py` |
| Adapter | A small file containing the changes learned by training | `outputs/qwen2.5-1.5b-bilingual-lora/` |
| Inference | Using the trained model to answer a new message | `local_bilingual_model.py` |

Computers do not directly learn from words such as “pain”. The tokenizer
turns text into numbers called **tokens**. The model processes those numbers,
and its learned values (called **weights**) are adjusted during training.

### 3. Preparing the examples

The Excel file has a category, an English question, a Hindi question, and
sometimes a safety note. For every row, the script creates three supervised
examples:

1. English instruction -> the English question
2. Hindi instruction -> the Hindi question
3. Hinglish instruction -> the Romanized Hindi question

Therefore, 83 source pairs become **249 training examples** (83 x 3). A sample
looks like this:

```text
System: Ask one question only. Never diagnose or prescribe medicine.
User: Continue the interview in English. Category: symptoms.
Assistant: Where exactly are you feeling the pain?
```

The `System`, `User`, and `Assistant` labels show the model which text is the
instruction and which text is the correct example answer. Hinglish examples
are made by converting Hindi letters into readable English letters.

### 4. What does “fine-tuning” mean?

Qwen is already trained on a large amount of general text. Starting with it is
like teaching a student who already knows how to read. **Fine-tuning** means
showing it our smaller collection of examples so it becomes better at this
specific job.

For each example, the model tries to predict the next token. It compares its
prediction with the real next token and calculates an error called **loss**.
Training changes the weights in a direction that usually lowers this loss.
This prediction-and-correction cycle is repeated over the examples.

### 5. Why did we use LoRA?

Changing all 1.5 billion values in Qwen would require a lot of memory and
time. **LoRA (Low-Rank Adaptation)** freezes the original Qwen model and adds
small trainable matrices to selected attention layers. Only these small
additional parts learn our question style.

This gives us two useful things:

- training needs less memory;
- the original model remains available, while our small adapter stores the
  project-specific behavior.

The saved adapter is loaded on top of the same base Qwen model when the API
receives a request.

### 6. The actual training settings

The successful bilingual run used:

- **249 examples** in English, Hindi, and Hinglish
- **3 epochs**: the model saw the complete example collection three times
- **Batch size 1**: one example was placed on the GPU at a time
- **Gradient accumulation 8**: eight small updates were combined before one
  weight update, acting somewhat like a larger batch
- **Learning rate `0.00005`**: the size of each weight adjustment
- **Maximum length 512 tokens**: prevents unusually long examples using
  unlimited memory
- **BF16 numbers on the RTX 4050**: a memory-saving number format
- **No 4-bit quantization in the successful run**: the requested
  `bitsandbytes` loader crashed on this Windows setup, so a normal BF16 LoRA
  run was used instead

The complete sequence is:

```text
Excel questions
    -> English/Hindi/Hinglish examples
    -> tokens (numbers)
    -> Qwen predicts the next token
    -> loss measures the mistake
    -> LoRA weights are adjusted
    -> repeat for 3 epochs
    -> save the adapter
```

### 7. What happens when a patient uses the app?

1. The API receives the patient's transcript.
2. The tokenizer converts the transcript and safety instruction into tokens.
3. The base Qwen model and our LoRA adapter read those tokens.
4. The model generates up to 64 new tokens, one token after another.
5. Sampling is disabled (`do_sample=False`) so the same input gives a
   repeatable result.
6. The tokenizer converts the generated tokens back into text.
7. The API returns one follow-up question.

The adapter is loaded lazily, which means the model is loaded the first time
`/ask-clinical-question` is called rather than when the server starts.

### 8. How to reproduce the bilingual training

From the project root, install the dependencies and run:

```powershell
pip install -r requirements.txt
pip install -r training\requirements.txt
python training\train_qwen_qlora.py `
  --dataset bilingual_clinical_conversation_questions.xlsx `
  --output training\outputs\qwen2.5-1.5b-bilingual-lora
```

The script needs a CUDA-enabled PyTorch installation and a compatible GPU for
the configured BF16 training run. It writes the adapter, tokenizer files, and
`training_summary.json` into the output directory.

### 9. How do we know what was trained?

The saved summary records the important facts:

- base model: Qwen2.5-1.5B-Instruct
- method: LoRA
- source pairs: 83
- expanded examples: 249
- languages: English, Hindi, Hinglish
- epochs: 3
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

This is reproducibility information: another developer can check what data,
method, and settings produced the adapter.

### 10. The separate prescription-image baseline

There is another experiment in this folder. It is **not** the bilingual
conversation model used by `/ask-clinical-question`.

It uses 102 valid prescription image/transcription pairs. The images are
converted to grayscale, resized to 64 x 64 pixels, and represented by pixel
values plus horizontal and vertical image gradients. That creates 12,288
numbers per image. A scikit-learn cosine **nearest-neighbor** model then finds
the most visually similar training image and returns its transcription.

This is a useful transparent baseline for a small dataset, but it is not a
modern OCR system and its measured accuracy is not clinically usable. It has
one deterministic fitting pass, not neural-network epochs. A production OCR
system would need more varied data and line-level text annotations.

## Quick reference: which file does what?

- `train_qwen_qlora.py` — trains the bilingual question adapter
- `local_bilingual_model.py` — loads that adapter and generates questions
- `prepare_dataset.py` — validates prescription image/JSON pairs and makes
  train/validation/test files
- `train.py` — trains the image nearest-neighbor baseline
- `evaluate.py` — measures the baseline's transcription results
- `inference.py` — runs the baseline on one prescription image

## Dataset

Dataset: `codinganirbanb/doctor-prescription-labelled-dataset`  
Source: downloaded with the required `kagglehub.dataset_download` call.  
The download contains 211 files: 105 JPG images and 105 JSON sidecars plus one helper Python file. JSON schemas are heterogeneous; 102 records expose a supported full transcription field, one JSON is empty, and two images are byte-for-byte duplicates. Images use JPG format, have varied dimensions, and all 105 images passed Pillow corruption checks. There are no bounding boxes or entity spans, so this run addresses image-to-text transcription retrieval, not object detection or medical entity extraction. No reliable class vocabulary can be inferred; medical fields are present as free-form text and vary by document.

The generated split is 81 train / 10 validation / 11 test (80% / 10% / 10% of 102 unique valid pairs). See `dataset/summary.json`.

## Hardware

GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB reported by `nvidia-smi`)  
CUDA: available through PyTorch `2.11.0+cu128`; BF16 training ran on the GPU.

## Model

Model: gradient-augmented grayscale image nearest-neighbor transcription baseline  
Parameters: 12,288 float32 image-gradient features; 76 training examples  
Framework: scikit-learn  
Precision: float32  
Quantization: none

This compact model is appropriate for the small, weakly standardized dataset and does not pretend to learn unsupported structured fields. A production OCR model should be trained after obtaining line-level boxes/transcriptions.

## Training and results

Training is deterministic and completes in one fitting pass (no neural epochs or GPU memory metric apply). Validation metrics: CER 0.8129, WER 4.8104, exact match 0.0. Test metrics: CER 0.7826, WER 4.7314, exact match 0.0. Per-image predictions are in `outputs/metrics.json`. These results are a baseline, not clinically usable extraction accuracy.

## Commands

```powershell
python training/prepare_dataset.py --root "C:\Users\LENOVO\.cache\kagglehub\datasets\codinganirbanb\doctor-prescription-labelled-dataset\versions\1\prepared_image_data"
python training/train.py
python training/evaluate.py
python training/inference.py --image "path\to\prescription.jpg"
```

## Output

Best model/checkpoint: `training/outputs/best_model/model.pkl`  
Configuration: `training/outputs/best_model/model_config.json`  
Training summary: `training/outputs/best_model/training_summary.txt`  
Metrics: `training/outputs/metrics.json`  
Preprocessing: `training/train.py`  
Inference script: `training/inference.py`

## Qwen adapter experiment

`train_qwen_qlora.py` fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` on the 81 available
prescription transcriptions. The requested 4-bit QLoRA loader crashed with a
native Windows bitsandbytes access violation before training, so the successful
run used a BF16 LoRA adapter instead. It completed 3 epochs on the RTX 4050;
this is a transcription-domain adapter and does not teach bilingual clinical
question behavior. Adapter files are in `outputs/qwen2.5-1.5b-qlora/`.

`train_qwen_qlora.py` also trains a separate multilingual clinical-question adapter
from `bilingual_clinical_conversation_questions.xlsx`. It uses 83 English/Hindi
question pairs expanded into 249 English/Hindi/Hinglish supervised examples, 3 epochs, and saves to
`outputs/qwen2.5-1.5b-bilingual-lora/`. The workbook contains questions only,
so this teaches safe multilingual question phrasing rather than full adaptive
dialogue policy. Hinglish examples are generated by Romanizing the Hindi column.
