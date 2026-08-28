# Prescription dataset training

## Dataset

Dataset: `codinganirbanb/doctor-prescription-labelled-dataset`  
Source: downloaded with the required `kagglehub.dataset_download` call.  
The download contains 211 files: 105 JPG images and 105 JSON sidecars plus one helper Python file. JSON schemas are heterogeneous; 102 records expose a supported full transcription field, one JSON is empty, and two images are byte-for-byte duplicates. Images use JPG format, have varied dimensions, and all 105 images passed Pillow corruption checks. There are no bounding boxes or entity spans, so this run addresses image-to-text transcription retrieval, not object detection or medical entity extraction. No reliable class vocabulary can be inferred; medical fields are present as free-form text and vary by document.

The generated split is 81 train / 10 validation / 11 test (80% / 10% / 10% of 102 unique valid pairs). See `dataset/summary.json`.

## Hardware

GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB reported by `nvidia-smi`)  
CUDA: system driver CUDA 13.1; installed PyTorch is `2.13.0+cpu`, so CUDA was unavailable to Python and training ran on CPU.

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
