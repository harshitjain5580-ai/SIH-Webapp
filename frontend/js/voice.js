/**
 * Web Speech API Voice Recognition & Synthesis Engine
 * Provides Dual-Mode voice input & auditory accessibility for elderly/low-literacy patients.
 */

export const VoiceEngine = {
  recognition: null,
  isListening: false,
  ttsEnabled: true,
  audioCtx: null,
  mediaRecorder: null,
  mediaStream: null,
  mediaChunks: [],
  recorderStarting: false,
  transcriptCallback: null,
  stateCallback: null,

  apiBase() {
    return (window.location.protocol.startsWith('http') && (window.location.port === '8000' || window.location.port === ''))
      ? window.location.origin
      : (window.MEDIKIOSK_API_BASE || 'http://127.0.0.1:8000');
  },

  /**
   * Initialize browser speech recognition and synthesis
   */
  init(onTranscriptCallback, onStateChangeCallback) {
    this.transcriptCallback = onTranscriptCallback;
    this.stateCallback = onStateChangeCallback;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-IN'; // Supports Indian English, Hindi, and Hinglish

      this.recognition.onstart = () => {
        this.isListening = true;
        if (onStateChangeCallback) onStateChangeCallback(true);
      };

      this.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        if (onTranscriptCallback) {
          onTranscriptCallback(finalTranscript || interimTranscript, Boolean(finalTranscript));
        }
      };

      this.recognition.onerror = (event) => {
        console.warn('Speech recognition notice:', event.error);
        this.isListening = false;
        if (onStateChangeCallback) onStateChangeCallback(false);
        if (event.error === 'network' || event.error === 'service-not-allowed') {
          this.startMediaRecorder();
        }
      };

      this.recognition.onend = () => {
        this.isListening = false;
        if (onStateChangeCallback) onStateChangeCallback(false);
      };
    } else {
      console.warn('Web Speech API is not supported in this browser; falling back to touch/text inputs.');
    }
  },

  /**
   * Toggle speech recognition recording
   */
  toggleListening(onTranscriptCallback, onStateChangeCallback) {
    this.transcriptCallback = onTranscriptCallback;
    this.stateCallback = onStateChangeCallback;
    if (this.recorderStarting) return true;
    if (this.mediaRecorder) {
      this.stopMediaRecorder();
      return false;
    }
    if (navigator.mediaDevices?.getUserMedia && window.MediaRecorder) {
      this.startMediaRecorder();
      return true;
    }
    if (!this.recognition) {
      this.init(onTranscriptCallback, onStateChangeCallback);
    }

    if (!this.recognition) {
      this.startMediaRecorder();
      return true;
    }

    if (this.isListening) {
      this.recognition.stop();
      this.isListening = false;
      if (onStateChangeCallback) onStateChangeCallback(false);
      return false;
    } else {
      try {
        this.recognition.start();
        return true;
      } catch (err) {
        console.error('Failed to start speech recognition:', err);
        return false;
      }
    }
  },

  async startMediaRecorder() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      alert('Microphone speech recognition is unavailable. Please type your answer instead.');
      return false;
    }
    if (this.isListening || this.recorderStarting) return true;

    try {
      this.recorderStarting = true;
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaChunks = [];
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
        .find((type) => MediaRecorder.isTypeSupported(type));
      this.mediaRecorder = new MediaRecorder(this.mediaStream, mimeType ? { mimeType } : undefined);
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size) this.mediaChunks.push(event.data);
      };
      this.mediaRecorder.onstop = async () => {
        const blob = new Blob(this.mediaChunks, { type: this.mediaRecorder?.mimeType || 'audio/webm' });
        this.mediaStream?.getTracks().forEach((track) => track.stop());
        this.mediaStream = null;
        this.mediaRecorder = null;
        if (!blob.size) return;

        const form = new FormData();
        form.append('file', blob, 'patient-voice.webm');
        try {
          const response = await fetch(`${this.apiBase()}/voice/transcribe?language=auto`, {
            method: 'POST',
            body: form
          });
          const result = await response.json();
          if (!response.ok) throw new Error('voice transcription unavailable');
          if (this.transcriptCallback) this.transcriptCallback(result.text, true);
        } catch (error) {
          console.error('Backend microphone transcription failed:', error);
          alert('We could not understand the recording. Please try again or type your answer.');
        }
      };
      this.mediaRecorder.start();
      this.recorderStarting = false;
      this.isListening = true;
      if (this.stateCallback) this.stateCallback(true);
      return true;
    } catch (error) {
      console.error('Microphone permission/start failed:', error);
      this.recorderStarting = false;
      this.isListening = false;
      if (this.stateCallback) this.stateCallback(false);
      alert('Microphone permission was denied or unavailable. Please allow microphone access or type your answer.');
      return false;
    }
  },

  stopMediaRecorder() {
    if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') return false;
    this.isListening = false;
    if (this.stateCallback) this.stateCallback(false);
    this.mediaRecorder.stop();
    return true;
  },

  /**
   * Text-to-Speech playback for accessibility
   */
  speak(text) {
    if (!this.ttsEnabled || !window.speechSynthesis) return;

    // Cancel ongoing speech
    window.speechSynthesis.cancel();

    // Clean markdown or brackets from speech
    const cleanText = text.replace(/[\(\)\[\]\*\#\_]/g, ' ').replace(/\s+/g, ' ').trim();
    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.95; // Slightly slower pace for elderly accessibility
    utterance.pitch = 1.0;

    // Attempt to pick a pleasant natural voice
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => v.lang.includes('en-IN') || v.name.includes('India') || v.name.includes('Natural')) || voices[0];
    if (preferredVoice) utterance.voice = preferredVoice;

    window.speechSynthesis.speak(utterance);
  },

  /**
   * Synthesize an audible chime alert using Web Audio API (for Emergency Red Flags)
   */
  playEmergencyChime() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      if (!this.audioCtx) this.audioCtx = new AudioContext();

      const ctx = this.audioCtx;
      if (ctx.state === 'suspended') ctx.resume();

      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(880, now); // A5
      osc1.frequency.exponentialRampToValueAtTime(440, now + 0.35);

      osc2.type = 'triangle';
      osc2.frequency.setValueAtTime(587.33, now); // D5
      osc2.frequency.exponentialRampToValueAtTime(293.66, now + 0.35);

      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.35);
      osc2.stop(now + 0.35);
    } catch (e) {
      console.warn('Audio chime notice:', e);
    }
  }
};
