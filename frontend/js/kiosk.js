/**
 * Patient Intake Kiosk Controller
 * Multi-turn interview driving two selectable patient-facing screen designs
 * ("One Question At A Time" and "Guided Steps"), quick-reply touch options,
 * dual-mode voice/text input, and red-flag alerts. SOCRATES/AYUSH tracking
 * lives on the triage/physician views, not here.
 */

import { ApiService } from './api.js';
import { VoiceEngine } from './voice.js';

const TOTAL_STEPS = 6;
const STEP_LABELS = [
  'Your symptom',
  'When & how it feels',
  'How severe',
  'Medicines & allergies',
  'Appetite & sleep',
  'Check & finish'
];

const STATIC_STRINGS = {
  en: {
    tapToSpeak: 'Tap and speak',
    yourAnswer: 'your answer',
    typeInstead: 'Type instead',
    sayAnswer: 'Or say your answer out loud',
    placeholder: 'Speak or type symptoms (बोलें या लिखें)...'
  },
  hi: {
    tapToSpeak: 'बोलने के लिए दबाएं',
    yourAnswer: 'अपना जवाब',
    typeInstead: 'टाइप करें',
    sayAnswer: 'या अपना जवाब बोलकर बताएं',
    placeholder: 'लक्षण बोलें या लिखें...'
  },
  hinglish: {
    tapToSpeak: 'Tap karke boliye',
    yourAnswer: 'apna jawab',
    typeInstead: 'Type karein',
    sayAnswer: 'Ya apna jawab bolkar bhi bata sakte hain',
    placeholder: 'Symptoms bolein ya likhein...'
  }
};

export const KioskModule = {
  currentPatient: {
    abha_id: '',
    name: 'Walk-in Patient',
    verified: false,
    dob: '',
    gender: ''
  },
  conversationHistory: [],
  isInterviewComplete: false,
  redFlagUrgent: false,
  layout: 'a',
  language: 'en',
  chiefComplaint: null,
  lastQuestion: '',
  lastOptions: [],

  /**
   * Initialize Kiosk Event Listeners & UI
   */
  init() {
    this.layout = localStorage.getItem('medikiosk_kiosk_layout') === 'b' ? 'b' : 'a';
    this.bindEvents();
    this.applyLayout();
    this.applyLanguage();
    // The interview itself doesn't start until AuthModule resolves a patient
    // identity at login and calls setPatientIdentity(), which starts a fresh
    // session for them.
  },

  bindEvents() {
    // Layout Switch
    const layoutToggle = document.getElementById('kiosk-layout-toggle');
    if (layoutToggle) {
      layoutToggle.querySelectorAll('.segmented-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          this.layout = btn.dataset.layout;
          localStorage.setItem('medikiosk_kiosk_layout', this.layout);
          this.applyLayout();
          this.renderCurrentStep();
        });
      });
    }

    // Language Toggle (both layouts share the same buttons)
    document.querySelectorAll('.kiosk-lang-toggle .lang-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.language = btn.dataset.lang;
        this.applyLanguage();
      });
    });

    // Layout A controls
    this.bindLayoutAControls();
    // Layout B controls
    this.bindLayoutBControls();

    // Restart Interview
    const resetBtn = document.getElementById('btn-reset-kiosk');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => this.resetSession());
    }

    // Generate Summary Early / Finish (both layouts)
    ['btn-finish-kiosk-a', 'btn-finish-kiosk-b'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', () => this.generateAndCompleteSummary());
    });
  },

  bindLayoutAControls() {
    const sendBtn = document.getElementById('kiosk-send-btn-a');
    const textInput = document.getElementById('kiosk-text-input-a');
    if (sendBtn && textInput) {
      sendBtn.addEventListener('click', () => this.handleSendMessage(null, textInput));
      textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.handleSendMessage(null, textInput);
      });
    }

    const typeToggle = document.getElementById('kiosk-type-toggle-a');
    const typeBox = document.getElementById('kiosk-typebox-a');
    if (typeToggle && typeBox) {
      typeToggle.addEventListener('click', () => {
        typeBox.hidden = !typeBox.hidden;
        if (!typeBox.hidden) textInput.focus();
      });
    }

    const repeatBtn = document.getElementById('kiosk-repeat-a');
    if (repeatBtn) {
      repeatBtn.addEventListener('click', () => {
        if (this.lastQuestion) VoiceEngine.speak(this.lastQuestion);
      });
    }

    const changeAnswerBtn = document.getElementById('kiosk-change-answer-a');
    if (changeAnswerBtn) {
      changeAnswerBtn.addEventListener('click', () => this.resetSession());
    }

    const micBtn = document.getElementById('kiosk-mic-btn-a');
    if (micBtn) this.bindMic(micBtn, textInput);
  },

  bindLayoutBControls() {
    const sendBtn = document.getElementById('kiosk-send-btn-b');
    const textInput = document.getElementById('kiosk-text-input-b');
    if (sendBtn && textInput) {
      sendBtn.addEventListener('click', () => this.handleSendMessage(null, textInput));
      textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.handleSendMessage(null, textInput);
      });
    }

    const repeatBtn = document.getElementById('kiosk-repeat-b');
    if (repeatBtn) {
      repeatBtn.addEventListener('click', () => {
        if (this.lastQuestion) VoiceEngine.speak(this.lastQuestion);
      });
    }

    const micBtn = document.getElementById('kiosk-mic-btn-b');
    if (micBtn) this.bindMic(micBtn, textInput);
  },

  bindMic(micBtn, textInput) {
    micBtn.addEventListener('click', () => {
      VoiceEngine.toggleListening(
        (transcript, isFinal) => {
          if (textInput) textInput.value = transcript;
          if (isFinal && transcript.trim()) {
            this.handleSendMessage(transcript, textInput);
          }
        },
        (isRecording) => {
          micBtn.classList.toggle('recording', isRecording);
        }
      );
    });
  },

  /**
   * Show only the active layout's card and mark its toggle button active.
   */
  applyLayout() {
    const a = document.getElementById('kiosk-layout-a');
    const b = document.getElementById('kiosk-layout-b');
    if (a) a.hidden = this.layout !== 'a';
    if (b) b.hidden = this.layout !== 'b';

    document.querySelectorAll('#kiosk-layout-toggle .segmented-toggle-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.layout === this.layout);
    });
  },

  /**
   * Refresh static (non-AI-generated) copy for the chosen language.
   */
  applyLanguage() {
    const strings = STATIC_STRINGS[this.language] || STATIC_STRINGS.en;

    document.querySelectorAll('.kiosk-lang-toggle .lang-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === this.language);
    });

    const micLabel = document.querySelector('.kiosk-v2-mic-label');
    if (micLabel) micLabel.innerHTML = `${strings.tapToSpeak}<br><span>${strings.yourAnswer}</span>`;

    const typeToggle = document.getElementById('kiosk-type-toggle-a');
    if (typeToggle) typeToggle.querySelector('span:last-child').textContent = strings.typeInstead;

    ['kiosk-text-input-a', 'kiosk-text-input-b'].forEach(id => {
      const input = document.getElementById(id);
      if (input) input.placeholder = strings.placeholder;
    });
  },

  /**
   * Reset the intake interview
   */
  async resetSession() {
    this.conversationHistory = [];
    this.isInterviewComplete = false;
    this.redFlagUrgent = false;
    this.chiefComplaint = null;
    this.lastQuestion = '';
    this.lastOptions = [];

    this.updateEmergencyBanner(false);

    ['btn-finish-kiosk-a', 'btn-finish-kiosk-b'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.style.display = 'none';
    });

    const typeBox = document.getElementById('kiosk-typebox-a');
    if (typeBox) typeBox.hidden = true;

    // Trigger initial question
    await this.fetchNextStep();
  },

  /**
   * Apply the patient identity resolved at login (ABHA-verified or walk-in)
   * to the kiosk header, and start a fresh interview for them.
   */
  setPatientIdentity(identity = {}) {
    this.currentPatient = {
      abha_id: identity.abha_id || '',
      name: identity.name || 'Walk-in Patient',
      verified: !!identity.verified,
      dob: identity.dob || '',
      gender: identity.gender || ''
    };

    const nameDisplay = document.getElementById('kiosk-patient-name');
    const abhaDisplay = document.getElementById('kiosk-patient-abha');
    const statusBadge = document.getElementById('kiosk-patient-status');

    if (nameDisplay) nameDisplay.textContent = this.currentPatient.name;
    if (abhaDisplay) {
      abhaDisplay.textContent = this.currentPatient.verified
        ? `ABHA: ${this.currentPatient.abha_id}`
        : 'ABHA ID: Not Verified';
    }
    if (statusBadge) {
      if (this.currentPatient.verified) {
        statusBadge.className = 'badge badge-success';
        statusBadge.textContent = 'ABHA Verified';
      } else {
        statusBadge.className = 'badge badge-warning';
        statusBadge.textContent = 'Self Intake Mode';
      }
    }

    this.resetSession();
  },

  /**
   * Send patient message. `sourceInput`, when given, is cleared after sending.
   */
  async handleSendMessage(forcedText = null, sourceInput = null) {
    const text = forcedText || (sourceInput ? sourceInput.value.trim() : '');
    if (!text) return;

    if (sourceInput) sourceInput.value = '';

    if (this.chiefComplaint === null) {
      this.chiefComplaint = text;
    }

    this.conversationHistory.push({ role: 'patient', content: text });

    await this.fetchNextStep();
  },

  /**
   * Request next question from API and render it into the active layout.
   */
  async fetchNextStep() {
    const step = await ApiService.converse(this.conversationHistory);
    if (!step) return;

    if (step.is_red_flag_urgent && !this.redFlagUrgent) {
      this.redFlagUrgent = true;
      this.updateEmergencyBanner(true);
      VoiceEngine.playEmergencyChime();
      window.App.showToast('ALERT: Urgent clinical symptom detected!', 'danger');
    }

    if (step.next_question) {
      this.conversationHistory.push({ role: 'assistant', content: step.next_question });
      this.lastQuestion = step.next_question;
      VoiceEngine.speak(step.next_question);
    }

    this.lastOptions = step.quick_reply_options || [];
    this.isInterviewComplete = !!step.is_complete;

    this.renderCurrentStep();
  },

  /**
   * Render the current question/options/progress into whichever layout is active.
   */
  renderCurrentStep() {
    const patientTurns = this.conversationHistory.filter(t => t.role === 'patient').length;
    const currentStep = Math.min(patientTurns + 1, TOTAL_STEPS);

    this.renderLayoutA(currentStep);
    this.renderLayoutB(currentStep);

    ['btn-finish-kiosk-a', 'btn-finish-kiosk-b'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) btn.style.display = this.isInterviewComplete ? 'inline-flex' : 'none';
    });
  },

  renderLayoutA(currentStep) {
    const label = document.getElementById('kiosk-progress-label-a');
    if (label) label.textContent = `Question ${currentStep} of ${TOTAL_STEPS}`;

    const bar = document.getElementById('kiosk-progress-bar-a');
    if (bar) {
      bar.innerHTML = '';
      for (let i = 1; i <= TOTAL_STEPS; i++) {
        const seg = document.createElement('span');
        if (i <= currentStep) seg.classList.add('filled');
        bar.appendChild(seg);
      }
    }

    const questionEl = document.getElementById('kiosk-question-a');
    if (questionEl) questionEl.textContent = this.lastQuestion;

    const grid = document.getElementById('kiosk-answer-grid-a');
    if (grid) {
      grid.innerHTML = '';
      this.lastOptions.forEach(opt => {
        const tile = document.createElement('button');
        tile.type = 'button';
        tile.className = 'kiosk-v2-answer-tile';
        tile.textContent = opt;
        tile.addEventListener('click', () => this.handleSendMessage(opt));
        grid.appendChild(tile);
      });
    }

    const sidebarCard = document.getElementById('kiosk-sidebar-card-a');
    const sidebarValue = document.getElementById('kiosk-sidebar-value-a');
    if (sidebarCard && sidebarValue) {
      if (this.chiefComplaint) {
        sidebarCard.hidden = false;
        sidebarValue.textContent = this.chiefComplaint;
      } else {
        sidebarCard.hidden = true;
      }
    }

    const hint = document.getElementById('kiosk-sidebar-hint-a');
    if (hint) {
      const remaining = TOTAL_STEPS - currentStep;
      hint.textContent = this.isInterviewComplete
        ? 'All done. Review your summary below.'
        : remaining > 0
          ? `${remaining} more question${remaining === 1 ? '' : 's'}. You can stop any time and a staff member will help.`
          : 'Last question. You can stop any time and a staff member will help.';
    }
  },

  renderLayoutB(currentStep) {
    const label = document.getElementById('kiosk-progress-label-b');
    if (label) label.textContent = `Step ${currentStep} of ${TOTAL_STEPS}`;

    const questionEl = document.getElementById('kiosk-question-b');
    if (questionEl) questionEl.textContent = this.lastQuestion;

    const rows = document.getElementById('kiosk-answer-rows-b');
    if (rows) {
      rows.innerHTML = '';
      this.lastOptions.forEach((opt, idx) => {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'kiosk-v2-answer-row';
        row.innerHTML = `<span class="kiosk-v2-answer-row-num">${idx + 1}</span><span class="kiosk-v2-answer-row-text"></span>`;
        row.querySelector('.kiosk-v2-answer-row-text').textContent = opt;
        row.addEventListener('click', () => this.handleSendMessage(opt));
        rows.appendChild(row);
      });
    }

    const rail = document.getElementById('kiosk-steps-rail-b');
    if (rail) {
      rail.innerHTML = '';
      STEP_LABELS.forEach((stepLabel, idx) => {
        const stepNum = idx + 1;
        const item = document.createElement('div');
        item.className = 'kiosk-step-item';
        if (stepNum < currentStep) item.classList.add('done');
        else if (stepNum === currentStep) item.classList.add('current');

        const marker = document.createElement('div');
        marker.className = 'kiosk-step-marker';
        marker.textContent = stepNum < currentStep ? '✓' : String(stepNum);

        const label = document.createElement('span');
        label.textContent = stepNum === 1 && this.chiefComplaint ? this.chiefComplaint : stepLabel;

        item.appendChild(marker);
        item.appendChild(label);
        rail.appendChild(item);
      });
    }
  },

  updateEmergencyBanner(active) {
    const banner = document.getElementById('global-emergency-banner');
    if (banner) banner.style.display = active ? 'flex' : 'none';
  },

  /**
   * Generate unified summary and redirect to Physician Review / Waiting Room
   */
  async generateAndCompleteSummary() {
    const transcript = this.conversationHistory
      .map(t => `${t.role.toUpperCase()}: ${t.content}`)
      .join('\n');

    window.App.showToast('Synthesizing clinical summary...', 'info');

    const summary = await ApiService.generateSummary(
      transcript,
      [],
      this.currentPatient.abha_id,
      this.currentPatient
    );

    if (summary) {
      window.App.showToast('Clinical summary generated & queued!', 'success');
      // Trigger update on triage and physician tabs
      window.App.refreshAllData();
      // Switch view to Doctor Review
      window.App.switchView('doctor-review', summary.id);
    }
  }
};
