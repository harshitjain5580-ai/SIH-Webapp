/**
 * Patient Intake Kiosk Controller
 * Multi-turn interview, SOCRATES pain extractor, AYUSH tracking, quick-reply touch chips, and red-flag alerts.
 */

import { ApiService } from './api.js';
import { VoiceEngine } from './voice.js';

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
  socratesState: {
    site: false,
    onset: false,
    character: false,
    radiation: false,
    associated: false,
    timing: false,
    exacerbating: false,
    severity: false
  },

  /**
   * Initialize Kiosk Event Listeners & UI
   */
  init() {
    this.bindEvents();
    this.resetSession();
  },

  bindEvents() {
    document.querySelectorAll('.gender-option-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.selectGender(btn.dataset.gender || 'Other or prefer not to say');
      });
    });

    // ABHA Verification
    const abhaBtn = document.getElementById('btn-verify-abha');
    const abhaInput = document.getElementById('kiosk-abha-input');
    if (abhaBtn && abhaInput) {
      abhaBtn.addEventListener('click', () => this.handleAbhaVerification());
      abhaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.handleAbhaVerification();
      });
    }

    // Quick Sample ABHA Buttons
    document.querySelectorAll('.sample-abha-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.target.dataset.abha;
        if (id && abhaInput) {
          abhaInput.value = id;
          this.handleAbhaVerification();
        }
      });
    });

    // Send Message Button & Enter Key
    const sendBtn = document.getElementById('kiosk-send-btn');
    const textInput = document.getElementById('kiosk-text-input');
    if (sendBtn && textInput) {
      sendBtn.addEventListener('click', () => this.handleSendMessage());
      textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.handleSendMessage();
      });
    }

    // Voice Microphone Button
    const micBtn = document.getElementById('kiosk-mic-btn');
    if (micBtn) {
      micBtn.addEventListener('click', () => {
        VoiceEngine.toggleListening(
          (transcript, isFinal) => {
            if (textInput) textInput.value = transcript;
            if (isFinal && transcript.trim()) {
              this.handleSendMessage();
            }
          },
          (isRecording) => {
            micBtn.classList.toggle('recording', isRecording);
            const wave = document.getElementById('kiosk-voice-wave');
            if (wave) wave.classList.toggle('recording', isRecording);
          }
        );
      });
    }

    // Restart Interview
    const resetBtn = document.getElementById('btn-reset-kiosk');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => this.resetSession());
    }

    // Generate Summary Early / Finish
    const finishBtn = document.getElementById('btn-finish-kiosk');
    if (finishBtn) {
      finishBtn.addEventListener('click', () => this.generateAndCompleteSummary());
    }
  },

  /**
   * Reset the intake interview
   */
  async resetSession() {
    this.conversationHistory = [];
    this.currentPatient.gender = '';
    this.isInterviewComplete = false;
    this.redFlagUrgent = false;
    this.socratesState = {
      site: false, onset: false, character: false, radiation: false,
      associated: false, timing: false, exacerbating: false, severity: false
    };

    this.updateSocratesTracker();
    this.updateEmergencyBanner(false);

    const chatBox = document.getElementById('kiosk-chat-stream');
    if (chatBox) chatBox.innerHTML = '';

    const quickBox = document.getElementById('kiosk-quick-replies');
    if (quickBox) quickBox.innerHTML = '';

    this.showGenderSelection();
  },

  showGenderSelection() {
    const selector = document.getElementById('kiosk-gender-selection');
    const chatBox = document.getElementById('kiosk-chat-stream');
    if (selector) selector.style.display = 'flex';
    if (chatBox) chatBox.style.display = 'none';
    this.renderQuickReplies([]);
  },

  async selectGender(gender) {
    this.currentPatient.gender = gender;
    const selector = document.getElementById('kiosk-gender-selection');
    const chatBox = document.getElementById('kiosk-chat-stream');
    if (selector) selector.style.display = 'none';
    if (chatBox) chatBox.style.display = 'flex';
    this.conversationHistory.push({
      role: 'patient',
      content: `Patient gender selected: ${gender}`
    });
    // Trigger initial question
    await this.fetchNextStep();
  },

  /**
   * ABHA Verification
   */
  async handleAbhaVerification() {
    const input = document.getElementById('kiosk-abha-input');
    const id = input ? input.value.trim() : '';
    if (!id) return;

    const result = await ApiService.verifyAbha(id);
    if (result && result.verified) {
      this.currentPatient = {
        abha_id: result.abha_id,
        name: result.patient_name || 'Verified Patient',
        verified: true,
        dob: result.date_of_birth,
        gender: result.gender
      };

      const nameDisplay = document.getElementById('kiosk-patient-name');
      const abhaDisplay = document.getElementById('kiosk-patient-abha');
      const statusBadge = document.getElementById('kiosk-patient-status');

      if (nameDisplay) nameDisplay.textContent = this.currentPatient.name;
      if (abhaDisplay) abhaDisplay.textContent = `ABHA: ${this.currentPatient.abha_id}`;
      if (statusBadge) {
        statusBadge.className = 'badge badge-success';
        statusBadge.textContent = 'ABHA Verified';
      }

      window.App.showToast(`ABHA verified for ${this.currentPatient.name}`, 'success');
    }
  },

  /**
   * Send patient message
   */
  async handleSendMessage(forcedText = null) {
    const input = document.getElementById('kiosk-text-input');
    const text = forcedText || (input ? input.value.trim() : '');
    if (!text) return;

    if (input) input.value = '';

    // Append patient turn
    this.addChatTurn('patient', text);
    this.conversationHistory.push({ role: 'patient', content: text });

    // Analyze text for SOCRATES & red flags
    this.analyzeSocratesTokens(text);

    // Fetch next kiosk step
    await this.fetchNextStep();
  },

  /**
   * Request next question from API
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
      this.addChatTurn('assistant', step.next_question);
      this.conversationHistory.push({ role: 'assistant', content: step.next_question });
      VoiceEngine.speak(step.next_question);
    }

    // Render quick-reply options
    this.renderQuickReplies(step.quick_reply_options || []);

    if (step.is_complete) {
      this.isInterviewComplete = true;
      const finishBtn = document.getElementById('btn-finish-kiosk');
      if (finishBtn) finishBtn.style.display = 'inline-flex';
    }
  },

  /**
   * Add chat bubble to transcript view
   */
  addChatTurn(role, content) {
    const chatBox = document.getElementById('kiosk-chat-stream');
    if (!chatBox) return;

    const turn = document.createElement('div');
    turn.className = `chat-turn ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'turn-avatar';
    avatar.textContent = role === 'assistant' ? 'Charaka' : 'PT';
    avatar.textContent = role === 'assistant' ? 'AI' : 'PT';

    const bubble = document.createElement('div');
    bubble.className = 'turn-bubble';
    bubble.innerHTML = `
      <p>${content}</p>
      <div class="turn-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
    `;

    turn.appendChild(avatar);
    turn.appendChild(bubble);
    chatBox.appendChild(turn);

    chatBox.scrollTop = chatBox.scrollHeight;
  },

  /**
   * Render Touch Quick-Reply Option Chips
   */
  renderQuickReplies(options) {
    const container = document.getElementById('kiosk-quick-replies');
    if (!container) return;

    container.innerHTML = '';
    if (!options || options.length === 0) return;

    options.forEach(opt => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'quick-reply-btn';
      btn.innerHTML = `<span>💬</span> <span>${opt}</span>`;
      btn.addEventListener('click', () => {
        this.handleSendMessage(opt);
      });
      container.appendChild(btn);
    });
  },

  /**
   * Real-time SOCRATES token extraction heuristics
   */
  analyzeSocratesTokens(text) {
    const t = text.toLowerCase();
    if (/(chest|head|stomach|abdomen|knee|back|throat|joint|pet|dard|gale)/.test(t)) this.socratesState.site = true;
    if (/(sudden|gradual|started|today|yesterday|days|months|hours|subah|kal)/.test(t)) this.socratesState.onset = true;
    if (/(burning|sharp|dull|throbbing|crushing|heavy|ache|pressure|jalan)/.test(t)) this.socratesState.character = true;
    if (/(radiat|spread|shoulder|arm|jaw|neck|back)/.test(t)) this.socratesState.radiation = true;
    if (/(nausea|sweat|fever|cough|vomit|dizzy|belch|heartburn|chakkar)/.test(t)) this.socratesState.associated = true;
    if (/(constant|intermittent|morning|night|after meal|khana)/.test(t)) this.socratesState.timing = true;
    if (/(worse|better|spicy|walking|rest|antacid|tablet|stairs)/.test(t)) this.socratesState.exacerbating = true;
    if (/(\d+\/10|severity|mild|moderate|severe|unbearable)/.test(t)) this.socratesState.severity = true;

    this.updateSocratesTracker();
  },

  updateSocratesTracker() {
    for (const [key, active] of Object.entries(this.socratesState)) {
      const el = document.getElementById(`soc-${key}`);
      if (el) el.classList.toggle('active', active);
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
