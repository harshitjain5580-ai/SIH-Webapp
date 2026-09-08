/**
 * Main Application Orchestrator & View Switcher
 */

import { ApiService } from './api.js';
import { VoiceEngine } from './voice.js';
import { KioskModule } from './kiosk.js';
import { ScannerModule } from './scanner.js';
import { TriageModule } from './triage.js';
import { DoctorModule } from './doctor.js';

export const App = {
  currentView: 'kiosk',

  init() {
    this.bindGlobalNavigation();
    this.bindAccessibilityControls();
    this.checkBackendConnection();

    // Initialize modules
    KioskModule.init();
    ScannerModule.init();
    TriageModule.init();
    DoctorModule.init();

    // Expose to window for inline onclick handlers and cross-module calls
    window.App = this;
  },

  bindGlobalNavigation() {
    // Nav Items
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const view = item.dataset.view;
        if (view) this.switchView(view);
      });
    });

    // Mobile Sidebar Toggle
    const mobileBtn = document.getElementById('mobile-toggle-btn');
    const sidebar = document.getElementById('app-sidebar');
    if (mobileBtn && sidebar) {
      mobileBtn.addEventListener('click', () => {
        sidebar.classList.toggle('mobile-open');
      });
    }
  },

  bindAccessibilityControls() {
    // Large Kiosk Font Toggle (for elderly/low-literacy accessibility)
    const largeFontBtn = document.getElementById('btn-toggle-kiosk-font');
    if (largeFontBtn) {
      largeFontBtn.addEventListener('click', () => {
        document.body.classList.toggle('kiosk-large-text');
        const isActive = document.body.classList.contains('kiosk-large-text');
        largeFontBtn.classList.toggle('active', isActive);
        this.showToast(isActive ? 'Large Kiosk Text Enabled (Elderly Mode)' : 'Standard Text Mode', 'info');
      });
    }

    // Voice Master Toggle (TTS read-aloud)
    const voiceToggleBtn = document.getElementById('btn-toggle-voice-assistant');
    if (voiceToggleBtn) {
      voiceToggleBtn.addEventListener('click', () => {
        VoiceEngine.ttsEnabled = !VoiceEngine.ttsEnabled;
        voiceToggleBtn.classList.toggle('active', VoiceEngine.ttsEnabled);
        this.showToast(VoiceEngine.ttsEnabled ? 'Voice Assistant Enabled' : 'Voice Assistant Muted', 'info');
      });
    }
  },

  async checkBackendConnection() {
    const isLive = await ApiService.checkHealth();
    const dot = document.getElementById('backend-status-dot');
    const label = document.getElementById('backend-status-label');

    if (dot && label) {
      if (isLive) {
        dot.className = 'status-dot';
        label.textContent = 'Connected';
      } else {
        dot.className = 'status-dot offline';
        label.textContent = 'Offline mode';
      }
    }
  },

  switchView(viewName, recordId = null) {
    this.currentView = viewName;

    // Update Nav Active State
    document.querySelectorAll('.nav-item').forEach(item => {
      item.classList.toggle('active', item.dataset.view === viewName);
    });

    // Switch View Panels
    document.querySelectorAll('.view-panel').forEach(panel => {
      panel.classList.toggle('active', panel.id === `view-${viewName}`);
    });

    // Close mobile sidebar if open
    const sidebar = document.getElementById('app-sidebar');
    if (sidebar) sidebar.classList.remove('mobile-open');

    // View specific actions
    if (viewName === 'triage') {
      TriageModule.loadData();
    } else if (viewName === 'doctor-review') {
      if (recordId) {
        DoctorModule.loadRecord(recordId);
      } else if (!DoctorModule.currentRecord) {
        // Load latest available record
        ApiService.getWaitingRoom().then(histories => {
          if (histories && histories.length > 0) {
            DoctorModule.loadRecord(histories[0].id);
          }
        });
      }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  refreshAllData() {
    TriageModule.loadData(true);
  },

  showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icon = type === 'success' ? '✓' : (type === 'danger' ? '⚠️' : 'ℹ️');
    const iconSpan = document.createElement('span');
    iconSpan.textContent = icon;
    const messageSpan = document.createElement('span');
    messageSpan.textContent = message;
    toast.append(iconSpan, document.createTextNode(' '), messageSpan);

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});
