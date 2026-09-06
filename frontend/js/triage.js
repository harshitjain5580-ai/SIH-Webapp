/**
 * Triage Waiting Room & Priority Alert Controller
 * Real-time queue, red-flag triage badges, and staff alert acknowledgment.
 */

import { ApiService } from './api.js';
import { VoiceEngine } from './voice.js';

export const TriageModule = {
  pollingTimer: null,

  init() {
    this.bindEvents();
    this.loadData();
    this.startPolling();
  },

  bindEvents() {
    const refreshBtn = document.getElementById('btn-refresh-triage');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.loadData());
    }
  },

  startPolling() {
    if (this.pollingTimer) clearInterval(this.pollingTimer);
    // Poll every 8 seconds for new kiosk submissions
    this.pollingTimer = setInterval(() => this.loadData(true), 8000);
  },

  async loadData(silent = false) {
    const histories = await ApiService.getWaitingRoom();
    const alerts = await ApiService.getPriorityAlerts();

    this.renderStats(histories, alerts);
    this.renderQueueTable(histories);
    this.renderPriorityAlerts(alerts);

    // If unacknowledged alerts exist, show global banner
    if (alerts.length > 0) {
      const banner = document.getElementById('global-emergency-banner');
      if (banner) banner.style.display = 'flex';
      const text = document.getElementById('emergency-banner-text');
      if (text) {
        text.textContent = `CRITICAL ALERT: ${alerts.length} patient(s) with urgent red-flag symptoms require immediate triage!`;
      }
      if (!silent) VoiceEngine.playEmergencyChime();
    }
  },

  renderStats(histories, alerts) {
    const totalEl = document.getElementById('stat-total-intake');
    const redEl = document.getElementById('stat-red-flags');
    const standardEl = document.getElementById('stat-standard-queue');
    const ayushEl = document.getElementById('stat-ayush-count');

    if (totalEl) totalEl.textContent = histories.length;
    if (redEl) redEl.textContent = alerts.length;
    if (standardEl) standardEl.textContent = histories.filter(h => !h.red_flags_detected).length;
    if (ayushEl) ayushEl.textContent = histories.filter(h => h.ayush_parameters && h.ayush_parameters.prakriti).length;

    // Update sidebar badge
    const badge = document.getElementById('sidebar-triage-badge');
    if (badge) {
      badge.textContent = alerts.length > 0 ? `${alerts.length} Alert` : histories.length;
      badge.className = alerts.length > 0 ? 'badge badge-danger' : 'badge badge-primary';
    }
  },

  renderQueueTable(histories) {
    const tbody = document.getElementById('triage-queue-tbody');
    if (!tbody) return;

    if (!histories || histories.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align:center; padding:30px;" class="text-muted">
            No patients currently in the intake queue.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = histories.map(h => {
      const timeStr = new Date(h.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const name = h.patient_name || (h.abha_id ? h.abha_id.split('@')[0] : 'Patient');
      const isRed = h.red_flags_detected;

      return `
        <tr class="triage-queue-row ${isRed ? 'emergency-row' : ''}">
          <td>
            <strong>${timeStr}</strong>
          </td>
          <td>
            <div style="font-weight:700; color:var(--utd-color-textprimary);">${name}</div>
            <div style="font-size:0.75rem; color:var(--utd-color-textmuted);">${h.abha_id || h.id.slice(0, 8)}</div>
          </td>
          <td style="max-width:320px;">
            <div style="font-size:0.88rem; color:var(--utd-color-textsecondary); text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">
              ${h.chief_complaint || 'No complaint recorded'}
            </div>
          </td>
          <td>
            ${isRed 
              ? `<span class="badge badge-danger">🚨 RED FLAG URGENT</span>` 
              : `<span class="badge badge-success">✓ STABLE</span>`}
          </td>
          <td>
            <span class="badge badge-primary">${h.ayush_parameters?.prakriti || 'Standard'}</span>
          </td>
          <td style="text-align:right;">
            <button class="btn btn-secondary btn-sm btn-open-review" data-id="${h.id}">
              Review Summary →
            </button>
          </td>
        </tr>
      `;
    }).join('');

    // Attach click events
    tbody.querySelectorAll('.btn-open-review').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.dataset.id;
        if (id) window.App.switchView('doctor-review', id);
      });
    });
  },

  renderPriorityAlerts(alerts) {
    const container = document.getElementById('priority-alerts-panel');
    const list = document.getElementById('priority-alerts-list');
    if (!container || !list) return;

    if (!alerts || alerts.length === 0) {
      container.style.display = 'none';
      return;
    }

    container.style.display = 'block';
    list.innerHTML = alerts.map(a => `
      <div style="padding:14px; background:#FEF2F2; border:1px solid #FECACA; border-radius:var(--utd-border-radius); display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:10px;">
        <div>
          <div style="font-weight:700; color:#991B1B;">⚠️ ${a.patient_name || 'Emergency Patient'} (${a.abha_id || 'Walk-in'})</div>
          <div style="font-size:0.85rem; color:#7F1D1D; margin-top:2px;">${a.chief_complaint}</div>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="btn btn-danger btn-sm btn-ack-alert" data-id="${a.id}">
            Acknowledge Alert
          </button>
          <button class="btn btn-secondary btn-sm btn-open-review" data-id="${a.id}">
            View
          </button>
        </div>
      </div>
    `).join('');

    list.querySelectorAll('.btn-ack-alert').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const id = e.currentTarget.dataset.id;
        if (id) {
          await ApiService.acknowledgeAlert(id);
          window.App.showToast('Red-flag alert acknowledged.', 'success');
          this.loadData();
        }
      });
    });

    list.querySelectorAll('.btn-open-review').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.dataset.id;
        if (id) window.App.switchView('doctor-review', id);
      });
    });
  }
};
