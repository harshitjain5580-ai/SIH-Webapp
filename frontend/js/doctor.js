/**
 * Doctor / Physician Review & Hospital Integration Controller
 * Structured summary review, inline draft editing, and ABHA / HIS FHIR push.
 */

import { ApiService } from './api.js';

export const DoctorModule = {
  currentRecord: null,

  init() {
    this.bindEvents();
  },

  bindEvents() {
    const saveBtn = document.getElementById('btn-save-physician-edits');
    if (saveBtn) {
      saveBtn.addEventListener('click', () => this.handleSaveEdits());
    }

    const hisPushBtn = document.getElementById('btn-push-to-his');
    if (hisPushBtn) {
      hisPushBtn.addEventListener('click', () => this.handlePushToHis());
    }

    const addMedBtn = document.getElementById('btn-add-medication-row');
    if (addMedBtn) {
      addMedBtn.addEventListener('click', () => this.addMedicationRow('', '', ''));
    }

    const closeHisModalBtn = document.getElementById('btn-close-his-modal');
    if (closeHisModalBtn) {
      closeHisModalBtn.addEventListener('click', () => {
        document.getElementById('his-success-modal').classList.remove('open');
      });
    }
  },

  async loadRecord(recordId) {
    window.App.showToast('Loading clinical intake record...', 'info');
    const record = await ApiService.getPatientHistory(recordId);
    if (!record) {
      window.App.showToast('This patient record is not available right now.', 'danger');
      return;
    }

    this.currentRecord = record;
    this.renderRecord(record);
  },

  renderRecord(r) {
    const headerName = document.getElementById('doctor-patient-name');
    const headerAbha = document.getElementById('doctor-patient-abha');
    const headerStatus = document.getElementById('doctor-redflag-status');

    if (headerName) headerName.textContent = r.patient_name || 'Patient Summary';
    if (headerAbha) headerAbha.textContent = `ID: ${r.id.slice(0, 13)}... | ABHA: ${r.abha_id || 'Not linked'}`;

    if (headerStatus) {
      if (r.red_flags_detected) {
        headerStatus.className = 'badge badge-danger';
        headerStatus.textContent = '🚨 RED FLAG DETECTED';
      } else {
        headerStatus.className = 'badge badge-success';
        headerStatus.textContent = '✓ STABLE / NO RED FLAGS';
      }
    }

    // Chief Complaint
    const ccInput = document.getElementById('doc-chief-complaint');
    if (ccInput) ccInput.value = r.chief_complaint || '';

    // HPI SOCRATES Narrative
    const hpiInput = document.getElementById('doc-hpi-socrates');
    if (hpiInput) hpiInput.value = r.hpi_socrates || '';

    // Past Medical History
    const pmhBox = document.getElementById('doc-past-history-list');
    if (pmhBox) {
      pmhBox.innerHTML = (r.past_medical_history || [])
        .map(h => `<span class="badge badge-primary">📋 ${h}</span>`)
        .join(' ') || '<span class="text-muted">No prior chronic illnesses noted</span>';
    }

    // Medications Table Rows
    const medContainer = document.getElementById('doc-medications-list');
    if (medContainer) {
      medContainer.innerHTML = '';
      const meds = r.current_medications || [];
      if (meds.length === 0) {
        this.addMedicationRow('', '', '');
      } else {
        meds.forEach(m => this.addMedicationRow(m.name, m.dosage, m.frequency));
      }
    }

    // AYUSH Dashavidha Pariksha Grid
    const ayushGrid = document.getElementById('doc-ayush-parameters-grid');
    if (ayushGrid && r.ayush_parameters) {
      const p = r.ayush_parameters;
      const params = [
        { label: 'Prakriti (Constitution)', val: p.prakriti },
        { label: 'Vikriti (Doshic Imbalance)', val: p.vikriti },
        { label: 'Sara (Tissue Quality)', val: p.sara || 'Madhyama' },
        { label: 'Samhanana (Compactness)', val: p.samhanana || 'Madhyama' },
        { label: 'Pramana (Proportions)', val: p.pramana || 'Normal' },
        { label: 'Satmya (Suitability)', val: p.satmya || 'Mishra' },
        { label: 'Sattva (Mental Resilience)', val: p.sattva || 'Madhyama' },
        { label: 'Ahara Shakti (Digestion/Agni)', val: p.ahara_shakti || 'Madhyama' },
        { label: 'Vyayama (Exercise Capacity)', val: p.vyayama_shakti || 'Madhyama' },
        { label: 'Vaya (Age Stage)', val: p.vaya || 'Madhyama' }
      ];

      ayushGrid.innerHTML = params.map(item => `
        <div class="ayush-param-card">
          <strong>${item.label}</strong>
          <span>${item.val || 'Not assessed'}</span>
        </div>
      `).join('');
    }
  },

  addMedicationRow(name = '', dosage = '', frequency = '') {
    const container = document.getElementById('doc-medications-list');
    if (!container) return;

    const row = document.createElement('div');
    row.className = 'med-row';
    row.style.cssText = 'display:flex; gap:10px; margin-bottom:8px; align-items:center;';

    row.innerHTML = `
      <input type="text" class="form-control med-name" placeholder="Medication name" value="${name}" style="flex:2;">
      <input type="text" class="form-control med-dosage" placeholder="Dosage (e.g. 500mg)" value="${dosage}" style="flex:1;">
      <input type="text" class="form-control med-freq" placeholder="Frequency (e.g. BD)" value="${frequency}" style="flex:1.5;">
      <button type="button" class="btn btn-secondary btn-icon-only btn-remove-med" style="color:var(--utd-color-danger);" title="Remove">✕</button>
    `;

    row.querySelector('.btn-remove-med').addEventListener('click', () => {
      row.remove();
    });

    container.appendChild(row);
  },

  async handleSaveEdits() {
    if (!this.currentRecord) return;

    const cc = document.getElementById('doc-chief-complaint')?.value;
    const hpi = document.getElementById('doc-hpi-socrates')?.value;

    const meds = [];
    document.querySelectorAll('#doc-medications-list .med-row').forEach(row => {
      const name = row.querySelector('.med-name')?.value.trim();
      const dosage = row.querySelector('.med-dosage')?.value.trim();
      const freq = row.querySelector('.med-freq')?.value.trim();
      if (name) {
        meds.push({ name, dosage: dosage || 'Standard', frequency: freq || 'As directed' });
      }
    });

    const patch = {
      chief_complaint: cc,
      hpi_socrates: hpi,
      current_medications: meds
    };

    window.App.showToast('Saving physician amendments...', 'info');
    const updated = await ApiService.updatePatientHistory(this.currentRecord.id, patch);

    if (updated) {
      this.currentRecord = updated;
      window.App.showToast('Clinical history amendments saved successfully!', 'success');
      window.App.refreshAllData();
    } else {
      window.App.showToast('We could not save the changes. Please try again.', 'danger');
    }
  },

  async handlePushToHis() {
    if (!this.currentRecord) return;

    window.App.showToast('Submitting record to Hospital Information System...', 'info');

    const abhaId = this.currentRecord.abha_id || 'abha.patient@abdm';
    const result = await ApiService.pushToHis(this.currentRecord.id, abhaId);

    if (result) {
      // Show confirmation modal
      const modal = document.getElementById('his-success-modal');
      const details = document.getElementById('his-modal-details');

      if (details) {
        details.innerHTML = `
          <div style="padding:14px; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:var(--utd-border-radius); margin:16px 0;">
            <p><strong>HIS Record ID:</strong> <code>${result.his_record_id}</code></p>
            <p><strong>Linked ABHA ID:</strong> ${result.abha_id}</p>
            <p><strong>FHIR Submission Status:</strong> <span class="badge badge-success">${result.status.toUpperCase()}</span></p>
            <p><strong>Timestamp:</strong> ${new Date().toLocaleString()}</p>
          </div>
          <p style="font-size:0.9rem; color:var(--utd-color-textsecondary);">
            The physician-reviewed clinical history and AYUSH Dashavidha Pariksha findings are now linked to the hospital EHR and available in the patient's Personal Health Record.
          </p>
        `;
      }

      if (modal) modal.classList.add('open');
      window.App.showToast('Successfully pushed to Hospital Information System!', 'success');
    }
  }
};
