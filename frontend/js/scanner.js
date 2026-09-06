/**
 * Medical Document Scanner & OCR Controller
 * Uploads prescription/lab images, parses diagnoses, medications, and highlights abnormal lab tests.
 */

import { ApiService } from './api.js';

export const ScannerModule = {
  lastExtractedDoc: null,

  init() {
    this.bindEvents();
  },

  bindEvents() {
    const dropzone = document.getElementById('scanner-dropzone');
    const fileInput = document.getElementById('scanner-file-input');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
      });

      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
      });

      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Sample Document Quick Buttons
    document.querySelectorAll('.sample-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        const type = e.target.dataset.sample;
        this.loadSampleDocument(type);
      });
    });

    // Merge into Summary Action
    const mergeBtn = document.getElementById('btn-merge-scanned-doc');
    if (mergeBtn) {
      mergeBtn.addEventListener('click', () => this.mergeExtractedDocument());
    }
  },

  async handleFileUpload(file) {
    window.App.showToast(`Analyzing ${file.name}...`, 'info');

    const result = await ApiService.uploadDocument(file);
    if (result && result.extracted_document) {
      this.lastExtractedDoc = result;
      this.renderExtractedFindings(result.extracted_document, file.name);
      window.App.showToast('Document analysis complete!', 'success');
    }
  },

  loadSampleDocument(type) {
    const sampleFile = new File(['mock_content'], `${type}_sample.jpg`, { type: 'image/jpeg' });
    this.handleFileUpload(sampleFile);
  },

  renderExtractedFindings(doc, filename) {
    const resultsContainer = document.getElementById('scanner-results-container');
    const emptyState = document.getElementById('scanner-empty-state');
    const filenameDisplay = document.getElementById('scanned-file-name');

    if (emptyState) emptyState.style.display = 'none';
    if (resultsContainer) resultsContainer.style.display = 'block';
    if (filenameDisplay) filenameDisplay.textContent = filename || 'Uploaded Medical Document';

    // Diagnoses Tags
    const diagBox = document.getElementById('scanned-diagnoses-list');
    if (diagBox) {
      diagBox.innerHTML = (doc.diagnoses || [])
        .map(d => `<span class="badge badge-primary">🩺 ${d}</span>`)
        .join(' ') || '<span class="text-muted">No explicit diagnosis found</span>';
    }

    // Investigations Table with Abnormal Badges
    const invTbody = document.getElementById('scanned-investigations-tbody');
    if (invTbody) {
      if (!doc.investigations || doc.investigations.length === 0) {
        invTbody.innerHTML = '<tr><td colspan="4" class="text-muted">No lab tests found in this document</td></tr>';
      } else {
        invTbody.innerHTML = doc.investigations.map(inv => `
          <tr class="${inv.is_abnormal ? 'abnormal-row' : ''}">
            <td><strong>${inv.test_name}</strong></td>
            <td>
              <span class="badge ${inv.is_abnormal ? 'badge-danger' : 'badge-success'}">
                ${inv.value}
              </span>
            </td>
            <td><span class="text-muted">${inv.reference_range || 'N/A'}</span></td>
            <td>
              ${inv.is_abnormal 
                ? '<span class="badge badge-danger">⚠️ ABNORMAL</span>' 
                : '<span class="badge badge-success">✓ NORMAL</span>'}
            </td>
          </tr>
        `).join('');
      }
    }

    // Medications Table
    const medTbody = document.getElementById('scanned-medications-tbody');
    if (medTbody) {
      if (!doc.medications || doc.medications.length === 0) {
        medTbody.innerHTML = '<tr><td colspan="3" class="text-muted">No active prescriptions detected</td></tr>';
      } else {
        medTbody.innerHTML = doc.medications.map(med => `
          <tr>
            <td><strong>💊 ${med.name}</strong></td>
            <td>${med.dosage}</td>
            <td><span class="badge badge-primary">${med.frequency}</span></td>
          </tr>
        `).join('');
      }
    }

    // Procedures
    const procBox = document.getElementById('scanned-procedures-list');
    if (procBox) {
      procBox.innerHTML = (doc.procedures || [])
        .map(p => `<span class="badge badge-warning">🔬 ${p}</span>`)
        .join(' ') || '<span class="text-muted">None specified</span>';
    }
  },

  async mergeExtractedDocument() {
    if (!this.lastExtractedDoc) return;

    window.App.showToast('Merging document findings into clinical summary...', 'info');

    const summary = await ApiService.generateSummary(
      'Document OCR intake merge.',
      [this.lastExtractedDoc],
      'PATIENT-DOC-' + Math.floor(1000 + Math.random() * 9000),
      { name: 'Document Intake Patient' }
    );

    if (summary) {
      window.App.showToast('Document record merged & added to triage queue!', 'success');
      window.App.refreshAllData();
      window.App.switchView('doctor-review', summary.id);
    }
  }
};
