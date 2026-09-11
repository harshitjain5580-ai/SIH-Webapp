/**
 * MediKiosk API Client & State Layer
 * Connects directly to FastAPI endpoints with automatic offline/simulation fallback.
 */

const API_BASE = (window.location.protocol.startsWith('http') && (window.location.port === '8000' || window.location.port === ''))
  ? window.location.origin
  : 'http://127.0.0.1:8000';

// Mock in-memory & localStorage store for standalone demo testing
const STORAGE_KEY = 'medikiosk_histories_v1';

function getStoredHistories() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.warn('Storage read error:', e);
  }
  
  // Default sample records if empty (strictly conforming to schema in main.py)
  const defaultRecords = [
    {
      id: 'd9b23f81-54c7-4b68-963a-23910c22fa10',
      created_at: new Date(Date.now() - 1000 * 60 * 12).toISOString(),
      chief_complaint: 'Severe chest discomfort and shortness of breath for 2 hours.',
      hpi_socrates: 'Site: Mid-sternal chest. Onset: Sudden while climbing stairs 2 hours ago. Character: Crushing, heavy pressure. Radiation: Radiating to left shoulder and jaw. Associated: Diaphoresis, nausea, mild dyspnoea. Time course: Constant and worsening. Exacerbating: Exertion; Relieving: None. Severity: 8/10.',
      past_medical_history: ['Hypertension (5 years)', 'Hyperlipidemia'],
      current_medications: [
        { name: 'Amlodipine', dosage: '5mg', frequency: 'Once daily (morning)' },
        { name: 'Atorvastatin', dosage: '20mg', frequency: 'Once daily (night)' }
      ],
      ayush_parameters: {
        prakriti: 'Pitta-Vata',
        vikriti: 'Pitta-Vata Vriddhi (Agitation, heat, chest congestion)',
        sara: 'Madhyama (Moderate)',
        samhanana: 'Madhyama (Medium build)',
        pramana: 'Pramana yukta (Well proportioned)',
        satmya: 'Mishra satmya',
        sattva: 'Avara (Anxious / Low psychic threshold)',
        ahara_shakti: 'Manda Agni (Sluggish appetite)',
        vyayama_shakti: 'Avara (Low stamina)',
        vaya: 'Madhyama (Adult 52 yrs)'
      },
      red_flags_detected: true,
      alert_acknowledged: false,
      patient_name: 'Rajesh Kumar',
      abha_id: 'rajesh.kumar@abdm',
      triage_level: 'Immediate'
    },
    {
      id: 'a47c19e3-82a1-4321-bf90-f9231849a901',
      created_at: new Date(Date.now() - 1000 * 60 * 35).toISOString(),
      chief_complaint: 'Burning epigastric pain and sour belching after oily meals.',
      hpi_socrates: 'Site: Epigastric region. Onset: Gradual over 3 weeks. Character: Burning, gnawing ache. Radiation: Upward into retrosternal area. Associated: Heartburn, water brash, mild nausea. Time course: Intermittent, worse 1 hour post meals. Exacerbating: Spicy/fried food; Relieving: Cold milk, antacid. Severity: 5/10.',
      past_medical_history: ['Acid peptic disease', 'No surgical history'],
      current_medications: [
        { name: 'Pantoprazole', dosage: '40mg', frequency: 'Before breakfast' }
      ],
      ayush_parameters: {
        prakriti: 'Pitta-Kapha',
        vikriti: 'Pitta Vidagdha (Amlapitta, excessive acid secretion)',
        sara: 'Pravara (Good muscle/tissue quality)',
        samhanana: 'Samhanana yukta (Compact build)',
        pramana: 'Normal',
        satmya: 'Katu-Amla asatmya (Sensitive to spicy/sour)',
        sattva: 'Madhyama (Moderate resilience)',
        ahara_shakti: 'Tikshnagni with Vidaha (Sharp appetite with burning)',
        vyayama_shakti: 'Madhyama (Moderate)',
        vaya: 'Yuva (34 yrs)'
      },
      red_flags_detected: false,
      alert_acknowledged: false,
      patient_name: 'Sunita Sharma',
      abha_id: 'sunita.sharma@abdm',
      triage_level: 'Standard'
    },
    {
      id: 'f1e8432a-1122-4455-8899-aabbccddeeff',
      created_at: new Date(Date.now() - 1000 * 60 * 75).toISOString(),
      chief_complaint: 'Bilateral knee stiffness and joint pain, worse in morning.',
      hpi_socrates: 'Site: Both knees (Right > Left). Onset: Insidious onset 6 months ago. Character: Dull aching, crepitus on flexion. Radiation: Localized to patellofemoral joint. Associated: Early morning stiffness lasting 15 mins, difficulty squatting. Time course: Progressive. Exacerbating: Walking downstairs, cold weather; Relieving: Hot fermentation, rest. Severity: 6/10.',
      past_medical_history: ['Osteoarthritis grade II', 'Type 2 Diabetes Mellitus (well controlled)'],
      current_medications: [
        { name: 'Metformin', dosage: '500mg', frequency: 'Twice daily after meals' },
        { name: 'Paracetamol', dosage: '650mg', frequency: 'As needed for joint pain' }
      ],
      ayush_parameters: {
        prakriti: 'Vata-Kapha',
        vikriti: 'Sandhigata Vata (Vata accumulation in joint spaces)',
        sara: 'Asthi Sara Avara (Depleted bone/joint tissue density)',
        samhanana: 'Sthula (Heavy / mild overweight)',
        pramana: 'Normal',
        satmya: 'Sheeta asatmya (Cold intolerant)',
        sattva: 'Pravara (Patient, calm demeanor)',
        ahara_shakti: 'Madhyama',
        vyayama_shakti: 'Avara (Limited due to joint stiffness)',
        vaya: 'Vriddha (Senior 64 yrs)'
      },
      red_flags_detected: false,
      alert_acknowledged: false,
      patient_name: 'Harish Chandra',
      abha_id: 'harish.c@abdm',
      triage_level: 'Standard'
    }
  ];

  localStorage.setItem(STORAGE_KEY, JSON.stringify(defaultRecords));
  return defaultRecords;
}

function saveHistories(histories) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(histories));
  } catch (e) {
    console.warn('Storage write error:', e);
  }
}

export const ApiService = {
  /**
   * Health check / connection detector
   */
  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET', mode: 'cors' });
      return res.ok;
    } catch {
      return false;
    }
  },

  /**
   * Mock ABHA Verification: POST /abdm/verify-abha
   */
  async verifyAbha(abhaId) {
    try {
      const res = await fetch(`${API_BASE}/abdm/verify-abha`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ abha_id: abhaId })
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.log('Using local ABHA verification simulation:', e.message);
    }

    // Realistic simulation fallback
    const names = {
      '91-9876543210@abdm': 'Aarav Sharma',
      'priya.patel@abdm': 'Priya Patel',
      'vikram.singh@abdm': 'Vikram Singh',
      'ananya.das@abdm': 'Ananya Das'
    };

    const cleanId = abhaId.trim().toLowerCase();
    const patientName = names[cleanId] || (abhaId.includes('@') ? abhaId.split('@')[0].replace('.', ' ').toUpperCase() : 'Patient ' + abhaId.slice(-4));

    return {
      abha_id: abhaId,
      verified: true,
      patient_name: patientName,
      date_of_birth: '1988-06-15',
      gender: 'Female'
    };
  },

  /**
   * Adaptive conversation turn: POST /converse
   */
  async converse(historyTurns) {
    try {
      const res = await fetch(`${API_BASE}/converse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: historyTurns })
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.log('Using local interview engine simulation:', e.message);
    }

    // Local conversational AI simulation conforming to ConversationStep
    const turnCount = historyTurns.filter(t => t.role === 'patient').length;
    const lastPatientMessage = (historyTurns.slice().reverse().find(t => t.role === 'patient')?.content || '').toLowerCase();

    // Red flag trigger detection
    const isRedFlag = /(chest pain|heart|breathing|breathless|faint|unconscious|stroke|paralysis|crushing pain|blood|dizziness|सीने में दर्द|सांस फूल)/i.test(lastPatientMessage);

    if (turnCount === 0) {
      return {
        next_question: "Namaste! Welcome to MediKiosk. Please tell me what brings you in today or what primary symptom you are experiencing?",
        quick_reply_options: [
          "Fever & Cough (बुखार और खांसी)",
          "Chest Discomfort (सीने में भारीपन)",
          "Stomach Pain (पेट दर्द)",
          "Joint & Knee Pain (जोड़ों का दर्द)"
        ],
        is_complete: false,
        is_red_flag_urgent: false
      };
    }

    if (turnCount === 1) {
      if (isRedFlag) {
        return {
          next_question: "I have noted this immediately. Is the pain or breathlessness radiating to your left arm, neck, or jaw right now?",
          quick_reply_options: ["Yes, radiating to arm/jaw", "No, localized in center", "Severe sweating / nausea"],
          is_complete: false,
          is_red_flag_urgent: true
        };
      }
      return {
        next_question: "Understood. When did this begin, and how would you describe the sensation (e.g., sharp, dull ache, burning, or throbbing)?",
        quick_reply_options: ["Started today suddenly", "2-3 days ago gradually", "Constant dull ache", "Sharp & burning"],
        is_complete: false,
        is_red_flag_urgent: false
      };
    }

    if (turnCount === 2) {
      return {
        next_question: "Are there any specific activities or foods that make it worse or better? Also, on a scale of 1 to 10, how severe is it?",
        quick_reply_options: ["Severity 3-4 (Mild)", "Severity 5-6 (Moderate)", "Severity 7-8 (Severe)", "Worse after eating"],
        is_complete: false,
        is_red_flag_urgent: isRedFlag
      };
    }

    if (turnCount === 3) {
      return {
        next_question: "Are you currently taking any daily medicines or do you have known allergies to any drugs (like penicillin, sulfa, or pain killers)?",
        quick_reply_options: ["No daily medicines", "Taking BP / Sugar tablets", "Allergic to Penicillin", "No known allergies"],
        is_complete: false,
        is_red_flag_urgent: isRedFlag
      };
    }

    if (turnCount === 4) {
      return {
        next_question: "Regarding your appetite, sleep, and bowel movements recently—how have they been?",
        quick_reply_options: ["Normal digestion & sleep", "Loss of appetite (भूख कम)", "Disturbed sleep due to pain", "Acidity & constipation"],
        is_complete: false,
        is_red_flag_urgent: isRedFlag
      };
    }

    // Interview completion
    return {
      next_question: "Thank you. I have captured your clinical history, pain characteristics (SOCRATES), and constitutional assessment. Preparing your clinical summary for the physician...",
      quick_reply_options: [],
      is_complete: true,
      is_red_flag_urgent: isRedFlag
    };
  },

  /**
   * Upload Document & OCR: POST /upload-document
   */
  async uploadDocument(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE}/upload-document`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.log('Using local OCR simulation:', e.message);
    }

    // Realistic document extraction simulation
    const fileName = file.name.toLowerCase();
    let sampleData;

    if (fileName.includes('blood') || fileName.includes('lab') || fileName.includes('report')) {
      sampleData = {
        diagnoses: ['Microcytic Hypochromic Anemia', 'Vitamin D Deficiency'],
        medications: [
          { name: 'Ferrous Ascorbate', dosage: '100mg', frequency: 'Once daily after dinner' },
          { name: 'Cholecalciferol', dosage: '60,000 IU', frequency: 'Once weekly for 8 weeks' }
        ],
        investigations: [
          { test_name: 'Hemoglobin (Hb)', value: '9.4 g/dL', reference_range: '12.0 - 15.5 g/dL', is_abnormal: true },
          { test_name: 'Total RBC Count', value: '3.8 mil/uL', reference_range: '4.0 - 5.2 mil/uL', is_abnormal: true },
          { test_name: 'Serum Ferritin', value: '11 ng/mL', reference_range: '20 - 200 ng/mL', is_abnormal: true },
          { test_name: 'Serum Vitamin D (25-OH)', value: '14.2 ng/mL', reference_range: '30.0 - 100.0 ng/mL', is_abnormal: true },
          { test_name: 'Fasting Blood Sugar', value: '94 mg/dL', reference_range: '70 - 100 mg/dL', is_abnormal: false }
        ],
        procedures: ['Routine venipuncture and complete hemogram']
      };
    } else {
      sampleData = {
        diagnoses: ['Acute Bronchitis', 'Mild Dehydration'],
        medications: [
          { name: 'Amoxicillin-Clavulanate', dosage: '625mg', frequency: 'Twice daily for 5 days' },
          { name: 'Levocetirizine + Montelukast', dosage: '5mg/10mg', frequency: 'Once daily at bedtime' },
          { name: 'Paracetamol', dosage: '650mg', frequency: 'Thrice daily if fever > 100°F' }
        ],
        investigations: [
          { test_name: 'Chest X-Ray (PA View)', value: 'Mild peribronchial thickening, no consolidation', reference_range: 'Normal clear lung fields', is_abnormal: true },
          { test_name: 'Total Leucocyte Count (TLC)', value: '11,800 /cu.mm', reference_range: '4,000 - 11,000 /cu.mm', is_abnormal: true },
          { test_name: 'SpO2 Room Air', value: '97%', reference_range: '95% - 100%', is_abnormal: false }
        ],
        procedures: ['Nebulization with Levosalbutamol given in triage']
      };
    }

    return {
      storage_path: `simulated_docs/${Date.now()}_${file.name}`,
      extracted_document: sampleData
    };
  },

  /**
   * Generate Summary & Persist: POST /generate-summary
   */
  async generateSummary(transcript, documents = [], patientId = null, patientMeta = {}) {
    try {
      const res = await fetch(`${API_BASE}/generate-summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: transcript,
          documents: documents,
          patient_id: patientId,
          patient_name: patientMeta.name || 'Walk-in Patient',
          abha_id: patientMeta.abha_id || (patientId || 'patient@abdm'),
          triage_level: patientMeta.triage_level || null
        })
      });
      if (res.ok) {
        const record = await res.json();
        // Update local cache
        const all = getStoredHistories();
        all.unshift(record);
        saveHistories(all);
        return record;
      }
    } catch (e) {
      console.log('Using local summary generation simulation:', e.message);
    }

    // Local summary compilation
    const lowerTrans = (transcript || '').toLowerCase();
    const isUrgent = /(chest pain|heart|breathing|breathless|faint|unconscious|stroke|सीने)/i.test(lowerTrans);

    const newRecord = {
      id: 'uuid-' + Math.random().toString(36).substring(2, 11) + '-' + Date.now().toString(36),
      created_at: new Date().toISOString(),
      chief_complaint: isUrgent
        ? 'Acute chest discomfort with exertion and diaphoresis.'
        : 'Intermittent abdominal discomfort and epigastric burning for 4 days.',
      hpi_socrates: isUrgent
        ? 'Site: Central substernal chest. Onset: Acute onset 2 hours ago. Character: Heavy squeezing pressure. Radiation: Radiates to left arm. Associated: Dyspnoea and cold sweating. Time course: Persistent. Exacerbating: Physical movement; Relieving: Rest. Severity: 8/10.'
        : 'Site: Epigastrium and upper abdomen. Onset: Subacute 4 days ago. Character: Burning ache. Radiation: Non-radiating. Associated: Mild nausea, dyspepsia, bloating. Time course: Waxes and wanes post meals. Exacerbating: Fatty food; Relieving: Antacids. Severity: 5/10.',
      past_medical_history: ['Known seasonal allergies', 'No prior major surgeries'],
      current_medications: [
        { name: 'Pantoprazole', dosage: '40mg', frequency: 'Once daily before breakfast' },
        { name: 'Multivitamin', dosage: '1 Tab', frequency: 'Once daily' }
      ],
      ayush_parameters: {
        prakriti: isUrgent ? 'Pitta-Vata' : 'Kapha-Pitta',
        vikriti: isUrgent ? 'Vata-Pitta Prakopa' : 'Pitta-Kaphaja Amlapitta',
        sara: 'Madhyama',
        samhanana: 'Madhyama',
        pramana: 'Normal',
        satmya: 'Mishra satmya',
        sattva: isUrgent ? 'Avara (Anxious)' : 'Madhyama',
        ahara_shakti: 'Manda Agni',
        vyayama_shakti: 'Madhyama',
        vaya: 'Madhyama (38 yrs)'
      },
      red_flags_detected: isUrgent,
      alert_acknowledged: false,
      patient_name: patientMeta.name || 'Walk-in Patient',
      abha_id: patientMeta.abha_id || (patientId || 'patient@abdm'),
      triage_level: isUrgent ? 'Immediate' : 'Standard'
    };

    const all = getStoredHistories();
    all.unshift(newRecord);
    saveHistories(all);
    return newRecord;
  },

  /**
   * Waiting Room Queue: GET /api/v1/waiting-room
   */
  async getWaitingRoom() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/waiting-room`);
      if (res.ok) {
        const data = await res.json();
        const histories = data.histories || [];
        if (histories.length > 0) {
          saveHistories(histories);
        }
        return histories;
      }
    } catch (e) {
      console.log('Using local waiting room data:', e.message);
    }
    return getStoredHistories();
  },

  /**
   * Priority Alerts: GET /api/v1/priority-alerts
   */
  async getPriorityAlerts() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/priority-alerts`);
      if (res.ok) {
        const data = await res.json();
        return data.histories || [];
      }
    } catch (e) {
      console.log('Using local priority alert data:', e.message);
    }
    return getStoredHistories().filter(h => h.red_flags_detected && !h.alert_acknowledged);
  },

  /**
   * Single Patient History: GET /patient-histories/{id}
   */
  async getPatientHistory(historyId) {
    try {
      const res = await fetch(`${API_BASE}/patient-histories/${historyId}`);
      if (res.ok) return await res.json();
    } catch (e) {
      console.log('Using local single history data:', e.message);
    }
    const all = getStoredHistories();
    return all.find(h => h.id === historyId) || all[0];
  },

  /**
   * Physician Edits: PATCH /patient-histories/{id}
   */
  async updatePatientHistory(historyId, patchData) {
    try {
      const res = await fetch(`${API_BASE}/patient-histories/${historyId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patchData)
      });
      if (res.ok) {
        const updated = await res.json();
        const all = getStoredHistories();
        const idx = all.findIndex(h => h.id === historyId);
        if (idx >= 0) all[idx] = updated;
        saveHistories(all);
        return updated;
      }
    } catch (e) {
      console.log('Using local history update simulation:', e.message);
    }

    const all = getStoredHistories();
    const idx = all.findIndex(h => h.id === historyId);
    if (idx >= 0) {
      all[idx] = { ...all[idx], ...patchData };
      saveHistories(all);
      return all[idx];
    }
    return null;
  },

  /**
   * Acknowledge Emergency Alert: POST /patient-histories/{id}/acknowledge-alert
   */
  async acknowledgeAlert(historyId) {
    try {
      const res = await fetch(`${API_BASE}/patient-histories/${historyId}/acknowledge-alert`, {
        method: 'POST'
      });
      if (res.ok) {
        const updated = await res.json();
        const all = getStoredHistories();
        const idx = all.findIndex(h => h.id === historyId);
        if (idx >= 0) all[idx] = updated;
        saveHistories(all);
        return updated;
      }
    } catch (e) {
      console.log('Using local alert acknowledgment simulation:', e.message);
    }

    const all = getStoredHistories();
    const idx = all.findIndex(h => h.id === historyId);
    if (idx >= 0) {
      all[idx].alert_acknowledged = true;
      saveHistories(all);
      return all[idx];
    }
    return null;
  },

  /**
   * Mock Push to HIS: POST /abdm/push-to-his
   */
  async pushToHis(historyId, abhaId) {
    try {
      const res = await fetch(`${API_BASE}/abdm/push-to-his`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history_id: historyId, abha_id: abhaId })
      });
      if (res.ok) return await res.json();
    } catch (e) {
      console.log('Using local push to HIS simulation:', e.message);
    }

    return {
      history_id: historyId,
      abha_id: abhaId,
      his_record_id: 'HIS-FHIR-' + Math.floor(100000 + Math.random() * 900000),
      status: 'submitted',
      timestamp: new Date().toISOString()
    };
  }
};
