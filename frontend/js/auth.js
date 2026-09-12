/**
 * Login Gate & Role Distribution
 *
 * Splits the patient-facing kiosk from the clinician-facing dashboard behind
 * a role-selection login screen (Patient / Doctor), matching the MediKiosk
 * Login design.
 *
 * Important: this is a navigation split for a public kiosk terminal, not a
 * security boundary. There is no backend user store, password check, OTP
 * delivery, or SSO provider behind this. Patient sign-in reuses the existing
 * mocked /abdm/verify-abha check (see api.js); doctor sign-in accepts any
 * non-empty credentials. Hiding nav items by role does not stop someone from
 * reaching another view by other means (e.g. the automatic handoff to
 * Physician Review after a patient finishes their interview still works).
 */

import { ApiService } from './api.js';
import { KioskModule } from './kiosk.js';

const SESSION_KEY = 'medikiosk_session';

export const AuthModule = {
  role: null,
  selectedDept: 'General Medicine',

  init() {
    this.bindEvents();

    const saved = this._loadSession();
    if (saved) {
      this._enterApp(saved.role, saved.identity, false);
    } else {
      this._showLogin();
    }
  },

  bindEvents() {
    const roleToggle = document.getElementById('login-role-toggle');
    if (roleToggle) {
      roleToggle.querySelectorAll('.segmented-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => this.switchTab(btn.dataset.loginRole));
      });
    }

    document.getElementById('btn-login-send-otp')?.addEventListener('click', () => this.loginWithAbha());
    document.getElementById('btn-login-mobile')?.addEventListener('click', () => this.loginWithAbha());
    document.getElementById('btn-login-walkin')?.addEventListener('click', () => this.loginAsWalkIn());

    const abhaInput = document.getElementById('login-abha-input');
    if (abhaInput) {
      abhaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') this.loginWithAbha();
      });
    }

    document.querySelectorAll('.login-sample-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (abhaInput) abhaInput.value = btn.dataset.abha;
        this.loginWithAbha();
      });
    });

    document.querySelectorAll('#login-dept-row .dept-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        this.selectedDept = chip.dataset.dept;
        document.querySelectorAll('#login-dept-row .dept-chip').forEach(c => c.classList.toggle('active', c === chip));
      });
    });

    document.getElementById('btn-login-doctor')?.addEventListener('click', () => this.loginAsDoctor(false));
    document.getElementById('btn-login-sso')?.addEventListener('click', () => this.loginAsDoctor(true));

    document.getElementById('btn-logout')?.addEventListener('click', () => this.logout());
  },

  switchTab(role) {
    document.querySelectorAll('#login-role-toggle .segmented-toggle-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.loginRole === role);
    });
    const isPatient = role === 'patient';
    const patientPanel = document.getElementById('login-patient-panel');
    const doctorPanel = document.getElementById('login-doctor-panel');
    const patientHeading = document.getElementById('login-patient-heading');
    const doctorHeading = document.getElementById('login-doctor-heading');
    if (patientPanel) patientPanel.hidden = !isPatient;
    if (doctorPanel) doctorPanel.hidden = isPatient;
    if (patientHeading) patientHeading.hidden = !isPatient;
    if (doctorHeading) doctorHeading.hidden = isPatient;
  },

  async loginWithAbha() {
    const input = document.getElementById('login-abha-input');
    const id = input ? input.value.trim() : '';
    if (!id) {
      window.App?.showToast('Enter an ABHA ID or number first.', 'danger');
      return;
    }

    const result = await ApiService.verifyAbha(id);
    const identity = (result && result.verified)
      ? {
        abha_id: result.abha_id,
        name: result.patient_name || 'Verified Patient',
        verified: true,
        dob: result.date_of_birth,
        gender: result.gender
      }
      : { abha_id: id, name: 'Walk-in Patient', verified: false };

    this._enterApp('patient', identity);
  },

  loginAsWalkIn() {
    this._enterApp('patient', { abha_id: '', name: 'Walk-in Patient', verified: false });
  },

  loginAsDoctor(viaSso) {
    const emailInput = document.getElementById('login-doctor-email');
    const email = viaSso ? 'Hospital SSO' : (emailInput ? emailInput.value.trim() : '');
    if (!viaSso && !email) {
      window.App?.showToast('Enter your hospital email or registration number.', 'danger');
      return;
    }
    this._enterApp('doctor', { email, department: this.selectedDept });
  },

  logout() {
    sessionStorage.removeItem(SESSION_KEY);
    this.role = null;
    this._showLogin();
  },

  _enterApp(role, identity, showWelcomeToast = true) {
    this.role = role;
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({ role, identity }));

    const loginScreen = document.getElementById('login-screen');
    const appContainer = document.getElementById('app-container');
    if (loginScreen) loginScreen.hidden = true;
    if (appContainer) appContainer.hidden = false;

    document.querySelectorAll('[data-role]').forEach(el => {
      el.hidden = el.dataset.role !== role;
    });

    if (role === 'patient') {
      KioskModule.setPatientIdentity(identity || {});
      window.App.switchView('kiosk');
      if (showWelcomeToast) window.App.showToast(`Welcome, ${(identity && identity.name) || 'patient'}`, 'success');
    } else {
      window.App.switchView('triage');
      if (showWelcomeToast) window.App.showToast(`Signed in — ${(identity && identity.department) || 'General Medicine'}`, 'success');
    }
  },

  _showLogin() {
    const loginScreen = document.getElementById('login-screen');
    const appContainer = document.getElementById('app-container');
    if (loginScreen) loginScreen.hidden = false;
    if (appContainer) appContainer.hidden = true;

    const abhaInput = document.getElementById('login-abha-input');
    if (abhaInput) abhaInput.value = '';
    const doctorEmail = document.getElementById('login-doctor-email');
    if (doctorEmail) doctorEmail.value = '';
    const doctorPassword = document.getElementById('login-doctor-password');
    if (doctorPassword) doctorPassword.value = '';

    this.switchTab('patient');
  },

  _loadSession() {
    try {
      const raw = sessionStorage.getItem(SESSION_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }
};
