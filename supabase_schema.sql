-- MediKiosk: Supabase schema
-- Creates the patient_histories table and the medical_documents storage
-- bucket, and locks both down to the backend's service_role key only.
--
-- The FastAPI backend authenticates with the service_role key, which
-- bypasses Row Level Security entirely. Row Level Security stays ENABLED
-- on both patient_histories and storage.objects, but with zero policies for
-- the anon/authenticated roles below -- i.e. default-deny. No client other
-- than the backend can read or write patient data.
--
-- The medical_documents bucket itself is still marked public=true so that
-- the OpenAI Vision model can fetch an uploaded image's public URL over
-- plain HTTP; object paths are random UUIDs (see main.py), so they are not
-- guessable/enumerable.

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- Table: patient_histories
-- ---------------------------------------------------------------------------
create table if not exists public.patient_histories (
    id                   uuid primary key default gen_random_uuid(),
    created_at           timestamptz not null default now(),
    chief_complaint      text not null,
    hpi_socrates         text not null,
    current_medications  jsonb not null default '[]'::jsonb,
    ayush_parameters     jsonb not null default '{}'::jsonb,
    red_flags_detected   boolean not null default false
);

-- Added for the physician edit/confirm workflow and red-flag triage alerts.
alter table public.patient_histories
    add column if not exists alert_acknowledged boolean not null default false;

-- ---------------------------------------------------------------------------
-- Row Level Security: patient_histories
-- ---------------------------------------------------------------------------
alter table public.patient_histories enable row level security;

-- Locked down: the anon/authenticated hackathon-testing policies have been
-- removed. The backend's service_role key bypasses RLS entirely, so no
-- policies are needed (or wanted) for other roles here.
drop policy if exists "Public insert access (hackathon testing)" on public.patient_histories;
drop policy if exists "Public select access (hackathon testing)" on public.patient_histories;
drop policy if exists "Public update access (hackathon testing)" on public.patient_histories;

-- ---------------------------------------------------------------------------
-- Storage: medical_documents bucket
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('medical_documents', 'medical_documents', true)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- Row Level Security: storage.objects (scoped to medical_documents bucket)
-- ---------------------------------------------------------------------------
-- Locked down the same way: uploads/listing via the storage API now require
-- service_role. Public read of existing objects still works through the
-- bucket's public=true flag below, which is a separate, unauthenticated
-- CDN-style route that does not consult these policies.
drop policy if exists "Public insert access to medical_documents (hackathon testing)" on storage.objects;
drop policy if exists "Public select access to medical_documents (hackathon testing)" on storage.objects;
