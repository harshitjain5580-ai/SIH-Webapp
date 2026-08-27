-- MediKiosk: Supabase schema
-- Creates the patient_histories table, the medical_documents storage bucket,
-- and permissive RLS policies for hackathon testing.
--
-- WARNING: The RLS policies below allow public (anon) INSERT and SELECT with
-- no auth checks. This is intentionally open for hackathon/demo purposes only
-- and must NOT be used with real patient data or in production.

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

-- ---------------------------------------------------------------------------
-- Row Level Security: patient_histories
-- ---------------------------------------------------------------------------
alter table public.patient_histories enable row level security;

create policy "Public insert access (hackathon testing)"
    on public.patient_histories
    for insert
    to anon, authenticated
    with check (true);

create policy "Public select access (hackathon testing)"
    on public.patient_histories
    for select
    to anon, authenticated
    using (true);

-- ---------------------------------------------------------------------------
-- Storage: medical_documents bucket
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public)
values ('medical_documents', 'medical_documents', true)
on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- Row Level Security: storage.objects (scoped to medical_documents bucket)
-- ---------------------------------------------------------------------------
create policy "Public insert access to medical_documents (hackathon testing)"
    on storage.objects
    for insert
    to anon, authenticated
    with check (bucket_id = 'medical_documents');

create policy "Public select access to medical_documents (hackathon testing)"
    on storage.objects
    for select
    to anon, authenticated
    using (bucket_id = 'medical_documents');
