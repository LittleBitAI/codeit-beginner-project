/** Backend 응답 형태. src/pipelines/web/api 의 route와 짝을 이룹니다. */

export type JobStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted';

export interface FieldSpec {
  name: string;
  type: 'string' | 'integer' | 'number' | 'boolean' | 'enum' | 'uri';
  default?: unknown;
  minimum?: number;
  choices?: string[];
  label: string;
  hint: string;
  pattern?: string;
  placeholder?: string;
  required?: boolean;
}

export interface Defaults {
  architecture: string;
  architecture_note: string;
  fields: FieldSpec[];
  data_fields: FieldSpec[];
  devices: { value: string; available: boolean; reason: string | null }[];
}

export interface FieldMessage {
  field: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: FieldMessage[];
  warnings: FieldMessage[];
  normalized: RuntimeConfig | null;
}

export interface RuntimeConfig {
  project: { name: string };
  execution: { mode: string };
  storage: Record<string, unknown>;
  train: Record<string, unknown>;
  inputs: { data: Record<string, string> };
}

export interface EpochRecord {
  epoch: number;
  train_loss: number | null;
  validation_loss: number | null;
  epoch_seconds: number | null;
  is_best: boolean | null;
}

export interface Progress {
  available: boolean;
  reason: string | null;
  message: string | null;
  run_id?: string | null;
  architecture?: string | null;
  device?: string | null;
  train_images?: number | null;
  validation_images?: number | null;
  class_count?: number | null;
  total_epochs: number | null;
  current_epoch: number | null;
  completed_epochs?: number;
  percent?: number | null;
  eta_seconds: number | null;
  epochs: EpochRecord[];
  best?: { epoch: number; validation_loss: number } | null;
  malformed_lines?: number;
}

export interface JobRecord {
  job_id: string;
  config_id: string;
  run_id: string;
  status: JobStatus;
  status_label: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number | null;
  exit_code: number | null;
  message: string | null;
  artifacts: Record<string, string>;
  summary: Record<string, unknown>;
  settings: Record<string, unknown>;
  data_inputs: Record<string, string>;
  progress: Progress;
  log_lines: number;
  orphan_note: string | null;
}

export interface JobListing {
  jobs: JobRecord[];
  active_job_id: string | null;
}

export interface LogLine {
  seq: number;
  stream: 'stdout' | 'stderr' | 'system';
  level: 'info' | 'warn' | 'error';
  text: string;
  ts: string;
}

export interface LogPage {
  lines: LogLine[];
  next: number;
  complete: boolean;
}

export interface GpuDevice {
  index: number | null;
  name: string | null;
  utilization_percent: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  temperature_c: number | null;
}

export interface GpuStatus {
  torch: { cuda_available: boolean; device_count: number; reason: string | null };
  telemetry: {
    source: string;
    reason: string | null;
    message: string | null;
    devices: GpuDevice[];
  };
  queried_at: string;
}

export interface MatchedArtifact {
  name: string;
  uri: string;
}

export interface ExaminedFile {
  name: string;
  uri: string;
  kind: 'manifest' | 'class_map' | 'summary' | 'unknown';
  problem: string | null;
}

/** 전처리 결과 폴더를 살펴본 결과. */
export interface DataSource {
  directory: string;
  complete: boolean;
  data: Record<string, string>;
  matched: Record<string, MatchedArtifact | null>;
  labels: Record<string, string>;
  missing: string[];
  problems: string[];
  examined: ExaminedFile[];
  available?: boolean;
  selected_at?: string | null;
}

export interface CreatedConfig {
  config_id: string;
  run_id: string;
  config: RuntimeConfig;
  warnings: FieldMessage[];
}
