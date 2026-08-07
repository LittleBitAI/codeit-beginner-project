/** Backend 응답 형태. src/pipelines/web/api 의 route와 짝을 이룹니다. */

export type JobStatus =
  | 'starting'
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
  defaults_by_optimizer?: Record<string, number>;
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
  /** 구버전 backend 응답에는 없을 수 있어 frontend도 fallback을 유지합니다. */
  train_capability?: TrainCapability;
  fields: FieldSpec[];
  data_fields: FieldSpec[];
  devices: { value: string; available: boolean; reason: string | null }[];
}

export interface TrainCapabilityChoice {
  default: string;
  choices: string[];
  selection_supported: boolean;
}

export interface TrainCapability {
  schema_version: 1;
  source: 'train' | 'legacy_fallback';
  fallback_reason: 'train_capability_unavailable' | 'train_capability_invalid' | null;
  model: TrainCapabilityChoice;
  optimizer: TrainCapabilityChoice;
}

export type CapabilityValueSource = 'record' | 'train' | 'legacy_fallback';

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
  /** 이름은 모델이 정합니다. 화면이 목록을 정해 두지 않고 받은 것을 그립니다. */
  train_loss_components?: Record<string, number> | null;
  validation_loss_components?: Record<string, number> | null;
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
  /** train이 완료 event를 보냈는지. 조기 종료여도 true입니다. */
  finished?: boolean;
  /** 완료 event가 없으면 null입니다. 모르는 것을 지어내지 않습니다. */
  stopped_early?: boolean | null;
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
  evaluation?: EvaluationState;
  registration?: RegistrationState;
  log_lines: number;
  orphan_note: string | null;
  cloud_run_id?: string | null;
  sync_revision?: number;
}

export interface TeamConfig {
  enabled: boolean;
  team_id: string | null;
  appsync_url: string | null;
  region: string;
  user_pool_id: string | null;
  user_pool_client_id: string | null;
  cognito_domain: string | null;
  /** 설정되어 있으면 로그인 없이 이 이름으로 팀에 기록하는 환경입니다. */
  actor: string | null;
}

export interface TeamRun {
  teamId: string;
  cloudRunId: string;
  localJobId: string;
  runId: string;
  actorSub: string;
  actorName: string;
  /** cognito는 로그인이 확인해 준 이름, iam은 실행하는 쪽이 직접 적은 이름입니다. */
  actorSource: 'cognito' | 'iam' | null;
  status: JobStatus;
  settings: Record<string, unknown>;
  dataInputs: Record<string, unknown>;
  progress: Record<string, unknown>;
  summary: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  /** 평가를 아직 돌리지 않았거나 이 field가 생기기 전 기록이면 빈 객체입니다. */
  evaluation: Record<string, unknown>;
  message: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  heartbeatAt: string;
  revision: number;
}

export interface TeamLogBatch {
  teamId: string;
  cloudRunId: string;
  startSeq: number;
  endSeq: number;
  lines: LogLine[];
  createdAt: string;
}

export interface JobListing {
  jobs: JobRecord[];
  active_job_id: string | null;
}

export interface ExperimentSummary {
  experiment_id: string;
  run_id: string;
  status: JobStatus;
  status_label: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_seconds: number | null;
  dataset: {
    identity: string | null;
    identity_source: 'dataset_id' | 'artifact_set' | 'unknown';
    artifacts_complete: boolean;
  };
  model: {
    architecture: string | null;
    pretrained: boolean | null;
    source: CapabilityValueSource;
  };
  optimizer: {
    name: string | null;
    source: CapabilityValueSource;
    learning_rate: number | null;
    momentum: number | null;
    weight_decay: number | null;
    beta1: number | null;
    beta2: number | null;
    epsilon: number | null;
  };
  training: {
    device: string | null;
    epochs: number | null;
    batch_size: number | null;
    num_workers: number | null;
    seed: number | null;
  };
  /** 학습이 남긴 loss와 evaluate가 낸 지표. 기록에 없으면 null이고 화면은 - 로 둡니다. */
  metrics: {
    best_epoch: number | null;
    best_validation_loss: number | null;
    final_train_loss: number | null;
    final_validation_loss: number | null;
    map: number | null;
    map50: number | null;
    map75: number | null;
    precision50: number | null;
    recall50: number | null;
  };
}

export interface ExperimentListing {
  experiments: ExperimentSummary[];
}

export interface ExperimentComparisonResult {
  experiments: ExperimentSummary[];
  missing: string[];
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
  /** 폴더를 직접 골랐는지, data pipeline이 준비해 준 것인지 */
  origin?: 'folder' | 'prepared';
  preparation?: Record<string, unknown> | null;
}

/**
 * 준비 subprocess가 stderr로 흘린 `data.progress/1` 진행 로그를 읽은 결과.
 *
 * 진행 줄이 한 번도 없으면 `available`이 false이고 나머지는 모두 null입니다.
 * 그때 가짜 진행률을 그리면 안 됩니다.
 */
export interface PreparationProgress {
  available: boolean;
  reason?: string | null;
  message?: string | null;
  /** listing / annotations / test_images / split / manifests / publish / completed */
  stage?: string | null;
  stage_label?: string | null;
  raw_prefix?: string | null;
  split_ratio?: string | null;
  seed?: number | null;
  split_method?: string | null;
  sources?: { train_images?: number; annotations?: number; test_images?: number } | null;
  read?: {
    stage: string | null;
    done: number | null;
    total: number | null;
    percent: number | null;
  } | null;
  completed?: {
    train_images?: number;
    validation_images?: number;
    category_count?: number;
  } | null;
  /** 관측된 읽기 속도로만 계산합니다. 관측이 부족하면 null입니다. */
  eta_seconds?: number | null;
  malformed_lines?: number;
}

/** 원본에서 artifact를 만드는 준비 실행의 상태. */
export interface PreparationState {
  status: 'idle' | 'running' | 'succeeded' | 'failed';
  split_ratio?: string;
  seed?: number;
  overwrite?: boolean;
  backend?: string;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string;
  supported?: boolean;
  exit_code?: number | null;
  artifacts?: Record<string, string>;
  summary?: Record<string, unknown>;
  selected?: boolean;
  progress?: PreparationProgress;
}

/** evaluate가 만드는 detection metric. 계산하지 않은 값은 null입니다. */
export interface DetectionMetrics {
  mAP?: number | null;
  mAP50?: number | null;
  mAP75?: number | null;
  precision50?: number | null;
  recall50?: number | null;
}

/**
 * 평가 subprocess가 stderr로 흘린 `evaluate.progress/1` 진행 상태입니다.
 *
 * 진행 줄이 한 번도 없으면 `available`이 false이고 나머지는 모두 비어 있습니다.
 * 그때 화면은 가짜 진행률을 그리지 않고 고정 안내 문구만 둡니다.
 */
export interface EvaluateProgress {
  available: boolean;
  reason?: string | null;
  message?: string | null;
  /** started / validation / test / metrics / submission / completed */
  stage?: string | null;
  stage_label?: string | null;
  run_id?: string | null;
  device?: string | null;
  images?: { validation_images?: number; test_images?: number } | null;
  predict?: {
    stage: string | null;
    done: number | null;
    total: number | null;
    percent: number | null;
  } | null;
  /** 계산하지 않은 지표는 0이 아니라 null입니다. */
  metrics?: { mAP?: number | null; mAP50?: number | null; mAP75?: number | null } | null;
  submission_rows?: number | null;
  completed?: { validation_images?: number; test_images?: number } | null;
  /** 관측된 추론 속도로만 계산합니다. 관측이 부족하면 null입니다. */
  eta_seconds?: number | null;
  malformed_lines?: number;
}

export interface EvaluationState {
  status: 'idle' | 'running' | 'succeeded' | 'failed';
  job_id?: string | null;
  run_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string;
  exit_code?: number | null;
  device?: string | null;
  score_threshold?: number;
  /** test manifest가 있어 submission.csv도 함께 만드는 실행인지 */
  submission_requested?: boolean;
  artifacts?: Record<string, string>;
  summary?: {
    metrics?: DetectionMetrics;
    image_count?: number;
    prediction_count?: number;
    annotation_count?: number;
    evaluated_class_count?: number;
    iou_thresholds?: number[];
    [key: string]: unknown;
  };
  /** 다른 학습의 평가가 돌고 있으면 그 job id */
  busy_with?: string | null;
  registration?: RegistrationState;
  progress?: EvaluateProgress;
}

export interface RegistrationState {
  status: 'idle' | 'running' | 'succeeded' | 'failed' | 'index_failed';
  message?: string;
}

export interface StorageEnvironment {
  bucket: string | null;
  bucket_configured: boolean;
  profile_configured: boolean;
  region: string | null;
  forced_backend: string | null;
  default_backend: string;
}

export interface PreparationResponse {
  split_ratios: string[];
  backends: string[];
  storage: StorageEnvironment;
  preparation: PreparationState;
}

/** 실제 data pipeline(--only data)을 불러 받은 결과. */
export interface DataVerification {
  ok: boolean;
  exit_code: number | null;
  message: string;
  artifacts: Record<string, string>;
  summary: Record<string, unknown>;
}

export interface CreatedConfig {
  config_id: string;
  run_id: string;
  config: RuntimeConfig;
  warnings: FieldMessage[];
}
