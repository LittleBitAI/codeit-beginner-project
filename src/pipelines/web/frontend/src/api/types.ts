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
  /** 고른 모델마다 기본값이 다른 칸입니다. 없으면 default가 그대로 기본값입니다. */
  defaults_by_architecture?: Record<string, number>;
  /** 여기 적힌 모델에서만 쓰는 칸입니다. 다른 모델이면 감춥니다. */
  only_for_architectures?: string[];
  /**
   * backbone만 다른 갈래를 편 표입니다(`{ dino: { resnet50: 'dino_r50_4scale' } }`).
   * 있으면 화면이 모델 칸 옆에 backbone 칸을 하나 더 그립니다. 보내는 값은 여전히
   * architecture 이름 하나입니다.
   */
  backbones?: Record<string, Record<string, string>>;
  /** 갈래를 처음 골랐을 때 놓을 backbone입니다. */
  backbone_defaults?: Record<string, string>;
  backbone_label?: string;
  backbone_hint?: string;
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
  /** 그 epoch의 마지막 batch가 쓴 값. schedule 이전의 옛 실행에는 없습니다. */
  learning_rate?: number | null;
}

/** 지금 지나고 있는 epoch 안의 batch 위치입니다. */
export interface StepProgress {
  phase: 'train' | 'validation';
  step: number;
  total_steps: number;
  percent: number;
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
  /** step event가 없는 예전 실행과 epoch 사이에서는 null입니다. */
  step?: StepProgress | null;
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
  epoch_sweep?: EpochSweepState;
  log_lines: number;
  orphan_note: string | null;
  cloud_run_id?: string | null;
  sync_revision?: number;
}

/**
 * 사람이 한 번 정해 두고 계속 쓰는 값.
 *
 * `evaluation_mode`가 `null`이면 아직 고른 적이 없다는 뜻이고, 그동안 서버는 자동
 * 평가를 하지 않습니다. 서버를 올렸다는 이유만으로 GPU가 도는 일은 없어야 합니다.
 */
export interface AppSettings {
  evaluation_mode: 'parallel' | 'serial' | null;
  /** epoch 훑기가 순위를 매길 지표 3개. 고른 순서가 곧 가중치(3:2:1)입니다. */
  epoch_metrics: string[] | null;
}

/** 고를 수 있는 지표. 서버(`settings.py`)의 목록과 같아야 합니다. */
export const EPOCH_METRIC_NAMES = [
  'mAP',
  'mAP50_95',
  'mAP50',
  'mAP75',
  'precision50',
  'recall50',
] as const;

export interface EpochSweepCandidate {
  epoch: number;
  checkpoint_uri: string;
  metrics?: Record<string, number | null>;
  normalized?: Record<string, number>;
  score?: number;
  failed?: boolean;
  message?: string;
}

export interface EpochSweepState {
  /** `interrupted`는 훑는 중에 서버가 다시 뜬 경우입니다. thread가 함께 사라집니다. */
  status: 'idle' | 'running' | 'succeeded' | 'failed' | 'interrupted';
  job_id?: string | null;
  busy_with?: string | null;
  message?: string;
  metrics?: string[];
  sample_size?: number;
  total?: number;
  done?: number;
  candidates?: EpochSweepCandidate[];
  winner?: (EpochSweepCandidate & { run_id?: string; full_metrics?: Record<string, number | null> }) | null;
  artifacts?: Record<string, string>;
  registration?: { status: string; message?: string };
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
  /**
   * 밤새 무인으로 대기열을 돌릴 때, 만료된 login token을 대신해 쓰는 이름입니다.
   * `actor`와 달리 **로그인 관문을 열지 않습니다.** 화면은 설정 여부를 확인하는
   * 용도로만 읽고, 이 값으로 로그인을 건너뛰면 안 됩니다.
   */
  unattended_actor: string | null;
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
    /** 학습 manifest가 든 폴더 이름. 표에 100자짜리 URI 대신 이것을 보여 줍니다. */
    label: string | null;
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
    /** 몇 batch를 모아 한 번 갱신했는지. 이 값을 모르던 옛 기록은 null입니다. */
    gradient_accumulation_steps: number | null;
    /** MMDetection 모델만 쓰는 입력 크기. 다른 모델이면 null입니다. */
    input_size: number | null;
    precision?: string | null;
    checkpoint_every?: number | null;
    /** 무엇에서 이어 학습했는지. 처음부터 학습한 실행은 null입니다. */
    resume_from?: string | null;
    /**
     * 아래 셋은 train이 받는 모양 그대로입니다.
     *
     * 평평하게 펴지 않는 이유는 '이 세팅으로 학습하기'가 이 값을 그대로 다시
     * 보내기 때문입니다. 안쪽 key를 모르는 채 펴면 되살릴 수 없습니다. 이 값을
     * 기록하지 않던 옛 실행은 null이라 화면이 - 로 둡니다.
     */
    augmentation?: { preset?: string } | null;
    lr_scheduler?: Record<string, unknown> | null;
    early_stopping?: Record<string, unknown> | null;
    seed: number | null;
  };
  /** evaluate가 간추린 약한 class. 평가 전이거나 옛 기록이면 null입니다. */
  per_class_summary?: {
    min_truth_count: number;
    top_n: number;
    counts: { weak: number; sparse: number; unmeasured: number };
    weak: WeakClassRow[];
    sparse: WeakClassRow[];
    /** 정답이 하나도 없어 잴 수 없었던 class. evaluate는 이것만 자르지 않습니다. */
    unmeasured?: WeakClassRow[];
  } | null;
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
    /** 사람이 Kaggle에 제출한 뒤 직접 기록한 실제 점수입니다. */
    kaggle_score?: number | null;
  };
  /**
   * 학습이 평가와 제출까지 갔는지. registry index에 있는 값으로만 판단합니다.
   *
   * 이 field가 생기기 전 backend가 아직 떠 있을 수 있어 선택입니다. 없다고 화면이
   * 통째로 죽으면 안 되므로 `completionOf()`를 거쳐서 읽습니다.
   */
  completion?: {
    evaluated: boolean;
    submission_generated?: boolean;
    submitted: boolean;
    submission_checked: boolean;
    submission_rows: number | null;
  };
}

/** 이 목록에 팀원의 실험도 들어오는지. registry index가 S3에 있을 때만 공용입니다. */
export interface RegistryScope {
  backend: string;
  shared: boolean;
}

export interface ExperimentListing {
  experiments: ExperimentSummary[];
  scope?: RegistryScope;
}

export interface ExperimentComparisonResult {
  experiments: ExperimentSummary[];
  missing: string[];
  /**
   * 실행 이름별 loss 곡선. 견주기 화면이 요청 하나로 표와 곡선을 다 받습니다.
   *
   * 예전에는 실행마다 상세를 또 불렀는데, 그 응답은 곡선에 쓰지 않는 평가 결과까지
   * 들고 오고 서버는 그때마다 registry index 전체를 훑었습니다.
   *
   * epoch 목록만이 아니라 `available`·`reason`까지 옵니다. 빈 배열 하나로는 못 읽은
   * 것과 아직 한 epoch도 안 끝난 것을 구별할 수 없어, 화면이 원인을 지어 말하게 됩니다.
   */
  curves: Record<string, ExperimentHistoryCurve>;
}

/** score threshold를 옮겨 가며 잰 precision·recall·F1 한 점입니다. */
export interface SweepPoint {
  threshold: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

/** 표본은 충분한데 점수가 낮은 class 한 줄. evaluate가 골라 둔 것을 그대로 씁니다. */
export interface WeakClassRow {
  category_id: number;
  name: string;
  ap: number | null;
  truth_count?: number;
  prediction_count?: number;
}

/**
 * `metrics.json`에서 화면이 쓸 것만 골라 온 결과입니다.
 *
 * confusion matrix와 per_image는 담기지 않습니다. 그 둘까지 넣으면 650KB입니다.
 */
/**
 * 정답이 무엇인데 무엇으로 봤는지, 그리고 몇 건인지.
 *
 * `background`는 class가 아니라 "없음"입니다. 정답이 background면 없는 것을
 * 찾아낸 것이고, 예측이 background면 놓친 것입니다.
 */
export interface ConfusionPair {
  truth_id: number | null;
  truth: string;
  predicted_id: number | null;
  predicted: string;
  count: number;
}

/** evaluate가 나누는 false positive의 원인 넷입니다. */
export interface FalsePositiveCauses {
  localization: number;
  classification: number;
  background: number;
  duplicate: number;
}

export interface ExperimentEvaluation {
  available: boolean;
  reason: string | null;
  /**
   * 아래 블록은 모두 선택입니다.
   *
   * 평가를 못 읽은 응답에는 이 key들이 통째로 없을 수 있어서, 화면이 available을
   * 확인하기 전에 파고들면 죽습니다. 실제로 상세 화면이 흰 채로 멈췄습니다.
   * 선택으로 두어 타입 검사가 방어를 강제하게 합니다.
   */
  metrics?: Record<string, number | null>;
  counts?: Record<string, number | null>;
  score_threshold?: number | null;
  max_detections_per_image?: number | null;
  /**
   * IoU label("0.50"/"0.75")별로 나뉩니다.
   *
   * 세 상태가 다릅니다. key가 없으면 재지 않은 것, `null`이면 기록은 있는데
   * 읽지 못한 것, 빈 배열이면 재서 지점이 하나도 없던 것입니다.
   */
  score_sweep?: Record<string, SweepPoint[] | null> | null;
  /** `null`은 최고점을 못 찍는다는 뜻입니다 — 표시가 없을 뿐 틀린 말은 안 합니다. */
  best_f1?: Record<string, SweepPoint | null> | null;
  /** 헷갈린 쌍입니다. 행렬 자체는 오지 않습니다 — 118종이면 119x119입니다. */
  confusions?: Record<string, ConfusionPair[]> | null;
  /**
   * 그 목록이 상위 몇 개로 잘렸는지. 말하지 않으면 잘린 것이 전부로 읽힙니다.
   * `null`은 기록은 있는데 **읽지 못했다**는 뜻이라, 재서 0건과 다릅니다.
   */
  confusion_counts?: Record<string, { pairs: number; shown: number } | null> | null;
  /** false positive를 원인별로 나눈 건수. */
  error_breakdown?: Record<string, FalsePositiveCauses | null> | null;
  per_class_summary?: {
    min_truth_count: number;
    top_n: number;
    counts: { weak: number; sparse: number; unmeasured: number };
    weak: WeakClassRow[];
    sparse: WeakClassRow[];
    unmeasured: WeakClassRow[];
  } | null;
}

export interface ExperimentHistoryCurve {
  available: boolean;
  reason: string | null;
  epochs?: EpochRecord[];
}

export interface ExperimentDetail {
  experiment: ExperimentSummary;
  evaluation: ExperimentEvaluation;
  history: ExperimentHistoryCurve;
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

/** evaluate가 per_class를 다시 배열해 준 한 줄입니다. 새 지표가 아닙니다. */
export interface PerClassRow {
  category_id: number;
  name: string;
  ap: number | null;
  ap50: number | null;
  ap75: number | null;
  truth_count: number;
  prediction_count: number;
}

/**
 * 57개 class 중 어디가 약한지 evaluate가 갈라 놓은 결과입니다.
 *
 * `weak`와 `sparse`를 섞지 않는 것이 핵심입니다. 정답이 몇 개뿐인 class는 AP가
 * 실력이 아니라 표본 수에 흔들려서, 같이 세우면 고칠 곳을 잘못 가리킵니다.
 */
export interface PerClassSummary {
  /** weak로 보려면 필요한 최소 정답 수 */
  min_truth_count: number;
  /** weak/sparse 목록에 남긴 최대 줄 수 */
  top_n: number;
  counts: { weak: number; sparse: number; unmeasured: number };
  weak: PerClassRow[];
  sparse: PerClassRow[];
  unmeasured: PerClassRow[];
}

/** 아직 시작하지 않은 학습 하나입니다. */
export interface QueueEntry {
  entry_id: string;
  config_id: string;
  run_id: string;
  queued_at: string;
}

export interface QueueState {
  entries: QueueEntry[];
  /** 멈춘 대기열은 앞 학습이 끝나도 다음을 시작하지 않습니다. */
  paused: boolean;
  /** 대기열에 넣자마자 시작된 학습. 뒤에 줄을 선 경우에는 null입니다. */
  started?: JobRecord | null;
}

/** 중단된 학습을 이어서 시작한 결과입니다. */
export interface ResumeResult {
  config_id: string;
  /** 이어서 하는 실행의 새 이름입니다. 중단된 실행과 섞이지 않게 새로 만듭니다. */
  run_id: string;
  resumed_from_job_id: string;
  /** train이 읽을 checkpoint 경로입니다. */
  resume_from: string;
  started?: JobRecord | null;
  entries: QueueEntry[];
  paused: boolean;
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

/** 다섯 숫자로 줄인 분포. 잰 값이 없으면 절 자체가 null입니다. */
export interface EdaDistribution {
  count: number;
  min: number;
  p10: number;
  median: number;
  p90: number;
  max: number;
}

/**
 * data pipeline이 model 없이 잰 dataset 리포트.
 *
 * 화면이 그림을 그리고 pipeline은 숫자만 냅니다. 재지 못한 값은 0이 아니라 null이라,
 * "0이었다"와 "재지 못했다"가 섞이지 않습니다.
 */
export interface EdaReport {
  schema_version: string;
  dataset_directory: string;
  shape: Record<string, {
    images: number;
    annotations: number;
    objects_per_image: Record<string, number>;
    images_with_a_repeated_class: number;
  }>;
  classes: {
    class_count: number;
    train_images_per_class: EdaDistribution | null;
    imbalance_ratio: number | null;
    classes_missing_from_train: number[];
    classes_missing_from_validation: number[];
    per_class: {
      category_id: number;
      name: string | null;
      train_images: number;
      validation_images: number;
    }[];
  };
  combinations: {
    train: { groups: number; images_per_group: EdaDistribution | null };
    validation: { groups: number; images_per_group: EdaDistribution | null };
    groups_in_both_splits: number;
    leaked_group_sample: string[];
    capture_conditions: Record<string, number>;
  };
  object_size: {
    train_annotation_fraction: EdaDistribution | null;
    validation_annotation_fraction: EdaDistribution | null;
    /** 픽셀로 잰 자가 정답을 얼마나 맞히는지. 못 믿으면 비교를 내주지 않습니다. */
    calibration: {
      images: number;
      measured_over_annotation: number | null;
      limits: [number, number];
      trustworthy: boolean;
    };
    train_foreground_fraction: EdaDistribution | null;
    test_foreground_fraction: EdaDistribution | null;
    test_over_train: { area_ratio: number; length_ratio: number } | null;
  };
  appearance: {
    train_background_color: number[] | null;
    test_background_color: number[] | null;
    train_foreground_color: number[] | null;
    test_foreground_color: number[] | null;
    background_color_distance: number | null;
    foreground_color_distance: number | null;
  };
  sources: Record<string, unknown>;
}

export interface EdaState {
  status: 'idle' | 'running' | 'succeeded' | 'failed';
  directory?: string;
  image_sample?: number;
  overwrite?: boolean;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string;
  artifacts?: Record<string, string>;
  summary?: Record<string, unknown>;
  progress?: PreparationProgress;
  report?: EdaReport | null;
  /** 지금 고른 dataset이 아닌 다른 폴더의 결과인지. 그러면 report는 비어 옵니다. */
  stale?: boolean;
}

export interface EdaResponse {
  eda: EdaState;
}

/** 고를 수 있는 전처리 폴더 하나. 파일을 열지 않고 이름과 유무만 봅니다. */
export interface ProcessedDataset {
  name: string;
  directory: string;
  /** 필수 artifact 4개가 다 있는지 */
  complete: boolean;
  missing: string[];
  has_test_manifest: boolean;
  has_eda_report: boolean;
  /** crop 은행이 함께 만들어졌는지. embedding 학습이 이것을 봅니다. */
  has_crop_bank?: boolean;
}

export interface ProcessedDatasets {
  backend: 'local' | 's3';
  root: string | null;
  datasets: ProcessedDataset[];
  problems: string[];
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

/**
 * 합칠 수 있는 실행 하나. **체크포인트가 있으면 후보**이고, 점수가 높은 것부터 옵니다.
 *
 * `ready`가 false면 test 예측이 아직 없다는 뜻입니다. 고를 수는 있고, 합치기를 누르면
 * 서버가 먼저 만듭니다 — 그 단계만 GPU를 씁니다.
 */
export interface EnsembleCandidate {
  run_id: string;
  checkpoint_uri: string;
  test_predictions_uri: string | null;
  ready: boolean;
  dataset_label: string | null;
  kaggle_score: number | null;
  created_at: string | null;
}

/** 합치기 전에 알 수 있는 것 하나. `level`이 `warn`이어도 실행은 막지 않습니다. */
export interface EnsembleCheck {
  id: string;
  level: 'ok' | 'warn';
  title: string;
  detail: string;
}

export interface EnsembleDiagnosis {
  run_ids: string[];
  checks: EnsembleCheck[];
  diversity: {
    agreement?: number;
    box_iou?: number;
    pairs: { runs: [string, string]; agreement: number; box_iou: number }[];
  };
  /** 결과가 떨어질 구간입니다. 일곱 개 융합이 평균에서 최고까지의 82% 지점이었습니다. */
  expected: { floor?: number; ceiling?: number; observed_ratio?: number };
  blocking: boolean;
}

/** 이 서버가 학습한 crop embedding 하나. 재순위에서 고를 후보이기도 합니다. */
export interface EmbeddingRun {
  run_id: string;
  job_id: string;
  status: string;
  backbone: string | null;
  epochs: number | null;
  checkpoint_uri: string | null;
  crop_bank_uri: string | null;
  created_at: string | null;
  /** 학습이 성공했고 checkpoint를 남겼을 때만 참입니다. */
  ready: boolean;
}

export interface EmbeddingDefaults {
  backbones: string[];
  devices: string[];
  run_id_pattern: string;
  defaults: {
    backbone: string;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    weight_decay: number;
    seed: number;
    pretrained: boolean;
    device: string;
  };
}

export interface EnsembleJob {
  status: 'idle' | 'running' | 'succeeded' | 'failed';
  run_id?: string;
  /** 합친 뒤 점수를 다시 매기는 데 쓴 embedding들입니다. */
  embedding_run_ids?: string[];
  /** 예측을 만드는 중(`harvest`)인지 합치는 중(`fuse`)인지. */
  stage?: 'harvest' | 'fuse';
  pending?: string[];
  harvesting?: string;
  harvest_progress?: [number, number];
  message?: string | null;
  artifacts?: Record<string, unknown>;
  summary?: Record<string, unknown>;
  logs?: string[];
}
