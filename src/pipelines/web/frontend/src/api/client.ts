import type {
  AppSettings,
  CreatedConfig,
  DataSource,
  DataVerification,
  Defaults,
  EnsembleCandidate,
  EnsembleDiagnosis,
  EnsembleJob,
  EpochSweepCandidate,
  EpochSweepState,
  EvaluationState,
  ExperimentDetail,
  PerClassSummary,
  QueueState,
  ResumeResult,
  ExperimentComparisonResult,
  ExperimentListing,
  GpuStatus,
  JobListing,
  JobRecord,
  LogPage,
  EdaResponse,
  PreparationResponse,
  ProcessedDatasets,
  RegistrationState,
  RuntimeConfig,
  TeamConfig,
  ValidationResult,
} from './types';

/** Backend가 4xx로 돌려준 오류. field별 메시지를 그대로 들고 옵니다. */
export class ApiError extends Error {
  readonly status: number;
  readonly fields: { field: string; message: string }[];

  constructor(status: number, message: string, fields: { field: string; message: string }[] = []) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.fields = fields;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, 'backend에 연결하지 못했습니다. 서버가 실행 중인지 확인해 주세요.');
  }

  const text = await response.text();
  const body: unknown = text ? safeParse(text) : null;

  if (!response.ok) {
    const detail = body as { message?: string; errors?: { field: string; message: string }[] } | null;
    throw new ApiError(
      response.status,
      detail?.message ?? `요청이 실패했습니다 (HTTP ${response.status}).`,
      detail?.errors ?? [],
    );
  }
  return body as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export interface ConfigDraftPayload {
  train: Record<string, unknown>;
  inputs: { data: Record<string, string> };
}

export const api = {
  defaults: () => request<Defaults>('/api/train/defaults'),

  validate: (payload: ConfigDraftPayload) =>
    request<ValidationResult>('/api/train/validate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createConfig: (payload: ConfigDraftPayload) =>
    request<CreatedConfig>('/api/train/configs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listJobs: () => request<JobListing>('/api/train/jobs'),

  listExperiments: () => request<ExperimentListing>('/api/train/experiments'),

  // overwrite는 사람이 "실제 mAP 수정"을 켜고 고칠 때만 true입니다. 서버는 그 말이
  // 없는 요청으로는 이미 기록된 점수를 바꾸지 않습니다.
  saveKaggleScore: (runId: string, score: number, overwrite = false) =>
    request<{ run_id: string; kaggle_score: number }>(
      `/api/train/experiments/${encodeURIComponent(runId)}/kaggle-score`,
      { method: 'PUT', body: JSON.stringify({ score, overwrite }) },
    ),

  // 목록에는 지표 9개 중 5개만 있고 loss 곡선은 없습니다. 상세는 record가 가리키는
  // artifact를 서버가 직접 읽어 화면이 쓸 만큼만 골라 돌려줍니다.
  experimentDetail: (runId: string) =>
    request<ExperimentDetail>(`/api/train/experiments/${encodeURIComponent(runId)}`),

  compareExperiments: (runIds: string[]) =>
    request<ExperimentComparisonResult>('/api/train/experiments/compare', {
      method: 'POST',
      body: JSON.stringify({ run_ids: runIds }),
    }),

  // 앙상블은 합치기 전에는 이득을 알 수 없고, 확인하는 방법이 Kaggle 제출뿐입니다.
  // 그래서 고른 조합을 먼저 진단해 보고 실행합니다.
  ensembleCandidates: () =>
    request<{ candidates: EnsembleCandidate[] }>('/api/ensemble/candidates'),

  // 예측 파일을 읽어야 해서 처음 한 번은 수 초 걸립니다. 재 본 쌍은 서버가 저장해
  // 두므로 후보를 하나씩 바꿔 볼 때는 빨라집니다.
  diagnoseEnsemble: (runIds: string[]) =>
    request<EnsembleDiagnosis>('/api/ensemble/diagnose', {
      method: 'POST',
      body: JSON.stringify({ run_ids: runIds }),
    }),

  startEnsemble: (payload: {
    run_ids: string[];
    run_id: string;
    allow_copied_images?: boolean;
  }) =>
    request<EnsembleJob>('/api/ensemble/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  ensembleStatus: () => request<EnsembleJob>('/api/ensemble/jobs'),

  getJob: (jobId: string) => request<JobRecord>(`/api/train/jobs/${jobId}`),

  getJobConfig: (jobId: string) =>
    request<{ config: RuntimeConfig }>(`/api/train/jobs/${jobId}/config`),

  startJob: (configId: string, accessToken?: string | null) =>
    request<JobRecord>('/api/train/jobs', {
      method: 'POST',
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
      body: JSON.stringify({ config_id: configId }),
    }),

  readQueue: () => request<QueueState>('/api/train/queue'),

  // 대기열도 `/jobs`와 같은 login token을 보냅니다. 항목을 꺼내 실제로 시작할 때
  // 팀 기록을 만들어야 하는데, token이 없으면 이미 로그인한 사람도 거절당합니다.
  addToQueue: (configId: string, accessToken?: string | null) =>
    request<QueueState>('/api/train/queue', {
      method: 'POST',
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
      body: JSON.stringify({ config_id: configId }),
    }),

  removeFromQueue: (entryId: string) =>
    request<QueueState>(`/api/train/queue/${entryId}`, { method: 'DELETE' }),

  // 서버가 다시 뜨면 저장해 둔 token이 사라지므로 다시 돌릴 때 새로 보내 줍니다.
  resumeQueue: (accessToken?: string | null) =>
    request<QueueState>('/api/train/queue/resume', {
      method: 'POST',
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
    }),

  teamConfig: () => request<TeamConfig>('/api/team/config'),

  settings: () => request<AppSettings>('/api/settings'),

  saveSettings: (body: AppSettings) =>
    request<AppSettings>('/api/settings', { method: 'PUT', body: JSON.stringify(body) }),

  /**
   * 이 학습을 이어서 **시도할 수 있는지**. 판단은 실제 저장소를 보는 서버가 합니다.
   *
   * `available`은 "이어갈 수 있다"가 아니라 "눌러 볼 수 있다"입니다. 저장소를 읽지 못한
   * 경우도 true로 오고 이유가 함께 옵니다 — 못 읽었다고 단추를 없애면 눌러서 알아낼 수
   * 있는 것까지 막습니다.
   */
  resumeAvailability: (jobId: string) =>
    request<{ available: boolean; reason: string | null }>(
      `/api/train/jobs/${jobId}/resume`,
    ),

  resumeJob: (jobId: string, epochs?: number | null, accessToken?: string | null) =>
    request<ResumeResult>(`/api/train/jobs/${jobId}/resume`, {
      method: 'POST',
      headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : undefined,
      body: JSON.stringify(epochs == null ? {} : { epochs }),
    }),

  cancelJob: (jobId: string) =>
    request<JobRecord>(`/api/train/jobs/${jobId}/cancel`, { method: 'POST' }),

  // 이 GUI가 들고 있던 기록만 지웁니다. checkpoint와 학습 결과 폴더, registry에
  // 등록된 실험, 팀에 공유된 기록은 그대로 남습니다.
  deleteJob: (jobId: string) =>
    request<JobListing>(`/api/train/jobs/${jobId}`, { method: 'DELETE' }),

  logs: (jobId: string, after: number, limit = 500) =>
    request<LogPage>(`/api/train/jobs/${jobId}/logs?after=${after}&limit=${limit}`),

  gpu: () => request<GpuStatus>('/api/gpu/status'),

  evaluationStatus: (jobId: string) =>
    request<{ evaluation: EvaluationState }>(`/api/train/jobs/${jobId}/evaluate`),

  // metrics.json이 650KB라 상태 polling에 얹지 않고 표를 펼칠 때만 부릅니다.
  evaluationPerClass: (jobId: string) =>
    request<{ summary: PerClassSummary | null }>(
      `/api/train/jobs/${jobId}/evaluate/per-class`,
    ),

  startEvaluation: (
    jobId: string,
    body: {
      score_threshold?: number;
      max_detections_per_image?: number;
      overwrite?: boolean;
      device?: string;
      test_manifest_uri?: string;
    },
  ) =>
    request<{ evaluation: EvaluationState }>(`/api/train/jobs/${jobId}/evaluate`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  retryRegistration: (jobId: string) =>
    request<{ registration: RegistrationState }>(`/api/train/jobs/${jobId}/register`, {
      method: 'POST',
    }),

  // 보관해 둔 epoch checkpoint와 지난 훑기 결과입니다.
  epochSweepStatus: (jobId: string) =>
    request<{
      epoch_sweep: EpochSweepState;
      candidates: EpochSweepCandidate[];
      metrics: string[] | null;
    }>(`/api/train/jobs/${jobId}/epoch-sweep`),

  startEpochSweep: (jobId: string, body: { sample_size?: number; device?: string }) =>
    request<{ epoch_sweep: EpochSweepState }>(`/api/train/jobs/${jobId}/epoch-sweep`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  inspectDirectory: (directory: string) =>
    request<DataSource>('/api/data/inspect', {
      method: 'POST',
      body: JSON.stringify({ directory }),
    }),

  // 고를 수 있는 전처리 폴더 목록. 파일을 열지 않으므로 판이 늘어도 빠릅니다.
  listDatasets: () => request<ProcessedDatasets>('/api/data/datasets'),

  getDataSource: () => request<{ source: DataSource | null }>('/api/data/source'),

  setDataSource: (directory: string) =>
    request<{ source: DataSource }>('/api/data/source', {
      method: 'POST',
      body: JSON.stringify({ directory }),
    }),

  clearDataSource: () =>
    request<{ source: null }>('/api/data/source', { method: 'DELETE' }),

  prepareStatus: () => request<PreparationResponse>('/api/data/prepare'),

  // EDA는 이미지를 전부 열어야 해서 오래 걸립니다. 시작만 시키고 상태를 물어봅니다.
  edaStatus: () => request<EdaResponse>('/api/data/eda'),

  startEda: (body: { image_sample?: number; overwrite?: boolean }) =>
    request<EdaResponse>('/api/data/eda', { method: 'POST', body: JSON.stringify(body) }),

  startPreparation: (body: {
    split_ratio: string;
    seed?: number;
    overwrite?: boolean;
    backend?: string;
  }) =>
    request<PreparationResponse>('/api/data/prepare', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** artifact URI를 알고 있으면 그대로, 아니면 위치를 넘겨 찾게 합니다. */
  verifyDataSource: (target: { data?: Record<string, string>; directory?: string }) =>
    request<{ inspected: DataSource | null; verification: DataVerification }>(
      '/api/data/verify',
      { method: 'POST', body: JSON.stringify(target) },
    ),
};
