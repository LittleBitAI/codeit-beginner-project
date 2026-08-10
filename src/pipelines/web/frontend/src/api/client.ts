import type {
  CreatedConfig,
  DataSource,
  DataVerification,
  Defaults,
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
  PreparationResponse,
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

  saveKaggleScore: (runId: string, score: number) =>
    request<{ run_id: string; kaggle_score: number }>(
      `/api/train/experiments/${encodeURIComponent(runId)}/kaggle-score`,
      { method: 'PUT', body: JSON.stringify({ score }) },
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

  inspectDirectory: (directory: string) =>
    request<DataSource>('/api/data/inspect', {
      method: 'POST',
      body: JSON.stringify({ directory }),
    }),

  getDataSource: () => request<{ source: DataSource | null }>('/api/data/source'),

  setDataSource: (directory: string) =>
    request<{ source: DataSource }>('/api/data/source', {
      method: 'POST',
      body: JSON.stringify({ directory }),
    }),

  clearDataSource: () =>
    request<{ source: null }>('/api/data/source', { method: 'DELETE' }),

  prepareStatus: () => request<PreparationResponse>('/api/data/prepare'),

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
