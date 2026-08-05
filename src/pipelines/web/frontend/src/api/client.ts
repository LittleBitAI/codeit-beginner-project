import type {
  CreatedConfig,
  DataSource,
  DataVerification,
  Defaults,
  GpuStatus,
  JobListing,
  JobRecord,
  LogPage,
  PreparationResponse,
  RuntimeConfig,
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

  getJob: (jobId: string) => request<JobRecord>(`/api/train/jobs/${jobId}`),

  getJobConfig: (jobId: string) =>
    request<{ config: RuntimeConfig }>(`/api/train/jobs/${jobId}/config`),

  startJob: (configId: string) =>
    request<JobRecord>('/api/train/jobs', {
      method: 'POST',
      body: JSON.stringify({ config_id: configId }),
    }),

  cancelJob: (jobId: string) =>
    request<JobRecord>(`/api/train/jobs/${jobId}/cancel`, { method: 'POST' }),

  logs: (jobId: string, after: number, limit = 500) =>
    request<LogPage>(`/api/train/jobs/${jobId}/logs?after=${after}&limit=${limit}`),

  gpu: () => request<GpuStatus>('/api/gpu/status'),

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
