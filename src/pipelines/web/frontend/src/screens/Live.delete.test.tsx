/**
 * 기록 삭제. 되돌릴 수 없는 동작이라 무엇이 사라지고 무엇이 남는지 먼저 말합니다.
 *
 * 화면에서 부를 곳이 없어지면 쌓인 실패 기록을 치울 방법이 사라집니다 —
 * 실제로 화면을 다시 짜면서 한 번 잃었습니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { JobListing, JobRecord } from '../api/types';
import { Live } from './Live';

let deleted: string[] = [];
/** 화면은 listing이 아니라 `/jobs/{id}`로 다시 읽습니다. 그 응답도 같은 것을 줍니다. */
let current: JobRecord | null = null;

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    job_id: 'job-1',
    config_id: 'cfg-1',
    run_id: 'retina-failed',
    status: 'failed',
    status_label: '실패',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T00:10:00Z',
    elapsed_seconds: 600,
    exit_code: 1,
    message: 'CUDA out of memory',
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: {
      available: false,
      reason: null,
      message: null,
      total_epochs: null,
      current_epoch: null,
      eta_seconds: null,
      epochs: [],
    },
    log_lines: 3,
    orphan_note: null,
    ...overrides,
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  deleted = [];
  current = null;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (init?.method === 'DELETE') {
        deleted.push(path);
        return jsonResponse({ jobs: [], active_job_id: null });
      }
      if (path.startsWith('/api/train/jobs/job-1/logs')) {
        return jsonResponse({ lines: [], next: 0, complete: true });
      }
      if (path === '/api/train/jobs/job-1') return jsonResponse(current ?? job());
      if (path === '/api/train/queue') return jsonResponse({ entries: [], paused: false });
      if (path === '/api/gpu/status') {
        return jsonResponse({
          torch: { cuda_available: false, device_count: 0, reason: 'fixture' },
          telemetry: { source: 'none', reason: 'fixture', message: null, devices: [] },
          queried_at: '2026-01-01T00:00:00Z',
        });
      }
      throw new Error(`fixture가 처리하지 않는 요청입니다: ${path}`);
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function show(record: JobRecord, onJobsChanged = () => {}) {
  current = record;
  const listing: JobListing = { jobs: [record], active_job_id: null };
  return render(
    <MemoryRouter initialEntries={['/monitor/job-1']}>
      <Live listing={listing} onNewExperiment={() => {}} onJobsChanged={onJobsChanged} />
    </MemoryRouter>,
  );
}

describe('Live 기록 삭제', () => {
  it('무엇이 사라지고 무엇이 남는지 먼저 말한 뒤에 지운다', async () => {
    const onJobsChanged = vi.fn();
    show(job(), onJobsChanged);

    fireEvent.click(await screen.findByRole('button', { name: '기록 지우기' }));

    expect(screen.getByText(/이 GUI가 들고 있는 실행 기록과 로그/)).toBeInTheDocument();
    expect(screen.getByText(/학습 결과 폴더와 checkpoint/)).toBeInTheDocument();
    expect(deleted).toHaveLength(0);

    fireEvent.click(screen.getByRole('button', { name: '지웁니다' }));

    await waitFor(() => expect(deleted).toEqual(['/api/train/jobs/job-1']));
    expect(onJobsChanged).toHaveBeenCalled();
  });

  it('취소하면 아무것도 지우지 않는다', async () => {
    show(job());

    fireEvent.click(await screen.findByRole('button', { name: '기록 지우기' }));
    fireEvent.click(screen.getByRole('button', { name: '취소' }));

    expect(screen.queryByText(/이 GUI가 들고 있는 실행 기록과 로그/)).not.toBeInTheDocument();
    expect(deleted).toHaveLength(0);
  });

  it('도는 학습은 지울 수 없다', async () => {
    show(job({ status: 'running', status_label: '학습 중', finished_at: null }));

    expect(await screen.findByRole('button', { name: '기록 지우기' })).toBeDisabled();
  });
});
