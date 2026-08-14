/**
 * 중지한 학습을 이어서 하는 부분입니다.
 *
 * 중지 단추를 누르면 이어서 학습할 방법이 화면에 없었습니다. epoch마다 저장한
 * checkpoint가 그대로 남아 있는데도 밤새 돌린 학습을 처음부터 다시 돌려야 했습니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { JobListing, JobRecord } from '../api/types';
import { Live } from './Live';

let posted: string[] = [];
let current: JobRecord | null = null;
/** 서버가 답할 이어하기 가능 여부. 화면은 이 답만 보고 단추를 세웁니다. */
let availability: { available: boolean; reason: string | null } = {
  available: true,
  reason: null,
};

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    job_id: 'job-1',
    config_id: 'cfg-1',
    run_id: 'retina-stopped',
    status: 'cancelled',
    status_label: '취소됨',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T02:00:00Z',
    elapsed_seconds: 7200,
    exit_code: null,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: {
      available: true,
      reason: null,
      message: null,
      total_epochs: 15,
      current_epoch: 7,
      completed_epochs: 7,
      eta_seconds: null,
      epochs: [{ epoch: 7, train_loss: 0.5, validation_loss: 0.55, epoch_seconds: 100, is_best: true }],
    },
    log_lines: 40,
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
  posted = [];
  current = null;
  availability = { available: true, reason: null };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path =
        typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (init?.method === 'POST') {
        posted.push(path);
        return jsonResponse({
          config_id: 'cfg-2',
          run_id: 'retina-stopped-resume-20260101T030000Z',
          resumed_from_job_id: 'job-1',
          resume_from: 'artifacts/experiments/completed/.retina-stopped.partial/last_checkpoint.pt',
          // 줄만 서고 시작하지는 않은 경우입니다. 시작하면 화면이 그쪽으로 넘어갑니다.
          started: null,
          entries: [],
          paused: false,
        });
      }
      if (path.startsWith('/api/train/jobs/job-1/logs')) {
        return jsonResponse({ lines: [], next: 0, complete: true });
      }
      // 이어갈 수 있는지는 서버가 답합니다. 화면은 그 답만 씁니다.
      if (path === '/api/train/jobs/job-1/resume') return jsonResponse(availability);
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

function show(record: JobRecord) {
  current = record;
  const listing: JobListing = { jobs: [record], active_job_id: null };
  return render(
    <MemoryRouter initialEntries={['/monitor/job-1']}>
      <Live listing={listing} onNewExperiment={() => {}} onJobsChanged={() => {}} />
    </MemoryRouter>,
  );
}

describe('Live 이어서 학습', () => {
  it('중지한 학습도 마친 epoch이 있으면 이어서 학습한다', async () => {
    show(job());

    fireEvent.click(await screen.findByRole('button', { name: '이어서 학습' }));

    await waitFor(() => expect(posted).toEqual(['/api/train/jobs/job-1/resume']));
    expect(
      await screen.findByText(/retina-stopped-resume-20260101T030000Z' 이름으로 대기열에 넣었습니다/),
    ).toBeInTheDocument();
  });

  // **마친 epoch 수로는 알 수 없는 경우입니다.** 이 학습은 epoch 7까지 마쳤지만
  // checkpoint는 주기로만 저장되고, 이어온 실행이면 앞선 실행의 epoch까지 섞여 있습니다.
  // 화면이 세면 단추를 세우고 서버는 거절합니다 — 그래서 서버 답만 봅니다.
  it('서버가 이어갈 수 없다고 하면 단추를 두지 않고 그 이유를 적는다', async () => {
    availability = {
      available: false,
      reason: '저장된 checkpoint가 없습니다. checkpoint 주기를 채우기 전에 끝난 학습입니다.',
    };
    show(job());

    // 서버 답이 도착한 **뒤에** 재야 합니다. 먼저 재면 아직 안 온 답 때문에 단추가
    // 없는 것을 보고 통과해 버립니다.
    expect(await screen.findByText(/저장된 checkpoint가 없습니다/)).toBeInTheDocument();
    expect(screen.getByText('학습을 중지했습니다')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '이어서 학습' })).toBeNull();
  });
});
