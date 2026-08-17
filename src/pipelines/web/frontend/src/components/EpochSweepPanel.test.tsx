/**
 * epoch 훑기 판.
 *
 * validation loss가 고른 best epoch이 정말 제일 잘 맞히는지 재 보는 자리입니다.
 * 보관한 epoch이 없거나 순위 기준을 고르지 않았으면 시작할 수 없어야 합니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EpochSweepState, JobRecord } from '../api/types';
import { EpochSweepPanel } from './EpochSweepPanel';

let posted: unknown[] = [];
let payload: {
  epoch_sweep: EpochSweepState;
  candidates: { epoch: number; checkpoint_uri: string }[];
  metrics: string[] | null;
};

function job(): JobRecord {
  return {
    job_id: 'job-1',
    config_id: 'cfg-1',
    run_id: 'retina-run',
    status: 'succeeded',
    status_label: '성공',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T02:00:00Z',
    elapsed_seconds: 7200,
    exit_code: 0,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: {
      available: true,
      reason: null,
      message: null,
      total_epochs: 20,
      current_epoch: 20,
      completed_epochs: 20,
      eta_seconds: null,
      epochs: [],
    },
    log_lines: 10,
    orphan_note: null,
  };
}

beforeEach(() => {
  posted = [];
  payload = {
    epoch_sweep: { status: 'idle' },
    candidates: [
      { epoch: 15, checkpoint_uri: 'epochs/epoch_015.pt' },
      { epoch: 16, checkpoint_uri: 'epochs/epoch_016.pt' },
    ],
    metrics: ['mAP', 'mAP50', 'recall50'],
  };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.method === 'POST') {
        posted.push(JSON.parse(String(init.body)));
        return new Response(JSON.stringify({ epoch_sweep: { status: 'running' } }), {
          status: 202,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify(payload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('EpochSweepPanel', () => {
  it('보관한 epoch이 없으면 판을 두지 않는다', async () => {
    payload = { epoch_sweep: { status: 'idle' }, candidates: [], metrics: ['mAP', 'mAP50', 'recall50'] };
    const { container } = render(<EpochSweepPanel job={job()} />);

    await waitFor(() => expect(container.textContent).toBe(''));
  });

  it('표본 크기를 받아 훑기를 시작한다', async () => {
    render(<EpochSweepPanel job={job()} />);

    const start = await screen.findByRole('button', { name: '훑기 시작' });
    fireEvent.change(screen.getByDisplayValue('300'), { target: { value: '250' } });
    fireEvent.click(start);

    await waitFor(() => expect(posted).toEqual([{ sample_size: 250 }]));
  });

  // 무엇이 Kaggle 점수를 예측하는지 모르는 것이 이 기능을 만든 이유입니다. 아무도
  // 고르지 않은 기준으로 순위를 매기면 그 질문 자체가 사라집니다.
  it('순위 기준을 고르지 않았으면 시작할 수 없다', async () => {
    payload = { ...payload, metrics: null };
    render(<EpochSweepPanel job={job()} />);

    expect(await screen.findByText(/설정 화면에서 지표 3개를 순서대로 고르면/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '훑기 시작' })).toBeDisabled();
  });

  // 훑기는 thread로만 돕니다. 서버가 다시 뜨면 그 thread가 함께 사라지므로, 잰 것이
  // 없는데 끝난 것처럼 보입니다. 왜 다시 눌러야 하는지 말해 주어야 합니다.
  it('서버가 다시 떠서 중단된 훑기를 알려 준다', async () => {
    payload = {
      ...payload,
      epoch_sweep: { status: 'interrupted', message: '서버가 다시 시작되어 훑기가 중단됐습니다.' },
    };
    render(<EpochSweepPanel job={job()} />);

    expect(await screen.findByText('훑기가 중단됐습니다')).toBeInTheDocument();
  });

  it('끝난 훑기는 이긴 epoch과 그 실행 이름을 보여 준다', async () => {
    payload = {
      ...payload,
      epoch_sweep: {
        status: 'succeeded',
        metrics: ['mAP', 'mAP50', 'recall50'],
        message: '끝났습니다.',
        candidates: [
          { epoch: 16, checkpoint_uri: 'e16.pt', metrics: { mAP: 0.62 }, score: 0.9 },
          { epoch: 15, checkpoint_uri: 'e15.pt', metrics: { mAP: 0.61 }, score: 0.4 },
        ],
        winner: { epoch: 16, checkpoint_uri: 'e16.pt', run_id: 'retina-run-e16', score: 0.9 },
        artifacts: { submission_uri: 's3://bucket/submissions/retina-run-e16/submission.csv' },
        registration: { status: 'succeeded' },
      },
    };
    render(<EpochSweepPanel job={job()} />);

    expect(await screen.findByText('epoch 16이 이겼습니다')).toBeInTheDocument();
    expect(screen.getByText('retina-run-e16')).toBeInTheDocument();
    expect(screen.getByText('0.6200')).toBeInTheDocument();
  });
});
