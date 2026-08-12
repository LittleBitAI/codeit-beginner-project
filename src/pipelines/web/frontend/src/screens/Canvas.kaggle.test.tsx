/**
 * Kaggle 점수는 사람이 직접 옮겨 적어야 하는 유일한 값입니다.
 *
 * 자동으로 채워지지 않고, 이 값이 있어야 '제출 완료'로 셉니다. 입력할 곳이 사라지면
 * 새 제출은 영영 기록되지 않습니다 — 그래서 화면 테스트를 남깁니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { ExperimentSummary } from '../api/types';
import { Canvas } from './Canvas';
import type { RunRecord } from '../lib/records';

let puts: { path: string; body: unknown }[] = [];
let kaggle: number | null = null;

function experiment(): ExperimentSummary {
  return {
    experiment_id: 'exp-1',
    run_id: 'retina-a7f3',
    status: 'succeeded',
    status_label: '완료',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T02:00:00Z',
    elapsed_seconds: 7200,
    dataset: { identity: 'sha:abc', identity_source: 'dataset_id', artifacts_complete: true, label: 'v5' },
    model: { architecture: 'retinanet_resnet50_fpn_v2', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.006,
      momentum: null,
      weight_decay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    training: {
      device: 'cuda',
      epochs: 15,
      batch_size: 4,
      num_workers: 4,
      gradient_accumulation_steps: 1,
      input_size: null,
      seed: 42,
    },
    metrics: {
      best_epoch: 12,
      best_validation_loss: 0.41,
      final_train_loss: 0.3,
      final_validation_loss: 0.44,
      map: 0.52,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      kaggle_score: kaggle,
    },
  };
}

function record(): RunRecord {
  return {
    runId: 'retina-a7f3',
    family: 'retinanet_resnet50_fpn_v2',
    datasetKey: 'v5',
    spec: 'e15 · b4',
    status: 'succeeded',
    statusLabel: '완료',
    at: '2026-01-01T02:00:00Z',
    jobId: null,
    registered: true,
    evaluated: true,
    submitted: false,
    metrics: {
      kaggle,
      map: 0.52,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      bestValidationLoss: 0.41,
      bestEpoch: 12,
      epochs: 15,
      elapsedSeconds: 7200,
    },
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  puts = [];
  kaggle = null;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (init?.method === 'PUT') {
        puts.push({ path, body: JSON.parse(String(init.body)) });
        kaggle = (JSON.parse(String(init.body)) as { score: number }).score;
        return jsonResponse({ run_id: 'retina-a7f3', kaggle_score: kaggle });
      }
      if (path === '/api/train/experiments/compare') {
        return jsonResponse({ experiments: [experiment()], missing: [] });
      }
      if (path.startsWith('/api/train/experiments/')) {
        return jsonResponse({
          experiment: experiment(),
          evaluation: { available: false, reason: 'fixture' },
          history: { available: false, reason: 'fixture', epochs: [] },
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

function show() {
  return render(
    <MemoryRouter initialEntries={['/canvas?run=retina-a7f3']}>
      <Canvas datasetKey="v5" records={[record()]} loading={false} />
    </MemoryRouter>,
  );
}

describe('Canvas Kaggle 점수', () => {
  it('점수가 없으면 곧바로 적을 수 있고 덮어쓰기를 요청하지 않는다', async () => {
    show();

    const input = await screen.findByRole('textbox', { name: 'Kaggle 점수' });
    fireEvent.change(input, { target: { value: '0.6123' } });
    fireEvent.click(screen.getByRole('button', { name: '저장' }));

    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]).toEqual({
      path: '/api/train/experiments/retina-a7f3/kaggle-score',
      // 처음 적는 값이라 overwrite를 켜지 않습니다.
      body: { score: 0.6123, overwrite: false },
    });
  });

  it('이미 적힌 점수는 잠가 두고 고치기를 눌러야 열린다', async () => {
    kaggle = 0.55;
    show();

    expect(await screen.findByText('0.5500')).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Kaggle 점수' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '고치기' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Kaggle 점수' }), {
      target: { value: '0.61' },
    });
    fireEvent.click(screen.getByRole('button', { name: '저장' }));

    // 이미 있는 값을 고치는 것이므로 서버에 덮어쓰기를 명시합니다.
    await waitFor(() => expect(puts).toHaveLength(1));
    expect(puts[0]?.body).toEqual({ score: 0.61, overwrite: true });
  });

  it('숫자가 아니면 보내지 않는다', async () => {
    show();

    const input = await screen.findByRole('textbox', { name: 'Kaggle 점수' });
    fireEvent.change(input, { target: { value: '아직' } });
    fireEvent.click(screen.getByRole('button', { name: '저장' }));

    expect(await screen.findByText('숫자를 적어 주세요.')).toBeInTheDocument();
    expect(puts).toHaveLength(0);
  });
});
