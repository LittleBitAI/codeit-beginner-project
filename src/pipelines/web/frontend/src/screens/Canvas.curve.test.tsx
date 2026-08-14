/**
 * 곡선이 비었을 때 화면이 **이유를 지어내지 않는지** 봅니다.
 *
 * 빈 곡선은 두 가지입니다: 아직 한 epoch도 안 끝났거나, 학습 기록 파일을 못 읽었거나.
 * 서버가 어느 쪽인지 함께 보내는데 화면이 그것을 버리고 늘 "epoch이 하나도 끝나지
 * 않았다"고 적으면, S3가 잠깐 흔들린 것도 그렇게 읽힙니다.
 */

import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { ExperimentSummary } from '../api/types';
import type { RunRecord } from '../lib/records';
import { Canvas } from './Canvas';

const REASON = '학습 기록 파일을 읽지 못했습니다.';

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
      kaggle_score: null,
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
      kaggle: null,
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

/** 곡선을 못 읽은 비교 응답 하나만 돌려주는 fixture입니다. */
function stubCompare(curve: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (path !== '/api/train/experiments/compare') {
        throw new Error(`fixture가 처리하지 않는 요청입니다: ${path}`);
      }
      return new Response(
        JSON.stringify({
          experiments: [experiment()],
          missing: [],
          curves: { 'retina-a7f3': curve },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }),
  );
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function show() {
  return render(
    <MemoryRouter initialEntries={['/canvas?run=retina-a7f3']}>
      <Canvas datasetKey="v5" records={[record()]} loading={false} onScoreSaved={() => {}} />
    </MemoryRouter>,
  );
}

describe('Canvas 곡선', () => {
  it('학습 기록을 못 읽었으면 그 이유를 적고 epoch 수를 단정하지 않는다', async () => {
    stubCompare({ available: false, reason: REASON, epochs: [] });

    show();

    expect(await screen.findByText(REASON)).toBeInTheDocument();
    expect(screen.queryByText(/epoch이 하나도 끝나지 않아/)).toBeNull();
  });

  it('정말로 한 epoch도 안 끝났으면 지금까지처럼 그렇게 적는다', async () => {
    stubCompare({ available: true, reason: null, epochs: [] });

    show();

    expect(await screen.findByText(/epoch이 하나도 끝나지 않아/)).toBeInTheDocument();
  });

  it('곡선이 있으면 그림을 그린다', async () => {
    stubCompare({
      available: true,
      reason: null,
      epochs: [
        { epoch: 1, train_loss: 0.9, validation_loss: 0.8, epoch_seconds: 10, is_best: false },
        { epoch: 2, train_loss: 0.5, validation_loss: 0.4, epoch_seconds: 10, is_best: true },
      ],
    });

    show();

    await waitFor(() => expect(document.querySelector('svg polyline')).not.toBeNull());
    expect(screen.queryByText(/epoch이 하나도 끝나지 않아/)).toBeNull();
  });
});
