/**
 * 약한 class 표와 그 옆 길들.
 *
 * 값이 없는 것을 0으로 그리면 화면이 거짓말을 합니다. 그리고 기록에서 누르면 이제
 * 이 화면으로 오므로, 이 컴퓨터가 돌린 실행의 로그로 가는 길이 여기 있어야 합니다.
 */

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { ExperimentSummary } from '../api/types';
import type { RunRecord } from '../lib/records';
import { Canvas } from './Canvas';

afterEach(cleanup);

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    runId: 'run-a',
    family: 'retinanet_resnet50_fpn_v2',
    datasetKey: 'v6',
    spec: 'e15 · b4',
    status: 'succeeded',
    statusLabel: '완료',
    at: '2026-08-05T00:00:00Z',
    jobId: null,
    registered: true,
    evaluated: true,
    submitted: false,
    metrics: {
      kaggle: null,
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      bestValidationLoss: null,
      bestEpoch: null,
      epochs: 15,
      elapsedSeconds: null,
    },
    ...overrides,
  } as RunRecord;
}

/** 재지 못한 AP(null)를 담은 약한 class 하나를 든 실험입니다. */
function experiment(): ExperimentSummary {
  return {
    experiment_id: 'run-a',
    run_id: 'run-a',
    status: 'succeeded',
    status_label: '등록 완료',
    created_at: '2026-08-05T00:00:00Z',
    started_at: null,
    finished_at: null,
    elapsed_seconds: null,
    dataset: { identity: 'v6', identity_source: 'artifact_set', artifacts_complete: true, label: 'v6' },
    model: { architecture: 'retinanet_resnet50_fpn_v2', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.0001,
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
      num_workers: 0,
      gradient_accumulation_steps: 1,
      input_size: null,
      seed: 42,
    },
    per_class_summary: {
      min_truth_count: 5,
      top_n: 10,
      counts: { weak: 2, sparse: 0, unmeasured: 0 },
      weak: [
        { category_id: 16548, name: '가바토파정 100mg', ap: 0.12 },
        // 표본은 충분한데 AP를 재지 못한 줄입니다. evaluate가 허용하는 모양입니다.
        { category_id: 16232, name: '리피토정 20mg', ap: null },
      ],
      sparse: [],
    },
    metrics: {
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      best_epoch: null,
      best_validation_loss: null,
      final_train_loss: null,
      final_validation_loss: null,
      kaggle_score: null,
    },
  } as unknown as ExperimentSummary;
}

function show(records: RunRecord[]) {
  return render(
    <MemoryRouter initialEntries={['/canvas?run=run-a']}>
      <Canvas
        datasetKey="v6"
        records={records}
        loading={false}
        onScoreSaved={() => {}}
        onNewExperiment={() => {}}
      />
    </MemoryRouter>,
  );
}

describe('약한 class 표', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({ experiments: [experiment()], missing: [], curves: {} }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it('재지 못한 AP를 0으로 그리지 않는다', async () => {
    show([record()]);

    expect(await screen.findByText('리피토정 20mg')).toBeTruthy();
    // 0.120은 실제로 잰 값이라 나와야 하고, null은 0.000이 되면 안 됩니다.
    expect(screen.getByText('0.120')).toBeTruthy();
    expect(screen.queryByText('0.000')).toBeNull();
  });

  it('이 컴퓨터가 돌린 실행이면 로그로 가는 길을 낸다', async () => {
    show([record({ jobId: 'job-77' })]);

    expect(await screen.findByText('로그 보기')).toBeTruthy();
  });

  it('팀원이 돌린 실행에는 로그 링크를 내지 않는다', async () => {
    show([record({ jobId: null })]);

    expect(await screen.findByText('리피토정 20mg')).toBeTruthy();
    expect(screen.queryByText('로그 보기')).toBeNull();
  });
});
