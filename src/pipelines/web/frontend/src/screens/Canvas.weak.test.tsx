/**
 * 약한 class 표와 그 옆 길들.
 *
 * 값이 없는 것을 0으로 그리면 화면이 거짓말을 합니다. 그리고 기록에서 누르면 이제
 * 이 화면으로 오므로, 이 컴퓨터가 돌린 실행의 로그로 가는 길이 여기 있어야 합니다.
 */

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

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
function experiment(weakCount = 2): ExperimentSummary {
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
      counts: { weak: weakCount, sparse: 0, unmeasured: 0 },
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
      <Routes>
        <Route
          path="/canvas"
          element={
            <Canvas
              datasetKey="v6"
              records={records}
              loading={false}
              onScoreSaved={() => {}}
              onNewExperiment={() => {}}
            />
          }
        />
        {/* 링크가 있는 것만으로는 부족합니다. 어디로 가는지까지 봅니다. */}
        <Route path="/monitor/:jobId" element={<div>모니터 화면</div>} />
      </Routes>
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

  it('이 컴퓨터가 돌린 실행이면 그 job의 로그 화면으로 보낸다', async () => {
    show([record({ jobId: 'job-77' })]);

    fireEvent.click(await screen.findByText('로그 보기'));

    expect(await screen.findByText('모니터 화면')).toBeTruthy();
  });

  it('목록이 잘려 있으면 없는 class를 약하지 않다고 적지 않는다', async () => {
    // run-b는 상위 1개만 받았고(counts 9 > 목록 1) 그 안에 가바토파정이 없습니다.
    // 실제로 약한데 순위 밖일 수 있으므로 "-"(약하지 않음)라고 말하면 안 됩니다.
    const other = {
      ...experiment(9),
      experiment_id: 'run-b',
      run_id: 'run-b',
      per_class_summary: {
        min_truth_count: 5,
        top_n: 1,
        counts: { weak: 9, sparse: 0, unmeasured: 0 },
        weak: [{ category_id: 99999, name: '다른 알약', ap: 0.05 }],
        sparse: [],
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({ experiments: [experiment(), other], missing: [], curves: {} }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );
    show([record(), record({ runId: 'run-b' })]);

    expect(await screen.findByText('가바토파정 100mg')).toBeTruthy();
    // run-b 칸이 "순위 밖"이어야 합니다. "-"로 적으면 약하지 않다고 단정하는 것입니다.
    expect(screen.getAllByText('순위 밖').length).toBeGreaterThan(0);
  });

  it('어느 칸이 어느 실행인지 머리글로 밝힌다', async () => {
    show([record()]);

    // 실행 이름은 곡선 범례에도 있습니다. 표 안에서 찾아야 머리글을 지킵니다.
    const table = (await screen.findByText('약한 class')).parentElement as HTMLElement;
    expect(within(table).getAllByText('run-a').length).toBeGreaterThan(0);
  });

  it('팀원이 돌린 실행에는 로그 링크를 내지 않는다', async () => {
    show([record({ jobId: null })]);

    expect(await screen.findByText('리피토정 20mg')).toBeTruthy();
    expect(screen.queryByText('로그 보기')).toBeNull();
  });
});
