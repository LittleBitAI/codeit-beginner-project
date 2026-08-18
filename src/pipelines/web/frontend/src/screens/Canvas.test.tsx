import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { RunRecord } from '../lib/records';
import { Canvas } from './Canvas';

afterEach(cleanup);

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    runId: 'retina-e15-b4-a7f3',
    family: 'retinanet_resnet50_fpn_v2',
    datasetKey: 'v4-seed42-8020-group',
    spec: 'e15 · b4 · lr 0.006 · seed 42',
    status: 'succeeded',
    statusLabel: '완료',
    at: '2026-08-05T00:00:00Z',
    jobId: null,
    registered: true,
    evaluated: true,
    submitted: false,
    metrics: {
      kaggle: 0.61,
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
    ...overrides,
  };
}

function show(records: RunRecord[], datasetKey: string | null = 'v4-seed42-8020-group') {
  return render(
    <MemoryRouter>
      <Canvas datasetKey={datasetKey} records={records} loading={false} onScoreSaved={() => {}} onNewExperiment={() => {}} onOpenDiagnosis={() => {}} />
    </MemoryRouter>,
  );
}

describe('Canvas', () => {
  it('어느 dataset 안에서 고르는 중인지 적는다', () => {
    show([record()]);

    expect(screen.getByText('v4-seed42-8020-group')).toBeInTheDocument();
  });

  it('넘겨받은 dataset의 기록만 목록에 올린다', () => {
    // App이 이미 dataset으로 걸러 넘깁니다. 화면은 받은 것만 그립니다 — 데이터가
    // 다른 실행을 나란히 세우면 모델 차이인지 데이터 차이인지 구별할 수 없습니다.
    show([record({ runId: 'v4-a' }), record({ runId: 'v4-b' })]);

    expect(screen.getByRole('button', { name: /v4-a/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /v4-b/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /v1-/ })).not.toBeInTheDocument();
  });

  it('등록되지 않은 실행은 고를 수 없다고 건수까지 말한다', () => {
    show([record({ runId: 'registered' }), record({ runId: 'local-only', registered: false })]);

    expect(screen.getByRole('button', { name: /registered/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /local-only/ })).not.toBeInTheDocument();
    expect(screen.getByText(/등록되지 않은 1건은 목록에 없습니다/)).toBeInTheDocument();
  });

  it('전부 미등록이면 왜 비었는지 말한다', () => {
    show([record({ registered: false })]);

    expect(screen.getByText(/registry에 등록되지 않았습니다/)).toBeInTheDocument();
  });
});
