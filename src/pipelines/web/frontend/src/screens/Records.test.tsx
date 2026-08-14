import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { RunRecord } from '../lib/records';
import { Records } from './Records';

afterEach(cleanup);

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    runId: 'retina-e15-b4-a7f3',
    family: 'retinanet_resnet50_fpn_v2',
    datasetKey: 'v5-118cls',
    spec: 'e15 · b4 · lr 0.006 · seed 42',
    status: 'succeeded',
    statusLabel: '완료',
    at: '2026-08-05T00:00:00Z',
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
    ...overrides,
  };
}

/** 결과를 남기지 못하고 끝난 기록. 검증 오차가 없다는 것이 판단 기준입니다. */
function failedRecord(overrides: Partial<RunRecord> = {}): RunRecord {
  return record({
    runId: 'oom',
    status: 'failed',
    statusLabel: '실패',
    registered: false,
    evaluated: false,
    metrics: { ...record().metrics, bestValidationLoss: null },
    ...overrides,
  });
}

function cancelledRecord(overrides: Partial<RunRecord> = {}): RunRecord {
  return failedRecord({
    runId: 'stopped',
    status: 'cancelled',
    statusLabel: '취소됨',
    ...overrides,
  });
}

function show(props: Partial<Parameters<typeof Records>[0]> = {}) {
  return render(
    <MemoryRouter>
      <Records
        datasets={[{ key: 'v5-118cls', sub: '1건의 기록', count: 1 }]}
        datasetKey="v5-118cls"
        onPickDataset={() => {}}
        records={[record()]}
        scope={{ backend: 'local', shared: false }}
        unnamedCount={0}
        error={null}
        onNewExperiment={() => {}}
        {...props}
      />
    </MemoryRouter>,
  );
}

describe('Records', () => {
  it('어떤 dataset의 기록을 보는지 고르고, 그것이 학습 입력이 아님을 말한다', () => {
    const onPickDataset = vi.fn();
    show({
      datasets: [
        { key: 'v5-118cls', sub: '1건의 기록', count: 1 },
        { key: 'v6-57cls', sub: '기록 없음 · 학습 전', count: 0 },
      ],
      onPickDataset,
    });

    // 학습에 실려 갈 데이터는 dataset 준비에서만 바뀝니다. 여기서 고르는 것은 보는 대상뿐입니다.
    expect(screen.getByText(/학습에 쓰는 데이터는 dataset 준비에서 고릅니다/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('DATASET'), { target: { value: 'v6-57cls' } });

    expect(onPickDataset).toHaveBeenCalledWith('v6-57cls');
  });

  it('모델별로 묶고 머리글에 건수와 가장 좋은 값을 적는다', () => {
    show({
      records: [
        record({ runId: 'retina-1', metrics: { ...record().metrics, kaggle: 0.61 } }),
        record({ runId: 'retina-2' }),
        record({ runId: 'faster-1', family: 'fasterrcnn_resnet50_fpn_v2' }),
      ],
    });

    expect(screen.getByRole('button', { name: /retinanet_resnet50_fpn_v2/ })).toHaveTextContent(
      '2건 · 최고 Kaggle 0.6100 · 최저 val 0.4100',
    );
    // 접으면 그 모델의 줄만 사라집니다.
    fireEvent.click(screen.getByRole('button', { name: /retinanet_resnet50_fpn_v2/ }));
    expect(screen.queryByText('retina-1')).toBeNull();
    expect(screen.getByText('faster-1')).toBeInTheDocument();
  });

  it('조건에 맞는 기록이 없으면 감췄다고 말한다', () => {
    show({ records: [record({ evaluated: false, registered: true })] });

    fireEvent.click(screen.getByRole('button', { name: /평가 완료/ }));

    expect(screen.getByText(/고른 조건에 맞는 기록이 없습니다/)).toBeInTheDocument();
  });

  it('Kaggle 점수가 없는 기록은 정렬해도 위로 올라오지 않는다', () => {
    show({
      records: [
        record({ runId: 'no-score', metrics: { ...record().metrics, kaggle: null } }),
        record({ runId: 'scored', metrics: { ...record().metrics, kaggle: 0.61 } }),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Kaggle' }));

    const ids = screen.getAllByText(/^(no-score|scored)$/).map((node) => node.textContent);
    expect(ids[0]).toBe('scored');
  });

  // 35건 중 32건이 결과 없이 끝난 기록이라 볼 것 3건이 가운데 묻혀 있었습니다.
  it('결과 없이 끝난 기록은 접어 두고 몇 건인지 말한다', () => {
    show({ records: [record({ runId: 'good' }), failedRecord(), cancelledRecord()] });

    expect(screen.getByText('good')).toBeInTheDocument();
    expect(screen.queryByText('oom')).toBeNull();
    expect(screen.queryByText('stopped')).toBeNull();
    expect(screen.getByText('2건 (실패 1 · 취소·중단 1)')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /결과 없이 끝남/ }));

    // 접은 것을 전부 되돌려 줘야 합니다. 하나라도 빠지면 그 기록은 어디에서도 못 봅니다.
    expect(screen.getByText('oom')).toBeInTheDocument();
    expect(screen.getByText('stopped')).toBeInTheDocument();
  });

  // 등록되지 않은 이유가 실패라면 배지 두 개가 같은 말을 두 번 합니다. 그렇다고 상태를
  // 통째로 지우면 취소·중단 줄에 아무 표시도 남지 않아, 성공한 기록과 구별되지 않습니다.
  it('끝난 이유는 한 번만, 그러나 반드시 적는다', () => {
    show({
      records: [
        failedRecord(),
        cancelledRecord({ metrics: { ...record().metrics, bestValidationLoss: 0.5 } }),
        // 중단은 이어서 학습할 대상이라 특히 눈에 띄어야 합니다. 이름만 "취소·중단"이라
        // 적고 취소만 넣으면, 중단 조건이 사라져도 이 test는 통과합니다.
        failedRecord({ runId: 'lost', status: 'interrupted', statusLabel: '중단됨' }),
        record({ runId: 'done', registered: false }),
      ],
    });

    // 결과가 남은 취소, 중단, 미등록 성공은 접히지 않습니다.
    expect(screen.getByText('취소됨')).toBeInTheDocument();
    expect(screen.getByText('중단됨')).toBeInTheDocument();
    expect(screen.getByText('미등록')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /결과 없이 끝남/ }));

    expect(screen.getByText('실패')).toBeInTheDocument();
    // 실패한 줄에는 미등록이 겹치지 않습니다 — 미등록 배지는 위의 성공 줄 하나뿐입니다.
    expect(screen.getAllByText('미등록')).toHaveLength(1);
  });

  // 접기는 전체 표에서만 합니다. 이미 좁혀 놓은 표에서 또 접으면 "12건이라는데 아무것도
  // 안 보인다"가 됩니다.
  it('미등록·실패 표에서는 접지 않는다', () => {
    show({ records: [record({ runId: 'good' }), failedRecord(), cancelledRecord()] });

    fireEvent.click(screen.getByRole('button', { name: /미등록·실패/ }));

    expect(screen.queryByText(/결과 없이 끝남/)).toBeNull();
    expect(screen.getByText('oom')).toBeInTheDocument();
    expect(screen.getByText('stopped')).toBeInTheDocument();
  });

  it('로컬 저장소면 팀원 기록이 왜 없는지 화면이 말한다', () => {
    show();

    expect(screen.getByText('이 컴퓨터')).toBeInTheDocument();
    expect(screen.getByText(/PILL_STORAGE_S3_BUCKET/)).toBeInTheDocument();
  });

  it('registry를 아직 못 읽었으면 이 컴퓨터뿐이라고 단정하지 않는다', () => {
    show({ scope: undefined });

    expect(screen.getByText('읽는 중')).toBeInTheDocument();
    expect(screen.queryByText('이 컴퓨터')).not.toBeInTheDocument();
  });

  it('dataset을 알 수 없어 감춘 기록이 있으면 몇 건인지 말한다', () => {
    show({ unnamedCount: 17 });

    expect(screen.getByText(/알 수 없는 기록 17건은 위 목록에 세우지 않았습니다/)).toBeInTheDocument();
  });
});
