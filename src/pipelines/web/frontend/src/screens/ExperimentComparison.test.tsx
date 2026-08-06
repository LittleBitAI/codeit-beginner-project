import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ExperimentSummary } from '../api/types';
import { datasetRelationship } from '../lib/experiments';

const listExperiments = vi.fn();
const compareExperiments = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    listExperiments: () => listExperiments(),
    compareExperiments: (runIds: string[]) => compareExperiments(runIds),
  },
}));

const { ExperimentComparison } = await import('./ExperimentComparison');

function makeExperiment(
  id: string,
  datasetIdentity: string | null,
): ExperimentSummary {
  return {
    experiment_id: id.repeat(32),
    run_id: `run-${id}`,
    status: 'succeeded',
    status_label: '성공',
    created_at: '2026-08-05T00:00:00Z',
    started_at: '2026-08-05T00:00:00Z',
    finished_at: '2026-08-05T00:01:00Z',
    elapsed_seconds: 60,
    dataset: {
      identity: datasetIdentity,
      identity_source: datasetIdentity ? 'artifact_set' : 'unknown',
      artifacts_complete: datasetIdentity !== null,
    },
    model: { architecture: 'fasterrcnn', pretrained: true, source: 'record' },
    optimizer: {
      name: 'SGD',
      source: 'legacy_fallback',
      learning_rate: 0.005,
      momentum: 0.9,
      weight_decay: 0.0005,
      beta1: null,
      beta2: null,
      epsilon: null,
    },
    training: { device: 'cpu', epochs: 2, batch_size: 1, num_workers: 0, seed: 42 },
    metrics: { best_epoch: 2, best_validation_loss: 0.4, map: null, map50: null },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('datasetRelationship', () => {
  it('같음, 다름, 판정 불가를 세 상태로 구분한다', () => {
    const first = makeExperiment('a', 'same');
    const second = makeExperiment('b', 'same');
    const different = makeExperiment('c', 'other');
    const unknown = makeExperiment('d', null);

    expect(datasetRelationship([first, second])).toBe('same');
    expect(datasetRelationship([first, different])).toBe('different');
    expect(datasetRelationship([first, unknown])).toBe('unknown');
  });
});

describe('ExperimentComparison', () => {
  it('두 실험을 선택하면 dataset 안내와 비교표를 보여 준다', async () => {
    listExperiments.mockResolvedValue({
      experiments: [makeExperiment('a', 'same'), makeExperiment('b', 'same')],
    });
    compareExperiments.mockResolvedValue({
      experiments: [makeExperiment('a', 'same'), makeExperiment('b', 'same')],
      missing: [],
    });
    render(
      <MemoryRouter>
        <ExperimentComparison />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByLabelText('run-a 비교 선택'));
    fireEvent.click(screen.getByLabelText('run-b 비교 선택'));

    expect(await screen.findByText('같은 dataset 입력으로 기록된 실험입니다')).toBeInTheDocument();
    expect(screen.getByText('BEST VAL LOSS')).toBeInTheDocument();
    // evaluate의 `mAP`는 mAP@[0.75:0.95]입니다. 그냥 mAP라고만 적으면 오해합니다.
    expect(screen.getByText('mAP@[0.75:0.95]')).toBeInTheDocument();
    expect(screen.getAllByText('0.4000')).toHaveLength(2);
    expect(screen.getAllByText('SGD (호환 기본값)')).toHaveLength(2);
    expect(compareExperiments).toHaveBeenCalledWith(['run-a', 'run-b']);
  });

  it('실험 하나만 골라도 모델과 하이퍼파라미터를 채워 준다', async () => {
    listExperiments.mockResolvedValue({ experiments: [makeExperiment('a', 'same')] });
    compareExperiments.mockResolvedValue({
      experiments: [makeExperiment('a', 'same')],
      missing: [],
    });
    render(
      <MemoryRouter>
        <ExperimentComparison />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByLabelText('run-a 비교 선택'));

    // 목록 응답은 registry index만 읽어 모델과 optimizer가 null입니다. record를
    // 읽는 비교 요청을 거쳐야 값이 채워집니다.
    expect(await screen.findByText('fasterrcnn')).toBeInTheDocument();
    expect(screen.getByText('SGD (호환 기본값)')).toBeInTheDocument();
    expect(compareExperiments).toHaveBeenCalledWith(['run-a']);
  });

  it('고르기 전에도 목록 각 행에 모델과 mAP 요약 한 줄을 보여 준다', async () => {
    const filled = makeExperiment('a', 'same');
    filled.metrics.map = 0.3125;
    const unknown = makeExperiment('b', 'same');
    unknown.model.architecture = null;
    listExperiments.mockResolvedValue({ experiments: [filled, unknown] });
    render(
      <MemoryRouter>
        <ExperimentComparison />
      </MemoryRouter>,
    );

    expect(await screen.findByText('fasterrcnn · mAP@[0.75:0.95] 0.3125')).toBeInTheDocument();
    // 기록에 없는 값은 추정하지 않고 - 로 둡니다.
    expect(screen.getByText('- · mAP@[0.75:0.95] -')).toBeInTheDocument();
    expect(compareExperiments).not.toHaveBeenCalled();
  });

  it('dataset 기록이 빠진 선택은 판정 불가로 알린다', async () => {
    listExperiments.mockResolvedValue({
      experiments: [makeExperiment('a', 'same'), makeExperiment('b', null)],
    });
    compareExperiments.mockResolvedValue({
      experiments: [makeExperiment('a', 'same'), makeExperiment('b', null)],
      missing: [],
    });
    render(
      <MemoryRouter>
        <ExperimentComparison />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByLabelText('run-a 비교 선택'));
    fireEvent.click(screen.getByLabelText('run-b 비교 선택'));

    expect(await screen.findByText('dataset 동일 여부를 판정할 수 없습니다')).toBeInTheDocument();
  });
});
