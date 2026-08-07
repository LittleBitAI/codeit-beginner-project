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
    metrics: {
      best_epoch: 2,
      best_validation_loss: 0.4,
      final_train_loss: null,
      final_validation_loss: null,
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
    },
  };
}

/** 주어진 실험을 모두 골라 비교표를 연 화면입니다. 초록 표시 test가 같은 길을 씁니다. */
async function renderComparison(experiments: ExperimentSummary[]) {
  listExperiments.mockResolvedValue({ experiments });
  compareExperiments.mockResolvedValue({ experiments, missing: [] });
  const view = render(
    <MemoryRouter>
      <ExperimentComparison />
    </MemoryRouter>,
  );

  for (const experiment of experiments) {
    fireEvent.click(await screen.findByLabelText(`${experiment.run_id} 비교 선택`));
  }
  await screen.findByRole('button', { name: '결과값' });
  return view;
}

/** 어떤 줄의 어떤 실험 칸이 초록으로 칠해졌는지 읽습니다. */
function bestRunIds(container: HTMLElement, label: string): string[] {
  return Array.from(
    container.querySelectorAll(`[data-row="${label}"][data-best="true"]`),
  ).map((cell) => cell.getAttribute('data-run') ?? '');
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

    fireEvent.click(screen.getByRole('button', { name: '학습 세팅' }));

    expect(screen.getAllByText('SGD (호환 기본값)')).toHaveLength(2);
    expect(compareExperiments).toHaveBeenCalledWith(['run-a', 'run-b']);
  });

  it('목록 polling이 돌아도 같은 선택을 다시 요청하지 않는다', async () => {
    // 목록은 3초마다 polling한다. 그때마다 새 배열이 만들어져 비교 요청이 다시
    // 나가면, 앞 응답은 정리 함수가 버린다. 비교가 polling 주기보다 오래 걸리면
    // 표는 영원히 채워지지 않는다. 실제로 20초 동안 비교 요청이 5번 나갔다.
    vi.useFakeTimers();
    try {
      listExperiments.mockImplementation(async () => ({
        // polling마다 새 객체를 돌려주는 실제 동작 그대로다.
        experiments: [makeExperiment('a', 'same'), makeExperiment('b', 'same')],
      }));
      compareExperiments.mockResolvedValue({
        experiments: [makeExperiment('a', 'same'), makeExperiment('b', 'same')],
        missing: [],
      });
      render(
        <MemoryRouter>
          <ExperimentComparison />
        </MemoryRouter>,
      );

      // 첫 목록 요청이 끝나 목록이 그려질 때까지 microtask를 흘려보낸다.
      for (let attempt = 0; attempt < 20; attempt += 1) {
        if (screen.queryByLabelText('run-a 비교 선택')) break;
        await vi.advanceTimersByTimeAsync(10);
      }
      fireEvent.click(screen.getByLabelText('run-a 비교 선택'));
      fireEvent.click(screen.getByLabelText('run-b 비교 선택'));
      await vi.advanceTimersByTimeAsync(10);
      const afterSelect = compareExperiments.mock.calls.length;
      expect(afterSelect).toBeGreaterThan(0);

      // polling을 여러 번 돌린다. 선택은 그대로다.
      await vi.advanceTimersByTimeAsync(12_000);

      expect(compareExperiments.mock.calls.length).toBe(afterSelect);
    } finally {
      vi.useRealTimers();
    }
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
    fireEvent.click(await screen.findByRole('button', { name: '학습 세팅' }));

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

  it('토글로 학습 세팅과 결과값 두 묶음을 갈아 끼운다', async () => {
    const { container } = await renderComparison([
      makeExperiment('a', 'same'),
      makeExperiment('b', 'same'),
    ]);

    // 기본은 결과값입니다. 사람이 비교하려는 이유가 결과이기 때문입니다.
    expect(screen.getByText('FINAL TRAIN LOSS')).toBeInTheDocument();
    expect(screen.queryByText('LEARNING RATE')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '학습 세팅' }));

    expect(screen.getByText('LEARNING RATE')).toBeInTheDocument();
    expect(screen.queryByText('FINAL TRAIN LOSS')).not.toBeInTheDocument();
    // 상태와 dataset 관계는 탭을 오가도 사라지지 않습니다.
    expect(container.querySelectorAll('[data-row="DATASET 관계"]').length).toBe(2);
  });

  it('loss는 낮은 쪽, 지표는 높은 쪽을 초록으로 표시한다', async () => {
    const low = makeExperiment('a', 'same');
    low.metrics = { ...low.metrics, best_validation_loss: 0.4, best_epoch: 2, map: 0.2 };
    const high = makeExperiment('b', 'same');
    high.metrics = { ...high.metrics, best_validation_loss: 0.7, best_epoch: 5, map: 0.5 };

    const { container } = await renderComparison([low, high]);

    expect(bestRunIds(container, 'BEST VAL LOSS')).toEqual(['run-a']);
    expect(bestRunIds(container, 'mAP@[0.75:0.95]')).toEqual(['run-b']);
    // BEST EPOCH는 크고 작음에 좋고 나쁨이 없습니다.
    expect(bestRunIds(container, 'BEST EPOCH')).toEqual([]);
  });

  it('값이 모두 같으면 아무 칸도 초록이 아니다', async () => {
    const { container } = await renderComparison([
      makeExperiment('a', 'same'),
      makeExperiment('b', 'same'),
    ]);

    expect(bestRunIds(container, 'BEST VAL LOSS')).toEqual([]);
  });

  it('값이 없는 칸은 이기지 않는다', async () => {
    const lower = makeExperiment('a', 'same');
    lower.metrics = { ...lower.metrics, map50: 0.5 };
    const higher = makeExperiment('b', 'same');
    higher.metrics = { ...higher.metrics, map50: 0.7 };
    const missing = makeExperiment('c', 'same');

    const { container } = await renderComparison([lower, higher, missing]);

    expect(missing.metrics.map50).toBeNull();
    expect(bestRunIds(container, 'mAP@0.5')).toEqual(['run-b']);
  });

  it('실험이 하나면 초록이 없다', async () => {
    const only = makeExperiment('a', 'same');
    only.metrics = { ...only.metrics, best_validation_loss: 0.4, map: 0.5 };

    const { container } = await renderComparison([only]);

    expect(bestRunIds(container, 'BEST VAL LOSS')).toEqual([]);
    expect(bestRunIds(container, 'mAP@[0.75:0.95]')).toEqual([]);
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
