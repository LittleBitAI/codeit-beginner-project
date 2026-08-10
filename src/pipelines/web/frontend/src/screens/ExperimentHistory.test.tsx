import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ExperimentListing, ExperimentSummary } from '../api/types';

const listExperiments = vi.fn();
const saveKaggleScore = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    listExperiments: () => listExperiments(),
    saveKaggleScore: (...args: unknown[]) => saveKaggleScore(...args),
  },
}));

const { ExperimentHistory } = await import('./ExperimentHistory');

function makeExperiment(
  runId: string,
  completion: Partial<ExperimentSummary['completion']> = {},
  kaggleScore: number | null = null,
): ExperimentSummary {
  return {
    experiment_id: runId,
    run_id: runId,
    status: 'succeeded',
    status_label: '등록 완료',
    created_at: '2026-08-09T00:00:00Z',
    started_at: null,
    finished_at: null,
    elapsed_seconds: null,
    dataset: {
      identity: 'same',
      identity_source: 'artifact_set',
      artifacts_complete: true,
      label: 'v3-seed42-8020-group',
    },
    model: { architecture: 'retinanet', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.0002,
      momentum: null,
      weight_decay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    training: { device: 'cuda', epochs: 15, batch_size: 4, num_workers: 0, seed: 42 },
    metrics: {
      best_epoch: 9,
      best_validation_loss: 0.06,
      final_train_loss: null,
      final_validation_loss: null,
      map: 0.9,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      kaggle_score: kaggleScore,
    },
    completion: {
      evaluated: true,
      submission_generated: true,
      submitted: kaggleScore !== null,
      submission_checked: true,
      submission_rows: 2942,
      ...completion,
    },
  };
}

function listing(experiments: ExperimentSummary[], shared = true): ExperimentListing {
  return { experiments, scope: { backend: shared ? 's3' : 'local', shared } };
}

function show() {
  render(
    <MemoryRouter>
      <ExperimentHistory />
    </MemoryRouter>,
  );
}

describe('ExperimentHistory', () => {
  beforeEach(() => {
    listExperiments.mockReset();
    saveKaggleScore.mockReset();
    saveKaggleScore.mockResolvedValue({ run_id: 'done', kaggle_score: 0.8123 });
  });

  it('평가 완료와 실제 제출 완료를 서로 다른 필터로 고른다', async () => {
    listExperiments.mockResolvedValue(
      listing([
        makeExperiment('submitted', {}, 0.8123),
        makeExperiment('evaluated-only'),
        makeExperiment('no-evaluation', { evaluated: false, submitted: false }),
      ]),
    );
    show();

    expect(await screen.findByText('submitted')).toBeInTheDocument();
    expect(screen.getByText('evaluated-only')).toBeInTheDocument();
    expect(screen.queryByText('no-evaluation')).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Kaggle 제출까지 끝난 실험만 보기'));

    expect(screen.getByText('submitted')).toBeInTheDocument();
    expect(screen.queryByText('evaluated-only')).not.toBeInTheDocument();
  });

  it('실험 행에서 Kaggle 점수를 직접 저장한다', async () => {
    listExperiments.mockResolvedValue(listing([makeExperiment('done')]));
    show();

    fireEvent.change(await screen.findByLabelText('done Kaggle 점수'), {
      target: { value: '0.8123' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'done Kaggle 점수 저장' }));

    await waitFor(() => expect(saveKaggleScore).toHaveBeenCalledWith('done', 0.8123));
  });

  it('자체평가 mAP와 실제 Kaggle 점수를 따로 정렬한다', async () => {
    const selfBest = makeExperiment('self-best', {}, 0.7);
    selfBest.metrics.map = 0.95;
    const actualBest = makeExperiment('actual-best', {}, 0.9);
    actualBest.metrics.map = 0.8;
    listExperiments.mockResolvedValue(listing([selfBest, actualBest]));
    show();

    fireEvent.click(await screen.findByRole('button', { name: 'mAP 높은 순(실제 점수)' }));

    const rows = document.querySelectorAll('[data-experiment-row]');
    expect(rows[0]).toHaveTextContent('actual-best');
    expect(screen.getByRole('button', { name: 'mAP 높은 순(자체평가)' })).toBeInTheDocument();
  });

  it('저장소가 팀 공용이 아니면 팀원 기록이 안 보인다고 알린다', async () => {
    listExperiments.mockResolvedValue(listing([makeExperiment('done')], false));
    show();

    expect(
      await screen.findByText('지금은 이 컴퓨터에 등록된 실험만 보입니다'),
    ).toBeInTheDocument();
  });

  it('completion이 없는 예전 backend 응답에도 화면이 죽지 않는다', async () => {
    // 이 field가 생기기 전 서버가 아직 떠 있으면 화면이 통째로 흰 채로 죽었습니다.
    const legacy = makeExperiment('legacy');
    delete (legacy as { completion?: unknown }).completion;
    listExperiments.mockResolvedValue(listing([legacy]));
    show();

    // 지표가 있으므로 평가는 끝난 것으로 보고, 제출은 알 수 없으니 아직으로 둡니다.
    fireEvent.click(await screen.findByLabelText('평가가 끝난 실험만 보기'));
    expect(screen.getByText('legacy')).toBeInTheDocument();
  });

  it('팀 공용 저장소면 그 안내를 띄우지 않는다', async () => {
    listExperiments.mockResolvedValue(listing([makeExperiment('done')], true));
    show();

    expect(await screen.findByText('done')).toBeInTheDocument();
    expect(
      screen.queryByText('지금은 이 컴퓨터에 등록된 실험만 보입니다'),
    ).not.toBeInTheDocument();
  });
});
