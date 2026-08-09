import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ExperimentListing, ExperimentSummary } from '../api/types';

const listExperiments = vi.fn();

vi.mock('../api/client', () => ({
  api: { listExperiments: () => listExperiments() },
}));

const { ExperimentHistory } = await import('./ExperimentHistory');

function makeExperiment(
  runId: string,
  completion: Partial<ExperimentSummary['completion']> = {},
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
    },
    completion: {
      evaluated: true,
      submitted: true,
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
  });

  it('평가와 제출까지 끝난 실험만 보여 주고 몇 건을 감췄는지 말한다', async () => {
    listExperiments.mockResolvedValue(
      listing([
        makeExperiment('done'),
        makeExperiment('no-submission', { submitted: false }),
        makeExperiment('no-evaluation', { evaluated: false, submitted: false }),
      ]),
    );
    show();

    expect(await screen.findByText('done')).toBeInTheDocument();
    expect(screen.queryByText('no-submission')).not.toBeInTheDocument();
    // 조용히 빼면 그만큼이 없는 줄 압니다. 항상 몇 건인지 말합니다.
    expect(screen.getByText('아직 끝나지 않은 2건을 감췄습니다')).toBeInTheDocument();
  });

  it('체크를 풀면 아직 끝나지 않은 실험도 보여 준다', async () => {
    listExperiments.mockResolvedValue(
      listing([makeExperiment('done'), makeExperiment('no-submission', { submitted: false })]),
    );
    show();

    fireEvent.click(await screen.findByLabelText('평가와 제출까지 끝난 실험만 보기'));

    expect(screen.getByText('no-submission')).toBeInTheDocument();
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
    fireEvent.click(await screen.findByLabelText('평가와 제출까지 끝난 실험만 보기'));
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
