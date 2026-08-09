import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ExperimentDetail as Detail } from '../api/types';

const experimentDetail = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: { experimentDetail: (runId: string) => experimentDetail(runId) },
}));

const { ExperimentDetail } = await import('./ExperimentDetail');

function detail(overrides: Partial<Detail> = {}): Detail {
  return {
    experiment: {
      experiment_id: 'v3-dataset-baseline',
      run_id: 'v3-dataset-baseline',
      status: 'succeeded',
      status_label: '등록 완료',
      created_at: '2026-08-08T08:31:00Z',
      started_at: null,
      finished_at: null,
      elapsed_seconds: 16020,
      dataset: {
        identity: 'same',
        identity_source: 'artifact_set',
        artifacts_complete: true,
        label: 'v3-seed42-8020-group',
      },
      model: { architecture: 'retinanet_resnet50_fpn_v2', pretrained: true, source: 'record' },
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
        best_validation_loss: 0.0575,
        final_train_loss: 0.02,
        final_validation_loss: 0.061,
        map: 0.9664,
        map50: 0.9891,
        map75: 0.982,
        precision50: 0.95,
        recall50: 0.93,
      },
      completion: {
        evaluated: true,
        submitted: true,
        submission_checked: true,
        submission_rows: 2942,
      },
    },
    evaluation: {
      available: true,
      reason: null,
      metrics: {
        mAP: 0.9664,
        mAP50_95: 0.81,
        mAP75_95: 0.9664,
        mAP50: 0.9891,
        mAP75: 0.982,
        precision50: 0.95,
        recall50: 0.93,
        precision75: 0.91,
        recall75: 0.89,
      },
      counts: {
        image_count: 500,
        annotation_count: 1200,
        prediction_count: 1500,
        evaluated_class_count: 57,
      },
      score_threshold: 0.5,
      max_detections_per_image: 100,
      score_sweep: {
        '0.50': [
          { threshold: 0.05, precision: 0.7, recall: 1, f1: 0.82 },
          { threshold: 0.5, precision: 0.95, recall: 0.93, f1: 0.94 },
        ],
      },
      best_f1: { '0.50': { threshold: 0.5, precision: 0.95, recall: 0.93, f1: 0.94 } },
      per_class_summary: {
        min_truth_count: 10,
        top_n: 5,
        counts: { weak: 3, sparse: 2, unmeasured: 0 },
        weak: [{ category_id: 1, name: '타이레놀', ap: 0.21, truth_count: 40 }],
        sparse: [],
        unmeasured: [],
      },
    },
    history: {
      available: true,
      reason: null,
      epochs: [
        { epoch: 1, train_loss: 0.9, validation_loss: 0.8, epoch_seconds: 60, is_best: false },
        { epoch: 2, train_loss: 0.02, validation_loss: 0.0575, epoch_seconds: 60, is_best: true },
      ],
    },
    ...overrides,
  };
}

function show() {
  render(
    <MemoryRouter initialEntries={['/history/v3-dataset-baseline']}>
      <Routes>
        <Route path="/history/:runId" element={<ExperimentDetail />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ExperimentDetail', () => {
  beforeEach(() => {
    experimentDetail.mockReset();
  });

  it('평가 결과를 먼저 열고 핵심 지표를 맨 위에 둔다', async () => {
    experimentDetail.mockResolvedValue(detail());
    show();

    // 결론 자리의 KPI는 값에 한 줄 해석이 붙습니다. 접힌 표에도 같은 label이 있어
    // 그 해석 문구로 구분합니다.
    expect(await screen.findByText('대회가 보는 값입니다. 높을수록 좋습니다.')).toBeInTheDocument();
    expect(screen.getAllByText('0.9664').length).toBeGreaterThanOrEqual(1);
    // counts.weak(=점수를 매길 수 있었던 class 수)를 약한 class 수로 읽으면 57개
    // 전부가 약해 보입니다. 다음에 손댈 곳은 그중 가장 낮은 class입니다.
    expect(screen.getByText('가장 낮은 CLASS')).toBeInTheDocument();
    expect(screen.getByText('타이레놀')).toBeInTheDocument();
    expect(screen.getByText('0.2100')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: '평가 결과' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  it('지표 9개를 모두 담는다', async () => {
    experimentDetail.mockResolvedValue(detail());
    show();

    fireEvent.click(await screen.findByText('지표 전체'));

    // 예전 화면은 5개만 보여 줬습니다.
    expect(screen.getByText('Precision@IoU0.75')).toBeInTheDocument();
    expect(screen.getByText('0.9100')).toBeInTheDocument();
    expect(screen.getByText('Recall@IoU0.75')).toBeInTheDocument();
  });

  it('F1이 가장 높은 score 기준을 짚어 준다', async () => {
    experimentDetail.mockResolvedValue(detail());
    show();

    expect(
      await screen.findByText(/0.50에서 F1 0.9400로 가장 높습니다/),
    ).toBeInTheDocument();
  });

  it('세팅 탭에서 하이퍼파라미터를 보여 준다', async () => {
    experimentDetail.mockResolvedValue(detail());
    show();

    fireEvent.click(await screen.findByRole('tab', { name: '세팅' }));

    expect(screen.getByText('retinanet_resnet50_fpn_v2')).toBeInTheDocument();
    expect(screen.getByText('v3-seed42-8020-group')).toBeInTheDocument();
    expect(screen.getByText('0.0002')).toBeInTheDocument();
    // AdamW는 momentum을 쓰지 않으므로 beta만 보여 줍니다.
    expect(screen.getByText('Beta 1')).toBeInTheDocument();
    expect(screen.queryByText('Momentum')).not.toBeInTheDocument();
  });

  it('평가 파일을 못 읽어도 세팅은 볼 수 있다', async () => {
    experimentDetail.mockResolvedValue(
      detail({
        evaluation: {
          available: false,
          reason: '평가 결과 파일을 읽지 못했습니다.',
          metrics: {},
          counts: {},
          score_threshold: null,
          max_detections_per_image: null,
          score_sweep: {},
          best_f1: {},
          per_class_summary: null,
        },
        history: { available: false, reason: '학습 기록 파일이 없습니다.' },
      }),
    );
    show();

    expect(await screen.findByText('이 실험에는 볼 수 있는 결과 파일이 없습니다')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('tab', { name: '세팅' }));
    expect(screen.getByText('retinanet_resnet50_fpn_v2')).toBeInTheDocument();
  });
});
