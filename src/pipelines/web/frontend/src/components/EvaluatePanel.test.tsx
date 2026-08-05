import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { EvaluationState, JobRecord, Progress } from '../api/types';

const evaluationStatus = vi.fn();
const startEvaluation = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    evaluationStatus: (...args: unknown[]) => evaluationStatus(...args),
    startEvaluation: (...args: unknown[]) => startEvaluation(...args),
  },
}));

const { EvaluatePanel } = await import('./EvaluatePanel');

const NO_PROGRESS: Progress = {
  available: false,
  reason: null,
  message: null,
  total_epochs: null,
  current_epoch: null,
  eta_seconds: null,
  epochs: [],
};

function job(withTestManifest: boolean): JobRecord {
  return {
    job_id: 'a'.repeat(32),
    config_id: 'b'.repeat(32),
    run_id: 'run-1',
    status: 'succeeded',
    status_label: '성공',
    created_at: '2026-08-05T00:00:00Z',
    started_at: '2026-08-05T00:00:01Z',
    finished_at: '2026-08-05T00:01:00Z',
    elapsed_seconds: 59,
    exit_code: 0,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: withTestManifest
      ? { test_manifest_uri: 'artifacts/data/test_manifest.json' }
      : {},
    progress: NO_PROGRESS,
    log_lines: 0,
    orphan_note: null,
  };
}

function succeeded(competition: boolean): EvaluationState {
  return {
    status: 'succeeded',
    submission_requested: competition,
    message: 'evaluate pipeline 실행 완료',
    artifacts: competition
      ? { submission_uri: 'submissions/run-1/submission.csv' }
      : { metrics_uri: 'artifacts/evaluate/run-1/metrics.json' },
    summary: {
      iou_thresholds: competition
        ? [0.75, 0.8, 0.85, 0.9, 0.95]
        : [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
      metrics: { mAP: 0.3123, mAP50: competition ? null : 0.5512 },
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  startEvaluation.mockResolvedValue({ evaluation: { status: 'running' } });
});

describe('EvaluatePanel · 대회 제출 흐름', () => {
  it('test manifest가 있으면 대회 지표와 submission 위치를 분명히 보여 준다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(true) });

    render(<EvaluatePanel job={job(true)} />);

    expect(await screen.findByRole('button', { name: '평가 및 submission 생성' })).toBeInTheDocument();
    expect(await screen.findByText('mAP@[0.75:0.95]')).toBeInTheDocument();
    expect(screen.getByText('대회 제출 파일 생성 완료')).toBeInTheDocument();
    expect(screen.getByText('submissions/run-1/submission.csv')).toBeInTheDocument();
  });

  it('기존 4-artifact 학습은 validation 평가 화면을 그대로 유지한다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(false) });

    render(<EvaluatePanel job={job(false)} />);

    expect(await screen.findByRole('button', { name: '평가 실행' })).toBeInTheDocument();
    expect(await screen.findByText('mAP@0.5:0.95')).toBeInTheDocument();
    expect(screen.queryByText('대회 제출 파일 생성 완료')).toBeNull();
  });

  it('완료된 학습에 test manifest URI를 붙여 submission 평가를 시작한다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(false) });

    render(<EvaluatePanel job={job(false)} />);

    const input = await screen.findByLabelText(/Test manifest URI/);
    fireEvent.change(input, {
      target: { value: 's3://bucket/datasets/test/test_manifest.json' },
    });
    fireEvent.click(screen.getByRole('button', { name: '평가 및 submission 생성' }));

    await waitFor(() =>
      expect(startEvaluation).toHaveBeenCalledWith(
        'a'.repeat(32),
        expect.objectContaining({
          test_manifest_uri: 's3://bucket/datasets/test/test_manifest.json',
        }),
      ),
    );
  });

  it('덮어쓰기를 체크하면 기존 평가 파일 안내를 숨긴다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(false) });

    render(<EvaluatePanel job={job(false)} />);

    fireEvent.change(await screen.findByLabelText(/Test manifest URI/), {
      target: { value: 's3://bucket/datasets/test/test_manifest.json' },
    });
    expect(screen.getByText('기존 평가 파일이 있습니다')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: '이미 있으면 덮어쓰기' }));

    expect(screen.queryByText('기존 평가 파일이 있습니다')).toBeNull();
  });
});
