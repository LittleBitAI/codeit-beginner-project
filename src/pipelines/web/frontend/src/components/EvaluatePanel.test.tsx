import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EvaluateProgress, EvaluationState, JobRecord, Progress } from '../api/types';

const evaluationStatus = vi.fn();
const startEvaluation = vi.fn();
const retryRegistration = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    evaluationStatus: (...args: unknown[]) => evaluationStatus(...args),
    startEvaluation: (...args: unknown[]) => startEvaluation(...args),
    retryRegistration: (...args: unknown[]) => retryRegistration(...args),
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
  retryRegistration.mockResolvedValue({ registration: { status: 'succeeded' } });
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

  it('연산 장치를 고르지 않으면 서버가 정하도록 비워서 보낸다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(false) });

    render(<EvaluatePanel job={job(true)} />);

    fireEvent.click(await screen.findByRole('button', { name: '평가 및 submission 생성' }));

    await waitFor(() => expect(startEvaluation).toHaveBeenCalled());
    expect(startEvaluation.mock.calls[0]![1]).not.toHaveProperty('device');
  });

  it('CPU를 고르면 그대로 실어 보낸다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(false) });

    render(<EvaluatePanel job={job(true)} />);

    fireEvent.change(await screen.findByLabelText(/연산 장치/), { target: { value: 'cpu' } });
    fireEvent.click(screen.getByRole('button', { name: '평가 및 submission 생성' }));

    await waitFor(() =>
      expect(startEvaluation).toHaveBeenCalledWith(
        'a'.repeat(32),
        expect.objectContaining({ device: 'cpu' }),
      ),
    );
  });

  it('CPU가 느리다는 것을 고를 때 알려 준다', async () => {
    evaluationStatus.mockResolvedValue({ evaluation: succeeded(false) });

    render(<EvaluatePanel job={job(true)} />);

    fireEvent.change(await screen.findByLabelText(/연산 장치/), { target: { value: 'cpu' } });

    expect(screen.getByText(/오래 걸립니다|느립니다/)).toBeInTheDocument();
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

  it('평가는 성공하고 등록만 실패한 경우 Registry 재시도를 제공한다', async () => {
    evaluationStatus.mockResolvedValue({
      evaluation: {
        ...succeeded(false),
        registration: { status: 'failed', message: 'Registry 저장 실패' },
      },
    });
    render(<EvaluatePanel job={job(false)} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Registry 등록 재시도' }));

    await waitFor(() => expect(retryRegistration).toHaveBeenCalledWith('a'.repeat(32)));
  });
});

const STARTED_AT = '2026-08-07T03:30:00Z';
/** 평가를 시작하고 74초가 지난 시점입니다. */
const NOW = Date.parse('2026-08-07T03:31:14Z');

function running(progress?: EvaluateProgress): EvaluationState {
  return {
    status: 'running',
    job_id: 'a'.repeat(32),
    started_at: STARTED_AT,
    finished_at: null,
    submission_requested: true,
    message: 'checkpoint로 검증 이미지를 추론하고 있습니다.',
    progress,
  };
}

function showRunning(progress?: EvaluateProgress) {
  evaluationStatus.mockResolvedValue({ evaluation: running(progress) });
  return render(<EvaluatePanel job={job(true)} />);
}

describe('EvaluatePanel · 평가 진행 상황', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(NOW);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('경과 시간을 초 단위로 보여 주고 계속 갱신한다', async () => {
    showRunning();

    expect(await screen.findByText('1분 14초')).toBeInTheDocument();

    // 가짜 시계를 4초 밀면 1초짜리 interval이 그 사이에 네 번 돕니다.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });

    expect(await screen.findByText('1분 18초')).toBeInTheDocument();
  });

  it('지금 어떤 단계인지 보여 준다', async () => {
    showRunning({
      available: true,
      stage: 'metrics',
      stage_label: '지표 계산 중',
      predict: null,
    });

    expect(await screen.findByText('지표 계산 중')).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });

  it('test 추론 중에는 done / total과 막대를 보여 준다', async () => {
    showRunning({
      available: true,
      stage: 'test',
      stage_label: 'test 추론 중',
      predict: { stage: 'test', done: 421, total: 842, percent: 50 },
      images: { validation_images: 46, test_images: 842 },
    });

    expect(await screen.findByText('test 추론 중')).toBeInTheDocument();
    expect(screen.getByText('421 / 842')).toBeInTheDocument();
    const bar = screen.getByRole('progressbar');
    expect(bar).toHaveAttribute('aria-valuenow', '50');
    expect(bar).toHaveAttribute('aria-valuemax', '100');
  });

  it('오래 걸리는 것이 정상임을 알 수 있게 test 이미지 장수를 함께 보여 준다', async () => {
    showRunning({
      available: true,
      stage: 'validation',
      stage_label: 'validation 추론 중',
      predict: { stage: 'validation', done: 10, total: 46, percent: 21.7 },
      images: { validation_images: 46, test_images: 842 },
    });

    expect(await screen.findByText(/test 이미지 842장/)).toBeInTheDocument();
  });

  it('관측된 남은 시간이 있으면 함께 보여 준다', async () => {
    showRunning({
      available: true,
      stage: 'test',
      stage_label: 'test 추론 중',
      predict: { stage: 'test', done: 421, total: 842, percent: 50 },
      eta_seconds: 95,
    });

    expect(await screen.findByText(/남은 시간 약 1분 35초/)).toBeInTheDocument();
  });

  it('남은 시간을 아직 모르면 그 자리를 비워 둔다', async () => {
    showRunning({
      available: true,
      stage: 'test',
      stage_label: 'test 추론 중',
      predict: { stage: 'test', done: 421, total: 842, percent: 50 },
      eta_seconds: null,
    });

    expect(await screen.findByText('test 추론 중')).toBeInTheDocument();
    expect(screen.queryByText(/남은 시간/)).not.toBeInTheDocument();
  });

  it('진행 정보가 없으면 가짜 진행률 대신 지금까지의 안내 문구를 보여 준다', async () => {
    showRunning({
      available: false,
      reason: 'evaluate_pipeline_no_progress_stream',
      message: 'evaluate pipeline이 진행 로그를 제공하지 않아 진행률을 알 수 없습니다.',
      stage: null,
      predict: null,
    });

    expect(
      await screen.findByText(/checkpoint로 검증 이미지를 추론하고 있습니다/),
    ).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    // 경과 시간은 진행 로그 없이도 알 수 있으므로 계속 보여 줍니다.
    expect(screen.getByText('1분 14초')).toBeInTheDocument();
  });

  it('진행 블록 자체가 없는 옛 서버 응답에서도 깨지지 않는다', async () => {
    showRunning(undefined);

    expect(
      await screen.findByText(/checkpoint로 검증 이미지를 추론하고 있습니다/),
    ).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });
});
