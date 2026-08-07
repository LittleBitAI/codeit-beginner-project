import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobRecord, Progress } from '../api/types';

const getJob = vi.fn();
const logs = vi.fn();
const gpu = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getJob: (...args: unknown[]) => getJob(...args),
    logs: (...args: unknown[]) => logs(...args),
    gpu: () => gpu(),
    cancelJob: vi.fn(),
  },
}));

const { LiveMonitor } = await import('./LiveMonitor');

const NO_PROGRESS: Progress = {
  available: false,
  reason: 'train_pipeline_no_progress_stream',
  message: 'train pipeline이 진행 로그를 제공하지 않아 진행률을 알 수 없습니다.',
  total_epochs: null,
  current_epoch: null,
  eta_seconds: null,
  epochs: [],
};

function makeJob(progress: Progress): JobRecord {
  return {
    job_id: 'a'.repeat(32),
    config_id: 'b'.repeat(32),
    run_id: 'exp-1',
    status: 'running',
    status_label: '실행 중',
    created_at: '2026-08-05T00:00:00Z',
    started_at: '2026-08-05T00:00:00Z',
    finished_at: null,
    elapsed_seconds: 61,
    exit_code: null,
    message: null,
    artifacts: {},
    summary: {},
    settings: { device: 'cuda', epochs: 50, batch_size: 2, seed: 42 },
    data_inputs: {},
    progress,
    log_lines: 0,
    orphan_note: null,
  };
}

function renderMonitor() {
  return render(
    <MemoryRouter initialEntries={[`/monitor/${'a'.repeat(32)}`]}>
      <Routes>
        <Route path="/monitor/:jobId" element={<LiveMonitor listing={null} />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  logs.mockResolvedValue({ lines: [], next: 0, complete: false });
  gpu.mockResolvedValue({
    torch: { cuda_available: false, device_count: 0, reason: null },
    telemetry: { source: 'unavailable', reason: 'nvidia_smi_not_found', message: 'nvidia-smi를 찾지 못했습니다.', devices: [] },
    queried_at: '2026-08-05T00:00:00Z',
  });
});

describe('LiveMonitor · 진행 로그가 없을 때', () => {
  it('진행률을 지어내지 않고 없다고 말한다', async () => {
    getJob.mockResolvedValue(makeJob(NO_PROGRESS));

    renderMonitor();

    expect(await screen.findByText('진행률 정보 없음')).toBeInTheDocument();
    expect(screen.getByText(/진행 로그를 제공하지 않아/)).toBeInTheDocument();
    // 퍼센트도 epoch 숫자도 만들어 내지 않습니다.
    expect(screen.queryByText(/epoch \d+ \//)).toBeNull();
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBeNull();
  });

  it('손실 곡선 자리에 빈 상태를 보여 준다', async () => {
    getJob.mockResolvedValue(makeJob(NO_PROGRESS));

    renderMonitor();

    expect(await screen.findByText(/그릴 데이터가 없습니다/)).toBeInTheDocument();
  });

  it('GPU 정보를 못 가져오면 0이 아니라 이유를 보여 준다', async () => {
    getJob.mockResolvedValue(makeJob(NO_PROGRESS));

    renderMonitor();

    expect(await screen.findByText('nvidia-smi를 찾지 못했습니다.')).toBeInTheDocument();
  });
});

const withProgress: Progress = {
  available: true,
  reason: null,
  message: null,
  total_epochs: 50,
  current_epoch: 2,
  completed_epochs: 2,
  percent: 4,
  eta_seconds: 720,
  epochs: [
    { epoch: 1, train_loss: 1.2, validation_loss: 1.1, epoch_seconds: 15, is_best: true },
    { epoch: 2, train_loss: 0.9, validation_loss: 1.0, epoch_seconds: 15, is_best: true },
  ],
  best: { epoch: 2, validation_loss: 1.0 },
};

describe('LiveMonitor · 진행 로그가 있을 때', () => {
  it('실제 epoch 숫자와 추정 남은 시간을 보여 준다', async () => {
    getJob.mockResolvedValue(makeJob(withProgress));

    renderMonitor();

    expect(await screen.findByText('epoch 2 / 50')).toBeInTheDocument();
    expect(screen.getByText('~12분')).toHaveClass('estimated');
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('4');
  });

  it('마지막 epoch의 손실을 KPI로 보여 준다', async () => {
    getJob.mockResolvedValue(makeJob(withProgress));

    renderMonitor();

    expect(await screen.findByText('0.9000')).toBeInTheDocument();
    // VAL LOSS와 BEST VAL LOSS가 같은 값이라 둘 다 나옵니다.
    expect(screen.getAllByText('1.0000').length).toBeGreaterThanOrEqual(1);
  });

  it('남은 시간을 아직 못 재면 추정하지 않는다', async () => {
    getJob.mockResolvedValue(
      makeJob({ ...withProgress, eta_seconds: null, epochs: [withProgress.epochs[0]!] }),
    );

    renderMonitor();

    await waitFor(() =>
      expect(screen.getByText('남은 시간을 추정할 수 없습니다')).toBeInTheDocument(),
    );
  });
});

describe('LiveMonitor · 조기 종료로 끝났을 때', () => {
  const stoppedEarly: Progress = {
    ...withProgress,
    finished: true,
    stopped_early: true,
    percent: 100,
    eta_seconds: 0,
  };

  it('계획 epoch가 남아 있어도 진행률을 다 채운다', async () => {
    getJob.mockResolvedValue(makeJob(stoppedEarly));

    renderMonitor();

    expect(await screen.findByText('epoch 2 / 50')).toBeInTheDocument();
    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('100');
  });

  it('남은 시간 대신 조기 종료로 끝났다고 알린다', async () => {
    getJob.mockResolvedValue(makeJob(stoppedEarly));

    renderMonitor();

    expect(await screen.findByText('조기 종료로 끝남')).toBeInTheDocument();
    expect(screen.queryByText(/남은 시간/)).toBeNull();
  });

  it('조기 종료가 아니면 그냥 끝났다고만 한다', async () => {
    getJob.mockResolvedValue(makeJob({ ...stoppedEarly, stopped_early: false }));

    renderMonitor();

    expect(await screen.findByText('학습 완료')).toBeInTheDocument();
    expect(screen.queryByText('조기 종료로 끝남')).toBeNull();
  });
});

describe('LiveMonitor · epoch별 손실 분해', () => {
  it('모델이 돌려준 이름과 값을 그대로 보여 준다', async () => {
    getJob.mockResolvedValue(
      makeJob({
        ...withProgress,
        epochs: [
          withProgress.epochs[0]!,
          {
            ...withProgress.epochs[1]!,
            train_loss_components: { classification: 0.72, bbox_regression: 0.53 },
            validation_loss_components: { classification: 0.79, bbox_regression: 0.59 },
          },
        ],
      }),
    );

    renderMonitor();

    expect(await screen.findByText('classification')).toBeInTheDocument();
    expect(screen.getByText('bbox_regression')).toBeInTheDocument();
    expect(screen.getByText('0.7200')).toBeInTheDocument();
    expect(screen.getByText('0.5900')).toBeInTheDocument();
  });

  it('상세 loss가 없으면 분해를 그리지 않는다', async () => {
    getJob.mockResolvedValue(makeJob(withProgress));

    renderMonitor();

    await screen.findByText('epoch 2 / 50');
    expect(screen.queryByText(/손실 분해/)).toBeNull();
  });
});
