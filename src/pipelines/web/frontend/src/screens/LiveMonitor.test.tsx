import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobRecord, Progress } from '../api/types';

const getJob = vi.fn();
const logs = vi.fn();
const gpu = vi.fn();
const resumeJob = vi.fn();
const getAccessToken = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    getJob: (...args: unknown[]) => getJob(...args),
    logs: (...args: unknown[]) => logs(...args),
    gpu: () => gpu(),
    cancelJob: vi.fn(),
    resumeJob: (...args: unknown[]) => resumeJob(...args),
  },
}));

vi.mock('../team/TeamContext', () => ({
  useTeam: () => ({ getAccessToken }),
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
  getAccessToken.mockResolvedValue('browser-token');
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

describe('LiveMonitor · epoch 안의 batch 진행', () => {
  it('지금 몇 번째 batch인지 phase와 함께 보여 준다', async () => {
    getJob.mockResolvedValue(
      makeJob({
        ...withProgress,
        step: { phase: 'train', step: 12, total_steps: 100, percent: 12 },
      }),
    );

    renderMonitor();

    expect(await screen.findByText('학습 batch 12 / 100')).toBeInTheDocument();
    // epoch 막대와 batch 막대가 각각 자기 진행률을 말합니다.
    const bars = screen.getAllByRole('progressbar');
    expect(bars.map((bar) => bar.getAttribute('aria-valuenow'))).toEqual(['4', '12']);
  });

  it('validation phase도 이름 그대로 알려 준다', async () => {
    getJob.mockResolvedValue(
      makeJob({
        ...withProgress,
        step: { phase: 'validation', step: 5, total_steps: 20, percent: 25 },
      }),
    );

    renderMonitor();

    expect(await screen.findByText('검증 batch 5 / 20')).toBeInTheDocument();
  });

  it('batch 정보가 없는 예전 실행에는 자리를 만들지 않는다', async () => {
    getJob.mockResolvedValue(makeJob(withProgress));

    renderMonitor();

    await screen.findByText('epoch 2 / 50');
    // 설정 줄의 `batch 2`(batch_size)와 헷갈리지 않도록 진행 형식으로만 찾습니다.
    expect(screen.queryByText(/batch \d+ \/ \d+/)).toBeNull();
    expect(screen.getAllByRole('progressbar')).toHaveLength(1);
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

describe('LiveMonitor · 중단된 학습', () => {
  function interrupted(): JobRecord {
    return {
      ...makeJob(NO_PROGRESS),
      status: 'interrupted',
      status_label: '중단됨',
      message: '이 서버는 저 학습을 더 이상 관리할 수 없습니다.',
    };
  }

  it('이어서 학습을 시작할 수 있다', async () => {
    getJob.mockResolvedValue(interrupted());
    resumeJob.mockResolvedValue({
      config_id: 'c'.repeat(32),
      run_id: 'web-resumed',
      resumed_from_job_id: 'a'.repeat(32),
      resume_from: 'artifacts/experiments/completed/.exp-1.partial/last_checkpoint.pt',
      started: null,
      entries: [
        {
          entry_id: 'queue-1',
          config_id: 'c'.repeat(32),
          run_id: 'web-resumed',
          queued_at: '2026-08-09T00:00:00Z',
        },
      ],
      paused: false,
    });

    renderMonitor();

    const button = await screen.findByRole('button', { name: '이어서 학습' });
    button.click();

    await waitFor(() =>
      expect(resumeJob).toHaveBeenCalledWith('a'.repeat(32), undefined, 'browser-token'),
    );
    expect(await screen.findByText(/'web-resumed' 이름으로 대기열에 넣었습니다/)).toBeInTheDocument();
  });

  it('실패하면 이유를 화면에 남긴다', async () => {
    getJob.mockResolvedValue(interrupted());
    resumeJob.mockRejectedValue(new Error('boom'));

    renderMonitor();

    (await screen.findByRole('button', { name: '이어서 학습' })).click();

    expect(
      await screen.findByText(/이어서 학습을 시작하지 못했습니다/),
    ).toBeInTheDocument();
  });

  it('끝난 학습에는 이어서 학습 단추를 두지 않는다', async () => {
    getJob.mockResolvedValue({ ...makeJob(NO_PROGRESS), status: 'succeeded' });

    renderMonitor();

    await screen.findByText('진행률 정보 없음');
    expect(screen.queryByRole('button', { name: '이어서 학습' })).toBeNull();
  });

  it('완료한 epoch가 있는 실패 학습도 이어서 학습할 수 있다', async () => {
    getJob.mockResolvedValue({
      ...makeJob({ ...NO_PROGRESS, completed_epochs: 15, epochs: [] }),
      status: 'failed',
      status_label: '실패',
      message: '학습 결과 JSON을 해석하지 못했습니다.',
    });

    renderMonitor();

    expect(await screen.findByRole('button', { name: '이어서 학습' })).toBeInTheDocument();
  });

  it('checkpoint가 생기기 전에 실패한 학습에는 이어서 학습 단추를 두지 않는다', async () => {
    getJob.mockResolvedValue({
      ...makeJob(NO_PROGRESS),
      status: 'failed',
      status_label: '실패',
    });

    renderMonitor();

    await screen.findByText('학습이 실패했습니다');
    expect(screen.queryByRole('button', { name: '이어서 학습' })).toBeNull();
  });
});
