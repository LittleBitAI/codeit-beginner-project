import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { PreparationProgress, PreparationState } from '../api/types';

const prepareStatus = vi.fn();
const startPreparation = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    prepareStatus: (...args: unknown[]) => prepareStatus(...args),
    startPreparation: (...args: unknown[]) => startPreparation(...args),
  },
}));

const { PreparePanel } = await import('./PreparePanel');

const STARTED_AT = '2026-08-07T03:30:00Z';
/** 준비를 시작하고 74초가 지난 시점입니다. */
const NOW = Date.parse('2026-08-07T03:31:14Z');

function running(progress?: PreparationProgress): PreparationState {
  return {
    status: 'running',
    split_ratio: '8:2',
    started_at: STARTED_AT,
    finished_at: null,
    message: '원본을 읽어 artifact를 만들고 있습니다.',
    progress,
  };
}

function show(state: PreparationState) {
  prepareStatus.mockResolvedValue({
    split_ratios: ['8:2', '9:1'],
    backends: ['auto', 'local'],
    storage: {
      bucket: null,
      bucket_configured: false,
      profile_configured: false,
      region: null,
      forced_backend: null,
      default_backend: 'local',
    },
    preparation: state,
  });
  return render(<PreparePanel onPrepared={vi.fn()} />);
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('PreparePanel · 참조 crop 은행', () => {
  it('은행을 켜고 실행하면 그 요청을 함께 보낸다', async () => {
    startPreparation.mockResolvedValue(undefined);
    show({ status: 'idle' });

    fireEvent.click(await screen.findByLabelText(/참조 crop 은행/));
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '데이터 준비 실행' }));
    });

    expect(startPreparation).toHaveBeenCalledWith(
      expect.objectContaining({ split_ratio: '8:2', crop_bank: true }),
    );
  });

});

describe('PreparePanel · 준비 진행 상황', () => {
  it('경과 시간을 초 단위로 보여 주고 계속 갱신한다', async () => {
    show(running());

    expect(await screen.findByText('1분 14초')).toBeInTheDocument();

    // 가짜 시계를 4초 밀면 1초짜리 interval이 그 사이에 네 번 돕니다.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });

    expect(await screen.findByText('1분 18초')).toBeInTheDocument();
  });

  it('지금 어떤 단계인지 보여 준다', async () => {
    show(
      running({
        available: true,
        stage: 'split',
        stage_label: '나누는 중',
        read: null,
        sources: { train_images: 1842, annotations: 1842, test_images: 842 },
      }),
    );

    expect(await screen.findByText('나누는 중')).toBeInTheDocument();
  });

  it('읽는 단계에서는 done / total과 막대를 보여 준다', async () => {
    show(
      running({
        available: true,
        stage: 'annotations',
        stage_label: 'annotation 읽는 중',
        read: { stage: 'annotations', done: 400, total: 1842, percent: 21.7 },
      }),
    );

    expect(await screen.findByText('annotation 읽는 중')).toBeInTheDocument();
    expect(screen.getByText('400 / 1842')).toBeInTheDocument();
    const bar = screen.getByRole('progressbar');
    expect(bar).toHaveAttribute('aria-valuenow', '21.7');
    expect(bar).toHaveAttribute('aria-valuemax', '100');
  });

  it('관측된 남은 시간이 있으면 함께 보여 준다', async () => {
    show(
      running({
        available: true,
        stage: 'annotations',
        stage_label: 'annotation 읽는 중',
        read: { stage: 'annotations', done: 400, total: 1842, percent: 21.7 },
        eta_seconds: 95,
      }),
    );

    expect(await screen.findByText(/남은 시간 약 1분 35초/)).toBeInTheDocument();
  });

  it('남은 시간을 아직 모르면 그 자리를 비워 둔다', async () => {
    show(
      running({
        available: true,
        stage: 'annotations',
        stage_label: 'annotation 읽는 중',
        read: { stage: 'annotations', done: 400, total: 1842, percent: 21.7 },
        eta_seconds: null,
      }),
    );

    expect(await screen.findByText('annotation 읽는 중')).toBeInTheDocument();
    expect(screen.queryByText(/남은 시간/)).not.toBeInTheDocument();
  });

  it('진행 정보가 없으면 가짜 진행률 대신 지금까지의 안내 문구를 보여 준다', async () => {
    show(
      running({
        available: false,
        reason: 'data_pipeline_no_progress_stream',
        message: 'data pipeline이 진행 로그를 제공하지 않아 진행률을 알 수 없습니다.',
        stage: null,
        read: null,
      }),
    );

    expect(await screen.findByText(/원본을 읽어 artifact를 만들고 있습니다/)).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    // 경과 시간은 진행 로그 없이도 알 수 있으므로 계속 보여 줍니다.
    expect(screen.getByText('1분 14초')).toBeInTheDocument();
  });

  it('진행 블록 자체가 없는 옛 서버 응답에서도 깨지지 않는다', async () => {
    show(running(undefined));

    expect(await screen.findByText(/원본을 읽어 artifact를 만들고 있습니다/)).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });

  it('total을 아직 모르면 막대를 그리지 않고 읽은 개수만 보여 준다', async () => {
    show(
      running({
        available: true,
        stage: 'annotations',
        stage_label: 'annotation 읽는 중',
        read: { stage: 'annotations', done: 400, total: null, percent: null },
      }),
    );

    expect(await screen.findByText('400개')).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });

  it('준비가 끝나면 진행 표시 대신 결과를 보여 준다', async () => {
    show({
      status: 'succeeded',
      split_ratio: '8:2',
      started_at: STARTED_AT,
      finished_at: '2026-08-07T03:37:54Z',
      message: '준비 완료',
      selected: true,
      summary: { train_images: 1473, validation_images: 369, category_count: 73 },
      progress: {
        available: true,
        stage: 'completed',
        stage_label: '준비 완료',
        read: null,
      },
    });

    expect(await screen.findByText(/데이터 준비 완료/)).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });
});

describe('PreparePanel · 원본 경로', () => {
  const idle: PreparationState = {
    status: 'idle',
    started_at: null,
    finished_at: null,
  };

  it('적은 경로를 그대로 실어 보낸다', async () => {
    startPreparation.mockResolvedValue({});
    show(idle);

    fireEvent.change(await screen.findByPlaceholderText('datasets/pill_detection/raw/<판>/'), {
      target: { value: ' datasets/pill_detection/raw/v90/ ' },
    });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '데이터 준비 실행' }));
    });

    expect(startPreparation).toHaveBeenCalledWith(
      expect.objectContaining({ raw_prefix: 'datasets/pill_detection/raw/v90/' }),
    );
  });

  it('비워 두면 보내지 않아 서버가 기본값을 쓰게 둔다', async () => {
    startPreparation.mockResolvedValue({});
    show(idle);

    await act(async () => {
      fireEvent.click(await screen.findByRole('button', { name: '데이터 준비 실행' }));
    });

    expect(startPreparation).toHaveBeenCalledWith(
      expect.not.objectContaining({ raw_prefix: expect.anything() }),
    );
  });
});
