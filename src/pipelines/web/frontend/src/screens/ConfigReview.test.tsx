import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { CreatedConfig, JobListing } from '../api/types';

const addToQueue = vi.fn();
const startJob = vi.fn();
const getAccessToken = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    addToQueue: (...args: unknown[]) => addToQueue(...args),
    startJob: (...args: unknown[]) => startJob(...args),
  },
}));

vi.mock('../team/TeamContext', () => ({
  useTeam: () => ({ getAccessToken }),
}));

const SAVED: CreatedConfig = {
  config_id: 'c'.repeat(32),
  run_id: 'exp-queue',
  config: {
    project: { name: 'pill-detection' },
    execution: { mode: 'local' },
    storage: {},
    train: {
      run_id: 'exp-queue',
      device: 'cuda',
      epochs: 3,
      batch_size: 2,
      learning_rate: 0.005,
      seed: 42,
      optimizer: 'SGD',
      momentum: 0.9,
      output_dir: 'artifacts/train',
    },
    inputs: { data: { train_manifest_uri: 'artifacts/data/train_manifest.json' } },
  },
  warnings: [],
};

vi.mock('../state/DraftContext', () => ({
  useDraft: () => ({ saved: SAVED }),
}));

const { ConfigReview } = await import('./ConfigReview');

/** 다른 학습이 이미 도는 중인 목록입니다. 대기열이 필요한 바로 그 상황입니다. */
const BUSY: JobListing = { jobs: [], active_job_id: 'a'.repeat(32) };
const IDLE: JobListing = { jobs: [], active_job_id: null };

function renderReview(listing: JobListing) {
  return render(
    <MemoryRouter>
      <ConfigReview defaults={null} listing={listing} onStarted={() => {}} />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  getAccessToken.mockResolvedValue('browser-token');
  addToQueue.mockResolvedValue({ started: null, entries: [], paused: false });
});

describe('설정 검토 · 대기열', () => {
  it('다른 학습이 도는 중에도 대기열에 넣고, 로그인 token을 함께 보낸다', async () => {
    /**
     * 줄을 세우는 기능을 정작 줄 세울 상황에서 막으면 쓸 데가 없습니다. 앞 학습이
     * 끝나기를 기다렸다가 사람이 다시 눌러야 한다면 그것은 대기열이 아닙니다.
     *
     * token까지 함께 봅니다. 대기열이 항목을 꺼내 실제로 시작할 때 팀 기록을
     * 만드는데, token이 없으면 이미 로그인한 사람에게도 "먼저 로그인해야 합니다"
     * 라고 답하며 멈춰 섭니다.
     */
    renderReview(BUSY);

    const button = screen.getByRole('button', { name: '대기열에 추가' });
    expect(button).toBeEnabled();

    fireEvent.click(button);

    await waitFor(() =>
      expect(addToQueue).toHaveBeenCalledWith(SAVED.config_id, 'browser-token'),
    );
  });

  it('바로 시작하기는 여전히 막고, 대기열로 가라고 알려 준다', async () => {
    /** 한 번에 하나만 돌릴 수 있다는 사실 자체는 그대로입니다. */
    renderReview(BUSY);

    expect(screen.getByRole('button', { name: '학습 시작' })).toBeDisabled();
    expect(screen.getByText(/대기열에 넣으면/)).toBeInTheDocument();
  });

  it('도는 학습이 없으면 학습 시작을 바로 누를 수 있다', async () => {
    renderReview(IDLE);

    expect(screen.getByRole('button', { name: '학습 시작' })).toBeEnabled();
  });
});
