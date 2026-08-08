import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobListing, QueueState } from '../api/types';

const readQueue = vi.fn();
const removeFromQueue = vi.fn();
const resumeQueue = vi.fn();

vi.mock('../api/client', () => ({
  api: {
    readQueue: () => readQueue(),
    removeFromQueue: (...args: unknown[]) => removeFromQueue(...args),
    resumeQueue: () => resumeQueue(),
    gpu: () =>
      Promise.resolve({
        torch: { cuda_available: false, device_count: 0, reason: null },
        telemetry: { source: 'unavailable', reason: null, message: '없음', devices: [] },
        queried_at: '2026-08-08T00:00:00Z',
      }),
  },
}));

vi.mock('../components/DataSourcePanel', () => ({ DataSourcePanel: () => null }));

const { TrainingOverview } = await import('./TrainingOverview');

const LISTING: JobListing = { jobs: [], active_job_id: null };

function queue(entries: number, paused: boolean): QueueState {
  return {
    paused,
    entries: Array.from({ length: entries }, (_, index) => ({
      entry_id: `e${index}`,
      config_id: 'c'.repeat(32),
      run_id: `run-${index + 1}`,
      queued_at: '2026-08-08T01:00:00Z',
    })),
  };
}

function renderOverview() {
  return render(
    <MemoryRouter>
      <TrainingOverview
        listing={LISTING}
        source={null}
        onSourceSelected={() => {}}
        onPrepared={() => {}}
      />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  removeFromQueue.mockResolvedValue(queue(0, false));
  resumeQueue.mockResolvedValue(queue(0, false));
});

describe('학습 대기열', () => {
  it('기다리는 학습을 순서대로 보여 준다', async () => {
    readQueue.mockResolvedValue(queue(2, false));

    renderOverview();

    expect(await screen.findByText('학습 대기열 (2)')).toBeInTheDocument();
    expect(screen.getByText('run-1')).toBeInTheDocument();
    expect(screen.getByText('run-2')).toBeInTheDocument();
  });

  it('대기열이 비면 자리를 차지하지 않는다', async () => {
    readQueue.mockResolvedValue(queue(0, false));

    renderOverview();

    await waitFor(() => expect(readQueue).toHaveBeenCalled());
    expect(screen.queryByText(/학습 대기열/)).toBeNull();
  });

  it('멈춰 있으면 왜 멈췄고 무엇을 눌러야 하는지 알려 준다', async () => {
    readQueue.mockResolvedValue(queue(1, true));

    renderOverview();

    expect(await screen.findByText('대기열이 멈춰 있습니다')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '대기열 다시 돌리기' }));

    await waitFor(() => expect(resumeQueue).toHaveBeenCalled());
  });

  it('돌고 있을 때는 다시 돌리기를 보여 주지 않는다', async () => {
    readQueue.mockResolvedValue(queue(1, false));

    renderOverview();

    await screen.findByText('run-1');
    expect(screen.queryByRole('button', { name: '대기열 다시 돌리기' })).toBeNull();
    expect(screen.getByText(/차례로 시작합니다/)).toBeInTheDocument();
  });

  it('기다리는 항목을 뺄 수 있다', async () => {
    readQueue.mockResolvedValue(queue(1, false));

    renderOverview();

    fireEvent.click(await screen.findByRole('button', { name: '빼기' }));

    await waitFor(() => expect(removeFromQueue).toHaveBeenCalledWith('e0'));
  });
});
