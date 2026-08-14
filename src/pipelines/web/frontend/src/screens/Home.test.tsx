import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { JobRecord, QueueState } from '../api/types';
import { Home } from './Home';

afterEach(cleanup);

function liveJob(): JobRecord {
  return {
    job_id: 'job-1',
    config_id: 'cfg-1',
    run_id: 'retina-live',
    status: 'running',
    status_label: '학습 중',
    created_at: '2026-08-05T00:00:00Z',
    started_at: '2026-08-05T00:00:00Z',
    finished_at: null,
    elapsed_seconds: 600,
    exit_code: null,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: {
      available: true,
      reason: null,
      message: null,
      architecture: 'retinanet_resnet50_fpn_v2',
      device: 'cuda',
      total_epochs: 15,
      current_epoch: 3,
      completed_epochs: 3,
      percent: 20,
      eta_seconds: 1800,
      epochs: [
        { epoch: 1, train_loss: 0.9, validation_loss: 0.88, epoch_seconds: 200, is_best: true },
        { epoch: 3, train_loss: 0.6, validation_loss: 0.62, epoch_seconds: 190, is_best: true },
      ],
      best: { epoch: 3, validation_loss: 0.62 },
    },
    log_lines: 120,
    orphan_note: null,
  };
}

function show(props: Partial<Parameters<typeof Home>[0]> = {}) {
  const queue: QueueState = { entries: [], paused: false };
  return render(
    <MemoryRouter>
      <Home
        liveJob={null}
        queue={queue}
        error={null}
        onNewExperiment={() => {}}
        onRemoveFromQueue={() => {}}
        onResumeQueue={() => {}}
        onCancelJob={() => {}}
        {...props}
      />
    </MemoryRouter>,
  );
}

describe('Home', () => {
  it('지금 도는 학습이 있으면 그 값을 맨 위에 크게 세운다', () => {
    show({ liveJob: liveJob() });

    expect(screen.getByText('지금 학습 중')).toBeInTheDocument();
    expect(screen.getAllByText('retina-live').length).toBeGreaterThan(0);
    // best validation loss가 큰 숫자 자리에 옵니다.
    expect(screen.getByText('0.6200')).toBeInTheDocument();
    // 첫 epoch에서 얼마나 내려왔는지. 0.88 - 0.62 입니다.
    expect(screen.getByText('↓ 0.2600')).toBeInTheDocument();
    // 남은 시간은 loss와 같은 자리·같은 크기입니다.
    expect(screen.getByText('남은 시간')).toBeInTheDocument();
    expect(screen.getByText('~30분')).toBeInTheDocument();
  });

  it('남은 시간을 못 재면 0이 아니라 모른다고 적는다', () => {
    const job = liveJob();
    show({ liveJob: { ...job, progress: { ...job.progress, eta_seconds: null } } });

    expect(screen.getAllByText('알 수 없음').length).toBeGreaterThan(0);
  });

  it('도는 학습이 없으면 라이브 카드 대신 시작할 길을 보여 준다', () => {
    const onNewExperiment = vi.fn();
    show({ onNewExperiment });

    expect(screen.queryByText('지금 학습 중')).not.toBeInTheDocument();
    expect(screen.getByText(/이 컴퓨터에서 도는 학습이 없습니다/)).toBeInTheDocument();

    // 빈 화면에서 다음에 할 일로 바로 갈 수 있어야 합니다.
    fireEvent.click(screen.getAllByRole('button', { name: '새 실험' })[0] as HTMLElement);
    expect(onNewExperiment).toHaveBeenCalled();
  });

  it('대기열에서 빼기를 누르면 그 항목만 알려 준다', () => {
    const onRemoveFromQueue = vi.fn();
    show({
      queue: {
        entries: [
          { entry_id: 'e1', config_id: 'c1', run_id: 'queued-1', queued_at: '2026-08-05T01:00:00Z' },
        ],
        paused: false,
      },
      onRemoveFromQueue,
    });

    fireEvent.click(screen.getByRole('button', { name: '빼기' }));

    expect(onRemoveFromQueue).toHaveBeenCalledWith('e1');
  });

  it('대기열이 멈춰 있으면 다시 돌릴 수 있다고 말한다', () => {
    const onResumeQueue = vi.fn();
    show({
      queue: {
        entries: [
          { entry_id: 'e1', config_id: 'c1', run_id: 'queued-1', queued_at: '2026-08-05T01:00:00Z' },
        ],
        paused: true,
      },
      onResumeQueue,
    });

    expect(screen.getByText('대기열이 멈춰 있습니다')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '다시 돌리기' }));
    expect(onResumeQueue).toHaveBeenCalled();
  });

  it('대기열은 지금 도는 학습부터 세우고 그 오른쪽에 취소를 둔다', () => {
    const onCancelJob = vi.fn();
    show({ liveJob: liveJob(), onCancelJob });

    expect(screen.getByText('지금')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '취소' }));
    expect(onCancelJob).toHaveBeenCalledWith('job-1');
  });

  it('추가 버튼은 대기열 맨 끝에 있다', () => {
    const onNewExperiment = vi.fn();
    show({
      liveJob: liveJob(),
      queue: {
        entries: [
          { entry_id: 'e1', config_id: 'c1', run_id: 'q1', queued_at: '2026-08-05T01:00:00Z' },
          { entry_id: 'e2', config_id: 'c2', run_id: 'q2', queued_at: '2026-08-05T02:00:00Z' },
        ],
        paused: false,
      },
      onNewExperiment,
    });

    const add = screen.getByRole('button', { name: '+ 대기열에 추가' });
    const last = screen.getByText('q2');
    // 마지막 대기열 줄보다 뒤에 있어야 합니다. 줄의 맨 끝이 새로 넣는 것이 들어갈 자리입니다.
    expect(last.compareDocumentPosition(add) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    fireEvent.click(add);
    expect(onNewExperiment).toHaveBeenCalled();
  });
});
