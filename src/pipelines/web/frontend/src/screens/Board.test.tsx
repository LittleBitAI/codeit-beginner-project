import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { JobRecord, TeamRun } from '../api/types';
import { Board } from './Board';

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
    data_inputs: { train_manifest_uri: 'artifacts/data/v5-118cls/train_manifest.json' },
    progress: {
      available: true,
      reason: null,
      message: null,
      total_epochs: 15,
      current_epoch: 3,
      completed_epochs: 3,
      eta_seconds: 1800,
      epochs: [],
      best: { epoch: 3, validation_loss: 0.62 },
    },
    log_lines: 120,
    orphan_note: null,
  };
}

function teamRun(overrides: Partial<TeamRun> = {}): TeamRun {
  return {
    teamId: 't',
    cloudRunId: 'c1',
    localJobId: 'job-9',
    runId: 'mate-run',
    actorSub: 'sub-1',
    actorName: '김팀원',
    actorSource: 'cognito',
    status: 'running',
    settings: {},
    dataInputs: {},
    progress: { current_epoch: 7, total_epochs: 15, best: { validation_loss: 0.0612 } },
    summary: {},
    artifacts: {},
    evaluation: {},
    message: null,
    createdAt: '2026-08-05T00:00:00Z',
    startedAt: '2026-08-05T00:00:00Z',
    finishedAt: null,
    heartbeatAt: new Date().toISOString(),
    revision: 1,
    ...overrides,
  };
}

function show(props: Partial<Parameters<typeof Board>[0]> = {}) {
  return render(
    <MemoryRouter>
      <Board liveJob={null} records={[]} teamRuns={[]} teamAvailable={false} {...props} />
    </MemoryRouter>,
  );
}

describe('Board', () => {
  it('사람별로 묶고, 같은 학습을 두 번 세지 않는다', () => {
    // 내 학습도 팀에 올라가 있습니다. 그대로 두면 같은 학습이 두 줄이 됩니다.
    const mine = teamRun({ cloudRunId: 'c2', runId: 'retina-live', actorName: '나' });

    show({ liveJob: liveJob(), teamRuns: [teamRun(), mine], teamAvailable: true });

    expect(screen.getByText('mate-run')).toBeInTheDocument();
    expect(screen.getByText('김팀원')).toBeInTheDocument();
    expect(screen.getAllByText('retina-live')).toHaveLength(1);
    // 내 줄은 이 컴퓨터 것으로 남아 모니터로 들어갈 수 있습니다.
    expect(screen.getByText('나 (이 컴퓨터)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '모니터 →' })).toBeInTheDocument();
    expect(screen.queryByText('나')).toBeNull();
  });

  it('무엇으로 돌리는지도 함께 적는다', () => {
    show({ liveJob: liveJob() });

    expect(screen.getByText(/v5-118cls/)).toBeInTheDocument();
  });

  it('heartbeat가 끊긴 학습은 목록에서 빼되 왜 뺐는지 말한다', () => {
    const stale = teamRun({
      cloudRunId: 'c9',
      runId: 'stale-run',
      actorName: '박팀원',
      // 2분을 훌쩍 넘긴 heartbeat. 도는 척만 하고 있습니다.
      heartbeatAt: '2026-08-05T00:00:00Z',
    });

    show({ teamRuns: [stale], teamAvailable: true });

    expect(screen.getByText('지금 돌고 있는 학습이 없습니다.')).toBeInTheDocument();
    expect(screen.getByText('연결이 끊긴 학습 1개')).toBeInTheDocument();
    expect(screen.getByText(/stale-run · 박팀원/)).toBeInTheDocument();
  });

  it('팀 연결이 꺼져 있으면 내 것만 보인다고 말한다', () => {
    show({ liveJob: liveJob(), teamAvailable: false });

    expect(screen.getByText(/팀 실시간 연결이 꺼져 있어/)).toBeInTheDocument();
  });
});
