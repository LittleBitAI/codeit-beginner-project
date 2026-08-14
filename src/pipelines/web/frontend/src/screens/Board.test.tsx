import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
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
      <Board
        liveJob={null}
        records={[]}
        teamRuns={[]}
        teamAvailable={false}
        teamLoaded={false}
        teamError={null}
        {...props}
      />
    </MemoryRouter>,
  );
}

describe('Board', () => {
  it('팀원 것과 내 것을 함께 세운다', () => {
    show({ liveJob: liveJob(), teamRuns: [teamRun()], teamAvailable: true, teamLoaded: true });

    expect(screen.getByText('mate-run')).toBeInTheDocument();
    expect(screen.getByText('김팀원')).toBeInTheDocument();
    expect(screen.getByText('retina-live')).toBeInTheDocument();
    // 내 줄은 이 컴퓨터 것으로 남아 모니터로 들어갈 수 있습니다.
    expect(screen.getByText('나 (이 컴퓨터)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '모니터 →' })).toBeInTheDocument();
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

    show({ teamRuns: [stale], teamAvailable: true, teamLoaded: true });

    expect(screen.getByText('지금 돌고 있는 학습이 없습니다.')).toBeInTheDocument();
    expect(screen.getByText('연결이 끊긴 학습 1개')).toBeInTheDocument();
    expect(screen.getByText(/stale-run · 박팀원/)).toBeInTheDocument();
  });

  it('팀 연결이 꺼져 있으면 내 것만 보인다고 말한다', () => {
    show({ liveJob: liveJob(), teamAvailable: false });

    expect(screen.getByText(/팀 실시간 연결이 꺼져 있어/)).toBeInTheDocument();
  });

  // 자동으로 지어지는 이름은 설정과 seed의 지문입니다. 팀원 둘이 같은 설정으로
  // 돌리면 이름이 같아지고, 그 이름으로 묶으면 한 사람의 학습이 통째로 사라집니다.
  it('이름이 같은 서로 다른 실행을 한 줄로 합치지 않는다', () => {
    const first = teamRun({ cloudRunId: 'c1', actorSub: 'sub-1', actorName: '김팀원' });
    const second = teamRun({ cloudRunId: 'c2', actorSub: 'sub-2', actorName: '박팀원' });

    show({ teamRuns: [first, second], teamAvailable: true, teamLoaded: true });

    expect(screen.getAllByText('mate-run')).toHaveLength(2);
    expect(screen.getByText('김팀원')).toBeInTheDocument();
    expect(screen.getByText('박팀원')).toBeInTheDocument();
  });

  it('이름이 같은 팀원 둘을 한 사람으로 묶지 않는다', () => {
    const one = teamRun({ cloudRunId: 'c1', actorSub: 'sub-1', runId: 'run-1' });
    const other = teamRun({ cloudRunId: 'c2', actorSub: 'sub-2', runId: 'run-2' });

    show({ teamRuns: [one, other], teamAvailable: true, teamLoaded: true });

    // 이름은 같지만 사람은 둘입니다. 묶음이 둘이어야 각자 몇 개를 돌리는지 맞습니다.
    expect(screen.getAllByText('김팀원')).toHaveLength(2);
    expect(screen.getAllByText('1개 학습 중')).toHaveLength(2);
  });

  it('내 학습이 팀에도 올라가 있으면 이 컴퓨터 줄만 남긴다', () => {
    // 같은 실행입니다. 팀 기록은 그것을 localJobId로 알려 줍니다.
    const mine = teamRun({ cloudRunId: 'c9', runId: 'retina-live', localJobId: 'job-1', actorName: '나' });

    show({ liveJob: liveJob(), teamRuns: [mine], teamAvailable: true, teamLoaded: true });

    expect(screen.getAllByText('retina-live')).toHaveLength(1);
    expect(screen.getByText('나 (이 컴퓨터)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '모니터 →' })).toBeInTheDocument();
  });

  // 같은 학습이 팀에 두 번 공유돼 있을 수 있습니다(다시 올렸거나 정리가 덜 됐거나).
  // 첫 하나만 빼면 남은 것이 또 한 줄이 되어 한 실행이 둘로 보입니다.
  it('같은 학습을 가리키는 팀 기록이 여럿이어도 한 줄만 남긴다', () => {
    const shared = [
      teamRun({ cloudRunId: 'c8', runId: 'retina-live', localJobId: 'job-1' }),
      teamRun({ cloudRunId: 'c9', runId: 'retina-live', localJobId: 'job-1' }),
    ];

    show({ liveJob: liveJob(), teamRuns: shared, teamAvailable: true, teamLoaded: true });

    expect(screen.getAllByText('retina-live')).toHaveLength(1);
    expect(screen.getByText('1개 학습 중')).toBeInTheDocument();
  });

  /**
   * key가 이름이면 React가 서로 다른 실행을 같은 줄로 봅니다.
   *
   * 그 자체는 조용합니다(경고 한 줄). 실제로 아픈 것은 줄에 붙은 **상태**입니다:
   * 펼쳐 둔 로그가 구독 갱신으로 순서가 바뀌는 순간 다른 사람의 실행으로 옮겨 갑니다.
   * 그래서 개수가 아니라 그 이동을 잽니다.
   */
  it('순서가 바뀌어도 펼쳐 둔 로그가 다른 실행으로 옮겨 가지 않는다', () => {
    const seven = teamRun({
      cloudRunId: 'c7',
      actorSub: 'sub-7',
      progress: { current_epoch: 7, total_epochs: 15 },
    });
    const three = teamRun({
      cloudRunId: 'c3',
      actorSub: 'sub-7',
      progress: { current_epoch: 3, total_epochs: 15 },
    });
    const { rerender } = show({ teamRuns: [seven, three], teamAvailable: true, teamLoaded: true });

    // 줄에는 **끝낸** epoch 수가 적히므로 7과 3은 6과 2로 보입니다.
    const rowOf = (done: string) =>
      screen.getByText(new RegExp(`epoch ${done} / 15`)).closest('div')?.parentElement
        ?.parentElement as HTMLElement;

    fireEvent.click(within(rowOf('2')).getByRole('button', { name: '로그 보기' }));
    expect(within(rowOf('2')).getByRole('button', { name: '로그 접기' })).toBeInTheDocument();

    // 구독 갱신으로 순서가 바뀝니다.
    rerender(
      <MemoryRouter>
        <Board
          liveJob={null}
          records={[]}
          teamRuns={[three, seven]}
          teamAvailable
          teamLoaded
          teamError={null}
        />
      </MemoryRouter>,
    );

    // 펼친 것은 여전히 같은 실행이어야 합니다.
    expect(within(rowOf('2')).getByRole('button', { name: '로그 접기' })).toBeInTheDocument();
    expect(within(rowOf('6')).getByRole('button', { name: '로그 보기' })).toBeInTheDocument();
  });

  it('팀 기록을 아직 못 읽었으면 "없다"고 단정하지 않는다', () => {
    show({ teamAvailable: true, teamLoaded: false });

    expect(screen.getByText('팀 기록을 읽고 있습니다.')).toBeInTheDocument();
    expect(screen.queryByText('지금 돌고 있는 학습이 없습니다.')).toBeNull();
  });

  it('팀 기록을 읽다 실패했으면 그 이유를 적는다', () => {
    show({ teamAvailable: true, teamLoaded: false, teamError: '연결이 끊겼습니다.' });

    expect(screen.getByText('팀 기록을 읽지 못했습니다')).toBeInTheDocument();
    expect(screen.getByText(/연결이 끊겼습니다/)).toBeInTheDocument();
    expect(screen.queryByText('지금 돌고 있는 학습이 없습니다.')).toBeNull();
  });
});
