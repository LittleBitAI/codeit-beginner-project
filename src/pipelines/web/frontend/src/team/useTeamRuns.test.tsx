/**
 * 팀 기록을 읽는 두 규칙을 지킵니다.
 *
 * 1. 목록 응답이 **구독으로 먼저 온 더 새로운 값**을 덮지 않는다. 덮으면 방금 시작한
 *    학습이 현황판에서 다시 사라져 다음 event가 올 때까지 보이지 않습니다.
 * 2. 읽을 수 없게 되면(로그아웃, 다른 사용자) 들고 있던 것을 버린다. 남겨 두면
 *    화면은 "이 컴퓨터 것만 보인다"고 적어 놓고 앞 사용자의 학습을 계속 보여 줍니다.
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { TeamRun } from '../api/types';

const team = {
  config: { team_id: 't', actor: null } as Record<string, unknown>,
  user: { name: '나' } as unknown,
  latestEvent: null as TeamRun | null,
};

vi.mock('./TeamContext', () => ({ useTeam: () => team }));

const listRuns = vi.fn();
vi.mock('./cloud', () => ({ listRuns: (...args: unknown[]) => listRuns(...args) }));

const { useTeamRuns } = await import('./useTeamRuns');

function run(overrides: Partial<TeamRun> = {}): TeamRun {
  return {
    teamId: 't',
    cloudRunId: 'c1',
    localJobId: 'job-1',
    runId: 'run-1',
    actorSub: 'sub-1',
    actorName: '김팀원',
    actorSource: 'cognito',
    status: 'running',
    settings: {},
    dataInputs: {},
    progress: {},
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

beforeEach(() => {
  listRuns.mockReset();
  team.config = { team_id: 't', actor: null };
  team.user = { name: '나' };
  team.latestEvent = null;
});

describe('useTeamRuns', () => {
  it('목록 응답이 구독으로 먼저 온 더 새로운 값을 덮지 않는다', async () => {
    let resolveList: (value: TeamRun[]) => void = () => {};
    listRuns.mockReturnValue(new Promise<TeamRun[]>((resolve) => { resolveList = resolve; }));

    const { result, rerender } = renderHook(() => useTeamRuns());

    // 목록이 아직 오지 않은 사이에 구독으로 최신 상태가 도착합니다.
    team.latestEvent = run({ revision: 5, status: 'running' });
    rerender();
    await waitFor(() => expect(result.current.runs).toHaveLength(1));

    // 뒤늦게 도착한 목록은 그 실행의 **옛** 모습을 담고 있습니다.
    await act(async () => {
      resolveList([run({ revision: 2, status: 'queued' })]);
    });

    await waitFor(() => expect(result.current.loaded).toBe(true));
    expect(result.current.runs).toHaveLength(1);
    expect(result.current.runs[0]?.revision).toBe(5);
    expect(result.current.runs[0]?.status).toBe('running');
  });

  it('목록에만 있는 실행은 그대로 받는다', async () => {
    listRuns.mockResolvedValue([run({ cloudRunId: 'c2', runId: 'run-2' })]);

    const { result } = renderHook(() => useTeamRuns());

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0]?.cloudRunId).toBe('c2');
  });

  it('읽을 수 없게 되면 들고 있던 팀 기록을 버린다', async () => {
    listRuns.mockResolvedValue([run()]);

    const { result, rerender } = renderHook(() => useTeamRuns());
    await waitFor(() => expect(result.current.runs).toHaveLength(1));

    // 로그아웃: 팀 기록을 읽을 수 없는 상태가 됩니다.
    team.config = { team_id: 't', actor: 'ci' };
    team.user = null;
    rerender();

    await waitFor(() => expect(result.current.available).toBe(false));
    expect(result.current.runs).toEqual([]);
    expect(result.current.loaded).toBe(false);
  });
});
