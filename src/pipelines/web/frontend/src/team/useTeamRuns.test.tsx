/**
 * 팀 기록을 읽는 규칙을 지킵니다.
 *
 * 1. 목록과 구독이 **같은 병합 규칙**을 쓴다: 새로운 소식(revision)의 값이 이기고,
 *    빈 자리는 오래된 쪽이 메운다. 규칙이 두 벌이면 도착 순서에 따라 값이 사라지거나
 *    끝난 학습이 다시 도는 것처럼 보입니다.
 * 2. 읽을 수 없게 되면(로그아웃, 다른 사용자) 들고 있던 것을 버리고, 그 전에 떠난
 *    응답도 받지 않는다. 남겨 두면 화면은 "이 컴퓨터 것만 보인다"고 적어 놓고 앞
 *    사용자의 학습을 계속 보여 줍니다.
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
  it('목록에만 있는 실행은 그대로 받는다', async () => {
    listRuns.mockResolvedValue([run({ cloudRunId: 'c2', runId: 'run-2' })]);

    const { result } = renderHook(() => useTeamRuns());

    await waitFor(() => expect(result.current.runs).toHaveLength(1));
    expect(result.current.runs[0]?.cloudRunId).toBe('c2');
  });

  // 목록과 구독이 규칙을 따로 쓰면, 도착 순서에 따라 값이 사라지거나 끝난 학습이
  // 다시 도는 것처럼 보입니다. 한 규칙만 씁니다.
  it('빈 field는 덮지 않고, 오래된 소식이 새 상태를 되돌리지 않는다', async () => {
    let resolveList: (value: TeamRun[]) => void = () => {};
    listRuns.mockReturnValue(new Promise<TeamRun[]>((resolve) => { resolveList = resolve; }));

    const { result, rerender } = renderHook(() => useTeamRuns());

    // 구독은 고른 field만 실어 옵니다. 여기서는 상태만 온 최신 소식입니다.
    team.latestEvent = run({ revision: 5, status: 'succeeded', settings: {}, dataInputs: {} });
    rerender();
    await waitFor(() => expect(result.current.runs).toHaveLength(1));

    // 뒤늦게 온 목록은 값이 다 있지만 옛 상태입니다.
    await act(async () => {
      resolveList([
        run({ revision: 2, status: 'running', settings: { architecture: 'retina' } }),
      ]);
    });

    await waitFor(() => expect(result.current.loaded).toBe(true));
    const [merged] = result.current.runs;
    // 상태는 새 쪽, 비어 있던 값은 옛 쪽이 메웁니다.
    expect(merged?.status).toBe('succeeded');
    expect(merged?.settings).toEqual({ architecture: 'retina' });
  });

  it('로그아웃 뒤 늦게 도착한 목록 응답은 받지 않는다', async () => {
    let resolveList: (value: TeamRun[]) => void = () => {};
    listRuns.mockReturnValue(new Promise<TeamRun[]>((resolve) => { resolveList = resolve; }));

    const { result, rerender } = renderHook(() => useTeamRuns());

    // 응답이 오기 전에 로그아웃합니다.
    team.config = { team_id: 't', actor: 'ci' };
    team.user = null;
    rerender();
    await waitFor(() => expect(result.current.available).toBe(false));

    await act(async () => {
      resolveList([run()]);
    });

    expect(result.current.runs).toEqual([]);
    expect(result.current.loaded).toBe(false);
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
