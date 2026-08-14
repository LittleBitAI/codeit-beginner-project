/**
 * 팀이 공유하는 학습 기록을 읽고, 그 뒤 변화는 구독으로 받아 이어 붙입니다.
 *
 * 팀 활동 화면과 기록 목록의 "학습 중" 표가 같은 것을 봅니다. 두 곳이 각자 읽으면
 * 같은 병합 규칙을 두 번 적게 되고, 한쪽만 고쳐지면 화면마다 다른 말을 합니다.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import type { TeamRun } from '../api/types';
import * as cloud from './cloud';
import { useTeam } from './TeamContext';

/** 값이 없는 것과 "아직 안 왔다"를 구분합니다. 빈 객체는 후자로 봅니다. */
function absent(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  return typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0;
}

/**
 * 같은 실행의 두 소식을 하나로 만듭니다. **목록과 구독이 이 규칙 하나만 씁니다.**
 *
 * 두 가지를 함께 지켜야 합니다.
 *
 * - **빈 field는 덮지 않는다.** AppSync subscription은 mutation이 고른 field만 실어
 *   줍니다. 좁게 고른 publisher가 섞여 들어오면 settings·summary·evaluation이 빈 채로
 *   도착하는데, 그걸 그대로 덮어쓰면 화면에서 모델명과 mAP가 통째로 `-`가 됩니다.
 * - **오래된 소식이 새 상태를 되돌리지 않는다.** 목록은 뜬 순간의 snapshot이라
 *   구독으로 온 최신 상태보다 뒤처져 있을 수 있습니다. `revision`이 그 순서입니다.
 *
 * 규칙이 두 벌이면 도착 순서에 따라 정보가 사라지거나, 끝난 학습이 다시 도는 것처럼
 * 보입니다. 그래서 새로운 쪽의 값이 이기고, 빈 자리는 오래된 쪽이 메웁니다.
 */
export function mergeRun(previous: TeamRun | undefined, incoming: TeamRun): TeamRun {
  if (!previous) return incoming;
  const [older, newer] =
    previous.revision > incoming.revision ? [incoming, previous] : [previous, incoming];
  const kept: Record<string, unknown> = { ...older };
  for (const [key, value] of Object.entries(newer)) {
    if (!absent(value)) kept[key] = value;
  }
  return kept as unknown as TeamRun;
}

/** 구독으로 온 소식 하나를 목록에 얹습니다. */
export function mergeRuns(previous: TeamRun[], incoming: TeamRun): TeamRun[] {
  const known = previous.find((run) => run.cloudRunId === incoming.cloudRunId);
  const without = previous.filter((run) => run.cloudRunId !== incoming.cloudRunId);
  return [mergeRun(known, incoming), ...without].sort((a, b) =>
    b.createdAt.localeCompare(a.createdAt),
  );
}

/** 아직 끝나지 않은 학습입니다. */
export function isActiveRun(run: TeamRun): boolean {
  return run.status === 'starting' || run.status === 'running' || run.status === 'queued';
}

/** heartbeat가 2분 넘게 끊긴 진행 중 학습. 도는 척만 하고 있을 수 있습니다. */
export function isStaleRun(run: TeamRun): boolean {
  return isActiveRun(run) && Date.now() - Date.parse(run.heartbeatAt) > 120_000;
}

export interface TeamRunsState {
  runs: TeamRun[];
  error: string | null;
  /** 팀 기록을 읽을 수 있는 환경인지. 아니면 `runs`가 비어 있어도 "없다"가 아닙니다. */
  available: boolean;
  /**
   * 한 번이라도 읽어 봤는지. 읽기 전에도 `runs`는 빈 배열이라, 이것 없이는 화면이
   * 첫 순간에 "팀에 도는 학습이 없다"고 단정합니다.
   */
  loaded: boolean;
  refresh: () => void;
}

export function useTeamRuns(): TeamRunsState {
  const team = useTeam();
  const teamId = team.config.team_id;
  // 팀 기록 조회는 Cognito 로그인만 받습니다. 로그인할 수 없는 환경에서 부르면
  // 반드시 실패하므로 아예 보내지 않습니다.
  const canRead = !team.config.actor || Boolean(team.user);
  const available = Boolean(teamId) && canRead;
  const [runs, setRuns] = useState<TeamRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  /**
   * 지금 누구의 자격으로 읽고 있는지. 그 자격이 바뀌면 이 값이 올라갑니다.
   *
   * 상태를 비우는 것만으로는 모자랍니다. 로그아웃 전에 떠난 목록 요청과 마지막 구독
   * event가 뒤늦게 도착해 비운 자리를 도로 채우면, 화면은 "이 컴퓨터 것만 보인다"고
   * 적어 놓고 앞 사용자의 학습을 다시 보여 줍니다.
   */
  const reader = useRef(0);

  const refresh = useCallback(() => {
    if (!teamId || !canRead) return;
    const asked = reader.current;
    void cloud.listRuns(teamId).then(
      (fetched) => {
        if (asked !== reader.current) return;
        // 목록 응답으로 통째로 덮으면 안 됩니다. 목록을 뜬 뒤에 일어난 변화가 구독으로
        // 먼저 도착해 있을 수 있습니다. 병합 규칙은 구독 경로와 같은 것을 씁니다.
        setRuns((previous) => {
          const known = new Map(previous.map((run) => [run.cloudRunId, run]));
          const merged = new Map(
            fetched.map((run) => [run.cloudRunId, mergeRun(known.get(run.cloudRunId), run)]),
          );
          // 목록에 없는데 구독으로만 들어온 실행도 남깁니다. 서버는 실행을 지우지
          // 않으므로, 없는 것은 목록을 뜬 뒤에 생겼다는 뜻입니다.
          for (const run of previous) if (!merged.has(run.cloudRunId)) merged.set(run.cloudRunId, run);
          return [...merged.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
        });
        setError(null);
        setLoaded(true);
      },
      (problem: unknown) => {
        if (asked !== reader.current) return;
        setError(problem instanceof Error ? problem.message : '팀 기록을 읽지 못했습니다.');
      },
    );
  }, [teamId, canRead]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // 읽을 수 없게 되면(로그아웃, 다른 사용자, 팀 설정 해제) 들고 있던 것을 버리고,
  // 그 전에 떠난 응답과 event를 더는 받지 않습니다.
  useEffect(() => {
    if (available) return;
    reader.current += 1;
    setRuns([]);
    setLoaded(false);
    setError(null);
  }, [available]);

  useEffect(() => {
    if (!available || !team.latestEvent) return;
    setRuns((previous) => mergeRuns(previous, team.latestEvent as TeamRun));
  }, [team.latestEvent, available]);

  return { runs, error, available, loaded, refresh };
}
