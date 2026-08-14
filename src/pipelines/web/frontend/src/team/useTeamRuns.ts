/**
 * 팀이 공유하는 학습 기록을 읽고, 그 뒤 변화는 구독으로 받아 이어 붙입니다.
 *
 * 팀 활동 화면과 기록 목록의 "학습 중" 표가 같은 것을 봅니다. 두 곳이 각자 읽으면
 * 같은 병합 규칙을 두 번 적게 되고, 한쪽만 고쳐지면 화면마다 다른 말을 합니다.
 */

import { useCallback, useEffect, useState } from 'react';

import type { TeamRun } from '../api/types';
import * as cloud from './cloud';
import { useTeam } from './TeamContext';

/** 값이 없는 것과 "아직 안 왔다"를 구분합니다. 빈 객체는 후자로 봅니다. */
function absent(value: unknown): boolean {
  if (value === null || value === undefined) return true;
  return typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0;
}

/**
 * AppSync subscription은 mutation이 고른 field만 실어 줍니다. 좁게 고른 publisher가
 * 섞여 들어오면 settings·summary·evaluation이 빈 채로 도착하는데, 그걸 그대로
 * 덮어쓰면 목록에서 모델명과 mAP가 통째로 `-`가 됩니다. 이미 아는 값은 지우지 않습니다.
 */
export function mergeRuns(previous: TeamRun[], incoming: TeamRun): TeamRun[] {
  const known = previous.find((run) => run.cloudRunId === incoming.cloudRunId);
  let merged = incoming;
  if (known) {
    const kept: Record<string, unknown> = { ...known };
    for (const [key, value] of Object.entries(incoming)) {
      if (!absent(value)) kept[key] = value;
    }
    merged = kept as unknown as TeamRun;
  }
  const without = previous.filter((run) => run.cloudRunId !== incoming.cloudRunId);
  return [merged, ...without].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
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

  const refresh = useCallback(() => {
    if (!teamId || !canRead) return;
    void cloud.listRuns(teamId).then(
      (fetched) => {
        setRuns(fetched);
        setError(null);
        setLoaded(true);
      },
      (problem: unknown) => {
        setError(problem instanceof Error ? problem.message : '팀 기록을 읽지 못했습니다.');
      },
    );
  }, [teamId, canRead]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (team.latestEvent) setRuns((previous) => mergeRuns(previous, team.latestEvent as TeamRun));
  }, [team.latestEvent]);

  return { runs, error, available, loaded, refresh };
}
