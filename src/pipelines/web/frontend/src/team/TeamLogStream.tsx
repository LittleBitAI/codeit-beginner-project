/**
 * 팀원이 지금 돌리는 학습의 실시간 로그.
 *
 * 밀린 것을 한 번 읽고, 그 뒤는 구독으로 이어 받습니다. 구독이 끊기면 다시 읽어
 * 메우고 "스트리밍 중"을 끕니다 — 상태만 보고 연결된 척하면, 로그가 멈춘 것이
 * 학습이 멈춘 것인지 구독이 끊긴 것인지 구별할 수 없습니다.
 */

import { useEffect, useState } from 'react';

import type { LogLine } from '../api/types';
import { LogStream } from '../components/LogStream';
import { color, type } from '../design/tokens';
import * as cloud from './cloud';
import { useTeam } from './TeamContext';

function mergeLines(previous: LogLine[], incoming: LogLine[]): LogLine[] {
  const bySequence = new Map(previous.map((line) => [line.seq, line]));
  for (const line of incoming) bySequence.set(line.seq, line);
  return [...bySequence.values()].sort((a, b) => a.seq - b.seq).slice(-2000);
}

export function TeamLogStream({ cloudRunId }: { cloudRunId: string }) {
  const team = useTeam();
  const teamId = team.config.team_id;
  // 팀 로그 조회는 Cognito 로그인만 받습니다. 로그인할 수 없는 환경에서 부르면
  // 반드시 실패하므로 아예 보내지 않습니다.
  const canRead = !team.config.actor || Boolean(team.user);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!teamId || !canRead) return;
    let active = true;
    let cursor = 0;
    setLines([]);
    setConnected(true);
    const catchUp = async () => {
      try {
        const batches = await cloud.listLogs(teamId, cloudRunId, cursor);
        if (!active) return;
        const received = batches.flatMap((batch) => batch.lines);
        if (received.length) cursor = Math.max(cursor, ...received.map((line) => line.seq));
        setLines((previous) => mergeLines(previous, received));
      } catch (problem) {
        if (active) setError(problem instanceof Error ? problem.message : '팀 로그를 읽지 못했습니다.');
      }
    };
    void catchUp();
    const subscription = cloud.subscribeLogs(
      teamId,
      cloudRunId,
      (batch) => {
        cursor = Math.max(cursor, batch.endSeq);
        setLines((previous) => mergeLines(previous, batch.lines));
      },
      () => {
        if (active) setConnected(false);
        void catchUp();
      },
    );
    return () => {
      active = false;
      setConnected(false);
      subscription.unsubscribe();
    };
  }, [teamId, cloudRunId, canRead]);

  if (!canRead) {
    return (
      <div style={{ ...type.note, color: color.textFaint, padding: '12px 0' }}>
        팀 로그를 읽으려면 로그인이 필요합니다.
      </div>
    );
  }

  return (
    <div style={{ padding: '12px 0 4px' }}>
      {error && (
        <div style={{ ...type.note, color: color.danger, marginBottom: 8 }}>{error}</div>
      )}
      <LogStream lines={lines} streaming={connected} height={260} />
    </div>
  );
}
