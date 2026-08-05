import { useCallback, useEffect, useMemo, useState } from 'react';

import type { LogLine, TeamRun } from '../api/types';
import { LogStream } from '../components/LogStream';
import { AlertRow, Button, Panel, ScreenIntro, StatusBadge } from '../components/primitives';
import { color, font } from '../design/tokens';
import * as cloud from '../team/cloud';
import { useTeam } from '../team/TeamContext';

function entries(value: Record<string, unknown>): [string, unknown][] {
  return Object.entries(value).filter(([, item]) => item !== null && item !== '');
}

function show(value: unknown): string {
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
}

function isActive(run: TeamRun): boolean {
  return run.status === 'starting' || run.status === 'running' || run.status === 'queued';
}

function isStale(run: TeamRun): boolean {
  return isActive(run) && Date.now() - Date.parse(run.heartbeatAt) > 120_000;
}

function mergeRuns(previous: TeamRun[], incoming: TeamRun): TeamRun[] {
  const without = previous.filter((run) => run.cloudRunId !== incoming.cloudRunId);
  return [incoming, ...without].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

function mergeLines(previous: LogLine[], incoming: LogLine[]): LogLine[] {
  const bySequence = new Map(previous.map((line) => [line.seq, line]));
  for (const line of incoming) bySequence.set(line.seq, line);
  return [...bySequence.values()].sort((a, b) => a.seq - b.seq).slice(-2000);
}

export function TeamActivity() {
  const team = useTeam();
  const teamId = team.config.team_id;
  const [runs, setRuns] = useState<TeamRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const selected = useMemo(
    () => runs.find((run) => run.cloudRunId === selectedId) ?? runs[0] ?? null,
    [runs, selectedId],
  );

  const refresh = useCallback(async () => {
    if (!teamId) return;
    try {
      setRuns(await cloud.listRuns(teamId));
      setError(null);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : '팀 기록을 읽지 못했습니다.');
    }
  }, [teamId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (team.latestEvent) setRuns((previous) => mergeRuns(previous, team.latestEvent!));
  }, [team.latestEvent]);

  useEffect(() => {
    if (!teamId || !selected) return;
    let active = true;
    let cursor = 0;
    setLines([]);
    const catchUp = async () => {
      try {
        const batches = await cloud.listLogs(teamId, selected.cloudRunId, cursor);
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
      selected.cloudRunId,
      (batch) => {
        cursor = Math.max(cursor, batch.endSeq);
        setLines((previous) => mergeLines(previous, batch.lines));
      },
      () => void catchUp(),
    );
    void catchUp();
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, [teamId, selected?.cloudRunId]);

  if (!team.config.enabled) {
    return (
      <AlertRow level="info" title="팀 동기화가 꺼져 있습니다">
        AWS stack 출력값을 서버 환경 변수에 넣고 PILL_TEAM_SYNC_ENABLED=true로 설정하세요.
      </AlertRow>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <ScreenIntro title="팀원들의 학습을 한곳에서 봅니다">
        시작·진행·완료 상태와 마스킹된 전체 로그가 실시간으로 공유됩니다. 로그는 AWS에서 30일 뒤
        만료됩니다.
      </ScreenIntro>
      {error && (
        <AlertRow level="warning" title="팀 cloud 연결을 확인해 주세요" action={<Button onClick={() => void refresh()}>다시 읽기</Button>}>
          {error}
        </AlertRow>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, .8fr) minmax(420px, 1.4fr)', gap: 14 }}>
        <Panel title={`팀 학습 ${runs.length}건`} bodyStyle={{ padding: 0 }}>
          {runs.length === 0 ? (
            <div style={{ padding: 16, color: color.textMuted }}>공유된 학습이 없습니다.</div>
          ) : (
            runs.map((run) => (
              <button
                key={run.cloudRunId}
                type="button"
                onClick={() => setSelectedId(run.cloudRunId)}
                style={{
                  width: '100%',
                  border: 0,
                  borderBottom: `1px solid ${color.borderInner}`,
                  background: selected?.cloudRunId === run.cloudRunId ? color.primaryTint : color.surface,
                  padding: '12px 14px',
                  textAlign: 'left',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  cursor: 'pointer',
                }}
              >
                <span style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                  <strong style={{ font: `600 12px/1.3 ${font.mono}`, color: color.text }}>{run.runId}</strong>
                  <StatusBadge status={run.status} />
                </span>
                <span style={{ font: `400 11px/1.4 ${font.sans}`, color: color.textBody }}>
                  {run.actorName} · {new Date(run.createdAt).toLocaleString('ko-KR')}
                </span>
                {isStale(run) && <span style={{ color: color.amber, fontSize: 11 }}>연결 끊김 의심 · 마지막 heartbeat 2분 초과</span>}
              </button>
            ))
          )}
        </Panel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {selected ? (
            <>
              <Panel title={`${selected.actorName} · ${selected.runId}`} right={<StatusBadge status={selected.status} />}>
                {selected.message && <p style={{ color: color.textBody }}>{selected.message}</p>}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
                  {entries(selected.settings).map(([key, value]) => (
                    <div key={key}>
                      <div style={{ font: `600 10px/1.4 ${font.mono}`, color: color.textMuted }}>{key}</div>
                      <div style={{ font: `400 11px/1.5 ${font.mono}`, overflowWrap: 'anywhere' }}>{show(value)}</div>
                    </div>
                  ))}
                </div>
              </Panel>
              <Panel title="실시간 로그" bodyStyle={{ padding: 0 }}>
                <LogStream lines={lines} streaming={isActive(selected)} height={280} />
              </Panel>
              {!isActive(selected) && (
                <Panel title="완료 결과와 산출물">
                  <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', font: `400 11px/1.6 ${font.mono}` }}>
                    {JSON.stringify(selected.summary, null, 2)}
                  </pre>
                  {entries(selected.artifacts).map(([key, value]) => {
                    const location = show(value);
                    return (
                      <div key={key} style={{ marginTop: 8, font: `400 11px/1.5 ${font.mono}` }}>
                        {key}: {location}{' '}
                        <strong style={{ color: location.startsWith('s3://') ? color.green : color.amber }}>
                          {location.startsWith('s3://') ? '· 팀 공유 가능' : '· 작성자 PC 전용'}
                        </strong>
                      </div>
                    );
                  })}
                </Panel>
              )}
            </>
          ) : (
            <Panel>왼쪽에서 학습을 선택하세요.</Panel>
          )}
        </div>
      </div>
    </div>
  );
}
