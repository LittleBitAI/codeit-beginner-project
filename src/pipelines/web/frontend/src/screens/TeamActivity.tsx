import { useCallback, useEffect, useMemo, useState } from 'react';

import type { Defaults, LogLine, TeamRun } from '../api/types';
import { LogStream } from '../components/LogStream';
import { AlertRow, Button, KpiCard, Panel, ScreenIntro, StatusBadge } from '../components/primitives';
import { color, font } from '../design/tokens';
import * as cloud from '../team/cloud';
import { useTeam } from '../team/TeamContext';

/**
 * 하이퍼파라미터 label은 /api/train/defaults에서 받아 씁니다. 여기 있는 것은 train
 * summary에만 나오는 key라 그 목록에 없는 값들입니다.
 */
const SUMMARY_LABELS: Record<string, string> = {
  augmentation: '증강 preset',
  train_images: '학습 이미지 수',
  validation_images: '검증 이미지 수',
  class_count: '클래스 수',
  best_epoch: 'Best epoch',
  best_validation_loss: 'Best validation loss',
};

/** evaluate가 metrics.json에 쓰는 이름 그대로입니다. `mAP`가 곧 mAP@[0.75:0.95]입니다. */
const MAP_LABEL = 'mAP@[0.75:0.95]';
const METRIC_LABELS: [string, string][] = [
  ['mAP', MAP_LABEL],
  ['mAP50', 'mAP@0.5'],
  ['mAP75', 'mAP@0.75'],
  ['precision50', 'Precision@IoU0.5 (score≥0.5)'],
  ['recall50', 'Recall@IoU0.5 (score≥0.5)'],
];

/** 맨 위 요약에 이미 나온 값은 아래 상세에서 다시 보여 주지 않습니다. */
const HEADLINE_KEYS = ['architecture', 'optimizer', 'epochs'];

function show(value: unknown): string {
  // 빈 값도 label과 함께 남겨야 "왜 비었는지"가 화면에 드러납니다.
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function metric(value: unknown): string {
  return typeof value === 'number' ? value.toFixed(4) : '-';
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
}

function metricsOf(run: TeamRun): Record<string, unknown> {
  return record(record(run.evaluation).metrics);
}

/** 학습이 끝나면 train이 정규화한 summary가 진짜입니다. 시작 시점 설정은 보조입니다. */
function detail(run: TeamRun, key: string): unknown {
  const recorded = run.summary?.[key];
  return recorded === undefined || recorded === null ? run.settings?.[key] : recorded;
}

function labelsOf(defaults: Defaults | null): Record<string, string> {
  const labels: Record<string, string> = { ...SUMMARY_LABELS };
  for (const field of defaults?.fields ?? []) labels[field.name] = field.label;
  return labels;
}

function rowsOf(value: Record<string, unknown>): [string, unknown][] {
  return Object.entries(value).filter(([key]) => !HEADLINE_KEYS.includes(key));
}

function summaryLine(run: TeamRun): string {
  return `${show(detail(run, 'architecture'))} · ${MAP_LABEL} ${metric(metricsOf(run).mAP)}`;
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

function ValueGrid({ rows }: { rows: [string, string][] }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
      {rows.map(([label, value]) => (
        <div key={label}>
          <div style={{ font: `600 10px/1.4 ${font.sans}`, color: color.textMuted }}>{label}</div>
          <div style={{ font: `400 11px/1.5 ${font.mono}`, overflowWrap: 'anywhere' }}>{value}</div>
        </div>
      ))}
    </div>
  );
}

export function TeamActivity({ defaults }: { defaults: Defaults | null }) {
  const team = useTeam();
  const teamId = team.config.team_id;
  const [runs, setRuns] = useState<TeamRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const labels = useMemo(() => labelsOf(defaults), [defaults]);
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

  const settingRows: [string, string][] = selected
    ? rowsOf(selected.settings).map(([key, value]) => [labels[key] ?? key, show(value)])
    : [];
  const summaryRows: [string, string][] = selected
    ? rowsOf(selected.summary).map(([key, value]) => [labels[key] ?? key, show(value)])
    : [];
  const evaluated = selected ? Object.keys(metricsOf(selected)).length > 0 : false;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <ScreenIntro title="팀원들의 학습을 한곳에서 봅니다">
        어떤 모델로 얼마나 나왔는지가 목록에 바로 보입니다. 마스킹된 전체 로그도 실시간으로
        공유되며 AWS에서 30일 뒤 만료됩니다.
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
                {/* 클릭하지 않아도 팀원끼리 비교되도록 맨 아래에 요약 한 줄을 둡니다. */}
                <span
                  style={{
                    font: `400 11px/1.4 ${font.mono}`,
                    color: color.textStrong,
                    overflowWrap: 'anywhere',
                  }}
                >
                  {summaryLine(run)}
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
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10 }}>
                  <KpiCard compact label={labels.architecture ?? '모델'} value={show(detail(selected, 'architecture'))} />
                  <KpiCard compact label={labels.optimizer ?? 'Optimizer'} value={show(detail(selected, 'optimizer'))} />
                  <KpiCard compact label={labels.epochs ?? 'Epochs'} value={show(detail(selected, 'epochs'))} />
                  <KpiCard compact label={MAP_LABEL} value={metric(metricsOf(selected).mAP)} />
                </div>
                <details style={{ marginTop: 12 }}>
                  <summary style={{ font: `600 11px/1.5 ${font.sans}`, color: color.textBody, cursor: 'pointer' }}>
                    하이퍼파라미터 자세히 ({settingRows.length}개)
                  </summary>
                  <div style={{ marginTop: 10 }}>
                    <ValueGrid rows={settingRows} />
                  </div>
                </details>
              </Panel>
              <Panel title="실시간 로그" bodyStyle={{ padding: 0 }}>
                <LogStream lines={lines} streaming={isActive(selected)} height={280} />
              </Panel>
              {!isActive(selected) && (
                <Panel title="완료 결과와 산출물">
                  <ValueGrid rows={summaryRows} />
                  <div style={{ marginTop: 14, borderTop: `1px solid ${color.borderRow}`, paddingTop: 12 }}>
                    {evaluated ? (
                      <ValueGrid
                        rows={METRIC_LABELS.map(([key, label]) => [
                          label,
                          metric(metricsOf(selected)[key]),
                        ])}
                      />
                    ) : (
                      <div style={{ font: `400 11px/1.5 ${font.sans}`, color: color.textMuted }}>
                        평가를 아직 돌리지 않아 {MAP_LABEL} 값이 없습니다.
                      </div>
                    )}
                  </div>
                  {Object.entries(selected.artifacts)
                    .filter(([, value]) => value !== null && value !== '')
                    .map(([key, value]) => {
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
