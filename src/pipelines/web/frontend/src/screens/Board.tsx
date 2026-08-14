/**
 * 현황판. **지금 누가 무엇을 돌리고 있나**에 답합니다.
 *
 * 사람별로 묶습니다. 이 화면에서 사람이 묻는 것은 "GPU가 비었나", "저 사람 것이
 * 아직 도나"처럼 대개 사람 단위이기 때문입니다. 내 학습도 함께 세웁니다 — 팀
 * 기록이 꺼져 있는 컴퓨터에서 이 화면이 통째로 비면 무엇이 잘못됐는지 알 수 없습니다.
 */

import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import type { JobRecord, Progress, TeamRun } from '../api/types';
import { AlertRow, EmptyState, LinkAction, LiveDot } from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { useElapsedSeconds } from '../hooks/useElapsedSeconds';
import { duration, loss } from '../lib/format';
import { epochsDone } from '../lib/progress';
import { isRunning, type RunRecord } from '../lib/records';
import { datasetLabel } from '../lib/runSpec';
import { TeamLogStream } from '../team/TeamLogStream';
import { isActiveRun, isStaleRun } from '../team/useTeamRuns';

/** 이 컴퓨터에서 도는 학습에 붙이는 이름입니다. */
const ME = '나 (이 컴퓨터)';

interface RunningRow {
  /**
   * 이 줄을 가르는 값. 화면의 key로도 씁니다.
   *
   * `runId`는 안 됩니다. 자동 이름은 설정과 seed의 지문이라 서로 다른 실행이 같은
   * 이름을 갖습니다. 그것을 key로 두면 React가 두 줄을 같은 것으로 보고, 펼쳐 둔
   * 로그가 다른 사람의 실행으로 옮겨 갑니다.
   */
  id: string;
  runId: string;
  /** 누가 돌리고 있는지. 이 컴퓨터면 그렇다고 적습니다. */
  who: string;
  /** 사람을 가르는 값. 이름은 겹칠 수 있으므로 묶을 때는 이것을 씁니다. */
  whoKey: string;
  epoch: string;
  valLoss: string;
  /** 무슨 데이터로 돌리는지. 모르면 적지 않습니다. */
  dataset: string | null;
  startedAt: string | null;
  /** 이 컴퓨터가 시작한 학습이면 모니터로 갈 수 있습니다. */
  jobId: string | null;
  /** 팀에 공유된 학습이면 여기서 실시간 로그를 펼칠 수 있습니다. */
  cloudRunId: string | null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function rowFromTeamRun(run: TeamRun): RunningRow {
  const progress = (run.progress ?? {}) as Record<string, unknown>;
  // 내 학습 줄과 같은 규칙으로 셉니다. 한 표 안에서 다르게 세면 안 됩니다.
  const counted = epochsDone({
    current_epoch: num(progress.current_epoch),
    completed_epochs: num(progress.completed_epochs) ?? undefined,
    total_epochs: num(progress.total_epochs),
  } as Progress);
  const current =
    num(progress.current_epoch) === null && num(progress.completed_epochs) === null ? null : counted;
  const total = num(progress.total_epochs);
  const best = (progress.best ?? null) as { validation_loss?: unknown } | null;
  return {
    id: `cloud:${run.cloudRunId}`,
    runId: run.runId,
    who: run.actorName,
    // 이름이 같은 팀원 둘이 한 사람으로 합쳐지지 않도록 고유한 값으로 가릅니다.
    whoKey: run.actorSub || run.actorName,
    epoch: current === null ? '-' : `epoch ${current}${total === null ? '' : ` / ${total}`}`,
    valLoss: loss(num(best?.validation_loss)),
    dataset: datasetLabel(run.dataInputs as Record<string, string>),
    startedAt: run.startedAt,
    jobId: null,
    cloudRunId: run.cloudRunId,
  };
}

function rowFromJob(job: JobRecord): RunningRow {
  const progress = job.progress;
  const current = progress.available ? epochsDone(progress) : null;
  return {
    id: `job:${job.job_id}`,
    runId: job.run_id,
    who: ME,
    whoKey: ME,
    epoch:
      current === null
        ? '-'
        : `epoch ${current}${progress.total_epochs === null ? '' : ` / ${progress.total_epochs}`}`,
    valLoss: loss(progress.best?.validation_loss),
    dataset: datasetLabel(job.data_inputs),
    startedAt: job.started_at ?? job.created_at,
    jobId: job.job_id,
    cloudRunId: job.cloud_run_id ?? null,
  };
}

function RunningRowView({ row, onOpen }: { row: RunningRow; onOpen: () => void }) {
  const elapsed = useElapsedSeconds(row.startedAt, true);
  // 팀에 공유된 학습은 여기서 로그를 펼쳐 봅니다. 내 학습은 모니터 화면이 더 많이
  // 보여 주므로 그쪽으로 보냅니다.
  const [openLog, setOpenLog] = useState(false);

  return (
    <div style={{ padding: '16px 0', borderTop: `1px solid ${color.borderRow}` }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: 18,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ minWidth: 0, flex: '1 1 22em' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 6 }}>
            <LiveDot size={7} pulse />
            <span style={{ ...type.monoId, color: color.text, overflowWrap: 'break-word' }}>
              {row.runId}
            </span>
          </div>
          <div style={{ ...type.monoSpec, color: color.textMuted, paddingLeft: 16 }}>
            {[
              row.epoch,
              `val ${row.valLoss}`,
              elapsed === null ? null : `${duration(elapsed)} 경과`,
              row.dataset,
            ]
              .filter((part): part is string => part !== null)
              .join(' · ')}
          </div>
        </div>
        {row.jobId ? (
          <LinkAction onClick={onOpen}>모니터 →</LinkAction>
        ) : (
          row.cloudRunId && (
            <LinkAction onClick={() => setOpenLog((value) => !value)}>
              {openLog ? '로그 접기' : '로그 보기'}
            </LinkAction>
          )
        )}
      </div>
      {openLog && row.cloudRunId && <TeamLogStream cloudRunId={row.cloudRunId} />}
    </div>
  );
}

/** 사람 한 명이 지금 돌리는 학습들. */
function PersonSection({
  who,
  rows,
  onOpen,
}: {
  who: string;
  rows: RunningRow[];
  onOpen: (jobId: string) => void;
}) {
  return (
    <div style={{ marginTop: 26 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 12,
          paddingBottom: 10,
          borderBottom: `1px solid ${color.border}`,
        }}
      >
        <span style={{ ...type.sectionTitle, color: color.text }}>{who}</span>
        <span style={{ font: `400 12.5px/1.4 ${font.mono}`, color: color.textMuted }}>
          {rows.length}개 학습 중
        </span>
      </div>
      {rows.map((row) => (
        <RunningRowView
          key={row.id}
          row={row}
          onOpen={() => row.jobId && onOpen(row.jobId)}
        />
      ))}
    </div>
  );
}

export function Board({
  liveJob,
  records,
  teamRuns,
  teamAvailable,
  teamLoaded,
  teamError,
}: {
  /** 이 컴퓨터가 지금 돌리는 학습. */
  liveJob: JobRecord | null;
  /** 이 컴퓨터의 기록. 그중 아직 도는 것을 함께 세웁니다. */
  records: RunRecord[];
  /** 팀이 공유하는 학습. 팀 기능이 꺼져 있으면 빈 배열입니다. */
  teamRuns: TeamRun[];
  /** 팀 기록을 읽을 수 있는 환경인지. 아니면 빈 것이 "없다"는 뜻이 아닙니다. */
  teamAvailable: boolean;
  /** 팀 기록을 한 번이라도 읽어 봤는지. 읽기 전에도 목록은 비어 있습니다. */
  teamLoaded: boolean;
  /** 팀 기록을 읽다 실패했으면 그 이유. 실패를 "없음"으로 말하면 안 됩니다. */
  teamError: string | null;
}) {
  const navigate = useNavigate();

  /**
   * 지금 도는 학습. 팀이 공유한 것과 이 컴퓨터 것을 합칩니다.
   *
   * **`run_id`로 합치면 안 됩니다.** 자동으로 지어지는 이름은 설정과 seed의 지문이라,
   * 팀원 둘이 같은 설정으로 돌리면 이름이 같습니다 — 그것이 중복 실험을 알아채라고
   * 일부러 그렇게 만든 값입니다. 그 이름으로 묶으면 한 사람의 학습이 다른 사람의
   * 줄에 덮여 통째로 사라집니다. 실행을 실제로 가르는 것은 팀 기록의 `cloudRunId`와
   * 이 컴퓨터의 `job_id`입니다.
   *
   * 대신 내 학습은 팀에도 올라가므로 같은 실행이 두 줄이 될 수 있습니다. 그것은
   * 팀 기록이 들고 있는 `localJobId`로 알아보고 이 컴퓨터 줄을 남깁니다 — 모니터로
   * 들어갈 수 있는 쪽이라서입니다.
   */
  const rows = useMemo(() => {
    const merged = new Map<string, RunningRow>();
    // heartbeat가 2분 넘게 끊긴 것은 여기 세우지 않습니다. 도는 척만 하고 있을 뿐이라
    // "지금 돌고 있는 것"이라는 이 표의 뜻과 어긋납니다. 아래에 따로 모읍니다.
    const active = teamRuns.filter((item) => isActiveRun(item) && !isStaleRun(item));
    for (const run of active) merged.set(`cloud:${run.cloudRunId}`, rowFromTeamRun(run));

    /**
     * 이 컴퓨터의 학습 하나를 세웁니다. 팀에 올라간 같은 실행은 뺍니다.
     *
     * 첫 하나만 빼면 안 됩니다. 같은 학습이 두 번 공유된 기록이 남아 있으면 그것이
     * 또 한 줄이 되어, 한 실행이 화면에 둘로 보입니다.
     */
    const mine = (jobId: string | null, row: RunningRow) => {
      if (jobId) {
        for (const run of active.filter((item) => item.localJobId === jobId)) {
          merged.delete(`cloud:${run.cloudRunId}`);
        }
      }
      merged.set(row.id, row);
    };

    if (liveJob) mine(liveJob.job_id, rowFromJob(liveJob));
    for (const record of records.filter(isRunning)) {
      const key = record.jobId ? `job:${record.jobId}` : `run:${record.runId}`;
      if (merged.has(key)) continue;
      mine(record.jobId, {
        id: key,
        runId: record.runId,
        who: ME,
        whoKey: ME,
        epoch: '-',
        valLoss: loss(record.metrics.bestValidationLoss),
        dataset: record.datasetKey,
        startedAt: record.at,
        jobId: record.jobId,
        cloudRunId: null,
      });
    }
    return [...merged.values()];
  }, [teamRuns, liveJob, records]);

  /**
   * 사람별로 묶습니다. 내 줄이 언제나 맨 위입니다.
   *
   * 묶는 값은 보이는 이름이 아니라 `actorSub`입니다. 이름은 사람이 정하는 표시용이라
   * 같은 이름을 쓰는 팀원 둘이 한 사람으로 합쳐질 수 있습니다.
   */
  const people = useMemo(() => {
    const byPerson = new Map<string, { key: string; who: string; rows: RunningRow[] }>();
    for (const row of rows) {
      const group = byPerson.get(row.whoKey);
      if (group) group.rows.push(row);
      else byPerson.set(row.whoKey, { key: row.whoKey, who: row.who, rows: [row] });
    }
    return [...byPerson.values()].sort((left, right) =>
      left.who === ME ? -1 : right.who === ME ? 1 : left.who.localeCompare(right.who),
    );
  }, [rows]);

  const stale = teamRuns.filter(isStaleRun);

  return (
    <div style={{ padding: '36px 40px 60px' }}>
      <h1 style={{ ...type.pageTitle, margin: '0 0 8px', color: color.textStrong }}>현황판</h1>
      <div style={{ ...type.body, color: color.textBody, marginBottom: 4, maxWidth: '46em', textWrap: 'pretty' }}>
        지금 팀이 돌리고 있는 학습입니다. 팀 기록이 켜져 있으면 팀원 것까지, 아니면 이
        컴퓨터 것만 보입니다.
      </div>

      {/* 팀 기록을 못 읽었으면 그렇게 말합니다. 읽지 못한 것을 "없다"로 말하면, 팀원이
          밤새 GPU를 돌리는 중에도 화면은 아무도 안 돌린다고 합니다. */}
      {teamAvailable && teamError && (
        <div style={{ marginTop: 22 }}>
          <AlertRow level="warning" title="팀 기록을 읽지 못했습니다">
            {teamError} 아래는 이 컴퓨터에서 돌리는 학습뿐입니다.
          </AlertRow>
        </div>
      )}

      {rows.length === 0 ? (
        <div style={{ marginTop: 30 }}>
          <EmptyState
            message={
              teamAvailable && !teamLoaded && !teamError
                ? '팀 기록을 읽고 있습니다.'
                : teamAvailable && teamError
                  ? '이 컴퓨터에서 돌고 있는 학습이 없습니다. 팀 것은 위 이유로 알 수 없습니다.'
                  : '지금 돌고 있는 학습이 없습니다.'
            }
          />
        </div>
      ) : (
        people.map((person) => (
          <PersonSection
            key={person.key}
            who={person.who}
            rows={person.rows}
            onOpen={(jobId) => navigate(`/monitor/${jobId}`)}
          />
        ))
      )}

      {/* 사라진 것을 조용히 빼지 않습니다. 왜 목록에 없는지 여기서 말합니다. */}
      {stale.length > 0 && (
        <div style={{ marginTop: 34, paddingTop: 18, borderTop: `1px solid ${color.border}` }}>
          <div style={{ ...type.sectionTitle, color: color.textMuted, marginBottom: 10 }}>
            연결이 끊긴 학습 {stale.length}개
          </div>
          {stale.map((run) => (
            <div key={run.cloudRunId} style={{ ...type.monoSpec, color: color.textFaint, padding: '6px 0' }}>
              {run.runId} · {run.actorName} · 마지막 소식 {new Date(run.heartbeatAt).toLocaleString('ko-KR')}
            </div>
          ))}
          <div style={{ ...type.note, color: color.textFaint, marginTop: 8, maxWidth: '40em' }}>
            2분 넘게 소식이 없습니다. 그 컴퓨터가 꺼졌거나 인터넷이 끊긴 것이라, 학습이 아직
            도는지는 이 화면으로 알 수 없습니다.
          </div>
        </div>
      )}

      {!teamAvailable && (
        <div style={{ ...type.note, color: color.textFaint, marginTop: 30, maxWidth: '40em' }}>
          팀 실시간 연결이 꺼져 있어 이 컴퓨터에서 돌리는 학습만 보입니다. 팀원 것까지 보려면
          팀 설정을 켜고 로그인하세요.
        </div>
      )}
    </div>
  );
}
