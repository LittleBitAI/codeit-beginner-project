/**
 * 고른 dataset의 기록과 대기열을 세우는 첫 화면입니다.
 *
 * 위에서부터 **지금 도는 것 → 기록 / 대기열** 순입니다. 화면을 열었을 때 사람이
 * 가장 먼저 묻는 것이 "지금 뭐가 돌고 있지"이고, 그다음이 "그래서 뭐가 제일
 * 낫지"이기 때문입니다.
 */

import { useMemo, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';

import type { JobRecord, Progress, QueueState, RegistryScope, TeamRun } from '../api/types';
import {
  AlertRow,
  Badge,
  Button,
  Chip,
  EmptyState,
  LinkAction,
  LiveDot,
  Metric,
  MetricGrid,
  SortToggle,
} from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { useElapsedSeconds } from '../hooks/useElapsedSeconds';
import { duration, loss, startedAt } from '../lib/format';
import { epochsDone, progressRatio } from '../lib/progress';
import {
  FILTER_LABEL,
  SORT_LABEL,
  countLabel,
  hasResult,
  isRunning,
  matchesFilter,
  sortRecords,
  type RecordFilter,
  type RecordSort,
  type RunRecord,
} from '../lib/records';
import { TeamLogStream } from '../team/TeamLogStream';
import { isActiveRun, isStaleRun } from '../team/useTeamRuns';

const FILTERS: RecordFilter[] = ['all', 'evaluated', 'submitted', 'running', 'unregistered'];
const SORTS: RecordSort[] = ['recent', 'kaggle', 'loss'];

function score(value: number | null): string {
  return value === null ? '-' : value.toFixed(4);
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/* ----------------------------------------------------------- 지금 학습 중 */

/**
 * 지금 도는 학습 한 장. 카드 바탕이 진행률만큼 차오릅니다.
 *
 * 막대를 따로 두지 않고 카드 자체를 채우는 것은, 이 화면에서 진행률이 곁다리
 * 정보가 아니라 카드 전체가 말하는 하나이기 때문입니다. 검증 손실과 **남은
 * 시간**을 같은 크기로 나란히 둡니다 — 카드를 열자마자 사람이 묻는 것이
 * "얼마나 잘 되나"와 "언제 끝나나" 둘이라, 하나를 작게 두면 매번 찾아야 합니다.
 */
function LiveCard({ job, onOpen }: { job: JobRecord; onOpen: () => void }) {
  const progress = job.progress;
  const latest = progress.epochs[progress.epochs.length - 1] ?? null;
  const best = progress.best ?? null;
  const done = epochsDone(progress);
  const total = progress.total_epochs;
  const ratio = progressRatio(progress);

  // 좋아진 폭입니다. 첫 epoch과 지금 best의 차이라 "얼마나 내려왔는지"가 됩니다.
  const first = progress.epochs.find((item) => item.validation_loss !== null)?.validation_loss ?? null;
  const delta = first !== null && best ? first - best.validation_loss : null;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') onOpen();
      }}
      style={{
        position: 'relative',
        background: color.panel,
        padding: '24px 26px',
        overflow: 'hidden',
        cursor: 'pointer',
      }}
    >
      {ratio !== null && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            bottom: 0,
            width: `${Math.max(0, Math.min(1, ratio)) * 100}%`,
            background: color.fill,
          }}
        />
      )}
      <div style={{ position: 'relative' }}>
        <div
          style={{
            ...type.monoValue,
            color: color.textStrong,
            marginBottom: 6,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {job.run_id}
        </div>
        <div
          style={{
            ...type.monoSpec,
            color: color.textMuted,
            marginBottom: 18,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {[progress.architecture, progress.device, job.status_label]
            .filter((part): part is string => Boolean(part))
            .join(' · ')}
        </div>

        {/* 큰 숫자 자리를 둘로 나눕니다: 얼마나 잘 되나 · 언제 끝나나. */}
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '20px 48px', flexWrap: 'wrap' }}>
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                font: `500 11.5px/1 ${font.mono}`,
                letterSpacing: '0.08em',
                color: color.textMid,
                marginBottom: 12,
              }}
            >
              VAL LOSS · EPOCH {done} / {total ?? '?'}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
              <span style={{ ...type.kpiHuge, color: color.textStrong }}>
                {loss(best?.validation_loss ?? latest?.validation_loss)}
              </span>
              {delta !== null && delta > 0 && (
                <span style={{ ...type.kpiMid, color: color.accent, whiteSpace: 'nowrap' }}>
                  ↓ {delta.toFixed(4)}
                </span>
              )}
            </div>
          </div>

          <div style={{ minWidth: 0 }}>
            <div style={{ font: `500 12px/1 ${font.sans}`, color: color.textMid, marginBottom: 12 }}>
              남은 시간
            </div>
            {/* 추정값은 측정값과 다르게 보여야 합니다. 못 재면 지어내지 않습니다. */}
            <div
              style={{
                ...type.kpiHuge,
                fontSize: 34,
                color: progress.eta_seconds === null ? color.textFaint : color.accent,
                fontStyle: progress.eta_seconds === null ? 'normal' : 'italic',
                whiteSpace: 'nowrap',
              }}
            >
              {progress.finished
                ? '끝남'
                : progress.eta_seconds === null
                  ? '알 수 없음'
                  : `~${duration(progress.eta_seconds)}`}
            </div>
            <div style={{ ...type.bodySmall, color: color.textBody, marginTop: 10 }}>
              {duration(job.elapsed_seconds)} 경과
            </div>
          </div>
        </div>
      </div>

      <MetricGrid
        style={{ position: 'relative', marginTop: 22, paddingTop: 18, borderTop: `1px solid ${color.fill}` }}
      >
        <Metric label="BEST EPOCH" value={best ? String(best.epoch) : '-'} />
        <Metric label="TRAIN LOSS" value={loss(latest?.train_loss)} />
        <Metric label="EPOCH 시간" value={duration(latest?.epoch_seconds)} />
        <Metric
          label="LEARNING RATE"
          value={latest?.learning_rate == null ? '-' : latest.learning_rate.toExponential(1)}
        />
        <Metric label="학습 이미지" value={progress.train_images?.toLocaleString('ko-KR') ?? '-'} />
        <Metric label="CLASS" value={progress.class_count?.toString() ?? '-'} />
      </MetricGrid>
    </div>
  );
}

/* ------------------------------------------------------------ 학습 중 목록 */

interface RunningRow {
  runId: string;
  /** 누가 돌리고 있는지. 이 컴퓨터면 그렇다고 적습니다. */
  who: string;
  epoch: string;
  valLoss: string;
  startedAt: string | null;
  /** 이 컴퓨터가 시작한 학습이면 모니터로 갈 수 있습니다. */
  jobId: string | null;
  /** 팀에 공유된 학습이면 여기서 실시간 로그를 펼칠 수 있습니다. */
  cloudRunId: string | null;
}

function rowFromTeamRun(run: TeamRun): RunningRow {
  const progress = (run.progress ?? {}) as Record<string, unknown>;
  // 내 학습 줄과 같은 규칙으로 셉니다. 한 표 안에서 다르게 세면 안 됩니다.
  const counted = epochsDone({
    current_epoch: num(progress.current_epoch),
    completed_epochs: num(progress.completed_epochs) ?? undefined,
    total_epochs: num(progress.total_epochs),
  } as Progress);
  const current = num(progress.current_epoch) === null && num(progress.completed_epochs) === null
    ? null
    : counted;
  const total = num(progress.total_epochs);
  const best = (progress.best ?? null) as { validation_loss?: unknown } | null;
  return {
    runId: run.runId,
    who: run.actorName,
    epoch: current === null ? '-' : `epoch ${current}${total === null ? '' : ` / ${total}`}`,
    valLoss: loss(num(best?.validation_loss)),
    startedAt: run.startedAt,
    jobId: null,
    cloudRunId: run.cloudRunId,
  };
}

function rowFromJob(job: JobRecord): RunningRow {
  const progress = job.progress;
  const current = progress.available ? epochsDone(progress) : null;
  return {
    runId: job.run_id,
    who: '나 (이 컴퓨터)',
    epoch:
      current === null
        ? '-'
        : `epoch ${current}${progress.total_epochs === null ? '' : ` / ${progress.total_epochs}`}`,
    valLoss: loss(progress.best?.validation_loss),
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
    <div style={{ padding: '18px 0', borderTop: `1px solid ${color.border}` }}>
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
            {[row.epoch, `val ${row.valLoss}`, elapsed === null ? null : `${duration(elapsed)} 경과`]
              .filter((part): part is string => part !== null)
              .join(' · ')}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, flex: 'none' }}>
          <span style={{ ...type.bodySmall, color: color.textBody }}>{row.who}</span>
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
      </div>
      {openLog && row.cloudRunId && <TeamLogStream cloudRunId={row.cloudRunId} />}
    </div>
  );
}

/* ------------------------------------------------------------------ 화면 */

function Tab({
  active,
  count,
  children,
  onClick,
}: {
  active: boolean;
  count: number;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'baseline',
        gap: 8,
        background: 'transparent',
        border: 0,
        padding: '0 0 8px',
        borderBottom: `2px solid ${active ? color.accent : 'transparent'}`,
      }}
    >
      <span style={{ ...type.sectionTitle, color: active ? color.text : color.textMuted }}>
        {children}
      </span>
      <span style={{ font: `400 13px/1 ${font.mono}`, color: color.textMuted }}>{count}</span>
    </button>
  );
}

export function Runs({
  datasetKey,
  records,
  liveJob,
  queue,
  scope,
  unnamedCount,
  teamRuns,
  teamAvailable,
  error,
  onNewExperiment,
  onRemoveFromQueue,
  onResumeQueue,
  onCancelJob,
}: {
  datasetKey: string | null;
  /** 이미 dataset으로 걸러진 기록들입니다. */
  records: RunRecord[];
  liveJob: JobRecord | null;
  queue: QueueState | null;
  scope: RegistryScope | undefined;
  /** 어떤 dataset으로 돌렸는지 이름을 댈 수 없어 왼쪽 목록에 줄이 없는 기록 수. */
  unnamedCount: number;
  /** 팀이 공유하는 학습. 팀 기능이 꺼져 있으면 빈 배열입니다. */
  teamRuns: TeamRun[];
  /** 팀 기록을 읽을 수 있는 환경인지. 아니면 빈 것이 "없다"는 뜻이 아닙니다. */
  teamAvailable: boolean;
  error: string | null;
  onNewExperiment: () => void;
  onRemoveFromQueue: (entryId: string) => void;
  onResumeQueue: () => void;
  onCancelJob: (jobId: string) => void;
}) {
  const navigate = useNavigate();
  const [tab, setTab] = useState<'records' | 'queue'>('records');
  const [filter, setFilter] = useState<RecordFilter>('all');
  const [sort, setSort] = useState<RecordSort>('recent');

  const shown = useMemo(
    () => sortRecords(records.filter((record) => matchesFilter(record, filter)), sort),
    [records, filter, sort],
  );

  /**
   * 결과 없이 끝난 기록은 목록 맨 아래로 접습니다.
   *
   * 전체를 볼 때만 접습니다. 다른 표는 사람이 이미 좁혀 놓은 것이라, 그 안에서 또
   * 접으면 "12건이라는데 아무것도 안 보인다"가 됩니다.
   */
  const folded = filter === 'all' ? shown.filter((record) => !hasResult(record)) : [];
  const listed = folded.length > 0 ? shown.filter(hasResult) : shown;

  const openRecord = (record: RunRecord) =>
    record.jobId
      ? navigate(`/monitor/${record.jobId}`)
      : navigate(`/canvas?run=${encodeURIComponent(record.runId)}`);

  /**
   * 지금 도는 학습. 팀이 공유한 것과 이 컴퓨터 것을 `run_id`로 합칩니다.
   *
   * 팀 기록이 켜져 있으면 내 학습도 거기 올라가므로 그대로 두면 같은 학습이 두 줄이
   * 됩니다. 이 컴퓨터 것이 이깁니다 — 모니터로 들어갈 수 있는 쪽이라서입니다.
   */
  const running = useMemo(() => {
    const rows = new Map<string, RunningRow>();
    // heartbeat가 2분 넘게 끊긴 것은 세우지 않습니다. 도는 척만 하고 있을 뿐이라,
    // "지금 돌고 있는 것"이라는 이 표의 뜻과 어긋납니다. 사라진 이유가 궁금하면
    // 팀 활동 화면이 상태 그대로 보여 줍니다.
    for (const run of teamRuns.filter((item) => isActiveRun(item) && !isStaleRun(item))) {
      rows.set(run.runId, rowFromTeamRun(run));
    }
    if (liveJob) rows.set(liveJob.run_id, rowFromJob(liveJob));
    for (const record of records.filter(isRunning)) {
      if (!rows.has(record.runId)) {
        rows.set(record.runId, {
          runId: record.runId,
          who: '나 (이 컴퓨터)',
          epoch: '-',
          valLoss: loss(record.metrics.bestValidationLoss),
          startedAt: record.at,
          jobId: record.jobId,
          cloudRunId: null,
        });
      }
    }
    return [...rows.values()];
  }, [teamRuns, liveJob, records]);

  const bestKaggle = records
    .map((record) => record.metrics.kaggle)
    .filter((value): value is number => value !== null);
  const bestLoss = records
    .map((record) => record.metrics.bestValidationLoss)
    .filter((value): value is number => value !== null);

  const stats = [
    `기록 ${records.length}건`,
    bestKaggle.length > 0 ? `최고 Kaggle ${Math.max(...bestKaggle).toFixed(4)}` : null,
    bestLoss.length > 0 ? `최저 val loss ${Math.min(...bestLoss).toFixed(4)}` : null,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');

  const entries = queue?.entries ?? [];
  const chipCount = (key: RecordFilter) =>
    key === 'running' ? running.length : records.filter((record) => matchesFilter(record, key)).length;

  return (
    <div style={{ padding: '36px 40px 60px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 28,
          marginBottom: 12,
        }}
      >
        <h1 style={{ ...type.pageTitle, margin: 0, color: color.textStrong, minWidth: 0, overflowWrap: 'break-word' }}>
          {datasetKey ?? '기록 없음'}
        </h1>
        <Button kind="primary" onClick={onNewExperiment} style={{ flex: 'none' }}>
          새 실험
        </Button>
      </div>

      <div style={{ ...type.body, color: color.textBody, marginBottom: 16 }}>{stats}</div>

      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 10,
          paddingBottom: 22,
          borderBottom: `1px solid ${color.border}`,
          maxWidth: '62em',
        }}
      >
        {/* registry를 아직 못 읽었으면 "이 컴퓨터"라고 단정하지 않습니다. index 전체를
            훑는 응답이라 수십 초가 걸리는데, 그동안 팀 기록이 없다고 말해 버리면
            사실이 아닌 것을 화면이 먼저 주장하는 셈입니다. */}
        <Badge tone={scope ? 'accent' : 'muted'}>
          {scope ? (scope.shared ? '팀 공유' : '이 컴퓨터') : '읽는 중'}
        </Badge>
        <span style={{ ...type.body, color: color.textBody, textWrap: 'pretty' }}>
          {!scope
            ? '등록된 실험 목록을 읽고 있습니다. 아래는 이 컴퓨터가 시작한 학습이고, 다 읽으면 팀 기록과 Kaggle 점수가 합쳐집니다.'
            : scope.shared
              ? '팀이 같은 S3 저장소를 쓰고 있어 팀원이 등록한 실험도 함께 나옵니다. 순위를 말할 수 있는 숫자는 Kaggle 점수뿐입니다 — 로컬 mAP는 참고용입니다.'
              : `지금 backend가 ${scope.backend}이라 이 컴퓨터에 등록된 실험만 보입니다. 팀원 것까지 보려면 PILL_STORAGE_S3_BUCKET을 설정한 뒤 서버를 다시 시작하세요.`}
        </span>
      </div>

      {error && (
        <div style={{ marginTop: 22 }}>
          <AlertRow level="error" title="backend에 연결하지 못했습니다">
            {error} 서버를 실행하려면 저장소 root에서{' '}
            <code style={{ fontFamily: font.mono }}>python -m src.pipelines.web.server</code>를
            실행하세요.
          </AlertRow>
        </div>
      )}

      <div style={{ marginTop: 34 }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            gap: 20,
            marginBottom: 18,
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', gap: 26, alignItems: 'baseline' }}>
            <Tab active={tab === 'records'} count={records.length} onClick={() => setTab('records')}>
              기록
            </Tab>
            <Tab
              active={tab === 'queue'}
              count={entries.length + (liveJob ? 1 : 0)}
              onClick={() => setTab('queue')}
            >
              학습 대기열
            </Tab>
          </div>
          <LinkAction onClick={() => navigate('/canvas')}>캔버스에서 견주기 →</LinkAction>
        </div>

        {tab === 'records' ? (
          <>
            {/* 학습 중 표에는 내 학습이 줄로 이미 들어갑니다. 카드까지 두면 같은
                학습이 한 화면에 두 번 나옵니다. */}
            {liveJob && filter !== 'running' && (
              <div style={{ marginBottom: 30 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                  <LiveDot pulse />
                  <span style={{ ...type.sectionTitle, color: color.text }}>지금 학습 중</span>
                  <LinkAction
                    tone="muted"
                    style={{ marginLeft: 'auto' }}
                    onClick={() => navigate(`/monitor/${liveJob.job_id}`)}
                  >
                    모니터 →
                  </LinkAction>
                </div>
                <LiveCard job={liveJob} onOpen={() => navigate(`/monitor/${liveJob.job_id}`)} />
              </div>
            )}

            {/* 조용히 빼면 그만큼이 없는 줄 압니다. 몇 건을 왜 뺐는지 늘 말합니다. */}
            {unnamedCount > 0 && (
              <div style={{ ...type.note, color: color.textFaint, marginBottom: 14 }}>
                어떤 dataset으로 돌렸는지 알 수 없는 기록 {unnamedCount}건은 왼쪽 목록에 세우지
                않았습니다. data artifact 위치가 <code style={{ fontFamily: font.mono }}>
                  …/&lt;dataset&gt;/train_manifest.json
                </code>{' '}
                모양이 아닌 옛 실행입니다.
              </div>
            )}

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '20px 24px',
                flexWrap: 'wrap',
                marginBottom: 4,
              }}
            >
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {FILTERS.map((key) => (
                  <Chip
                    key={key}
                    active={filter === key}
                    count={chipCount(key)}
                    onClick={() => setFilter(key)}
                  >
                    {FILTER_LABEL[key]}
                  </Chip>
                ))}
              </div>
              {filter !== 'running' && (
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
                  <span
                    style={{
                      font: `400 11px/1.4 ${font.mono}`,
                      letterSpacing: '0.08em',
                      color: color.textFaint,
                    }}
                  >
                    SORT
                  </span>
                  {SORTS.map((key) => (
                    <SortToggle key={key} active={sort === key} onClick={() => setSort(key)}>
                      {SORT_LABEL[key]}
                    </SortToggle>
                  ))}
                </div>
              )}
            </div>

            {filter === 'running' ? (
              <>
                {running.length === 0 ? (
                  <EmptyState message="지금 돌고 있는 학습이 없습니다." />
                ) : (
                  running.map((row) => (
                    <RunningRowView
                      key={row.runId}
                      row={row}
                      onOpen={() => row.jobId && navigate(`/monitor/${row.jobId}`)}
                    />
                  ))
                )}
                {!teamAvailable && (
                  <div style={{ ...type.note, color: color.textFaint, marginTop: 16 }}>
                    팀 실시간 연결이 꺼져 있어 이 컴퓨터에서 돌리는 학습만 보입니다. 팀원 것까지
                    보려면 팀 설정을 켜고 로그인하세요.
                  </div>
                )}
              </>
            ) : shown.length === 0 ? (
              <EmptyState
                message={
                  records.length === 0
                    ? '이 dataset에는 아직 기록이 없습니다. 오른쪽 위 새 실험으로 첫 학습을 걸어 보세요.'
                    : '고른 조건에 맞는 기록이 없습니다. 위 표를 바꾸면 나머지가 보입니다.'
                }
                action={
                  records.length === 0 ? (
                    <Button kind="primary" onClick={onNewExperiment}>
                      새 실험
                    </Button>
                  ) : undefined
                }
              />
            ) : (
              <>
                {listed.map((record) => (
                  <RecordRow key={record.runId} record={record} onOpen={() => openRecord(record)} />
                ))}
                {folded.length > 0 && <FoldedRecords records={folded} onOpen={openRecord} />}
              </>
            )}
          </>
        ) : (
          <QueueTab
            liveJob={liveJob}
            queue={queue}
            onNewExperiment={onNewExperiment}
            onRemoveFromQueue={onRemoveFromQueue}
            onResumeQueue={onResumeQueue}
            onCancelJob={onCancelJob}
            onOpenMonitor={(jobId) => navigate(`/monitor/${jobId}`)}
          />
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- 대기열 탭 */

/**
 * 지금 도는 것을 맨 위에 두고 그 아래로 줄을 세웁니다.
 *
 * 대기열은 "다음에 무엇이 도는가"의 목록이고, 지금 도는 학습이 그 0번입니다.
 * 따로 떼어 놓으면 순서를 머릿속에서 이어 붙여야 합니다. 추가 버튼은 언제나 줄의
 * 맨 끝에 있습니다 — 새로 넣는 것이 실제로 들어가는 자리가 거기라서입니다.
 */
function QueueTab({
  liveJob,
  queue,
  onNewExperiment,
  onRemoveFromQueue,
  onResumeQueue,
  onCancelJob,
  onOpenMonitor,
}: {
  liveJob: JobRecord | null;
  queue: QueueState | null;
  onNewExperiment: () => void;
  onRemoveFromQueue: (entryId: string) => void;
  onResumeQueue: () => void;
  onCancelJob: (jobId: string) => void;
  onOpenMonitor: (jobId: string) => void;
}) {
  const entries = queue?.entries ?? [];
  const paused = Boolean(queue?.paused);

  return (
    <div>
      {paused && entries.length > 0 && (
        <div style={{ marginBottom: 18 }}>
          <AlertRow
            level="warning"
            title="대기열이 멈춰 있습니다"
            action={<Button onClick={onResumeQueue}>다시 돌리기</Button>}
          >
            중지했거나 서버가 다시 시작돼서 멈췄습니다. 눌러야 다음 학습이 시작됩니다.
          </AlertRow>
        </div>
      )}

      {liveJob ? (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 18,
            padding: '18px 0',
            borderTop: `1px solid ${color.border}`,
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, minWidth: 0, flex: '1 1 20em' }}>
            <span style={{ font: `500 12.5px/1.5 ${font.mono}`, color: color.accent, flex: 'none' }}>
              지금
            </span>
            <div style={{ minWidth: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <LiveDot size={6} pulse />
                <button
                  type="button"
                  onClick={() => onOpenMonitor(liveJob.job_id)}
                  style={{
                    ...type.monoId,
                    color: color.text,
                    background: 'transparent',
                    border: 0,
                    padding: 0,
                    textAlign: 'left',
                    overflowWrap: 'break-word',
                  }}
                >
                  {liveJob.run_id}
                </button>
              </div>
              <div style={{ ...type.monoSpec, color: color.textMuted, paddingLeft: 15 }}>
                epoch {epochsDone(liveJob.progress)} /{' '}
                {liveJob.progress.total_epochs ?? '?'} · 남은 시간{' '}
                {liveJob.progress.eta_seconds === null
                  ? '알 수 없음'
                  : `~${duration(liveJob.progress.eta_seconds)}`}
              </div>
            </div>
          </div>
          <Button kind="danger" onClick={() => onCancelJob(liveJob.job_id)} style={{ flex: 'none' }}>
            취소
          </Button>
        </div>
      ) : (
        <div
          style={{
            padding: '18px 0',
            borderTop: `1px solid ${color.border}`,
            ...type.body,
            color: color.textMuted,
          }}
        >
          지금 도는 학습이 없습니다. 대기열에 넣으면 곧바로 시작합니다.
        </div>
      )}

      {entries.map((entry, index) => (
        <div
          key={entry.entry_id}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 18,
            padding: '16px 0',
            borderTop: `1px solid ${color.borderRow}`,
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, minWidth: 0, flex: '1 1 20em' }}>
            <span style={{ font: `500 12.5px/1.5 ${font.mono}`, color: color.textFaint, flex: 'none' }}>
              {index + 1}
            </span>
            <div style={{ minWidth: 0 }}>
              <div style={{ ...type.monoId, color: color.textBody, overflowWrap: 'break-word' }}>
                {entry.run_id || '(이름 없음)'}
              </div>
              <div style={{ ...type.note, color: color.textMuted }}>
                {startedAt(entry.queued_at)} 추가
              </div>
            </div>
          </div>
          <LinkAction
            tone="muted"
            onClick={() => onRemoveFromQueue(entry.entry_id)}
            style={{ flex: 'none', borderBottom: `1px solid ${color.border}` }}
          >
            빼기
          </LinkAction>
        </div>
      ))}

      {/* 줄의 맨 끝. 새로 넣는 것이 실제로 들어가는 자리입니다. */}
      <div style={{ paddingTop: 20, borderTop: `1px solid ${color.borderRow}` }}>
        <Button kind="secondary" onClick={onNewExperiment}>
          + 대기열에 추가
        </Button>
      </div>
    </div>
  );
}

/* ------------------------------------------------- 결과 없이 끝난 기록 구역 */

/**
 * 결과 없이 끝난 기록을 접어 두는 구역입니다.
 *
 * 지우지 않고 접습니다. 왜 실패했는지는 로그를 봐야 알 수 있고, 그 로그로 가는
 * 길이 이 줄뿐입니다. 대신 몇 건을 무슨 이유로 접었는지 머리글이 늘 말합니다 —
 * 조용히 빼면 그만큼이 없는 줄 압니다.
 */
function FoldedRecords({
  records,
  onOpen,
}: {
  records: RunRecord[];
  onOpen: (record: RunRecord) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ borderTop: `1px solid ${color.border}` }}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 10,
          width: '100%',
          padding: '18px 0',
          background: 'transparent',
          border: 0,
          textAlign: 'left',
        }}
      >
        <span style={{ ...type.body, color: color.textMuted }}>
          {open ? '▾' : '▸'} 결과 없이 끝남
        </span>
        <span style={{ font: `400 13px/1.4 ${font.mono}`, color: color.textFaint }}>
          {countLabel(records)}
        </span>
      </button>
      {open &&
        records.map((record) => (
          <RecordRow key={record.runId} record={record} onOpen={() => onOpen(record)} />
        ))}
    </div>
  );
}

/* -------------------------------------------------------------- 기록 한 줄 */

/** 기록 한 줄. 이름 → 식별자 → 설정 → 지표 순으로 내려갑니다. */
function RecordRow({ record, onOpen }: { record: RunRecord; onOpen: () => void }) {
  const running = isRunning(record);
  return (
    <div
      role="button"
      tabIndex={0}
      data-row-hover=""
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') onOpen();
      }}
      style={{ padding: '20px 0', borderTop: `1px solid ${color.border}`, cursor: 'pointer' }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: 18,
          marginBottom: 5,
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 0 }}>
          <span style={{ ...type.listName, color: color.text, minWidth: 0 }}>{record.family}</span>
          {record.submitted && <Badge>제출</Badge>}
          {/* 실패·취소한 줄에 미등록까지 붙이면 배지 둘이 같은 말을 합니다. 등록될
              수 있었는데 아직 안 된 것, 곧 성공으로 끝난 학습에만 붙입니다. */}
          {record.status === 'succeeded' && !record.registered && <Badge tone="muted">미등록</Badge>}
          {/* 끝난 이유는 반드시 적습니다. 미등록을 성공에만 붙이기로 하면서 취소·중단
              줄에는 아무 표시도 남지 않았습니다. 특히 중단은 이어서 학습할 대상이라
              성공한 기록과 눈으로 구별되어야 합니다. */}
          {record.status === 'failed' && <Badge tone="danger">{record.statusLabel}</Badge>}
          {(record.status === 'cancelled' || record.status === 'interrupted') && (
            <Badge tone="muted">{record.statusLabel}</Badge>
          )}
          {running && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 7, flex: 'none' }}>
              <LiveDot size={6} pulse />
              <span style={{ font: `500 12px/1.4 ${font.mono}`, color: color.accent }}>
                {record.statusLabel}
              </span>
            </span>
          )}
        </span>
        <span style={{ ...type.monoSpec, color: color.textMuted, flex: 'none', whiteSpace: 'nowrap' }}>
          {startedAt(record.at)}
        </span>
      </div>
      <div style={{ ...type.monoId, color: color.textBody, marginBottom: 4, overflowWrap: 'break-word' }}>
        {record.runId}
      </div>
      {record.spec !== '' && (
        <div style={{ ...type.monoSpec, color: color.textMuted, marginBottom: 16 }}>{record.spec}</div>
      )}
      <MetricGrid>
        <Metric
          label="KAGGLE"
          value={score(record.metrics.kaggle)}
          strong
          tone={record.metrics.kaggle === null ? 'muted' : 'accent'}
        />
        <Metric label="BEST VAL LOSS" value={loss(record.metrics.bestValidationLoss)} strong />
        <Metric label="mAP" value={score(record.metrics.map)} tone="muted" />
        <Metric label="mAP50" value={score(record.metrics.map50)} tone="muted" />
        <Metric label="mAP75" value={score(record.metrics.map75)} tone="muted" />
        <Metric label="PRECISION50" value={score(record.metrics.precision50)} tone="muted" />
        <Metric label="RECALL50" value={score(record.metrics.recall50)} tone="muted" />
        <Metric label="BEST EPOCH" value={record.metrics.bestEpoch?.toString() ?? '-'} tone="muted" />
        <Metric label="경과" value={duration(record.metrics.elapsedSeconds)} tone="muted" />
      </MetricGrid>
    </div>
  );
}
