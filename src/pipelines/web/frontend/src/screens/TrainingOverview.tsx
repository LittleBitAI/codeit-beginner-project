import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type {
  DataSource,
  GpuStatus,
  JobListing,
  JobRecord,
  QueueState,
} from '../api/types';
import { DataSourcePanel } from '../components/DataSourcePanel';
import {
  AlertRow,
  Button,
  EmptyState,
  KpiCard,
  Panel,
  ScreenIntro,
  StatusBadge,
} from '../components/primitives';
import { color, font, radius, type } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { duration, loss, megabytes, percent, startedAt } from '../lib/format';
import { countLabel, hasResult, specLine, stagesOf } from '../lib/runSpec';
import { useTeam } from '../team/TeamContext';

const COLUMNS = '1.6fr 132px .72fr .8fr .7fr .9fr 40px';
const HEADINGS = ['실행 이름', '단계', 'mAP', 'VAL LOSS', '경과', '시작', ''];

export function TrainingOverview({
  listing,
  source,
  onSourceSelected,
  onPrepared,
  onJobsChanged,
}: {
  listing: JobListing | null;
  source: DataSource | null;
  onSourceSelected: (source: DataSource) => void;
  onPrepared: () => void;
  onJobsChanged?: () => void;
}) {
  const navigate = useNavigate();
  const gpu = usePolling<GpuStatus>(() => api.gpu(), 5000);
  // 지우기 전에 무엇이 사라지고 무엇이 남는지 보여 줍니다.
  const [pendingDelete, setPendingDelete] = useState<JobRecord | null>(null);

  const jobs = listing?.jobs ?? [];
  const active = jobs.find((job) => job.job_id === listing?.active_job_id) ?? null;
  // 결과가 남은 학습과 결과 없이 끝난 기록을 나눕니다. 판단 기준은 lib/runSpec에 있습니다.
  const kept = useMemo(() => jobs.filter(hasResult), [jobs]);
  const discarded = useMemo(() => jobs.filter((job) => !hasResult(job)), [jobs]);

  const latestBest = useMemo(() => {
    const scored = jobs
      .filter((job) => job.status === 'succeeded')
      .map((job) => job.summary?.best_validation_loss)
      .filter((value): value is number => typeof value === 'number');
    return scored.length > 0 ? Math.min(...scored) : null;
  }, [jobs]);

  const failed = jobs.find((job) => job.status === 'failed') ?? null;
  const device = gpu.data?.telemetry.devices[0] ?? null;
  const telemetryDown = gpu.data && gpu.data.telemetry.source !== 'nvidia-smi';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1320 }}>
      <ScreenIntro
        title="지금 무엇이 돌고 있고 무엇이 끝났는지 봅니다"
        terms={[
          { term: 'epoch', meaning: '학습 데이터 전체를 한 번 훑는 단위입니다' },
          { term: 'validation loss', meaning: '학습에 쓰지 않은 데이터에서의 오차. 낮을수록 좋습니다' },
          { term: 'checkpoint', meaning: '학습 중간에 저장한 모델 파일입니다' },
        ]}
      >
        이 화면은 이 GUI로 시작한 학습만 보여 줍니다. 터미널에서 직접 돌린 학습은 여기에 나오지 않습니다.
        학습은 한 번에 하나만 실행할 수 있습니다.
      </ScreenIntro>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(172px, 1fr))',
          gap: 10,
        }}
      >
        <KpiCard
          label="실행 중"
          value={active ? '1' : '0'}
          compact
          note={active ? `${active.run_id} 진행 중입니다.` : '지금 돌고 있는 학습이 없습니다.'}
        />
        <KpiCard
          label="GPU · LOCAL"
          value={device?.name ?? (gpu.data?.torch.cuda_available ? 'CUDA 사용 가능' : '없음')}
          compact
          note={
            telemetryDown
              ? gpu.data?.telemetry.message ?? 'GPU 사용률 정보를 가져올 수 없습니다.'
              : `${percent(device?.utilization_percent)} · ${megabytes(device?.memory_used_mb)} / ${megabytes(device?.memory_total_mb)} · ${device?.temperature_c ?? '-'}°C`
          }
        />
        <KpiCard label="총 실행" value={String(jobs.length)} compact note="이 GUI가 시작한 학습 수입니다." />
        <KpiCard
          label="최고 VAL LOSS"
          value={loss(latestBest)}
          compact
          valueColor={latestBest === null ? color.textMuted : color.tealDark}
          note={
            latestBest === null
              ? '성공한 학습이 아직 없습니다.'
              : '성공한 학습 중 가장 낮은 검증 오차입니다.'
          }
        />
      </div>

      <DataSourcePanel
        source={source}
        onSelected={onSourceSelected}
        onPrepared={onPrepared}
      />

      <TrainingQueue />

      <Panel
        title="학습 실행"
        right={
          <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMuted }}>
            {jobs.length}건
          </span>
        }
        bodyStyle={{ padding: 0 }}
      >
        {jobs.length === 0 ? (
          <EmptyState
            message="아직 이 GUI로 시작한 학습이 없습니다."
            action={
              <Button kind="primary" onClick={() => navigate('/new')}>
                새 실험 만들기
              </Button>
            }
          />
        ) : (
          <>
            <RunGroup
              title="결과가 있는 학습"
              detail={`${kept.length}건`}
              jobs={kept}
              open
              onOpen={(job) => navigate(`/monitor/${job.job_id}`)}
              onDelete={setPendingDelete}
            />
            <RunGroup
              title="결과 없이 끝난 기록"
              detail={countLabel(discarded)}
              jobs={discarded}
              open={false}
              onOpen={(job) => navigate(`/monitor/${job.job_id}`)}
              onDelete={setPendingDelete}
            />
          </>
        )}
      </Panel>

      {failed && (
        <AlertRow
          level="error"
          title={`${failed.run_id} 학습이 실패했습니다`}
          action={
            <Button onClick={() => navigate(`/monitor/${failed.job_id}`)}>로그 보기</Button>
          }
        >
          {failed.message ?? '원인을 알 수 없습니다.'}
        </AlertRow>
      )}

      {pendingDelete && (
        <DeleteRecordDialog
          job={pendingDelete}
          onClose={() => setPendingDelete(null)}
          onDeleted={() => {
            setPendingDelete(null);
            onJobsChanged?.();
          }}
        />
      )}
    </div>
  );
}

/**
 * 기록을 지우기 전 확인입니다.
 *
 * 무엇이 사라지고 무엇이 남는지를 나란히 적습니다. "정말 지울까요?"만 묻는 창은
 * 학습 결과까지 날아가는 줄 알게 만들거나, 반대로 날아가는 줄 모르고 누르게 합니다.
 */
function DeleteRecordDialog({
  job,
  onClose,
  onDeleted,
}: {
  job: JobRecord;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await api.deleteJob(job.job_id);
      onDeleted();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '기록을 지우지 못했습니다.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${job.run_id} 기록 삭제`}
      onKeyDown={(event) => {
        if (event.key === 'Escape') onClose();
      }}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 60,
        background: 'rgba(17,28,46,.34)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
      }}
    >
      <div
        style={{
          width: 'min(460px, 100%)',
          background: color.surface,
          border: `1px solid ${color.border}`,
          borderRadius: radius.panel,
          padding: '18px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        <span style={{ font: `600 13.5px/1.4 ${font.sans}`, color: color.text }}>
          {job.run_id} 기록을 지울까요?
        </span>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          <span style={{ ...type.body, color: color.textBody }}>
            <b style={{ color: color.red }}>지웁니다</b> — 이 GUI가 들고 있는 실행 기록과 로그.
          </span>
          <span style={{ ...type.body, color: color.textBody }}>
            <b style={{ color: color.greenDark }}>그대로 둡니다</b> — 학습 결과 폴더와 checkpoint,
            registry에 등록된 실험, 팀에 공유된 기록, 이 학습이 쓴 설정 파일.
          </span>
          <span style={{ ...type.plainNote, color: color.textMuted }}>
            되돌릴 수 없습니다. 목록에서만 사라지고 학습 자체는 남습니다.
          </span>
        </div>
        {error && <AlertRow level="error" title="지우지 못했습니다">{error}</AlertRow>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <Button onClick={onClose} disabled={busy}>
            취소
          </Button>
          <Button kind="danger" onClick={() => void remove()} disabled={busy}>
            {busy ? '지우는 중…' : '기록 지우기'}
          </Button>
        </div>
      </div>
    </div>
  );
}

/**
 * 한 구역과 그 안의 표입니다. 비어 있으면 머리글도 만들지 않습니다.
 *
 * 열 머리글을 구역 안에 두는 것은, 접힌 구역을 폈을 때 그 표가 무슨 열인지 바로
 * 위에 있어야 하기 때문입니다.
 */
function RunGroup({
  title,
  detail,
  jobs,
  open,
  onOpen,
  onDelete,
}: {
  title: string;
  detail: string;
  jobs: JobRecord[];
  open: boolean;
  onOpen: (job: JobRecord) => void;
  onDelete: (job: JobRecord) => void;
}) {
  if (jobs.length === 0) return null;

  return (
    <details open={open}>
      <summary
        style={{
          padding: '9px 14px',
          font: `600 12px/1.4 ${font.sans}`,
          color: color.textStrong,
          background: color.surfaceAlt,
          borderBottom: `1px solid ${color.borderInner}`,
          cursor: 'pointer',
          display: 'flex',
          gap: 8,
          alignItems: 'baseline',
        }}
      >
        {title}
        <span style={{ ...type.plainNote, color: color.textMuted }}>{detail}</span>
      </summary>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: COLUMNS,
          background: color.surfaceTableHead,
          borderBottom: `1px solid ${color.border}`,
        }}
      >
        {HEADINGS.map((heading, index) => (
          <span
            key={heading || `spacer-${index}`}
            // 한글 머리글에는 mono와 자간을 쓰지 않습니다. "실 행  이 름"처럼 벌어져
            // 두 단어로 읽힙니다. 값은 그대로 mono라 소수점은 계속 맞습니다.
            style={{ font: `600 11.5px/1.3 ${font.sans}`, color: '#66707E', padding: '9px 12px' }}
          >
            {heading}
          </span>
        ))}
      </div>
      {jobs.map((job) => (
        <JobRow
          key={job.job_id}
          job={job}
          onOpen={() => onOpen(job)}
          onDelete={() => onDelete(job)}
        />
      ))}
    </details>
  );
}

/** 한 단계가 끝났는지. 색만으로 뜻을 전하지 않도록 점과 글자를 함께 둡니다. */
function StagePips({ job }: { job: JobRecord }) {
  return (
    <span style={{ display: 'flex', gap: 7, padding: '8px 12px', alignItems: 'center' }}>
      {stagesOf(job).map((stage) => (
        <span
          key={stage.key}
          title={stage.done ? `${stage.label} 끝남` : `${stage.label} 아직`}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 3,
            font: `${stage.done ? 600 : 400} 11px/1 ${font.sans}`,
            color: stage.done ? color.tealDark : color.textFaint,
          }}
        >
          <span
            aria-hidden
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: stage.done ? color.teal : 'transparent',
              border: `1px solid ${stage.done ? color.teal : color.borderControl}`,
            }}
          />
          {stage.label}
        </span>
      ))}
    </span>
  );
}

/**
 * 행마다 열리는 작은 메뉴입니다.
 *
 * 행 전체가 눌리는 버튼이라 메뉴 버튼은 그 클릭을 막아야 합니다. Escape와 초점이
 * 밖으로 나가면 닫히므로 마우스 없이도 빠져나올 수 있습니다.
 */
function RowMenu({
  label,
  items,
}: {
  label: string;
  items: { label: string; onSelect: () => void; danger?: boolean }[];
}) {
  const [open, setOpen] = useState(false);

  return (
    <span
      style={{ position: 'relative', display: 'flex', justifyContent: 'center' }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') setOpen(false);
      }}
    >
      <button
        type="button"
        aria-label={label}
        aria-expanded={open}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((value) => !value);
        }}
        style={{
          border: 0,
          background: 'transparent',
          color: color.textMuted,
          font: `600 13px/1 ${font.sans}`,
          padding: '6px 8px',
          borderRadius: radius.control,
        }}
      >
        ⋯
      </button>
      {open && (
        <span
          role="menu"
          style={{
            position: 'absolute',
            top: '100%',
            right: 0,
            zIndex: 10,
            minWidth: 148,
            background: color.surface,
            border: `1px solid ${color.borderControl}`,
            borderRadius: radius.control,
            boxShadow: '0 4px 14px rgba(17,28,46,.12)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {items.map((item) => (
            <button
              key={item.label}
              type="button"
              role="menuitem"
              onClick={(event) => {
                event.stopPropagation();
                setOpen(false);
                item.onSelect();
              }}
              style={{
                border: 0,
                background: color.surface,
                textAlign: 'left',
                padding: '8px 11px',
                font: `500 12px/1.4 ${font.sans}`,
                color: item.danger ? color.red : color.textStrong,
              }}
            >
              {item.label}
            </button>
          ))}
        </span>
      )}
    </span>
  );
}

function JobRow({
  job,
  onOpen,
  onDelete,
}: {
  job: JobRecord;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const best = job.summary?.best_validation_loss;
  const map = job.evaluation?.summary?.metrics?.mAP;
  const spec = specLine(job);
  const cells = [
    typeof map === 'number' ? map.toFixed(4) : '-',
    typeof best === 'number' ? loss(best) : '-',
    duration(job.elapsed_seconds),
    startedAt(job.started_at ?? job.created_at),
  ];

  return (
    <div
      role="button"
      tabIndex={0}
      // global.css가 이 표시를 보고 마우스를 올렸을 때 배경을 바꿉니다.
      data-row-hover=""
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') onOpen();
      }}
      style={{
        display: 'grid',
        gridTemplateColumns: COLUMNS,
        alignItems: 'center',
        borderBottom: `1px solid ${color.borderInner}`,
        cursor: 'pointer',
      }}
    >
      <span style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 7, minWidth: 0 }}>
          <span
            style={{
              font: `600 12.5px/1.3 ${font.sans}`,
              color: color.text,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
            title={job.run_id}
          >
            {job.run_id}
          </span>
          <StatusBadge status={job.status} label={job.status_label} />
        </span>
        {/* 예전에는 이 자리에 job_id 앞 8자가 있었습니다. 무엇으로 돌린 학습인지를 둡니다. */}
        {spec !== '' && (
          <span
            style={{
              font: `400 11px/1.35 ${font.mono}`,
              color: color.textFaint,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {spec}
          </span>
        )}
      </span>
      <StagePips job={job} />
      {cells.map((value, index) => (
        <span
          key={index}
          style={{ padding: '8px 12px', font: `400 12px/1.3 ${font.mono}`, color: color.textStrong }}
        >
          {value}
        </span>
      ))}
      <RowMenu
        label={`${job.run_id} 학습 메뉴`}
        items={[
          { label: '자세히 보기', onSelect: onOpen },
          // 실행 중인 학습은 backend도 거절하지만, 누를 수 없는 편이 낫습니다.
          ...(job.status === 'running' || job.status === 'queued' || job.status === 'starting'
            ? []
            : [{ label: '기록 삭제', onSelect: onDelete, danger: true }]),
        ]}
      />
    </div>
  );
}

/**
 * 아직 시작하지 않은 학습들. 여러 설정을 줄 세워 놓고 자러 갈 때 씁니다.
 *
 * 앞 학습이 자연스럽게 끝나면 실패했더라도 다음으로 넘어갑니다. 하나가 OOM으로
 * 죽었다고 나머지가 안 돌면 밤을 통째로 버리기 때문입니다. 대신 사람이 중지를
 * 누르거나 서버가 다시 뜨면 멈춰 서서, 다시 돌릴지 사람이 정하게 합니다.
 */
function TrainingQueue() {
  const queue = usePolling<QueueState>(() => api.readQueue(), 3000);
  const team = useTeam();
  const [busy, setBusy] = useState(false);
  const entries = queue.data?.entries ?? [];
  const paused = Boolean(queue.data?.paused);

  if (entries.length === 0) return null;

  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    try {
      await action();
      queue.refresh();
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel
      title={`학습 대기열 (${entries.length})`}
      right={
        paused ? (
          <Button
            kind="primary"
            disabled={busy}
            // 서버가 다시 뜨면 저장해 둔 login token이 사라집니다. 다시 돌리는 사람의
            // token을 함께 보내지 않으면 남은 항목을 하나도 시작하지 못합니다.
            onClick={() => void act(async () => api.resumeQueue(await team.getAccessToken()))}
          >
            대기열 다시 돌리기
          </Button>
        ) : (
          <span style={{ font: `400 12px/1 ${font.sans}`, color: color.textMuted }}>
            앞 학습이 끝나면 위에서부터 차례로 시작합니다
          </span>
        )
      }
    >
      {paused && (
        <div style={{ marginBottom: 10 }}>
          <AlertRow level="warning" title="대기열이 멈춰 있습니다">
            중지했거나 서버가 다시 시작돼서 멈췄습니다. 눌러야 다음 학습이 시작됩니다.
          </AlertRow>
        </div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {entries.map((entry, index) => (
          <div
            key={entry.entry_id}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '7px 10px',
              border: `1px solid ${color.borderInner}`,
              borderRadius: 5,
            }}
          >
            <span style={{ font: `600 12px/1 ${font.mono}`, color: color.textMuted, width: 18 }}>
              {index + 1}
            </span>
            <span style={{ font: `600 12.5px/1 ${font.mono}`, color: color.text, flex: 1 }}>
              {entry.run_id || '(이름 없음)'}
            </span>
            <span style={{ font: `400 11.5px/1 ${font.sans}`, color: color.textMuted }}>
              {startedAt(entry.queued_at)} 추가
            </span>
            <Button
              disabled={busy}
              onClick={() => void act(() => api.removeFromQueue(entry.entry_id))}
            >
              빼기
            </Button>
          </div>
        ))}
      </div>
    </Panel>
  );
}
