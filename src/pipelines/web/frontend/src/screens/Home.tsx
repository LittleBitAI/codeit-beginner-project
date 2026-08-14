/**
 * 첫 화면. **내 학습이 지금 어떤 상태인가**만 답합니다.
 *
 * 위에서부터 지금 도는 학습 한 장 → 대기열 순입니다. 이 화면을 열었을 때 사람이
 * 묻는 것이 "지금 뭐가 돌고 있지"와 "다음은 뭐지" 둘이기 때문입니다. 팀원 것은
 * 현황판이, 지난 기록은 기록 화면이 맡습니다 — 한 화면이 셋을 다 하면 무엇을 보러
 * 왔든 나머지 둘을 지나쳐야 합니다.
 */

import { useNavigate } from 'react-router-dom';

import type { JobRecord, QueueState } from '../api/types';
import {
  AlertRow,
  Button,
  EmptyState,
  LinkAction,
  LiveDot,
  Metric,
  MetricGrid,
} from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { duration, loss, startedAt } from '../lib/format';
import { epochsDone, progressRatio } from '../lib/progress';

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

/**
 * 지금 도는 것을 맨 위에 두고 그 아래로 줄을 세웁니다.
 *
 * 대기열은 "다음에 무엇이 도는가"의 목록이고, 지금 도는 학습이 그 0번입니다.
 * 따로 떼어 놓으면 순서를 머릿속에서 이어 붙여야 합니다. 추가 버튼은 언제나 줄의
 * 맨 끝에 있습니다 — 새로 넣는 것이 실제로 들어가는 자리가 거기라서입니다.
 */
function Queue({
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

export function Home({
  liveJob,
  queue,
  error,
  onNewExperiment,
  onRemoveFromQueue,
  onResumeQueue,
  onCancelJob,
}: {
  /** 이 컴퓨터가 지금 돌리고 있는 학습. */
  liveJob: JobRecord | null;
  queue: QueueState | null;
  error: string | null;
  onNewExperiment: () => void;
  onRemoveFromQueue: (entryId: string) => void;
  onResumeQueue: () => void;
  onCancelJob: (jobId: string) => void;
}) {
  const navigate = useNavigate();

  return (
    <div style={{ padding: '36px 40px 60px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 28,
          marginBottom: 8,
        }}
      >
        <h1 style={{ ...type.pageTitle, margin: 0, color: color.textStrong, minWidth: 0 }}>
          내 학습
        </h1>
        <Button kind="primary" onClick={onNewExperiment} style={{ flex: 'none' }}>
          새 실험
        </Button>
      </div>
      <div style={{ ...type.body, color: color.textBody, marginBottom: 30 }}>
        이 컴퓨터에서 도는 학습과 다음 차례입니다. 팀원 것은 현황판에, 지난 기록은 기록에
        있습니다.
      </div>

      {error && (
        <div style={{ marginBottom: 22 }}>
          <AlertRow level="error" title="backend에 연결하지 못했습니다">
            {error} 서버를 실행하려면 저장소 root에서{' '}
            <code style={{ fontFamily: font.mono }}>python -m src.pipelines.web.server</code>를
            실행하세요.
          </AlertRow>
        </div>
      )}

      {liveJob ? (
        <div style={{ marginBottom: 40 }}>
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
      ) : (
        <div style={{ marginBottom: 40 }}>
          <EmptyState
            message="이 컴퓨터에서 도는 학습이 없습니다. dataset을 고른 뒤 새 실험으로 시작하세요."
            action={
              <Button kind="primary" onClick={onNewExperiment}>
                새 실험
              </Button>
            }
          />
        </div>
      )}

      <div style={{ ...type.sectionTitle, color: color.text, marginBottom: 16 }}>학습 대기열</div>
      <Queue
        liveJob={liveJob}
        queue={queue}
        onNewExperiment={onNewExperiment}
        onRemoveFromQueue={onRemoveFromQueue}
        onResumeQueue={onResumeQueue}
        onCancelJob={onCancelJob}
        onOpenMonitor={(jobId) => navigate(`/monitor/${jobId}`)}
      />
    </div>
  );
}
