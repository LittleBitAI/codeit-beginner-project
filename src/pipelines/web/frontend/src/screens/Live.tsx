/**
 * 학습 하나를 지켜보는 화면입니다.
 *
 * 맨 위에 지금 값 하나(검증 손실)를 크게 두고, 그 아래로 곡선 → 로그 → 대기열
 * 순으로 내려갑니다. 밤에 흘깃 보는 화면이라 "지금 어디까지 왔나"가 스크롤 없이
 * 읽혀야 합니다.
 */

import { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type { GpuStatus, JobListing, JobRecord, QueueState } from '../api/types';
import { EvaluatePanel } from '../components/EvaluatePanel';
import { ChartHead, LossChart } from '../components/LossChart';
import { LossBreakdown } from '../components/LossBreakdown';
import { LogStream } from '../components/LogStream';
import { LrChart } from '../components/LrChart';
import {
  AlertRow,
  Button,
  EmptyState,
  EstimatedValue,
  LinkAction,
  LiveDot,
  Metric,
  MetricGrid,
  MicroLabel,
  ProgressBar,
  StatusBadge,
} from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { useJobStream } from '../hooks/useJobStream';
import { usePolling } from '../hooks/usePolling';
import { duration, loss, megabytes, percent } from '../lib/format';
import { epochsDone, progressRatio } from '../lib/progress';
import { useTeam } from '../team/TeamContext';

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
    <>
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, background: 'rgba(8,6,4,.55)', zIndex: 65 }}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`${job.run_id} 기록 삭제`}
        onKeyDown={(event) => {
          if (event.key === 'Escape') onClose();
        }}
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 70,
          width: 'min(460px, calc(100vw - 40px))',
          background: color.sheet,
          border: `1px solid ${color.border}`,
          borderRadius: 4,
          padding: '20px 22px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}
      >
        <span style={{ ...type.subTitle, color: color.text, overflowWrap: 'anywhere' }}>
          {job.run_id} 기록을 지울까요?
        </span>
        <span style={{ ...type.bodySmall, color: color.textBody }}>
          <b style={{ color: color.danger }}>지웁니다</b> — 이 GUI가 들고 있는 실행 기록과 로그.
        </span>
        <span style={{ ...type.bodySmall, color: color.textBody }}>
          <b style={{ color: color.ok }}>그대로 둡니다</b> — 학습 결과 폴더와 checkpoint, registry에
          등록된 실험, 팀에 공유된 기록, 이 학습이 쓴 설정 파일.
        </span>
        <span style={{ ...type.note, color: color.textMuted }}>
          되돌릴 수 없습니다. 목록에서만 사라지고 학습 자체는 남습니다.
        </span>
        {error && (
          <AlertRow level="error" title="지우지 못했습니다">
            {error}
          </AlertRow>
        )}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <Button kind="ghost" onClick={onClose} disabled={busy}>
            취소
          </Button>
          <Button kind="danger" onClick={() => void remove()} disabled={busy}>
            {busy ? '지우는 중…' : '지웁니다'}
          </Button>
        </div>
      </div>
    </>
  );
}

export function Live({
  listing,
  onNewExperiment,
  onJobsChanged,
}: {
  listing: JobListing | null;
  onNewExperiment: () => void;
  /** 기록을 지운 뒤 목록을 다시 읽게 합니다. */
  onJobsChanged: () => void;
}) {
  const navigate = useNavigate();
  const team = useTeam();
  const params = useParams<{ jobId?: string }>();
  const fallback = listing?.active_job_id ?? listing?.jobs[0]?.job_id;
  const jobId = params.jobId ?? fallback;
  const { job, lines, error, streaming } = useJobStream(jobId);
  const gpu = usePolling<GpuStatus>(() => api.gpu(), 5000, Boolean(job));
  const queue = usePolling<QueueState>(() => api.readQueue(), 5000);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [resumed, setResumed] = useState<string | null>(null);
  // 지우기 전에 무엇이 사라지고 무엇이 남는지 보여 줍니다.
  const [confirmDelete, setConfirmDelete] = useState(false);

  if (!jobId) {
    return (
      <div style={{ padding: '36px 40px 60px' }}>
        <LinkAction onClick={() => navigate('/')}>← 목록</LinkAction>
        <div style={{ marginTop: 30 }}>
          <EmptyState
            message="아직 이 GUI로 시작한 학습이 없습니다. 새 실험으로 첫 학습을 걸어 보세요."
            action={
              <Button kind="primary" onClick={onNewExperiment}>
                새 실험
              </Button>
            }
          />
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div style={{ padding: '36px 40px 60px' }}>
        <LinkAction onClick={() => navigate('/')}>← 목록</LinkAction>
        <div style={{ marginTop: 30, ...type.body, color: color.textMuted }}>
          {error ?? '학습 정보를 불러오고 있습니다.'}
        </div>
      </div>
    );
  }

  const progress = job.progress;
  const active = job.status === 'running' || job.status === 'queued' || job.status === 'starting';
  const epochs = progress.epochs ?? [];
  const last = epochs.length > 0 ? epochs[epochs.length - 1] : undefined;
  const best = progress.best ?? null;
  const done = epochsDone(progress);
  const ratio = progressRatio(progress);
  const first = epochs.find((item) => item.validation_loss !== null)?.validation_loss ?? null;
  const delta = first !== null && best ? first - best.validation_loss : null;
  // 이어서 학습할 수 있는 상태입니다. 실패든 사람이 중지했든, 마친 epoch이 있으면
  // 그 지점의 checkpoint가 남아 있습니다. 없으면 이어갈 것이 없으므로 단추를 두지
  // 않습니다 — 눌러 봐야 서버가 같은 이유로 거절합니다.
  const completedEpochs = Math.max(progress.completed_epochs ?? 0, epochs.length);
  const resumable =
    (job.status === 'failed' || job.status === 'cancelled') && completedEpochs > 0;
  const resumeButton = resumable ? (
    <Button onClick={() => void resume()} disabled={resuming}>
      {resuming ? '시작하는 중…' : '이어서 학습'}
    </Button>
  ) : undefined;
  const device = gpu.data?.telemetry.devices[0] ?? null;
  const telemetryDown = gpu.data && gpu.data.telemetry.source !== 'nvidia-smi';
  const gpuLine = telemetryDown
    ? (gpu.data?.telemetry.message ?? 'GPU 정보 없음')
    : device
      ? `${device.name ?? 'GPU'} · ${percent(device.utilization_percent)} · ${megabytes(device.memory_used_mb)} / ${megabytes(device.memory_total_mb)}`
      : 'GPU 정보를 불러오고 있습니다';
  const entries = queue.data?.entries ?? [];

  async function cancel() {
    setCancelling(true);
    setCancelError(null);
    try {
      await api.cancelJob(job!.job_id);
    } catch (caught) {
      setCancelError(caught instanceof ApiError ? caught.message : '중지 요청이 실패했습니다.');
    } finally {
      setCancelling(false);
    }
  }

  async function resume() {
    setResuming(true);
    setResumeError(null);
    try {
      // epochs를 비워 두면 중단된 실행의 전체 목표를 그대로 이어갑니다.
      const result = await api.resumeJob(job!.job_id, undefined, await team.getAccessToken());
      setResumed(result.run_id);
      if (result.started) navigate(`/monitor/${result.started.job_id}`);
    } catch (caught) {
      setResumeError(
        caught instanceof ApiError ? caught.message : '이어서 학습을 시작하지 못했습니다.',
      );
    } finally {
      setResuming(false);
    }
  }

  return (
    <div style={{ padding: '36px 40px 60px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 20 }}>
        <LinkAction onClick={() => navigate('/')}>← 목록</LinkAction>
        <Button kind="primary" onClick={onNewExperiment} style={{ flex: 'none' }}>
          새 실험
        </Button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 11, margin: '24px 0 8px', flexWrap: 'wrap' }}>
        {active && <LiveDot size={8} pulse />}
        <span style={{ font: `600 20px/1.4 ${font.sans}`, color: color.textStrong }}>
          {String(job.settings?.architecture ?? progress.architecture ?? '학습')}
        </span>
        <StatusBadge status={job.status} label={job.status_label} />
      </div>
      <div style={{ font: `400 13.5px/1.7 ${font.mono}`, color: color.textMuted, overflowWrap: 'break-word' }}>
        {job.run_id}
      </div>
      <div style={{ ...type.monoSpec, color: color.textMuted, marginTop: 4 }}>
        {[
          `device ${String(job.settings?.device ?? '-')}`,
          `epochs ${String(job.settings?.epochs ?? '-')}`,
          `batch ${String(job.settings?.batch_size ?? '-')}`,
          `seed ${String(job.settings?.seed ?? '-')}`,
        ].join(' · ')}
      </div>

      <div
        style={{
          position: 'relative',
          background: color.panel,
          padding: '24px 26px',
          overflow: 'hidden',
          marginTop: 26,
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
        {/* 검증 손실과 남은 시간을 같은 크기로 나란히 둡니다 — 카드를 열자마자 사람이
            묻는 것이 "얼마나 잘 되나"와 "언제 끝나나" 둘입니다. */}
        <div
          style={{
            position: 'relative',
            display: 'flex',
            alignItems: 'flex-end',
            gap: '24px 56px',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div style={{ marginBottom: 12, color: color.textMid }}>
              <span style={{ font: `500 11.5px/1 ${font.mono}`, letterSpacing: '0.08em' }}>
                VAL LOSS
              </span>
              {/* 한글은 sans로 이어 붙입니다. mono는 낱자를 전각으로 벌려 놓습니다. */}
              {progress.available && progress.total_epochs ? (
                <span style={{ font: `500 11.5px/1 ${font.mono}`, letterSpacing: '0.08em' }}>
                  {` · EPOCH ${done} / ${progress.total_epochs}`}
                </span>
              ) : (
                <span style={{ font: `500 12px/1 ${font.sans}` }}> · 진행률 정보 없음</span>
              )}
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
              <span
                style={{ font: `600 52px/1 ${font.mono}`, letterSpacing: '-0.035em', color: color.textStrong }}
              >
                {loss(best?.validation_loss ?? last?.validation_loss)}
              </span>
              {delta !== null && delta > 0 && (
                <span style={{ font: `500 22px/1 ${font.mono}`, color: color.accent, whiteSpace: 'nowrap' }}>
                  ↓ {delta.toFixed(4)}
                </span>
              )}
            </div>
          </div>

          <div style={{ minWidth: 0 }}>
            <div style={{ font: `500 12px/1 ${font.sans}`, color: color.textMid, marginBottom: 12 }}>
              남은 시간
            </div>
            <div
              style={{
                font: `600 38px/1 ${font.mono}`,
                letterSpacing: '-0.03em',
                color: progress.eta_seconds === null || progress.finished ? color.textFaint : color.accent,
                whiteSpace: 'nowrap',
              }}
            >
              {progress.finished ? (
                progress.stopped_early ? (
                  '조기 종료'
                ) : (
                  '완료'
                )
              ) : progress.eta_seconds === null ? (
                '알 수 없음'
              ) : (
                <EstimatedValue>{duration(progress.eta_seconds)}</EstimatedValue>
              )}
            </div>
            <div style={{ ...type.bodySmall, color: color.textBody, marginTop: 10 }}>
              {duration(job.elapsed_seconds)} 경과
            </div>
          </div>

          <div style={{ ...type.monoSpec, color: color.textMuted, marginLeft: 'auto', overflowWrap: 'anywhere' }}>
            {gpuLine}
          </div>
        </div>

        {/* epoch 하나가 20분씩 걸리면 위 진행률만으로는 멈춘 것과 구별되지 않습니다.
            train이 batch 위치를 알려 준 실행에서만 이 줄을 그립니다. */}
        {progress.step && (
          <div style={{ position: 'relative', marginTop: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 6 }}>
              <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMid }}>
                {progress.step.phase === 'validation' ? '검증' : '학습'} batch {progress.step.step} /{' '}
                {progress.step.total_steps}
              </span>
              <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMid }}>
                {percent(progress.step.percent)}
              </span>
            </div>
            <ProgressBar ratio={progress.step.percent / 100} />
          </div>
        )}

        <MetricGrid
          style={{ position: 'relative', marginTop: 22, paddingTop: 18, borderTop: `1px solid ${color.fill}` }}
        >
          <Metric label="TRAIN LOSS" value={loss(last?.train_loss)} />
          <Metric label="BEST EPOCH" value={String(best?.epoch ?? job.summary?.best_epoch ?? '-')} />
          <Metric
            label="LEARNING RATE"
            value={typeof last?.learning_rate === 'number' ? last.learning_rate.toExponential(2) : '-'}
          />
          <Metric label="EPOCH 시간" value={duration(last?.epoch_seconds)} />
          <Metric label="온도" value={device?.temperature_c == null ? '-' : `${device.temperature_c}°C`} />
          <Metric label="로그 줄" value={job.log_lines.toLocaleString('ko-KR')} />
        </MetricGrid>
      </div>

      {!progress.available && (
        <div style={{ marginTop: 16, ...type.note, color: color.textMuted }}>
          {progress.message ??
            'train pipeline이 진행 로그를 제공하지 않아 몇 번째 epoch인지 알 수 없습니다.'}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 22 }}>
        {cancelError && (
          <AlertRow level="error" title="중지하지 못했습니다">
            {cancelError}
          </AlertRow>
        )}
        {job.status === 'failed' && (
          <AlertRow level="error" title="학습이 실패했습니다" action={resumeButton}>
            {job.message ?? '원인을 알 수 없습니다. 아래 로그를 확인해 주세요.'}
            {resumable &&
              ' 완료한 epoch가 있어 checkpoint가 저장소에 남아 있으면 새 실행 이름으로 이어갑니다.'}
            {resumed && ` '${resumed}' 이름으로 대기열에 넣었습니다.`}
            {resumeError && ` ${resumeError}`}
          </AlertRow>
        )}
        {job.status === 'cancelled' && (
          <AlertRow level="warning" title="학습을 중지했습니다" action={resumeButton}>
            {resumable
              ? `epoch ${completedEpochs}까지 마쳤습니다. 그 지점의 checkpoint가 남아 있으면 새 실행 이름으로 이어갑니다 — 남은 epoch이 아니라 원래 계획한 전체 epoch까지 돕니다.`
              : '마친 epoch이 없어 저장된 checkpoint가 없습니다. 이어서 학습할 수 없고 처음부터 다시 돌려야 합니다.'}
            {job.orphan_note &&
              ` ${job.orphan_note} 정리는 train pipeline이 소유한 영역이라 이 화면에서 지우지 않습니다.`}
            {resumed && ` '${resumed}' 이름으로 대기열에 넣었습니다.`}
            {resumeError && ` ${resumeError}`}
          </AlertRow>
        )}
        {job.status === 'interrupted' && (
          <AlertRow
            level="warning"
            title="서버가 다시 시작되어 상태를 잃었습니다"
            action={
              <Button onClick={() => void resume()} disabled={resuming}>
                {resuming ? '시작하는 중…' : '이어서 학습'}
              </Button>
            }
          >
            {job.message ?? '이 학습의 실제 결과는 알 수 없습니다.'} epoch마다 저장한 checkpoint가
            있으면 그 지점부터 이어서 학습합니다. 결과가 섞이지 않도록 새 실행 이름으로 시작하고,
            남은 epoch이 아니라 원래 계획한 전체 epoch까지 돕니다.
            {resumed && ` '${resumed}' 이름으로 대기열에 넣었습니다.`}
            {resumeError && ` ${resumeError}`}
          </AlertRow>
        )}
      </div>

      <div style={{ margin: '34px 0 0' }}>
        <ChartHead label="LOSS" right="실선 val · 점선 train" />
        <LossChart epochs={epochs} totalEpochs={progress.total_epochs} height={230} />
      </div>

      {/* schedule을 쓴 학습에서만 뜻이 있습니다. 값이 없으면 그렇다고 말합니다. */}
      <div style={{ marginTop: 30 }}>
        <ChartHead label="LEARNING RATE" />
        <LrChart epochs={epochs} totalEpochs={progress.total_epochs} />
      </div>

      <div style={{ marginTop: 30 }}>
        <LossBreakdown epochs={epochs} />
      </div>

      {/* 평가는 학습이 성공으로 끝난 뒤에만 할 수 있습니다. checkpoint가 있어야 합니다.
          key를 주어 학습을 바꾸면 이전 학습의 평가 상태가 남지 않게 합니다. */}
      {job.status === 'succeeded' && (
        <div style={{ marginTop: 30 }}>
          <EvaluatePanel key={job.job_id} job={job} />
        </div>
      )}

      {job.status === 'succeeded' && Object.keys(job.artifacts).length > 0 && (
        <div style={{ marginTop: 34, paddingTop: 24, borderTop: `1px solid ${color.border}` }}>
          <MicroLabel style={{ marginBottom: 16 }}>결과 파일</MicroLabel>
          {Object.entries(job.artifacts).map(([key, value]) => (
            <div
              key={key}
              style={{
                display: 'flex',
                gap: 16,
                flexWrap: 'wrap',
                padding: '10px 0',
                borderTop: `1px solid ${color.borderRow}`,
              }}
            >
              <span style={{ ...type.monoSpec, color: color.textMuted, minWidth: 170 }}>{key}</span>
              <span style={{ ...type.monoSpec, color: color.text, overflowWrap: 'anywhere', flex: 1 }}>
                {value}
              </span>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 36, paddingTop: 24, borderTop: `1px solid ${color.border}` }}>
        <MicroLabel style={{ marginBottom: 14 }}>실행 로그</MicroLabel>
        <LogStream lines={lines} streaming={streaming} />
      </div>

      <div style={{ marginTop: 34, paddingTop: 24, borderTop: `1px solid ${color.border}` }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            justifyContent: 'space-between',
            gap: 20,
            marginBottom: 16,
            flexWrap: 'wrap',
          }}
        >
          <span style={{ font: `600 14px/1.4 ${font.sans}`, color: color.text }}>
            대기열 <span style={{ fontWeight: 400, color: color.textMuted }}>{entries.length}</span>
          </span>
          <span style={{ ...type.bodySmall, color: color.textMuted }}>
            {queue.data?.paused
              ? '멈춰 있습니다. 목록 화면에서 다시 돌릴 수 있습니다.'
              : '이 학습이 끝나면 위에서부터 차례로 시작합니다'}
          </span>
        </div>
        {entries.map((entry, index) => (
          <div
            key={entry.entry_id}
            style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              gap: 16,
              padding: '13px 0',
              borderTop: `1px solid ${color.borderRow}`,
            }}
          >
            <span style={{ display: 'flex', alignItems: 'baseline', gap: 12, minWidth: 0 }}>
              <span style={{ font: `500 12.5px/1.5 ${font.mono}`, color: color.textFaint, flex: 'none' }}>
                {index + 1}
              </span>
              <span style={{ ...type.monoId, color: color.textBody, overflowWrap: 'break-word' }}>
                {entry.run_id || '(이름 없음)'}
              </span>
            </span>
            <LinkAction
              tone="muted"
              onClick={() => void api.removeFromQueue(entry.entry_id).then(() => queue.refresh())}
              style={{ flex: 'none', borderBottom: `1px solid ${color.border}` }}
            >
              빼기
            </LinkAction>
          </div>
        ))}

        <div style={{ marginTop: 22, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Button kind="secondary" disabled={!active || cancelling} onClick={() => void cancel()}>
            {cancelling ? '중지 요청 중…' : '학습 중지'}
          </Button>
          {/* 도는 학습은 backend도 거절하지만, 누를 수 없는 편이 낫습니다. */}
          <Button kind="danger" disabled={active} onClick={() => setConfirmDelete(true)}>
            기록 지우기
          </Button>
        </div>
      </div>

      {confirmDelete && (
        <DeleteRecordDialog
          job={job}
          onClose={() => setConfirmDelete(false)}
          onDeleted={() => {
            setConfirmDelete(false);
            onJobsChanged();
            navigate('/');
          }}
        />
      )}
    </div>
  );
}
