import { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type { GpuStatus, JobListing } from '../api/types';
import { ChartLegend, LossChart } from '../components/LossChart';
import { LrChart } from '../components/LrChart';
import { LossBreakdown } from '../components/LossBreakdown';
import { EvaluatePanel } from '../components/EvaluatePanel';
import { LogStream } from '../components/LogStream';
import {
  AlertRow,
  Button,
  EstimatedValue,
  KpiCard,
  Panel,
  ProgressBar,
  ScreenIntro,
  StatusBadge,
} from '../components/primitives';
import { color, font } from '../design/tokens';
import { useJobStream } from '../hooks/useJobStream';
import { usePolling } from '../hooks/usePolling';
import { duration, loss, megabytes, percent } from '../lib/format';
import { useTeam } from '../team/TeamContext';

export function LiveMonitor({ listing }: { listing: JobListing | null }) {
  const navigate = useNavigate();
  const team = useTeam();
  const params = useParams<{ jobId?: string }>();
  const fallback = listing?.active_job_id ?? listing?.jobs[0]?.job_id;
  const jobId = params.jobId ?? fallback;
  const { job, lines, error, streaming } = useJobStream(jobId);
  const gpu = usePolling<GpuStatus>(() => api.gpu(), 5000, Boolean(job));
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
  const [resumed, setResumed] = useState<string | null>(null);

  if (!jobId) {
    return (
      <div style={{ maxWidth: 900 }}>
        <ScreenIntro title="아직 실행한 학습이 없습니다">
          새 실험 화면에서 설정을 만들고 저장한 뒤 학습을 시작하면 여기에서 진행 상황을 볼 수 있습니다.
        </ScreenIntro>
        <Button kind="primary" onClick={() => navigate('/new')}>
          새 실험 만들기
        </Button>
      </div>
    );
  }

  if (!job) {
    return <Panel>{error ?? '학습 정보를 불러오는 중입니다.'}</Panel>;
  }

  const progress = job.progress;
  const active = job.status === 'running' || job.status === 'queued';
  const epochs = progress.epochs ?? [];
  const last = epochs.length > 0 ? epochs[epochs.length - 1] : undefined;
  const device = gpu.data?.telemetry.devices[0] ?? null;
  const telemetryDown = gpu.data && gpu.data.telemetry.source !== 'nvidia-smi';

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
      if (result.started) {
        navigate(`/monitor/${result.started.job_id}`);
      }
    } catch (caught) {
      setResumeError(
        caught instanceof ApiError ? caught.message : '이어서 학습을 시작하지 못했습니다.',
      );
    } finally {
      setResuming(false);
    }
  }

  return (
    <div style={{ maxWidth: 1320, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Panel bodyStyle={{ padding: '14px 18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            {active && (
              <span
                className="pulse-dot"
                style={{ width: 7, height: 7, borderRadius: '50%', background: color.teal }}
              />
            )}
            <span style={{ font: `700 15px/1.2 ${font.mono}`, color: color.text }}>{job.run_id}</span>
            <StatusBadge status={job.status} label={job.status_label} />
          </div>

          <div style={{ flex: '1 1 300px', minWidth: 240, display: 'flex', flexDirection: 'column', gap: 5 }}>
            {progress.available && progress.total_epochs ? (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <span style={{ font: `600 12.5px/1 ${font.mono}`, color: color.text }}>
                    epoch {progress.current_epoch ?? 0} / {progress.total_epochs}
                  </span>
                  <span style={{ font: `400 12px/1 ${font.sans}`, color: color.textMuted }}>
                    {progress.finished ? (
                      // 끝난 학습에 남은 시간을 말하면 아직 도는 것처럼 읽힙니다.
                      progress.stopped_early ? '조기 종료로 끝남' : '학습 완료'
                    ) : progress.eta_seconds === null ? (
                      '남은 시간을 추정할 수 없습니다'
                    ) : (
                      <>
                        남은 시간 <EstimatedValue>{duration(progress.eta_seconds)}</EstimatedValue> · 추정
                      </>
                    )}
                  </span>
                </div>
                {/* 조기 종료로 계획 epoch가 남아도 끝난 학습은 다 채웁니다. */}
                <ProgressBar
                  ratio={
                    progress.percent !== null && progress.percent !== undefined
                      ? progress.percent / 100
                      : null
                  }
                />
                {/* epoch 하나가 20분씩 걸리면 위 막대만으로는 멈춘 것과 구별되지 않습니다.
                    train이 batch 위치를 알려 준 실행에서만 두 번째 막대를 그립니다. */}
                {progress.step && (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                      <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMuted }}>
                        {progress.step.phase === 'validation' ? '검증' : '학습'} batch{' '}
                        {progress.step.step} / {progress.step.total_steps}
                      </span>
                      <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMuted }}>
                        {percent(progress.step.percent)}
                      </span>
                    </div>
                    <ProgressBar ratio={progress.step.percent / 100} tint={color.tealDark} />
                  </>
                )}
              </>
            ) : (
              <>
                <span style={{ font: `600 12.5px/1 ${font.mono}`, color: color.textMuted }}>
                  진행률 정보 없음
                </span>
                <ProgressBar ratio={null} />
                <span style={{ font: `400 11.5px/1.5 ${font.sans}`, color: color.textMuted }}>
                  {progress.message ??
                    'train pipeline이 진행 로그를 제공하지 않아 몇 번째 epoch인지 알 수 없습니다.'}
                </span>
              </>
            )}
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            <Button kind="danger" disabled={!active || cancelling} onClick={() => void cancel()}>
              {cancelling ? '중지 요청 중…' : '중지'}
            </Button>
            <Button onClick={() => navigate('/')}>학습 개요</Button>
          </div>
        </div>

        <div
          style={{
            marginTop: 10,
            font: `400 12px/1.5 ${font.mono}`,
            color: color.textMuted,
            display: 'flex',
            gap: 14,
            flexWrap: 'wrap',
          }}
        >
          <span>device {String(job.settings?.device ?? '-')}</span>
          <span>epochs {String(job.settings?.epochs ?? '-')}</span>
          <span>batch {String(job.settings?.batch_size ?? '-')}</span>
          <span>seed {String(job.settings?.seed ?? '-')}</span>
          <span>경과 {duration(job.elapsed_seconds)}</span>
        </div>
      </Panel>

      {cancelError && (
        <AlertRow level="error" title="중지하지 못했습니다">
          {cancelError}
        </AlertRow>
      )}

      {job.status === 'failed' && (
        <AlertRow level="error" title="학습이 실패했습니다">
          {job.message ?? '원인을 알 수 없습니다. 아래 로그를 확인해 주세요.'}
        </AlertRow>
      )}
      {job.status === 'cancelled' && job.orphan_note && (
        <AlertRow level="warning" title="중지된 학습의 임시 파일이 남아 있을 수 있습니다">
          {job.orphan_note} 정리는 train pipeline이 소유한 영역이라 이 화면에서 지우지 않습니다.
        </AlertRow>
      )}
      {job.status === 'interrupted' && (
        <AlertRow
          level="warning"
          title="서버가 다시 시작되어 상태를 잃었습니다"
          action={
            <Button onClick={resume} disabled={resuming}>
              {resuming ? '시작하는 중…' : '이어서 학습'}
            </Button>
          }
        >
          {job.message ?? '이 학습의 실제 결과는 알 수 없습니다.'} epoch마다 저장한
          checkpoint가 있으면 그 지점부터 이어서 학습합니다. 결과가 섞이지 않도록 새 실행
          이름으로 시작하고, 남은 epoch이 아니라 원래 계획한 전체 epoch까지 돕니다.
          {resumed && ` '${resumed}' 이름으로 대기열에 넣었습니다.`}
          {resumeError && ` ${resumeError}`}
        </AlertRow>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(126px, 1fr))',
          gap: 8,
        }}
      >
        <KpiCard label="TRAIN LOSS" value={loss(last?.train_loss)} compact />
        <KpiCard label="VAL LOSS" value={loss(last?.validation_loss)} compact />
        <KpiCard
          label="BEST VAL LOSS"
          value={loss(progress.best?.validation_loss ?? (job.summary?.best_validation_loss as number | undefined))}
          compact
          valueColor={color.tealDark}
        />
        <KpiCard
          label="BEST EPOCH"
          value={String(progress.best?.epoch ?? job.summary?.best_epoch ?? '-')}
          compact
        />
        <KpiCard
          label="LEARNING RATE"
          value={
            typeof last?.learning_rate === 'number'
              ? last.learning_rate.toExponential(2)
              : '-'
          }
          compact
        />
        <KpiCard label="EPOCH TIME" value={duration(last?.epoch_seconds)} compact />
        <KpiCard label="경과" value={duration(job.elapsed_seconds)} compact />
      </div>

      {/* 평가는 학습이 성공으로 끝난 뒤에만 할 수 있습니다. checkpoint가 있어야 합니다. */}
      {/* key를 주어 학습을 바꾸면 이전 학습의 평가 상태가 남지 않게 합니다. */}
      {job.status === 'succeeded' && <EvaluatePanel key={job.job_id} job={job} />}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-start' }}>
        <div style={{ flex: '3 1 380px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Panel title="손실 곡선" bodyStyle={{ padding: '12px 16px 0' }}>
            <LossChart
              epochs={epochs}
              totalEpochs={progress.total_epochs}
              currentEpoch={progress.current_epoch}
            />
            <ChartLegend />
          </Panel>
          {/* schedule을 쓴 학습에서만 의미가 있습니다. 값이 없으면 그렇다고 말합니다. */}
          <Panel title="Learning rate" bodyStyle={{ padding: '12px 16px 0' }}>
            <LrChart epochs={epochs} totalEpochs={progress.total_epochs} />
          </Panel>
          <LossBreakdown epochs={epochs} />

          {job.status === 'succeeded' && (
            <Panel title="학습 결과">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: 10,
                  }}
                >
                  {Object.entries(job.summary).map(([key, value]) => (
                    <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      <span style={{ font: `500 11px/1.3 ${font.mono}`, color: color.textMuted }}>
                        {key}
                      </span>
                      <span style={{ font: `600 12.5px/1.2 ${font.mono}`, color: color.text }}>
                        {typeof value === 'number' ? value.toString() : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
                <div
                  style={{
                    borderTop: `1px solid ${color.borderInner}`,
                    paddingTop: 10,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                  }}
                >
                  {Object.entries(job.artifacts).map(([key, value]) => (
                    <div key={key} style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                      <span
                        style={{
                          font: `500 11.5px/1.5 ${font.mono}`,
                          color: color.textMuted,
                          minWidth: 160,
                        }}
                      >
                        {key}
                      </span>
                      <span
                        style={{
                          font: `400 12px/1.5 ${font.mono}`,
                          color: color.textStrong,
                          overflowWrap: 'anywhere',
                        }}
                      >
                        {value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
          )}
        </div>

        <div style={{ flex: '2 1 286px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Panel title="GPU">
            {telemetryDown ? (
              <span style={{ font: `400 12.5px/1.6 ${font.sans}`, color: color.textBody }}>
                {gpu.data?.telemetry.message ?? 'GPU 사용률 정보를 가져올 수 없습니다.'}
              </span>
            ) : device ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <span style={{ font: `600 12.5px/1.2 ${font.mono}`, color: color.text }}>
                  {device.name}
                </span>
                <LabeledBar
                  label="사용률"
                  text={percent(device.utilization_percent)}
                  ratio={device.utilization_percent === null ? null : device.utilization_percent / 100}
                />
                <LabeledBar
                  label="메모리"
                  text={`${megabytes(device.memory_used_mb)} / ${megabytes(device.memory_total_mb)}`}
                  ratio={
                    device.memory_total_mb && device.memory_used_mb !== null
                      ? device.memory_used_mb / device.memory_total_mb
                      : null
                  }
                />
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ font: `400 11px/1 ${font.mono}`, color: color.textMuted }}>온도</span>
                  <span style={{ font: `600 12.5px/1 ${font.mono}`, color: color.text }}>
                    {device.temperature_c === null ? '-' : `${device.temperature_c}°C`}
                  </span>
                </div>
              </div>
            ) : (
              <span style={{ font: `400 12.5px/1.6 ${font.sans}`, color: color.textBody }}>
                GPU 정보를 불러오는 중입니다.
              </span>
            )}
          </Panel>

          <Panel title="실행 로그" bodyStyle={{ padding: 0 }}>
            <LogStream lines={lines} streaming={streaming} />
          </Panel>
        </div>
      </div>
    </div>
  );
}

function LabeledBar({
  label,
  text,
  ratio,
}: {
  label: string;
  text: string;
  ratio: number | null;
}) {
  const tint = ratio === null ? color.textFaint : ratio > 0.9 ? color.red : ratio > 0.75 ? color.amber : color.teal;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ font: `400 11px/1 ${font.mono}`, color: color.textMuted }}>{label}</span>
        <span style={{ font: `600 12px/1 ${font.mono}`, color: color.text }}>{text}</span>
      </div>
      <ProgressBar ratio={ratio} tint={tint} />
    </div>
  );
}
