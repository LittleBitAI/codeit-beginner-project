import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api } from '../api/client';
import type { DataSource, GpuStatus, JobListing, JobRecord, JobStatus } from '../api/types';
import { DataSourcePanel } from '../components/DataSourcePanel';
import {
  AlertRow,
  Button,
  Chip,
  EmptyState,
  KpiCard,
  Panel,
  ScreenIntro,
  StatusBadge,
} from '../components/primitives';
import { color, font } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { duration, loss, megabytes, percent, startedAt } from '../lib/format';

const FILTERS: { key: string; label: string; match: (status: JobStatus) => boolean }[] = [
  { key: 'all', label: '전체', match: () => true },
  { key: 'running', label: '실행 중', match: (status) => status === 'running' || status === 'queued' },
  { key: 'succeeded', label: '성공', match: (status) => status === 'succeeded' },
  { key: 'failed', label: '실패', match: (status) => status === 'failed' },
  { key: 'cancelled', label: '취소·중단', match: (status) => status === 'cancelled' || status === 'interrupted' },
];

const COLUMNS = '1.5fr .7fr .6fr .9fr .95fr .7fr .9fr';

export function TrainingOverview({
  listing,
  source,
  onSourceSelected,
}: {
  listing: JobListing | null;
  source: DataSource | null;
  onSourceSelected: (source: DataSource) => void;
}) {
  const navigate = useNavigate();
  const [filter, setFilter] = useState('all');
  const gpu = usePolling<GpuStatus>(() => api.gpu(), 5000);

  const jobs = listing?.jobs ?? [];
  const active = jobs.find((job) => job.job_id === listing?.active_job_id) ?? null;
  const visible = useMemo(() => {
    const rule = FILTERS.find((item) => item.key === filter) ?? FILTERS[0]!;
    return jobs.filter((job) => rule.match(job.status));
  }, [jobs, filter]);

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
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1560 }}>
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

      <DataSourcePanel source={source} onSelected={onSourceSelected} />

      <Panel
        title="학습 실행"
        right={
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            {FILTERS.map((item) => (
              <Chip key={item.key} active={filter === item.key} onClick={() => setFilter(item.key)}>
                {item.label}
              </Chip>
            ))}
            <span style={{ font: `400 11px/1 ${font.mono}`, color: color.textMuted, marginLeft: 4 }}>
              {visible.length}건
            </span>
          </div>
        }
        bodyStyle={{ padding: 0 }}
      >
        {visible.length === 0 ? (
          <EmptyState
            message="이 조건에 맞는 학습이 0건입니다."
            action={
              filter === 'all' ? (
                <Button kind="primary" onClick={() => navigate('/new')}>
                  새 실험 만들기
                </Button>
              ) : (
                <Button onClick={() => setFilter('all')}>필터 초기화</Button>
              )
            }
          />
        ) : (
          <>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: COLUMNS,
                background: color.surfaceTableHead,
                borderBottom: `1px solid ${color.border}`,
              }}
            >
              {['실행 이름', 'DEVICE', 'EPOCHS', '상태', 'BEST VAL LOSS', '경과', '시작'].map(
                (heading) => (
                  <span
                    key={heading}
                    style={{
                      font: `600 10px/1.3 ${font.mono}`,
                      letterSpacing: '.04em',
                      color: '#66707E',
                      padding: '9px 12px',
                    }}
                  >
                    {heading}
                  </span>
                ),
              )}
            </div>
            {visible.map((job) => (
              <JobRow key={job.job_id} job={job} onOpen={() => navigate(`/monitor/${job.job_id}`)} />
            ))}
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
    </div>
  );
}

function JobRow({ job, onOpen }: { job: JobRecord; onOpen: () => void }) {
  const best = job.summary?.best_validation_loss;
  const cells: (string | number)[] = [
    '',
    String(job.settings?.device ?? '-'),
    String(job.settings?.epochs ?? '-'),
    '',
    typeof best === 'number' ? loss(best) : '-',
    duration(job.elapsed_seconds),
    startedAt(job.started_at ?? job.created_at),
  ];

  return (
    <div
      role="button"
      tabIndex={0}
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
      <span style={{ padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: 1 }}>
        <span style={{ font: `600 12px/1.3 ${font.sans}`, color: color.text }}>{job.run_id}</span>
        <span style={{ font: `400 10px/1.35 ${font.mono}`, color: color.textFaint }}>
          {job.job_id.slice(0, 8)}
        </span>
      </span>
      {cells.slice(1, 3).map((value, index) => (
        <span
          key={index}
          style={{ padding: '8px 12px', font: `400 11px/1.3 ${font.mono}`, color: color.textStrong }}
        >
          {value}
        </span>
      ))}
      <span style={{ padding: '8px 12px' }}>
        <StatusBadge status={job.status} label={job.status_label} />
      </span>
      {cells.slice(4).map((value, index) => (
        <span
          key={index}
          style={{ padding: '8px 12px', font: `400 11px/1.3 ${font.mono}`, color: color.textStrong }}
        >
          {value}
        </span>
      ))}
    </div>
  );
}
