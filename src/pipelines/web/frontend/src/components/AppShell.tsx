/**
 * 화면 전체의 틀입니다.
 *
 * 왼쪽은 **할 일의 차례**입니다. 학습 한 번은 `dataset 준비 → EDA → 새 실험 →
 * 기록`으로 흐르므로 메뉴도 그 순서로 세웁니다. 예전에는 여기에 dataset 목록이
 * 있었는데, 그것은 "무엇을 볼지" 고르는 값이지 화면이 아니라 학습에 쓸 데이터를
 * 바꾸는 것으로 읽혔습니다. 지금 그 고르기는 기록 화면 안에 있고, 학습에 실제로
 * 쓰이는 데이터는 dataset 준비에서만 바뀝니다.
 *
 * 맨 위 제목이 첫 화면(내 학습 현황)으로 가는 길입니다.
 */

import type { ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import { color, font, type } from '../design/tokens';
import type { GpuStatus } from '../api/types';
import { LiveDot } from './primitives';
import { ThemeToggle } from './ThemeToggle';

/** 색은 `<svg>`의 `color`에 한 번 얹고 안쪽은 `currentColor`로 물려받습니다. */
const iconBox = { flex: 'none' as const };

function IconDataset() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={iconBox}>
      <path
        d="M1.5 4.2c0-1 2.9-1.9 6.5-1.9s6.5.9 6.5 1.9-2.9 1.9-6.5 1.9S1.5 5.2 1.5 4.2Z"
        stroke="currentColor"
        strokeWidth="1.3"
      />
      <path
        d="M1.5 4.2V8c0 1 2.9 1.9 6.5 1.9S14.5 9 14.5 8V4.2"
        stroke="currentColor"
        strokeWidth="1.3"
      />
      <path
        d="M1.5 8v3.8c0 1 2.9 1.9 6.5 1.9s6.5-.9 6.5-1.9V8"
        stroke="currentColor"
        strokeWidth="1.3"
      />
    </svg>
  );
}

function IconSettings() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={iconBox}>
      <path d="M3.2 3.4h9.6M3.2 8h9.6M3.2 12.6h9.6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
      <circle cx="6" cy="3.4" r="1.7" style={{ fill: color.rail }} stroke="currentColor" strokeWidth="1.3" />
      <circle cx="10.4" cy="8" r="1.7" style={{ fill: color.rail }} stroke="currentColor" strokeWidth="1.3" />
      <circle cx="5.4" cy="12.6" r="1.7" style={{ fill: color.rail }} stroke="currentColor" strokeWidth="1.3" />
    </svg>
  );
}

function IconEda() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={iconBox}>
      <path d="M2 13.5V9M6 13.5V4M10 13.5V6.5M14 13.5V2" stroke="currentColor" strokeWidth="1.3" />
    </svg>
  );
}

function IconRecords() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={iconBox}>
      <path
        d="M2.5 4h11M2.5 8h11M2.5 12h7"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconBoard() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" style={iconBox}>
      <circle cx="5.2" cy="5" r="2.1" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="11.2" cy="5" r="2.1" stroke="currentColor" strokeWidth="1.3" />
      <path
        d="M1.8 13.2c0-1.7 1.5-3 3.4-3s3.4 1.3 3.4 3M8.4 13.2c0-1.7 1.5-3 3.4-3 1.2 0 2.3.5 2.9 1.3"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** 왼쪽 메뉴 한 줄. 화면으로 가거나 오른쪽 시트를 엽니다. */
function RailItem({
  icon,
  active,
  children,
  right,
  onClick,
}: {
  icon: ReactNode;
  active?: boolean;
  children: ReactNode;
  right?: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-current={active ? 'page' : undefined}
      data-row-hover={active ? undefined : ''}
      onClick={onClick}
      style={{
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        width: '100%',
        padding: '10px 20px',
        textAlign: 'left',
        border: 0,
        background: active ? color.fill : 'transparent',
        color: active ? color.text : color.textMuted,
        font: `${active ? 500 : 400} 12.5px/1.4 ${font.sans}`,
      }}
    >
      {active && (
        <span
          style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 2, background: color.accent }}
        />
      )}
      {icon}
      <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{children}</span>
      {right && <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center' }}>{right}</span>}
    </button>
  );
}

/**
 * GPU 한 장의 메모리와 사용률입니다.
 *
 * 값을 못 읽으면 0%를 그리지 않고 이유를 적습니다. 빈 막대는 "GPU가 놀고 있다"로
 * 읽히는데, 사실은 조회에 실패한 것이라 정반대의 말이 됩니다.
 */
function GpuGauge({ gpu }: { gpu: GpuStatus | null }) {
  const device = gpu?.telemetry.devices[0];
  const usedMb = device?.memory_used_mb ?? null;
  const totalMb = device?.memory_total_mb ?? null;
  const ratio = usedMb !== null && totalMb ? Math.min(1, usedMb / totalMb) : null;
  const name = device?.name ?? (gpu?.torch.cuda_available ? 'GPU' : null);
  const reason = gpu?.telemetry.reason ?? gpu?.torch.reason ?? null;

  return (
    <div style={{ marginTop: 22, padding: '20px 20px 0', borderTop: `1px solid ${color.border}` }}>
      <div style={{ ...type.note, color: color.textMuted, marginBottom: 12 }}>
        {name ?? 'GPU 없음'}
      </div>
      {ratio === null ? (
        <div style={{ font: `400 11.5px/1.6 ${font.mono}`, color: color.textFaint }}>
          {reason ?? '메모리 사용량을 읽지 못했습니다'}
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
            <span
              style={{
                font: `600 20px/1 ${font.mono}`,
                letterSpacing: '-0.02em',
                color: color.text,
                fontVariantNumeric: 'tabular-nums',
              }}
            >
              {((usedMb as number) / 1024).toFixed(1)}
            </span>
            <span style={{ font: `400 12.5px/1 ${font.mono}`, color: color.textMuted }}>
              / {((totalMb as number) / 1024).toFixed(1)} GB
            </span>
          </div>
          <div style={{ height: 6, background: color.border, borderRadius: 3, overflow: 'hidden', marginBottom: 10 }}>
            <div style={{ width: `${ratio * 100}%`, height: '100%', background: color.accent }} />
          </div>
          <div style={{ font: `400 11.5px/1 ${font.mono}`, color: color.textMuted }}>
            {device?.utilization_percent === null || device?.utilization_percent === undefined
              ? '사용률 -'
              : `사용률 ${device.utilization_percent}%`}
          </div>
        </>
      )}
    </div>
  );
}

export function AppShell({
  children,
  gpu,
  running,
  onOpenPrepare,
  onOpenEda,
  onOpenSettings,
}: {
  children: ReactNode;
  gpu: GpuStatus | null;
  /** 이 컴퓨터에서 지금 도는 학습이 있는지. 제목 옆에 점을 답니다. */
  running: boolean;
  onOpenPrepare: () => void;
  onOpenEda: () => void;
  onOpenSettings: () => void;
}) {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  return (
    <div
      style={{
        background: color.page,
        color: color.text,
        minHeight: '100vh',
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 232px) minmax(0, 1fr)',
      }}
    >
      <nav
        style={{
          background: color.rail,
          borderRight: `1px solid ${color.border}`,
          padding: '24px 0 20px',
          display: 'flex',
          flexDirection: 'column',
          position: 'sticky',
          top: 0,
          height: '100vh',
          overflowY: 'auto',
        }}
      >
        {/* 제목이 곧 첫 화면(내 학습 현황)으로 가는 길입니다. */}
        <button
          type="button"
          aria-current={pathname === '/' ? 'page' : undefined}
          onClick={() => navigate('/')}
          style={{
            padding: '0 20px 26px',
            textAlign: 'left',
            border: 0,
            background: 'transparent',
          }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ font: `600 14px/1.4 ${font.sans}`, color: color.text }}>알약 객체 탐지</span>
            {running && <LiveDot size={6} pulse />}
          </span>
          <span
            style={{
              display: 'block',
              font: `400 11.5px/1.5 ${font.mono}`,
              color: pathname === '/' ? color.accent : color.textMuted,
            }}
          >
            {running ? '학습 중' : 'Training'}
          </span>
        </button>

        <RailItem icon={<IconDataset />} onClick={onOpenPrepare}>
          dataset 준비
        </RailItem>
        {/* 고른 dataset을 model 없이 뜯어봅니다. 준비 바로 아래에 두는 것은
            "만들고 → 살펴본다"가 한 가지 일의 순서이기 때문입니다. */}
        <RailItem icon={<IconEda />} onClick={onOpenEda}>
          EDA
        </RailItem>
        <RailItem
          icon={<IconRecords />}
          active={pathname.startsWith('/records') || pathname.startsWith('/canvas')}
          onClick={() => navigate('/records')}
        >
          기록
        </RailItem>
        <RailItem
          icon={<IconBoard />}
          active={pathname.startsWith('/board')}
          onClick={() => navigate('/board')}
        >
          현황판
        </RailItem>
        <RailItem icon={<IconSettings />} onClick={onOpenSettings}>
          설정
        </RailItem>

        <div style={{ marginTop: 'auto' }}>
          <GpuGauge gpu={gpu} />
        </div>
      </nav>

      {/* 팀원 학습 시작 알림 토스트는 두지 않습니다. 화면 오른쪽 아래를 가리는 값에
          비해 얻는 것이 적었습니다. 같은 정보는 현황판에 있습니다. */}
      <div style={{ minWidth: 0, position: 'relative' }}>{children}</div>
      <ThemeToggle />
    </div>
  );
}
