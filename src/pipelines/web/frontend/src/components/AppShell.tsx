/**
 * 화면 전체의 틀입니다.
 *
 * 왼쪽은 **dataset 목록**입니다. 화면 이름을 늘어놓는 대신 "어떤 데이터의 기록을
 * 보고 있는지"를 세로로 세웁니다. 이 도구에서 사람이 실제로 갈아 끼우는 것이
 * 화면이 아니라 dataset이기 때문입니다. 화면 사이 이동은 본문 안의 링크가 합니다.
 */

import type { ReactNode } from 'react';

import { color, font, type } from '../design/tokens';
import type { GpuStatus } from '../api/types';
import { LiveDot, MicroLabel } from './primitives';
import { ThemeToggle } from './ThemeToggle';

/** 왼쪽 목록의 한 줄. 기록에서 뽑은 dataset 하나입니다. */
export interface DatasetOption {
  /** 고르기·비교에 쓰는 값. dataset 이름 그대로입니다. */
  key: string;
  /** 목록에 적는 짧은 이름. */
  short: string;
  /** 이름 아래 한 줄 설명. */
  sub: string;
  /** 이 dataset으로 남은 기록 수. */
  count: number;
}

/** 색은 `<svg>`의 `color`에 한 번 얹고 안쪽은 `currentColor`로 물려받습니다. */
const iconBox = { flex: 'none' as const, color: color.textMuted };

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

function RailAction({ icon, children, onClick }: { icon: ReactNode; children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        font: `400 12.5px/1 ${font.sans}`,
        color: color.textMuted,
        background: 'transparent',
        border: 0,
        padding: 0,
      }}
    >
      {icon}
      {children}
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
  datasets,
  activeDataset,
  onPickDataset,
  gpu,
  running,
  onOpenPrepare,
  onOpenEda,
  onOpenSettings,
}: {
  children: ReactNode;
  datasets: DatasetOption[];
  activeDataset: string | null;
  onPickDataset: (key: string) => void;
  gpu: GpuStatus | null;
  /** 지금 도는 학습이 있는지. 목록의 해당 dataset 옆에 점을 답니다. */
  running: string | null;
  onOpenPrepare: () => void;
  onOpenEda: () => void;
  onOpenSettings: () => void;
}) {
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
        <div style={{ padding: '0 20px 26px' }}>
          <div style={{ font: `600 14px/1.4 ${font.sans}`, color: color.text }}>알약 객체 탐지</div>
          <div style={{ font: `400 11.5px/1.5 ${font.mono}`, color: color.textMuted }}>Training</div>
        </div>

        <MicroLabel style={{ padding: '0 20px 12px' }}>DATASETS</MicroLabel>

        {datasets.length === 0 ? (
          <div style={{ padding: '0 20px', ...type.note, color: color.textFaint }}>
            아직 기록이 없습니다. 아래 <b style={{ color: color.textMuted }}>dataset 준비</b>로
            전처리를 먼저 돌리세요.
          </div>
        ) : (
          datasets.map((item) => {
            const on = item.key === activeDataset;
            return (
              <button
                key={item.key}
                type="button"
                aria-current={on ? 'true' : undefined}
                data-row-hover={on ? undefined : ''}
                onClick={() => onPickDataset(item.key)}
                style={{
                  padding: '12px 20px',
                  position: 'relative',
                  textAlign: 'left',
                  border: 0,
                  background: on ? color.fill : 'transparent',
                  width: '100%',
                }}
              >
                {on && (
                  <span
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: 2,
                      background: color.accent,
                    }}
                  />
                )}
                <span
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    gap: 10,
                  }}
                >
                  <span
                    style={{
                      font: `500 12.5px/1.4 ${font.mono}`,
                      color: on ? color.text : color.textBody,
                      minWidth: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {item.short}
                  </span>
                  <span
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 7,
                      flex: 'none',
                      font: `500 12px/1 ${font.mono}`,
                      color: color.textMuted,
                    }}
                  >
                    {running === item.key && <LiveDot size={6} pulse />}
                    {item.count}
                  </span>
                </span>
                <span
                  style={{
                    display: 'block',
                    font: `400 12px/1.5 ${font.sans}`,
                    color: color.textMuted,
                    marginTop: 5,
                  }}
                >
                  {item.sub}
                </span>
              </button>
            );
          })
        )}

        <div
          style={{
            marginTop: 'auto',
            padding: '26px 20px 0',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            gap: 14,
          }}
        >
          <RailAction icon={<IconDataset />} onClick={onOpenPrepare}>
            dataset 준비
          </RailAction>
          {/* 고른 dataset을 model 없이 뜯어봅니다. 준비 바로 아래에 두는 것은
              "만들고 → 살펴본다"가 한 가지 일의 순서이기 때문입니다. */}
          <RailAction icon={<IconEda />} onClick={onOpenEda}>
            EDA
          </RailAction>
          <RailAction icon={<IconSettings />} onClick={onOpenSettings}>
            설정
          </RailAction>
        </div>

        <GpuGauge gpu={gpu} />
      </nav>

      {/* 팀원 학습 시작 알림 토스트는 두지 않습니다. 화면 오른쪽 아래를 가리는 값에
          비해 얻는 것이 적었습니다. 같은 정보는 기록 목록의 "학습 중" 표에 있습니다. */}
      <div style={{ minWidth: 0, position: 'relative' }}>{children}</div>
      <ThemeToggle />
    </div>
  );
}
