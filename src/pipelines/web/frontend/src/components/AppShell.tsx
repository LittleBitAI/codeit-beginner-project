import type { ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

import { color, font, radius } from '../design/tokens';
import type { JobRecord } from '../api/types';

const NAV_ITEMS = [
  { to: '/', label: '학습 개요', end: true },
  { to: '/new', label: '새 실험', end: false },
  { to: '/review', label: '설정 검토', end: false },
  { to: '/monitor', label: '라이브 모니터', end: false },
];

const PAGE_TITLES: Record<string, string> = {
  '/': '학습 개요',
  '/new': '새 실험',
  '/review': '설정 검토',
  '/monitor': '라이브 모니터',
};

function pageTitle(pathname: string): string {
  if (pathname.startsWith('/monitor')) return '라이브 모니터';
  return PAGE_TITLES[pathname] ?? '학습';
}

function Badge({
  children,
  tone,
}: {
  children: ReactNode;
  tone: 'teal' | 'neutral' | 'blue';
}) {
  const palette = {
    teal: { color: color.tealDark, background: color.tealTint, border: '#B8E5E1' },
    neutral: { color: color.textBody, background: '#F1F4F8', border: color.border },
    blue: { color: color.primaryHover, background: color.primaryTint, border: color.primaryBorder },
  }[tone];
  return (
    <span
      style={{
        font: `500 10.5px/1 ${font.mono}`,
        color: palette.color,
        background: palette.background,
        border: `1px solid ${palette.border}`,
        borderRadius: radius.badge,
        padding: '4px 7px',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        maxWidth: 320,
      }}
    >
      {children}
    </span>
  );
}

export function AppShell({
  children,
  activeJob,
}: {
  children: ReactNode;
  activeJob: JobRecord | null;
}) {
  const location = useLocation();
  const running = activeJob?.status === 'running' || activeJob?.status === 'queued';

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: color.surfacePage }}>
      <nav
        style={{
          width: 216,
          flex: 'none',
          background: color.navy,
          display: 'flex',
          flexDirection: 'column',
          padding: '15px 0 14px',
          position: 'sticky',
          top: 0,
          height: '100vh',
          overflowY: 'auto',
        }}
      >
        <div style={{ padding: '0 15px 12px' }}>
          <div style={{ font: `650 13px/1.3 ${font.sans}`, color: '#fff' }}>알약 객체 탐지</div>
          <div style={{ font: `400 10px/1.4 ${font.mono}`, color: color.railLabel }}>
            Training GUI
          </div>
        </div>
        <div
          style={{
            font: `600 10px/1.3 ${font.mono}`,
            letterSpacing: '.09em',
            color: color.railLabel,
            padding: '11px 15px 4px',
          }}
        >
          TRAINING
        </div>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            style={({ isActive }) => ({
              // 활성 항목은 배경만 바꿉니다. 왼쪽 강조 막대는 쓰지 않습니다.
              padding: '6px 15px 6px 17px',
              font: `${isActive ? 600 : 500} 12.5px/1.5 ${font.sans}`,
              color: isActive ? '#fff' : color.railIdle,
              background: isActive ? 'rgba(255,255,255,.09)' : 'transparent',
              display: 'flex',
              alignItems: 'center',
              gap: 7,
            })}
          >
            {item.label}
            {item.to === '/monitor' && running && (
              <span
                className="pulse-dot"
                style={{ width: 6, height: 6, borderRadius: '50%', background: color.teal }}
              />
            )}
          </NavLink>
        ))}
      </nav>

      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ height: 3, background: color.teal, flex: 'none' }} />
        <header
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 20,
            background: '#FBFCFE',
            borderBottom: `1px solid ${color.border}`,
            padding: '9px 22px',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          <span style={{ font: `650 14.5px/1.3 ${font.sans}`, color: color.text }}>
            {pageTitle(location.pathname)}
          </span>
          <span style={{ width: 1, height: 15, background: '#DFE3E8' }} />
          <Badge tone="neutral">Faster R-CNN</Badge>
          {activeJob ? (
            <Badge tone={running ? 'teal' : 'blue'}>
              {activeJob.run_id} · {activeJob.status_label}
            </Badge>
          ) : (
            <Badge tone="neutral">실행 중인 학습 없음</Badge>
          )}
        </header>
        <main style={{ flex: 1, padding: '18px 22px 60px', minWidth: 0 }}>{children}</main>
      </div>
    </div>
  );
}
