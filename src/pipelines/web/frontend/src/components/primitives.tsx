import type { CSSProperties, ReactNode } from 'react';

import { color, font, radius, type } from '../design/tokens';
import type { JobStatus } from '../api/types';
import { IconCheck, IconError, IconInfo, IconWarning } from './Icon';

/* -------------------------------------------------------------- Panel */

export function Panel({
  title,
  right,
  children,
  style,
  bodyStyle,
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  style?: CSSProperties;
  bodyStyle?: CSSProperties;
}) {
  return (
    // 패널은 그림자가 아니라 1px 테두리로만 구분합니다.
    <section
      style={{
        background: color.surface,
        border: `1px solid ${color.border}`,
        borderRadius: radius.panel,
        overflow: 'hidden',
        ...style,
      }}
    >
      {title !== undefined && (
        <header
          style={{
            padding: '11px 16px',
            borderBottom: `1px solid ${color.borderInner}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          <span style={{ ...type.panelHeader, color: color.text }}>{title}</span>
          {right}
        </header>
      )}
      <div style={{ padding: '14px 16px', ...bodyStyle }}>{children}</div>
    </section>
  );
}

/* -------------------------------------------------------------- Badge */

const STATUS_STYLE: Record<JobStatus, { fg: string; bg: string }> = {
  starting: { fg: color.primaryHover, bg: color.primaryTint },
  running: { fg: color.tealDark, bg: color.tealTint },
  queued: { fg: color.textBody, bg: '#F1F4F8' },
  succeeded: { fg: color.greenDark, bg: color.greenTint },
  failed: { fg: color.red, bg: color.redTint },
  cancelled: { fg: color.textBody, bg: '#F1F4F8' },
  interrupted: { fg: color.amber, bg: color.amberTint },
};

export function StatusBadge({ status, label }: { status: JobStatus; label?: string }) {
  const palette = STATUS_STYLE[status] ?? STATUS_STYLE.queued;
  return (
    <span
      style={{
        ...type.badge,
        color: palette.fg,
        background: palette.bg,
        borderRadius: radius.badge,
        padding: '4px 6px',
        textTransform: 'uppercase',
        whiteSpace: 'nowrap',
      }}
    >
      {label ?? status}
    </span>
  );
}

/* -------------------------------------------------------------- Alert */

export type AlertLevel = 'error' | 'warning' | 'success' | 'info';

const ALERT_ICON = {
  error: { Component: IconError, fg: color.red },
  warning: { Component: IconWarning, fg: color.amber },
  success: { Component: IconCheck, fg: color.green },
  info: { Component: IconInfo, fg: '#4C7FBF' },
} as const;

/**
 * 경고와 오류는 가로 한 줄입니다. 색은 아이콘과 제목에만 씁니다.
 * 왼쪽 세로 강조 막대나 큰 색면 패널은 디자인상 금지입니다.
 */
export function AlertRow({
  level,
  title,
  children,
  action,
}: {
  level: AlertLevel;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const { Component, fg } = ALERT_ICON[level];
  return (
    <div
      style={{
        background: color.surface,
        border: `1px solid ${color.border}`,
        borderRadius: 5,
        padding: '11px 13px',
        display: 'flex',
        gap: 11,
        alignItems: 'flex-start',
      }}
    >
      <span style={{ marginTop: 2, flex: 'none' }}>
        <Component color={fg} />
      </span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0, flex: 1 }}>
        <span style={{ font: `600 12.5px/1.4 ${font.sans}`, color: fg }}>{title}</span>
        {children && (
          <span style={{ font: `400 11.5px/1.55 ${font.sans}`, color: color.textBody }}>
            {children}
          </span>
        )}
      </div>
      {action && <div style={{ flex: 'none', display: 'flex', gap: 6 }}>{action}</div>}
    </div>
  );
}

/* -------------------------------------------------------------- Buttons */

type ButtonKind = 'primary' | 'secondary' | 'teal' | 'danger';

const BUTTON_STYLE: Record<ButtonKind, CSSProperties> = {
  primary: { color: '#fff', background: color.primary, border: `1px solid ${color.primary}` },
  secondary: {
    color: color.textStrong,
    background: color.surface,
    border: `1px solid ${color.borderControl}`,
  },
  teal: { color: '#fff', background: color.teal, border: `1px solid ${color.teal}` },
  danger: { color: color.red, background: color.surface, border: '1px solid #F3C9C9' },
};

export function Button({
  kind = 'secondary',
  children,
  onClick,
  disabled,
  title,
  style,
  type: buttonType = 'button',
}: {
  kind?: ButtonKind;
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
  style?: CSSProperties;
  type?: 'button' | 'submit';
}) {
  return (
    <button
      type={buttonType}
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        ...BUTTON_STYLE[kind],
        font: `600 11.5px/1 ${font.sans}`,
        borderRadius: radius.control,
        padding: '8px 12px',
        opacity: disabled ? 0.5 : 1,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

export function Chip({
  active,
  children,
  onClick,
}: {
  active?: boolean;
  children: ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        font: `${active ? 600 : 500} 10.5px/1 ${font.mono}`,
        color: active ? '#fff' : color.textBody,
        background: active ? color.primary : color.surface,
        border: `1px solid ${active ? color.primary : color.borderControl}`,
        borderRadius: radius.chip,
        padding: '5px 9px',
      }}
    >
      {children}
    </button>
  );
}

/* -------------------------------------------------------------- KPI */

export function KpiCard({
  label,
  value,
  note,
  valueColor,
  compact,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  valueColor?: string;
  compact?: boolean;
}) {
  return (
    <div
      style={{
        background: color.surface,
        border: `1px solid ${color.border}`,
        borderRadius: 5,
        padding: '11px 13px',
        display: 'flex',
        flexDirection: 'column',
        gap: 5,
        minWidth: 0,
      }}
    >
      <span style={{ ...type.microLabel, color: color.textMuted }}>{label}</span>
      <span
        style={{
          ...(compact ? type.kpiCompact : type.kpiLarge),
          color: valueColor ?? color.text,
          overflowWrap: 'anywhere',
        }}
      >
        {value}
      </span>
      {note && (
        <span
          style={{
            ...type.plainNote,
            color: color.textBody,
            borderTop: `1px solid ${color.borderRow}`,
            paddingTop: 5,
          }}
        >
          {note}
        </span>
      )}
    </div>
  );
}

/* -------------------------------------------------------------- 기타 */

/** 추정값은 측정값과 반드시 다르게 보여야 합니다: 기울임 + ~ + 점선 밑줄. */
export function EstimatedValue({ children }: { children: ReactNode }) {
  return (
    <span className="estimated" style={{ color: color.textBody, fontFamily: font.mono }}>
      ~{children}
    </span>
  );
}

export function ProgressBar({ ratio, tint }: { ratio: number | null; tint?: string }) {
  const clamped = ratio === null ? null : Math.max(0, Math.min(1, ratio));
  return (
    <div
      style={{
        height: 7,
        borderRadius: 4,
        background: '#EDF1F6',
        overflow: 'hidden',
      }}
      role="progressbar"
      aria-valuenow={clamped === null ? undefined : Math.round(clamped * 100)}
    >
      {clamped !== null && (
        <div style={{ width: `${clamped * 100}%`, height: '100%', background: tint ?? color.teal }} />
      )}
    </div>
  );
}

export function EmptyState({ message, action }: { message: string; action?: ReactNode }) {
  return (
    <div
      style={{
        padding: '46px 20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 10,
      }}
    >
      <span style={{ ...type.body, color: color.textBody }}>{message}</span>
      {action}
    </div>
  );
}

export function ScreenIntro({
  title,
  children,
  terms,
}: {
  title: string;
  children: ReactNode;
  terms?: { term: string; meaning: string }[];
}) {
  return (
    <div
      style={{
        background: color.surface,
        border: `1px solid ${color.border}`,
        borderRadius: radius.panel,
        marginBottom: 12,
        overflow: 'hidden',
        maxWidth: 1720,
      }}
    >
      <div style={{ display: 'flex', gap: 10, padding: '13px 16px', alignItems: 'flex-start' }}>
        <span style={{ marginTop: 1, flex: 'none' }}>
          <IconInfo color="#4C7FBF" />
        </span>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <span style={{ ...type.introTitle, color: color.text }}>{title}</span>
          <span style={{ ...type.body, color: color.textBody, maxWidth: 960 }}>{children}</span>
        </div>
      </div>
      {terms && terms.length > 0 && (
        <div
          style={{
            background: color.surfaceAlt,
            borderTop: `1px solid ${color.borderInner}`,
            padding: '9px 16px 10px',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '7px 22px',
          }}
        >
          {terms.map((item) => (
            <span key={item.term} style={{ flex: '1 1 300px', ...type.plainNote, color: color.textBody }}>
              <b style={{ font: `600 11px/1.5 ${font.mono}`, color: color.textStrong }}>
                {item.term}
              </b>{' '}
              {item.meaning}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 5, minWidth: 0 }}>
      <span style={{ ...type.fieldLabel, color: color.textStrong }}>{label}</span>
      {children}
      {error ? (
        <span style={{ ...type.fieldHint, color: color.red }}>{error}</span>
      ) : hint ? (
        <span style={{ ...type.fieldHint, color: color.textMuted }}>{hint}</span>
      ) : null}
    </label>
  );
}

export const controlStyle: CSSProperties = {
  padding: '8px 10px',
  border: `1px solid ${color.borderControl}`,
  borderRadius: radius.control,
  font: `500 12px/1.3 ${font.mono}`,
  outline: 'none',
  background: color.surface,
  color: color.text,
  width: '100%',
};

export const invalidControlStyle: CSSProperties = {
  ...controlStyle,
  borderColor: color.red,
};
