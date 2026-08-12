/**
 * Training Console 디자인의 공통 조각입니다.
 *
 * 면은 각지고, 구역은 그림자 대신 1px 선으로만 나눕니다. 강조색(amber)은
 * 누를 수 있는 것과 지금 도는 것에만 씁니다. 색으로 등급을 매기지 않습니다.
 */

import type { CSSProperties, ReactNode } from 'react';

import { color, font, radius, type } from '../design/tokens';
import type { JobStatus } from '../api/types';
import { IconCheck, IconError, IconInfo, IconWarning } from './Icon';

/* -------------------------------------------------------------- 글머리 */

/**
 * 한글에는 mono와 넓은 자간을 쓰지 않습니다.
 *
 * `실 험  이 름`처럼 벌어져 두 단어로 읽힙니다. 자간은 대문자 라틴 글자를 줄지어
 * 놓을 때만 도움이 되고, 한글에서는 낱자를 떼어 놓기만 합니다. 숫자와 식별자는
 * 그대로 mono라 소수점은 계속 맞습니다.
 */
const HANGUL = /[가-힣]/;

function labelFont(children: ReactNode, mono: CSSProperties): CSSProperties {
  return typeof children === 'string' && HANGUL.test(children)
    ? { font: `600 12px/1.4 ${font.sans}` }
    : mono;
}

/** 구역 머리말. 라틴 대문자면 자간을 벌린 mono, 한글이면 sans 한 줄입니다. */
export function MicroLabel({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{ ...labelFont(children, type.microLabel), color: color.textMuted, ...style }}>
      {children}
    </div>
  );
}

/** 제목 왼쪽, 오른쪽에 보조 정보를 두는 구역 머리. 밑줄(baseline)을 맞춥니다. */
export function SectionHead({
  title,
  right,
  style,
}: {
  title: ReactNode;
  right?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        gap: 20,
        flexWrap: 'wrap',
        marginBottom: 16,
        ...style,
      }}
    >
      <span style={{ ...type.sectionTitle, color: color.text }}>{title}</span>
      {right}
    </div>
  );
}

/* -------------------------------------------------------------- 면 */

/**
 * 떠 있는 한 면. 원본 디자인의 카드는 모서리를 굴리지 않고 배경색으로만 뜹니다.
 * `title`을 주면 위에 머리말 한 줄이 붙습니다.
 */
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
    <section style={{ background: color.panel, minWidth: 0, ...style }}>
      {title !== undefined && (
        <header
          style={{
            padding: '14px 20px',
            borderBottom: `1px solid ${color.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          <span style={{ ...type.sectionTitle, color: color.text }}>{title}</span>
          {right}
        </header>
      )}
      <div style={{ padding: '18px 20px', ...bodyStyle }}>{children}</div>
    </section>
  );
}

/** 구역을 가르는 가로선. 목록 줄 사이에는 `faint`를 씁니다. */
export function Rule({ faint, style }: { faint?: boolean; style?: CSSProperties }) {
  return (
    <div
      style={{ height: 1, background: faint ? color.borderRow : color.border, ...style }}
      role="presentation"
    />
  );
}

/* -------------------------------------------------------------- 배지 */

/** 지금 도는 것 옆의 점. `pulse`를 주면 깜빡입니다. */
export function LiveDot({ size = 7, pulse }: { size?: number; pulse?: boolean }) {
  return (
    <span
      className={pulse ? 'pulse-dot' : undefined}
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        background: color.accent,
        display: 'inline-block',
        flex: 'none',
      }}
    />
  );
}

/** 테두리만 두른 작은 표. 바탕을 채우지 않아 줄 사이에서 튀지 않습니다. */
export function Badge({
  children,
  tone = 'accent',
}: {
  children: ReactNode;
  tone?: 'accent' | 'muted' | 'danger';
}) {
  const palette = {
    accent: { fg: color.accent, line: color.accentLine },
    muted: { fg: color.textMuted, line: color.border },
    danger: { fg: color.danger, line: color.dangerLine },
  }[tone];
  return (
    <span
      style={{
        ...labelFont(children, type.badge),
        color: palette.fg,
        border: `1px solid ${palette.line}`,
        borderRadius: radius.badge,
        padding: '3px 7px',
        whiteSpace: 'nowrap',
        flex: 'none',
      }}
    >
      {children}
    </span>
  );
}

/**
 * 학습 상태. 실패만 색을 달리합니다 — 나머지를 색으로 갈라 두면 화면이
 * 신호등이 되어 정작 실패한 줄이 묻힙니다.
 */
const STATUS_TONE: Record<JobStatus, 'accent' | 'muted' | 'danger'> = {
  starting: 'accent',
  running: 'accent',
  queued: 'muted',
  succeeded: 'muted',
  failed: 'danger',
  cancelled: 'muted',
  interrupted: 'danger',
};

export function StatusBadge({ status, label }: { status: JobStatus; label?: string }) {
  return <Badge tone={STATUS_TONE[status] ?? 'muted'}>{label ?? status}</Badge>;
}

/* -------------------------------------------------------------- 알림 */

export type AlertLevel = 'error' | 'warning' | 'success' | 'info';

const ALERT_ICON = {
  error: { Component: IconError, fg: color.danger },
  warning: { Component: IconWarning, fg: color.warn },
  success: { Component: IconCheck, fg: color.ok },
  info: { Component: IconInfo, fg: color.textMid },
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
        background: color.panel,
        border: `1px solid ${level === 'error' ? color.dangerLine : color.border}`,
        borderRadius: radius.control,
        padding: '12px 14px',
        display: 'flex',
        gap: 11,
        alignItems: 'flex-start',
      }}
    >
      <span style={{ marginTop: 2, flex: 'none' }}>
        <Component color={fg} />
      </span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0, flex: 1 }}>
        <span style={{ font: `600 13px/1.4 ${font.sans}`, color: fg }}>{title}</span>
        {children && (
          <span style={{ ...type.note, color: color.textBody }}>{children}</span>
        )}
      </div>
      {action && <div style={{ flex: 'none', display: 'flex', gap: 6 }}>{action}</div>}
    </div>
  );
}

/* -------------------------------------------------------------- 버튼 */

type ButtonKind = 'primary' | 'secondary' | 'ghost' | 'danger';

const BUTTON_STYLE: Record<ButtonKind, CSSProperties> = {
  primary: {
    color: color.onAccent,
    background: color.accent,
    border: '0',
    font: `600 13px/1 ${font.sans}`,
    padding: '11px 18px',
  },
  secondary: {
    color: color.text,
    background: 'transparent',
    border: `1px solid ${color.accentLine}`,
    font: `500 13px/1 ${font.sans}`,
    padding: '11px 17px',
  },
  ghost: {
    color: color.textBody,
    background: 'transparent',
    border: `1px solid ${color.border}`,
    font: `500 13px/1 ${font.sans}`,
    padding: '11px 17px',
  },
  danger: {
    color: color.danger,
    background: 'transparent',
    border: `1px solid ${color.dangerLine}`,
    font: `500 13px/1 ${font.sans}`,
    padding: '11px 17px',
  },
};

export function Button({
  kind = 'ghost',
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
        borderRadius: radius.control,
        opacity: disabled ? 0.45 : 1,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

/** 버튼처럼 보이지 않는 이동 장치. 원본의 "캔버스에서 견주기 →" 같은 줄입니다. */
export function LinkAction({
  children,
  onClick,
  tone = 'accent',
  style,
}: {
  children: ReactNode;
  onClick?: () => void;
  tone?: 'accent' | 'muted';
  style?: CSSProperties;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        font: `500 12.5px/1 ${font.sans}`,
        color: tone === 'accent' ? color.accent : color.textMuted,
        background: 'transparent',
        border: 0,
        padding: 0,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

/** 고르는 표. 이름과 개수를 함께 답니다. */
export function Chip({
  active,
  children,
  count,
  onClick,
}: {
  active?: boolean;
  children: ReactNode;
  count?: ReactNode;
  onClick?: () => void;
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
        border: `1px solid ${active ? color.accentLine : color.border}`,
        borderRadius: radius.control,
        padding: '8px 12px',
        background: active ? color.fill : 'transparent',
      }}
    >
      <span style={{ font: `500 13px/1.3 ${font.sans}`, color: active ? color.text : color.textBody }}>
        {children}
      </span>
      {count !== undefined && (
        <span
          style={{
            font: `500 12px/1.3 ${font.mono}`,
            color: active ? color.accent : color.textMuted,
          }}
        >
          {count}
        </span>
      )}
    </button>
  );
}

/** 밑줄로만 지금 고른 것을 말하는 정렬 손잡이. */
export function SortToggle({
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
      aria-pressed={active}
      onClick={onClick}
      style={{
        font: `${active ? 500 : 400} 13px/1.4 ${font.sans}`,
        color: active ? color.text : color.textMuted,
        background: 'transparent',
        border: 0,
        padding: '0 0 3px',
        borderBottom: `1px solid ${active ? color.accent : 'transparent'}`,
      }}
    >
      {children}
    </button>
  );
}

/* -------------------------------------------------------------- 지표 */

/**
 * 지표 한 칸. 이름은 작은 mono, 값은 tabular-nums입니다.
 * 값이 없으면 0이 아니라 `-`입니다 — 모르는 것을 지어내지 않습니다.
 */
export function Metric({
  label,
  value,
  strong,
  tone,
}: {
  label: ReactNode;
  value: ReactNode;
  /** 그 칸이 이 묶음의 핵심일 때만 켭니다. */
  strong?: boolean;
  tone?: 'accent' | 'muted';
}) {
  return (
    <div style={{ minWidth: 0 }}>
      <div
        style={{
          ...labelFont(label, type.metricLabel),
          color: color.textMuted,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          marginBottom: 7,
        }}
      >
        {label}
      </div>
      <div
        style={{
          font: strong ? `600 17px/1 ${font.mono}` : `400 14px/1 ${font.mono}`,
          color: tone === 'accent' ? color.accent : tone === 'muted' ? color.textMuted : color.text,
          fontVariantNumeric: 'tabular-nums',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {value}
      </div>
    </div>
  );
}

/** 지표 여러 칸을 같은 폭으로 흘려 놓는 격자. */
export function MetricGrid({
  children,
  min = 124,
  style,
}: {
  children: ReactNode;
  min?: number;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))`,
        gap: '16px 20px',
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/** 화면에서 가장 큰 숫자 하나. 이름 + 값 두 줄입니다. */
export function KpiCard({
  label,
  value,
  note,
  valueColor,
  compact,
}: {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  valueColor?: string;
  compact?: boolean;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>
      <span style={{ ...labelFont(label, type.fieldLabel), color: color.textMuted }}>{label}</span>
      <span
        style={{
          ...(compact ? type.kpiSmall : type.kpiLarge),
          color: valueColor ?? color.textStrong,
          fontVariantNumeric: 'tabular-nums',
          overflowWrap: 'anywhere',
        }}
      >
        {value}
      </span>
      {note && <span style={{ ...type.note, color: color.textMuted }}>{note}</span>}
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
      style={{ height: 6, borderRadius: 3, background: color.border, overflow: 'hidden' }}
      role="progressbar"
      aria-valuenow={clamped === null ? undefined : Math.round(clamped * 100)}
    >
      {clamped !== null && (
        <div style={{ width: `${clamped * 100}%`, height: '100%', background: tint ?? color.accent }} />
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
        gap: 12,
      }}
    >
      <span style={{ ...type.body, color: color.textMuted, textAlign: 'center' }}>{message}</span>
      {action}
    </div>
  );
}

/**
 * 화면이 무엇을 하는 곳인지 한 문단. 원본 디자인의 "믿을 수 있는 값" 줄처럼
 * 표 하나 + 문장 하나로 둡니다. 색면 상자를 쓰지 않습니다.
 */
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
    <div style={{ paddingBottom: 22, borderBottom: `1px solid ${color.border}`, maxWidth: '62em' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
        <Badge>{title}</Badge>
        <span style={{ ...type.body, color: color.textBody, textWrap: 'pretty', flex: '1 1 20em' }}>
          {children}
        </span>
      </div>
      {terms && terms.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '7px 24px', marginTop: 12 }}>
          {terms.map((item) => (
            <span key={item.term} style={{ ...type.note, color: color.textMuted }}>
              <b style={{ font: `500 12px/1.5 ${font.mono}`, color: color.textMid }}>{item.term}</b>{' '}
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
    <label style={{ display: 'flex', flexDirection: 'column', gap: 9, minWidth: 0 }}>
      <span style={{ ...labelFont(label, type.fieldLabel), color: color.textMuted }}>{label}</span>
      {children}
      {error ? (
        <span style={{ font: `400 11.5px/1.45 ${font.sans}`, color: color.danger }}>{error}</span>
      ) : hint ? (
        <span style={{ font: `400 11.5px/1.45 ${font.sans}`, color: color.textMuted }}>{hint}</span>
      ) : null}
    </label>
  );
}

export const controlStyle: CSSProperties = {
  padding: '11px 12px',
  border: `1px solid ${color.border}`,
  borderRadius: radius.control,
  font: `400 14px/1 ${font.mono}`,
  // outline은 비우고 global.css의 :focus-visible에 맡깁니다. 여기서 none으로
  // 막으면 키보드로 칸을 옮길 때 어디에 있는지 보이지 않습니다.
  background: color.page,
  color: color.text,
  width: '100%',
};

export const invalidControlStyle: CSSProperties = {
  ...controlStyle,
  borderColor: color.dangerLine,
  color: color.danger,
};

/* -------------------------------------------------------------- 시트 */

/**
 * 오른쪽에서 밀려 나오는 판. 새 실험과 설정이 이 위에 섭니다.
 *
 * 화면을 갈아 끼우지 않고 덮기 때문에, 뒤에 있던 목록이 그대로 남아
 * "무엇을 보다가 이걸 열었는지"를 잃지 않습니다.
 */
export function Sheet({
  title,
  subtitle,
  onClose,
  footer,
  children,
}: {
  title: string;
  subtitle?: ReactNode;
  onClose: () => void;
  footer?: ReactNode;
  children: ReactNode;
}) {
  return (
    <>
      <div
        onClick={onClose}
        style={{ position: 'fixed', inset: 0, background: 'rgba(8,6,4,.55)', zIndex: 55 }}
      />
      <aside
        role="dialog"
        aria-label={title}
        style={{
          position: 'fixed',
          right: 0,
          top: 0,
          bottom: 0,
          width: 'min(520px, 100vw)',
          background: color.sheet,
          borderLeft: `1px solid ${color.border}`,
          padding: '32px 34px 34px',
          zIndex: 60,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
            gap: 16,
            marginBottom: subtitle ? 8 : 34,
          }}
        >
          <h2 style={{ ...type.sheetTitle, margin: 0, color: color.textStrong }}>{title}</h2>
          <button
            type="button"
            aria-label="닫기"
            onClick={onClose}
            style={{
              font: `400 22px/1 ${font.sans}`,
              color: color.textMuted,
              background: 'transparent',
              border: 0,
              padding: 0,
            }}
          >
            ×
          </button>
        </div>
        {subtitle && (
          <div style={{ ...type.monoId, color: color.textMuted, marginBottom: 34 }}>{subtitle}</div>
        )}
        {children}
        {footer && (
          <div style={{ marginTop: 'auto', paddingTop: 32, display: 'flex', alignItems: 'center', gap: 12 }}>
            {footer}
          </div>
        )}
      </aside>
    </>
  );
}
