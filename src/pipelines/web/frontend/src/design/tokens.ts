/**
 * design_handoff_pill_detect_platform/README.md의 토큰 표를 그대로 옮긴 값입니다.
 * 색을 여기 말고 다른 곳에 직접 쓰지 않습니다. CSS 변수도 이 값에서 만들어집니다.
 */

export const color = {
  navy: '#0B2545',
  navyInk: '#172030',

  primary: '#1A56A8',
  primaryHover: '#164A94',
  primaryTint: '#EAF1FB',
  primaryBorder: '#C9DCF3',

  teal: '#0D8B84',
  tealDark: '#0A6E68',
  tealTint: '#E2F5F3',

  green: '#1F8A3B',
  greenDark: '#166B2D',
  greenTint: '#EAF6EC',

  amber: '#B5760A',
  amberTint: '#FBF4E8',

  red: '#C1332D',
  redTint: '#FBE4E4',

  text: '#111C2E',
  textStrong: '#31405A',
  textBody: '#5C6470',
  textMuted: '#8A929E',
  textFaint: '#9AA2AD',

  border: '#E4E7EB',
  borderInner: '#EEF0F3',
  borderRow: '#F2F4F7',
  borderControl: '#D8DCE2',
  borderChart: '#DFE3E8',

  surface: '#FFFFFF',
  surfacePage: '#F6F7F9',
  surfaceAlt: '#FAFBFC',
  surfaceSunken: '#F6F7F9',
  surfaceTableHead: '#F7F9FC',

  railLabel: '#5C7290',
  railIdle: '#9AA2AD',

  logText: '#C6D4E6',
  logWarn: '#E0A93F',
  logError: '#F08A8A',
  logGood: '#8FE0A8',
} as const;

/** 곡선 색: train은 primary, validation은 amber (design 05 라이브 모니터). */
export const chartColor = {
  train: color.primary,
  validation: color.amber,
  now: color.teal,
} as const;

export const font = {
  sans: "'Pretendard Variable', Pretendard, 'Malgun Gothic', system-ui, sans-serif",
  /** 숫자·식별자·경로·로그는 예외 없이 mono입니다. 소수점 정렬이 비교의 전제입니다. */
  mono: "'IBM Plex Mono', Consolas, 'Courier New', monospace",
} as const;

/** 반경은 badge 3~4px, control 4px, panel 5~6px를 넘지 않습니다. */
export const radius = {
  badge: 3,
  chip: 4,
  control: 4,
  panel: 6,
} as const;

export const space = (steps: number): number => steps * 4;

export const type = {
  pageTitle: { font: `650 14.5px/1.3 ${font.sans}` },
  panelHeader: { font: `600 13px/1 ${font.sans}` },
  introTitle: { font: `600 13px/1.4 ${font.sans}` },
  body: { font: `400 12.5px/1.7 ${font.sans}` },
  plainNote: { font: `400 11px/1.5 ${font.sans}` },
  fieldLabel: { font: `600 11.5px/1 ${font.sans}` },
  fieldHint: { font: `400 10.5px/1.45 ${font.sans}` },
  tableCell: { font: `400 11px/1.3 ${font.mono}` },
  tableHead: { font: `600 10px/1.3 ${font.mono}`, letterSpacing: '.04em' },
  microLabel: { font: `500 10px/1.3 ${font.mono}`, letterSpacing: '.05em' },
  kpiLarge: { font: `600 22px/1 ${font.mono}` },
  kpiCompact: { font: `600 15px/1 ${font.mono}` },
  badge: { font: `600 10px/1.3 ${font.mono}` },
  logLine: { font: `400 10.5px/1.6 ${font.mono}` },
  code: { font: `400 10.5px/1.65 ${font.mono}` },
} as const;

/** global.css가 쓰는 CSS 변수. 단일 출처를 지키려고 여기서 주입합니다. */
export const cssVariables: Record<string, string> = {
  '--color-navy': color.navy,
  '--color-navy-ink': color.navyInk,
  '--color-primary': color.primary,
  '--color-text': color.text,
  '--color-border-control': color.borderControl,
  '--color-surface-page': color.surfacePage,
  '--color-scrollbar': '#C9D3E0',
  '--font-sans': font.sans,
  '--font-mono': font.mono,
};

export function applyCssVariables(root: HTMLElement = document.documentElement): void {
  for (const [name, value] of Object.entries(cssVariables)) {
    root.style.setProperty(name, value);
  }
}
