/**
 * Training Console 디자인의 토큰입니다.
 * 색을 여기 말고 다른 곳에 직접 쓰지 않습니다.
 *
 * 강조색 하나로만 말하는 단색 계열이고, 어두운 판과 밝은 판 둘이 있습니다.
 * **`color.x`가 돌려주는 것은 hex가 아니라 `var(--color-x)`입니다.** 화면 코드는
 * 그대로 두고 `:root`의 변수만 갈아 끼우면 테마가 바뀝니다 — 350군데를 다시
 * 쓰지 않으려면 이 방법뿐입니다.
 */

export type ThemeName = 'dark' | 'light';

const DARK = {
  /** 본문 바탕. 화면에서 가장 어두운 면입니다. */
  page: '#130F0C',
  /** 왼쪽 목록 바탕. 본문보다 한 단계 더 어둡습니다. */
  rail: '#0E0B09',
  /** 떠 있는 면: 지금 학습 중 카드, 코드 블록. */
  panel: '#1D1713',
  /** 오른쪽에서 밀려 나오는 시트와 캔버스 왼쪽 목록. */
  sheet: '#171310',
  /** 고른 줄의 바탕, 진행 막대의 채운 부분. */
  fill: '#382A20',

  /** 강조색. 누를 수 있는 것, 지금 도는 것, 값이 좋아진 방향에만 씁니다. */
  accent: '#E0A96D',
  /** 강조색 위에 얹는 글자색(버튼 안쪽 글씨). */
  onAccent: '#130F0C',
  /** 강조를 두르는 선. 배지 테두리와 보조 버튼에 씁니다. */
  accentLine: '#543D28',

  /** 제목과 큰 숫자. */
  textStrong: '#FDF4E8',
  /** 기본 글자색. */
  text: '#F4E8D8',
  /** 설명 문장. */
  textBody: '#B09A85',
  /** 카드 안 지표 이름처럼 바탕이 밝을 때의 보조 글자. */
  textMid: '#C4AE97',
  /** 보조 정보, 단위, 경로. */
  textMuted: '#8A7663',
  /** 축 눈금처럼 있는 줄만 알면 되는 글자. */
  textFaint: '#5E4E42',

  /** 구역을 가르는 선. */
  border: '#29211B',
  /** 목록 줄 사이의 더 옅은 선. */
  borderRow: '#221B16',

  /**
   * 상태색. 원본 디자인은 amber 하나로만 말하지만, 실패와 성공을 같은 색으로
   * 두면 밤새 돌린 학습이 왜 멈췄는지 화면이 말해 주지 못합니다. 어두운 바탕에서
   * 읽히는 최소한의 세 가지만 남깁니다.
   */
  danger: '#E08A7A',
  dangerLine: '#5C2F26',
  warn: '#E0A96D',
  ok: '#8FC79A',

  /** 로그 줄. 심각도별로 글자색만 다릅니다. */
  logText: '#B09A85',
  logWarn: '#E0A96D',
  logError: '#E08A7A',
  logGood: '#8FC79A',
} as const;

/**
 * 밝은 판. 어두운 판을 그대로 뒤집되 **강조색은 그대로 못 씁니다** —
 * `#E0A96D`를 흰 바탕에 두면 대비가 1.9:1이라 글자가 안 읽힙니다. 같은 amber
 * 계열에서 대비를 맞춘 값으로 내립니다(본문 4.5:1 이상).
 */
const LIGHT: Record<keyof typeof DARK, string> = {
  page: '#FAF7F2',
  rail: '#F3EDE4',
  panel: '#FFFFFF',
  sheet: '#FFFFFF',
  fill: '#F0E4D4',

  accent: '#A25E22',
  onAccent: '#FFFFFF',
  accentLine: '#E0C9AC',

  textStrong: '#17120D',
  text: '#241C14',
  textBody: '#5A4B3C',
  textMid: '#6B5A48',
  textMuted: '#857462',
  textFaint: '#A69684',

  border: '#E5DACB',
  borderRow: '#F0E8DC',

  danger: '#A63A28',
  dangerLine: '#E7C4BC',
  warn: '#A25E22',
  ok: '#2E7D4F',

  logText: '#5A4B3C',
  logWarn: '#A25E22',
  logError: '#A63A28',
  logGood: '#2E7D4F',
};

/** 실제 hex 값. 테마를 켤 때만 읽습니다. */
export const palette: Record<ThemeName, Record<keyof typeof DARK, string>> = {
  dark: DARK,
  light: LIGHT,
};

/** 캔버스에서 실행을 여러 개 겹칠 때 쓰는 순서색. 강조색이 언제나 첫 번째입니다. */
const SERIES: Record<ThemeName, readonly string[]> = {
  dark: ['#E0A96D', '#8FC79A', '#9DB4D8', '#D28FA8', '#C4AE97'],
  light: ['#A25E22', '#2E7D4F', '#3B5F97', '#9A3B62', '#6B5A48'],
};

/** `textMuted` -> `--color-text-muted` */
function cssName(name: string): string {
  return `--color-${name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`;
}

const NAMES = Object.keys(DARK) as (keyof typeof DARK)[];

/**
 * 화면 코드가 쓰는 색. 값은 hex가 아니라 CSS 변수 참조입니다.
 * `` `1px solid ${color.border}` `` 처럼 문자열에 섞어 써도 그대로 동작합니다.
 */
export const color = Object.fromEntries(
  NAMES.map((name) => [name, `var(${cssName(name)})`]),
) as Record<keyof typeof DARK, string>;

/** 곡선 색: validation은 강조색 실선, train은 뒤로 물러난 점선입니다. */
export const chartColor = {
  train: color.textMuted,
  validation: color.accent,
  grid: color.borderRow,
  axis: color.border,
} as const;

export const seriesColor = [0, 1, 2, 3, 4].map((index) => `var(--series-${index})`);

export const font = {
  sans: "'Pretendard Variable', Pretendard, 'Malgun Gothic', system-ui, sans-serif",
  /** 숫자·식별자·경로·로그는 예외 없이 mono입니다. 소수점 정렬이 비교의 전제입니다. */
  mono: "'IBM Plex Mono', Consolas, 'Courier New', monospace",
} as const;

/** 면은 각지게 둡니다. 반경은 배지 3px, 컨트롤 4px을 넘지 않습니다. */
export const radius = {
  badge: 3,
  control: 4,
} as const;

export const space = (steps: number): number => steps * 4;

export const type = {
  /** 화면 제목. dataset 이름처럼 식별자라 mono입니다. */
  pageTitle: { font: `600 27px/1.25 ${font.mono}`, letterSpacing: '-0.02em' },
  sheetTitle: { font: `700 24px/1.3 ${font.sans}`, letterSpacing: '-0.015em' },
  sectionTitle: { font: `600 15px/1 ${font.sans}` },
  subTitle: { font: `600 16px/1 ${font.sans}` },
  listName: { font: `500 14.5px/1.5 ${font.sans}` },

  body: { font: `400 13.5px/1.7 ${font.sans}` },
  bodySmall: { font: `400 13px/1.6 ${font.sans}` },
  note: { font: `400 12.5px/1.6 ${font.sans}` },

  /** 구역 머리말. 대문자 + 넓은 자간. */
  microLabel: { font: `500 11px/1 ${font.mono}`, letterSpacing: '0.1em' },
  fieldLabel: { font: `500 11.5px/1 ${font.mono}`, letterSpacing: '0.06em' },
  metricLabel: { font: `400 11px/1.4 ${font.mono}`, letterSpacing: '0.04em' },

  /** 식별자 한 줄(run_id, 경로). */
  monoId: { font: `400 13px/1.6 ${font.mono}` },
  monoSpec: { font: `400 12.5px/1.6 ${font.mono}` },
  monoValue: { font: `500 15px/1 ${font.mono}` },
  tableCell: { font: `400 13px/1.5 ${font.mono}` },

  /** 지금 보고 있는 화면에서 가장 큰 숫자 하나에만 씁니다. */
  kpiHuge: { font: `600 46px/1 ${font.mono}`, letterSpacing: '-0.035em' },
  kpiLarge: { font: `600 40px/1 ${font.mono}`, letterSpacing: '-0.03em' },
  kpiMid: { font: `500 20px/1 ${font.mono}` },
  kpiSmall: { font: `600 15px/1 ${font.mono}` },

  badge: { font: `600 11px/1.5 ${font.mono}`, letterSpacing: '0.05em' },
  logLine: { font: `400 12px/1.7 ${font.mono}` },
  code: { font: `400 12.5px/1.9 ${font.mono}` },
} as const;

const STORAGE_KEY = 'pill-training-theme';

/** 저장해 둔 테마. 고른 적이 없으면 어두운 판입니다. */
export function readTheme(): ThemeName {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'light' ? 'light' : 'dark';
  } catch {
    // localStorage를 못 써도 화면은 계속 동작해야 합니다.
    return 'dark';
  }
}

/**
 * 고른 판을 `:root`에 씌웁니다.
 *
 * `color-scheme`도 함께 적어야 스크롤바와 목록 상자 같은 브라우저 기본 부품이
 * 따라옵니다. 이걸 빼면 밝은 화면에 검은 스크롤바가 남습니다.
 */
export function applyTheme(theme: ThemeName, root: HTMLElement = document.documentElement): void {
  for (const name of NAMES) root.style.setProperty(cssName(name), palette[theme][name]);
  SERIES[theme].forEach((value, index) => root.style.setProperty(`--series-${index}`, value));
  root.style.setProperty('--font-sans', font.sans);
  root.style.setProperty('--font-mono', font.mono);
  root.style.setProperty('color-scheme', theme);
  root.dataset.theme = theme;
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // 저장을 못 해도 이번 세션에서는 바뀐 채로 씁니다.
  }
}
