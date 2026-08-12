/**
 * 아이콘은 전부 inline SVG입니다. 디자인 제약상 emoji나 ✓ 같은 글자 아이콘은 쓰지
 * 않습니다. stroke는 1.25~1.4로 통일합니다.
 *
 * 색은 `<svg>`의 `color`에 한 번만 얹고 안쪽은 `currentColor`로 물려받습니다.
 * 토큰이 `var(--color-x)` 문자열이라 presentation attribute로는 풀리지 않기
 * 때문이고, 덕분에 path마다 색을 다시 적을 일도 없어집니다.
 */

interface IconProps {
  size?: number;
  color: string;
  title?: string;
}

const base = (size: number, color: string) => ({
  width: size,
  height: size,
  viewBox: '0 0 16 16',
  fill: 'none',
  style: { flex: 'none' as const, display: 'block' as const, color },
});

export function IconError({ size = 13, color, title }: IconProps) {
  return (
    <svg {...base(size, color)} role={title ? 'img' : 'presentation'} aria-label={title}>
      <circle cx="8" cy="8" r="6.4" stroke="currentColor" strokeWidth="1.25" />
      <path
        d="M5.9 5.9l4.2 4.2M10.1 5.9l-4.2 4.2"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function IconWarning({ size = 13, color, title }: IconProps) {
  return (
    <svg {...base(size, color)} role={title ? 'img' : 'presentation'} aria-label={title}>
      <path d="M8 2.4l5.6 10.2H2.4L8 2.4z" stroke="currentColor" strokeWidth="1.25" strokeLinejoin="round" />
      <path d="M8 6.4v3" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      <circle cx="8" cy="11.1" r="0.65" fill="currentColor" />
    </svg>
  );
}

export function IconCheck({ size = 13, color, title }: IconProps) {
  return (
    <svg {...base(size, color)} role={title ? 'img' : 'presentation'} aria-label={title}>
      <circle cx="8" cy="8" r="6.4" stroke="currentColor" strokeWidth="1.3" />
      <path
        d="M5.4 8.2l1.9 1.9 3.4-3.9"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconInfo({ size = 15, color, title }: IconProps) {
  return (
    <svg {...base(size, color)} role={title ? 'img' : 'presentation'} aria-label={title}>
      <circle cx="8" cy="8" r="6.4" stroke="currentColor" strokeWidth="1.25" />
      <path d="M8 7.3v3.4" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      <circle cx="8" cy="5.2" r="0.7" fill="currentColor" />
    </svg>
  );
}

export function IconShield({ size = 13, color, title }: IconProps) {
  return (
    <svg {...base(size, color)} role={title ? 'img' : 'presentation'} aria-label={title}>
      <path
        d="M8 2l4.6 1.7v4c0 3-2 5.3-4.6 6.3-2.6-1-4.6-3.3-4.6-6.3v-4L8 2z"
        stroke="currentColor"
        strokeWidth="1.25"
        strokeLinejoin="round"
      />
    </svg>
  );
}
