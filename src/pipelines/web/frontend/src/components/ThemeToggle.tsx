/**
 * 밝은 판과 어두운 판을 오가는 단추. 화면 오른쪽 아래에 떠 있습니다.
 *
 * 색은 전부 `:root`의 CSS 변수라, 여기서 변수만 갈아 끼우면 화면 전체가 따라
 * 바뀝니다. React state를 위로 올려 다시 그릴 필요가 없습니다.
 */

import { useState } from 'react';

import { applyTheme, color, font, radius, readTheme, type ThemeName } from '../design/tokens';

function IconSun() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ flex: 'none' }}>
      <circle cx="8" cy="8" r="3.1" stroke="currentColor" strokeWidth="1.3" />
      <path
        d="M8 1.4v1.8M8 12.8v1.8M1.4 8h1.8M12.8 8h1.8M3.3 3.3l1.3 1.3M11.4 11.4l1.3 1.3M12.7 3.3l-1.3 1.3M4.6 11.4l-1.3 1.3"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconMoon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" style={{ flex: 'none' }}>
      <path
        d="M13.4 9.6A5.8 5.8 0 0 1 6.4 2.6a5.9 5.9 0 1 0 7 7Z"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<ThemeName>(readTheme);
  const next: ThemeName = theme === 'dark' ? 'light' : 'dark';

  return (
    <button
      type="button"
      aria-label={`${next === 'light' ? '밝은' : '어두운'} 화면으로 바꾸기`}
      title={`${next === 'light' ? '밝은' : '어두운'} 화면으로`}
      onClick={() => {
        applyTheme(next);
        setTheme(next);
      }}
      style={{
        position: 'fixed',
        right: 18,
        bottom: 18,
        zIndex: 70,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '9px 13px',
        borderRadius: radius.control,
        border: `1px solid ${color.border}`,
        background: color.panel,
        color: color.textBody,
        font: `500 12.5px/1 ${font.sans}`,
      }}
    >
      {theme === 'dark' ? <IconSun /> : <IconMoon />}
      {theme === 'dark' ? '밝게' : '어둡게'}
    </button>
  );
}
