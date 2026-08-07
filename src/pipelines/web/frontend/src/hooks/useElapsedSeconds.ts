import { useEffect, useState } from 'react';

/** 초를 "7분 54초"처럼 읽기 쉬운 한국어로 바꿉니다. */
export function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return minutes > 0 ? `${minutes}분 ${rest}초` : `${rest}초`;
}

/**
 * 어떤 일을 시작한 뒤 지난 시간을 초 단위로 계속 셉니다.
 *
 * 데이터 준비는 8분, 평가는 20분 넘게 걸립니다. 화면에 움직이는 것이 하나도 없으면
 * 멈춘 줄 알고 취소하기 때문에, 진행 로그가 하나도 없을 때에도 이 숫자만은 계속
 * 움직입니다.
 */
export function useElapsedSeconds(
  startedAt: string | null | undefined,
  active: boolean,
): number | null {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active]);

  if (!startedAt) return null;
  const started = Date.parse(startedAt);
  if (Number.isNaN(started)) return null;
  return Math.max(0, (now - started) / 1000);
}
