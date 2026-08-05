/** 숫자와 시간을 화면에 그대로 쓸 문자열로 바꿉니다. 값이 없으면 "-"입니다. */

export function loss(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toFixed(4);
}

export function integer(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toLocaleString('ko-KR');
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '-';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  // 0인 자리는 붙이지 않습니다. "12분 0초"보다 "12분"이 읽기 쉽습니다.
  if (hours > 0) return minutes > 0 ? `${hours}시간 ${minutes}분` : `${hours}시간`;
  if (minutes > 0) return rest > 0 ? `${minutes}분 ${rest}초` : `${minutes}분`;
  return `${rest}초`;
}

export function startedAt(value: string | null | undefined): string {
  if (!value) return '-';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '-';
  return parsed.toLocaleString('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function megabytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return '-';
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`;
  return `${value} MB`;
}

export function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : `${value}%`;
}
