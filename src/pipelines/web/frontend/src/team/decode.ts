/**
 * AppSync `AWSJSON` field을 화면이 쓸 값으로 되돌립니다.
 *
 * AWSJSON은 **문자열 하나**로 오갑니다. resolver가 이미 문자열이 된 값을 그대로
 * 돌려주면 AppSync가 그것을 한 번 더 감싸서, browser에는 두 겹으로 감싼 문자열이
 * 도착합니다. 한 번만 풀면 객체가 아니라 문자열이 나오고, 화면은 그 값을 통째로
 * 잃습니다. 팀 활동에서 모델명과 optimizer, mAP가 `-`로만 보이고 실시간 로그가
 * 한 줄도 늘지 않던 원인이 이것입니다.
 *
 * 그래서 원하는 모양이 나올 때까지 풀되, 서버를 고쳐 한 겹으로 오는 값도 그대로
 * 읽습니다. 두 모양은 배포 시점 차이로 한동안 섞입니다.
 */

import type { LogLine } from '../api/types';

/** 실수로 무한히 도는 일이 없도록 둔 상한입니다. 두 겹이면 충분합니다. */
const MAX_DEPTH = 4;

function unwrap(value: unknown): unknown {
  let current = value;
  for (let depth = 0; depth < MAX_DEPTH; depth += 1) {
    if (typeof current !== 'string') return current;
    try {
      current = JSON.parse(current);
    } catch {
      // 더 풀 수 없는 문자열입니다. 판단은 부르는 쪽 type 검사에 맡깁니다.
      return current;
    }
  }
  return current;
}

/** object가 아니면 빈 object입니다. 화면이 "값이 아직 없다"로 그리게 합니다. */
export function decodeJson(value: unknown): Record<string, unknown> {
  const decoded = unwrap(value);
  if (!decoded || typeof decoded !== 'object' || Array.isArray(decoded)) return {};
  return decoded as Record<string, unknown>;
}

/** 배열이 아니면 빈 목록입니다. */
export function decodeLines(value: unknown): LogLine[] {
  const decoded = unwrap(value);
  return Array.isArray(decoded) ? (decoded as LogLine[]) : [];
}
