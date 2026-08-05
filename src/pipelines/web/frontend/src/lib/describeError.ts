import { ApiError } from '../api/client';

/**
 * 오류를 사람이 고칠 수 있는 문장으로 바꿉니다.
 *
 * backend는 "설정을 확인해 주세요." 같은 요약과 함께 어느 칸이 왜 잘못됐는지를
 * `errors`에 담아 보냅니다. 요약만 보여 주면 무엇을 고쳐야 할지 알 수 없습니다.
 */
export function describeError(caught: unknown, fallback: string): string {
  if (!(caught instanceof ApiError)) return fallback;
  if (caught.fields.length === 0) return caught.message || fallback;
  const details = caught.fields.map((item) => `${item.field}: ${item.message}`).join(' · ');
  return `${caught.message} ${details}`.trim();
}
