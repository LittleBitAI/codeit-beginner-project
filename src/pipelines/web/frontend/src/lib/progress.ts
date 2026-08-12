/**
 * 진행 상태를 화면이 믿을 수 있는 두 숫자로 읽습니다.
 *
 * 이어서 학습한 실행은 앞선 epoch가 진행 log로 오지 않습니다. 그래서 옛 backend가
 * 남긴 기록은 `completed_epochs`를 0으로 적어 두고 `current_epoch`만 11이라고
 * 말합니다. 그대로 그리면 11 epoch째 도는 학습이 화면에서는 0 epoch입니다.
 * 두 값 중 큰 쪽을 쓰면 그런 기록도 제대로 읽힙니다.
 */

import type { Progress } from '../api/types';

export function epochsDone(progress: Progress): number {
  const completed = progress.completed_epochs ?? 0;
  const running = progress.current_epoch ?? 0;
  // 지금 도는 epoch은 아직 끝나지 않았으므로 하나 앞까지가 끝난 수입니다.
  return Math.max(completed, running > 0 ? running - 1 : 0);
}

/**
 * 진행률(0~1). 계획 epoch를 모르면 `null`입니다.
 *
 * backend의 `percent`를 쓰지 않는 것은 그 값이 `completed_epochs`에서 나와 같은
 * 이유로 0에 머물기 때문입니다. 끝난 학습은 조기 종료로 계획 epoch가 남아 있어도
 * 100%입니다.
 */
export function progressRatio(progress: Progress): number | null {
  if (progress.finished) return 1;
  if (!progress.total_epochs) return null;
  return Math.max(0, Math.min(1, epochsDone(progress) / progress.total_epochs));
}
