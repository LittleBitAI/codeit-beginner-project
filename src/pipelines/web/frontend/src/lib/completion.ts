/**
 * 실험이 어디까지 갔는지를 안전하게 읽습니다.
 *
 * `completion`은 backend에 나중에 생긴 field입니다. 그 값을 그대로 파고들면,
 * 아직 옛 backend가 떠 있는 동안 화면이 통째로 흰 채로 죽습니다. 실제로 그렇게
 * 죽었습니다. 모르는 것은 "아직 안 했다"로 두고 화면은 계속 그립니다.
 */

import type { ExperimentSummary } from '../api/types';

export interface Completion {
  evaluated: boolean;
  submitted: boolean;
  submission_checked: boolean;
  submission_rows: number | null;
}

export function completionOf(experiment: ExperimentSummary): Completion {
  const recorded = experiment.completion;
  if (recorded) return recorded;
  // 옛 응답에도 지표는 있습니다. 평가 여부는 그것으로 알 수 있고, 제출은 알 수 없습니다.
  return {
    evaluated: experiment.metrics.map !== null,
    submitted: false,
    submission_checked: false,
    submission_rows: null,
  };
}

/** 요청한 기준입니다: 학습을 마치고 검증(평가)과 제출까지 끝낸 실험. */
export function isComplete(experiment: ExperimentSummary): boolean {
  const completion = completionOf(experiment);
  return completion.evaluated && completion.submitted;
}
