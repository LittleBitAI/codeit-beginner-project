import type { ExperimentSummary } from '../api/types';

export type DatasetRelationship = 'same' | 'different' | 'unknown';

/** 선택된 experiment가 같은 데이터 artifact 묶음을 썼는지 판정합니다. */
export function datasetRelationship(
  experiments: ExperimentSummary[],
): DatasetRelationship {
  if (experiments.length < 2) return 'unknown';
  const identities = experiments.map((experiment) => experiment.dataset.identity);
  if (identities.some((identity) => identity === null)) return 'unknown';
  const first = identities[0];
  return identities.every((identity) => identity === first) ? 'same' : 'different';
}
