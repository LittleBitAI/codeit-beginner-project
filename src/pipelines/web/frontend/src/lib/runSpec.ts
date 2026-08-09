/**
 * 학습 하나를 목록에서 알아보는 데 필요한 값들을 뽑습니다.
 *
 * 표에는 실행 이름만 크게 두고 나머지는 이름 아래 한 줄로 내립니다. 예전에는 그
 * 자리에 job_id 앞 8자(`7d851928`)가 있었는데, 어떤 데이터로 무슨 설정을 돌렸는지는
 * 알려 주지 않으면서 자리만 차지했습니다.
 */

import type { JobRecord } from '../api/types';

/** 학습이 어디까지 갔는지. 학습 -> 평가 -> 제출 순서 그대로입니다. */
export interface Stage {
  key: 'train' | 'evaluate' | 'submit';
  label: string;
  done: boolean;
}

/**
 * data artifact 위치에서 데이터셋을 가리키는 폴더 이름만 꺼냅니다.
 *
 * `s3://bucket/datasets/pill_detection/processed/v3-seed42-8020-group/train_manifest.json`
 * 에서 `v3-seed42-8020-group`을 얻습니다. 전체 URI는 100자가 넘어 표에 넣을 수
 * 없고, 팀이 데이터셋을 구별할 때 실제로 부르는 이름이 이 폴더 이름입니다.
 */
export function datasetLabel(dataInputs: Record<string, string> | null | undefined): string | null {
  const uri = dataInputs?.train_manifest_uri;
  if (typeof uri !== 'string' || uri.trim() === '') return null;
  const parts = uri.replace(/\\/g, '/').split('/').filter((part) => part !== '');
  // 마지막은 파일 이름이므로 그 앞이 폴더입니다. 폴더가 없으면 알 수 없습니다.
  const folder = parts.length >= 2 ? parts[parts.length - 2] : undefined;
  if (folder === undefined || folder.endsWith(':')) return null;
  return folder;
}

function text(value: unknown): string | null {
  if (typeof value === 'string' && value.trim() !== '') return value.trim();
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return null;
}

/**
 * 이름 아래에 붙일 한 줄입니다. 모르는 값은 지어내지 않고 빼기만 합니다.
 *
 * 전부 모르면 빈 문자열이라 화면이 그 줄을 아예 그리지 않습니다.
 */
export function specLine(job: JobRecord): string {
  const seed = text(job.settings?.seed);
  return [
    datasetLabel(job.data_inputs),
    text(job.settings?.device),
    text(job.settings?.optimizer),
    seed === null ? null : `seed ${seed}`,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');
}

/**
 * 학습 -> 평가 -> 제출 세 단계입니다. 새 backend field 없이 이미 있는 상태로만 셉니다.
 *
 * 제출은 evaluate가 submission.csv를 실제로 만들었고 registry 등록까지 끝난
 * 경우입니다. 판단을 `submission_requested`가 아니라 **artifact가 있는지**로 하는
 * 이유는, 그 field가 나중에 생겨서 이미 제출을 만든 예전 기록에는 아예 없기
 * 때문입니다. 요청했는지보다 결과물이 남았는지가 사람이 알고 싶은 것이기도 합니다.
 * 등록이 index_failed면 실험 목록에 안 나오므로 여기서도 끝난 것으로 보지 않습니다.
 */
export function stagesOf(job: JobRecord): Stage[] {
  const trained = job.status === 'succeeded';
  const evaluated = trained && job.evaluation?.status === 'succeeded';
  const submitted =
    evaluated &&
    Boolean(job.evaluation?.artifacts?.submission_uri) &&
    job.registration?.status === 'succeeded';
  return [
    { key: 'train', label: '학습', done: trained },
    { key: 'evaluate', label: '평가', done: Boolean(evaluated) },
    { key: 'submit', label: '제출', done: Boolean(submitted) },
  ];
}

/**
 * 목록에서 어느 구역에 둘지.
 *
 * 실패와 취소가 성공한 학습과 한 줄에 섞이면 눈으로 골라내야 합니다. 실제로 35건
 * 중 32건이 결과 없이 끝난 기록이라 3건이 가운데 묻혀 있었습니다.
 *
 * 중단(interrupted)은 위에 둡니다. epoch마다 저장한 checkpoint가 남아 있어 이어서
 * 학습할 수 있으므로, 사람이 아직 판단할 것이 있는 기록입니다. 실패·취소여도 검증
 * 오차가 기록에 남았으면 결과가 있는 것이므로 함께 올립니다.
 */
export function hasResult(job: JobRecord): boolean {
  if (job.status !== 'failed' && job.status !== 'cancelled') return true;
  return typeof job.summary?.best_validation_loss === 'number';
}

/** 접어 둔 구역의 머리글에 쓸 내역입니다. 몇 건을 감췄는지 항상 말해 줍니다. */
export function countLabel(jobs: JobRecord[]): string {
  const failed = jobs.filter((job) => job.status === 'failed').length;
  const cancelled = jobs.length - failed;
  const parts = [
    failed > 0 ? `실패 ${failed}` : null,
    cancelled > 0 ? `취소·중단 ${cancelled}` : null,
  ].filter((part): part is string => part !== null);
  return parts.length > 0 ? `${jobs.length}건 (${parts.join(' · ')})` : `${jobs.length}건`;
}
