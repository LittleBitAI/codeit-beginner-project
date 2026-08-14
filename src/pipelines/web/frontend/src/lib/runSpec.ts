/**
 * 학습 하나를 목록에서 알아보는 데 필요한 값들을 뽑습니다.
 *
 * 표에는 실행 이름만 크게 두고 나머지는 이름 아래 한 줄로 내립니다. 예전에는 그
 * 자리에 job_id 앞 8자(`7d851928`)가 있었는데, 어떤 데이터로 무슨 설정을 돌렸는지는
 * 알려 주지 않으면서 자리만 차지했습니다.
 */

/** data pipeline이 내놓는 학습 manifest의 파일 이름입니다. */
const MANIFEST_FILE = 'train_manifest.json';

/**
 * data artifact 위치에서 데이터셋을 가리키는 폴더 이름만 꺼냅니다.
 *
 * `s3://bucket/datasets/pill_detection/processed/v3-seed42-8020-group/train_manifest.json`
 * 에서 `v3-seed42-8020-group`을 얻습니다. 전체 URI는 100자가 넘어 표에 넣을 수
 * 없고, 팀이 데이터셋을 구별할 때 실제로 부르는 이름이 이 폴더 이름입니다.
 *
 * "폴더 이름이 곧 데이터셋 이름"은 **파일이 실제로 `train_manifest.json`일 때만**
 * 성립합니다. 그렇지 않으면 그냥 담고 있던 폴더 이름이 잡힙니다. 실제로 값 대신
 * field 이름이 적힌 `artifacts/data/train_manifest_uri.json`이 `data`를,
 * pytest 임시 폴더의 `.../fixtures/train.json`이 `fixtures`를 데이터셋인 척
 * 왼쪽 목록에 올려 놓았습니다. 이름을 댈 수 없으면 지어내지 않습니다.
 */
export function datasetLabel(dataInputs: Record<string, string> | null | undefined): string | null {
  const uri = dataInputs?.train_manifest_uri;
  if (typeof uri !== 'string' || uri.trim() === '') return null;
  const parts = uri.replace(/\\/g, '/').split('/').filter((part) => part !== '');
  if (parts[parts.length - 1] !== MANIFEST_FILE) return null;
  // 마지막은 파일 이름이므로 그 앞이 폴더입니다. 폴더가 없으면 알 수 없습니다.
  const folder = parts.length >= 2 ? parts[parts.length - 2] : undefined;
  if (folder === undefined || folder.endsWith(':')) return null;
  return folder;
}

