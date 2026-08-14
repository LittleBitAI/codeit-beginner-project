/**
 * data artifact 위치에서 데이터셋 이름을 읽습니다.
 *
 * 왼쪽 dataset 목록과 기록 줄이 같은 이름을 쓰려면 한 곳에서 뽑아야 합니다. 규칙이
 * 두 벌이 되면 같은 학습이 두 dataset으로 갈라져 보입니다.
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
