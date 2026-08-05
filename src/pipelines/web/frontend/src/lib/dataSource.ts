import type { DataSource } from '../api/types';
import { ALL_DATA_KEYS } from './dataKeys';

/**
 * 고른 데이터셋을 구별하는 값입니다.
 *
 * 필수 artifact와 선택 test manifest URI를 그대로 이어 붙입니다. 위치 이름이나 시각이 아니라 실제로 읽을
 * 파일이 바뀌었는지를 기준으로 삼아야, 같은 폴더에서 다시 준비했을 때도 알아챕니다.
 */
export function sourceKeyOf(source: DataSource | null): string | null {
  if (!source?.complete) return null;
  return ALL_DATA_KEYS.map((key) => source.data[key] ?? '').join('|');
}

/**
 * 지금 입력 칸의 값이 고른 데이터셋과 같은지 확인합니다.
 *
 * 다르면 화면에는 새 데이터셋이 보이는데 실제로는 다른 데이터로 학습됩니다.
 * 실제로 그렇게 잘못 학습된 적이 있어 눈에 띄게 알려야 합니다.
 */
export function dataMatchesSource(
  draftData: Record<string, string>,
  source: DataSource | null,
): boolean {
  if (!source?.complete) return true;
  return ALL_DATA_KEYS.every(
    (key) => (draftData[key] ?? '').trim() === (source.data[key] ?? ''),
  );
}
