/**
 * dataset 준비 시트.
 *
 * 이미 전처리된 폴더를 고르는 일과, 원본에서 새로 만드는 일을 한 판에 둡니다.
 * 둘은 순서가 있는 한 가지 일입니다 — 없으면 만들고, 있으면 고릅니다.
 */

import type { DataSource } from '../api/types';
import { DataSourcePanel } from '../components/DataSourcePanel';
import { MicroLabel, Sheet } from '../components/primitives';
import { color, type } from '../design/tokens';

export function PrepareSheet({
  source,
  onSelected,
  onPrepared,
  onClose,
}: {
  source: DataSource | null;
  onSelected: (source: DataSource) => void;
  onPrepared: () => void;
  onClose: () => void;
}) {
  return (
    <Sheet title="dataset 준비" onClose={onClose}>
      <MicroLabel style={{ marginBottom: 12 }}>지금 고른 데이터</MicroLabel>
      <div style={{ ...type.body, color: color.textBody, marginBottom: 22, textWrap: 'pretty' }}>
        여기서 고른 폴더의 artifact 위치가 새 실험의 네 칸에 자동으로 채워집니다. 폴더가 아직
        없으면 아래에서 원본을 전처리해 만드세요.
      </div>
      <DataSourcePanel source={source} onSelected={onSelected} onPrepared={onPrepared} />
    </Sheet>
  );
}
