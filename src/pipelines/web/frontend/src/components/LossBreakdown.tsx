import type { EpochRecord } from '../api/types';
import { color, type } from '../design/tokens';
import { loss } from '../lib/format';
import { Panel } from './primitives';

/**
 * 마지막으로 상세 loss가 들어온 epoch 하나를 이름별로 펼쳐 보여 줍니다.
 *
 * 이름 목록은 모델이 정합니다(Faster R-CNN과 RetinaNet이 서로 다릅니다). 그래서
 * 화면이 이름을 정해 두지 않고 **받은 데이터에서 뽑아** 그립니다. 상세 loss가 아예
 * 없는 실행에서는 아무것도 그리지 않습니다. 빈 표를 보여 주면 값이 0인 것처럼 읽힙니다.
 */
export function LossBreakdown({ epochs }: { epochs: EpochRecord[] }) {
  const latest = [...epochs]
    .reverse()
    .find((item) => item.train_loss_components || item.validation_loss_components);
  if (!latest) return null;

  const train = latest.train_loss_components ?? {};
  const validation = latest.validation_loss_components ?? {};
  const names = [...new Set([...Object.keys(train), ...Object.keys(validation)])].sort();
  if (names.length === 0) return null;

  return (
    <Panel title={`손실 분해 · epoch ${latest.epoch}`}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ display: 'flex', gap: 10, ...type.tableHead, color: color.textMuted }}>
          <span style={{ flex: '1 1 auto', minWidth: 0 }}>NAME</span>
          <span style={{ width: 66, textAlign: 'right' }}>TRAIN</span>
          <span style={{ width: 66, textAlign: 'right' }}>VAL</span>
        </div>
        {names.map((name) => (
          <div
            key={name}
            style={{
              display: 'flex',
              gap: 10,
              ...type.tableCell,
              color: color.textBody,
              borderTop: `1px solid ${color.borderRow}`,
              paddingTop: 5,
            }}
          >
            <span style={{ flex: '1 1 auto', minWidth: 0, overflowWrap: 'anywhere' }}>{name}</span>
            <span style={{ width: 66, textAlign: 'right', color: color.text }}>
              {loss(train[name])}
            </span>
            <span style={{ width: 66, textAlign: 'right', color: color.text }}>
              {loss(validation[name])}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}
