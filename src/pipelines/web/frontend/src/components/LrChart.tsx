import { color, type } from '../design/tokens';
import type { EpochRecord } from '../api/types';
import { Chart } from './LossChart';

/**
 * Learning rate 곡선. 손실 곡선과 자릿수가 달라 같은 축에 그리면 바닥에 붙습니다.
 *
 * 값은 train이 `epoch_completed`로 실제로 알려 준 것뿐입니다. 없는 epoch은 건너뛰고
 * 0으로 채우지 않습니다. 채우면 learning rate가 0까지 떨어진 것처럼 보입니다.
 */
export function LrChart({
  epochs,
  totalEpochs,
}: {
  epochs: EpochRecord[];
  totalEpochs: number | null;
}) {
  const points = epochs
    .filter(
      (item): item is EpochRecord & { learning_rate: number } =>
        typeof item.learning_rate === 'number' && Number.isFinite(item.learning_rate),
    )
    .map((item) => ({ x: item.epoch, y: item.learning_rate }));

  if (points.length === 0) {
    return (
      <div
        style={{
          height: 96,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          ...type.note,
          color: color.textMuted,
          textAlign: 'center',
          padding: '0 16px',
        }}
      >
        이 학습은 learning rate를 기록하지 않았습니다. 이 기록이 생기기 전에 돌린 실행입니다.
      </div>
    );
  }

  const total = Math.max(totalEpochs ?? 0, ...points.map((point) => point.x), 1);
  return (
    <Chart
      xMax={total}
      height={130}
      series={[{ label: 'learning rate', color: color.textMid, width: 1.8, points }]}
    />
  );
}
