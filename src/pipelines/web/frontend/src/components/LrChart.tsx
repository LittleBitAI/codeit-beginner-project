import { chartColor, color, font } from '../design/tokens';
import type { EpochRecord } from '../api/types';

const WIDTH = 640;
const HEIGHT = 120;
const LEFT = 56;
const RIGHT = 628;
const TOP = 14;
const BOTTOM = 88;

/** 눈금 표기입니다. learning rate는 0.0001처럼 작아 소수점으로는 읽히지 않습니다. */
function tick(value: number): string {
  return value === 0 ? '0' : value.toExponential(1);
}

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
  const points = epochs.filter(
    (item): item is EpochRecord & { learning_rate: number } =>
      typeof item.learning_rate === 'number' && Number.isFinite(item.learning_rate),
  );

  if (points.length === 0) {
    return (
      <div
        style={{
          height: 96,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          font: `400 12.5px/1.6 ${font.sans}`,
          color: color.textMuted,
          textAlign: 'center',
          padding: '0 16px',
        }}
      >
        이 학습은 learning rate를 기록하지 않았습니다. 이 기록이 생기기 전에 돌린 실행입니다.
      </div>
    );
  }

  const total = Math.max(totalEpochs ?? 0, ...points.map((item) => item.epoch), 1);
  const values = points.map((item) => item.learning_rate);
  const max = Math.max(...values) * 1.1 || 1;

  const x = (epoch: number) => LEFT + (epoch / total) * (RIGHT - LEFT);
  const y = (value: number) => BOTTOM - (value / max) * (BOTTOM - TOP);

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={LEFT} x2={RIGHT} y1={BOTTOM} y2={BOTTOM} stroke={color.borderChart} />
      <line x1={LEFT} x2={LEFT} y1={TOP} y2={BOTTOM} stroke={color.borderChart} />
      {[0, max].map((value, index) => (
        <text
          key={value}
          x={LEFT - 6}
          y={[BOTTOM, TOP][index]! + 3}
          textAnchor="end"
          fontFamily={font.mono}
          fontSize="9"
          fill={color.textFaint}
        >
          {tick(value)}
        </text>
      ))}
      <polyline
        points={points.map((item) => `${x(item.epoch)},${y(item.learning_rate)}`).join(' ')}
        fill="none"
        stroke={chartColor.now}
        strokeWidth="1.8"
      />
      <text x={LEFT} y={HEIGHT - 6} fontFamily={font.mono} fontSize="9" fill={color.textFaint}>
        learning rate · epoch 1
      </text>
      <text
        x={RIGHT}
        y={HEIGHT - 6}
        textAnchor="end"
        fontFamily={font.mono}
        fontSize="9"
        fill={color.textFaint}
      >
        epoch {total}
      </text>
    </svg>
  );
}
