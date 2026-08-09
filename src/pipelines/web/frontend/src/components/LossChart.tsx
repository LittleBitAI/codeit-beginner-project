import { chartColor, color, font } from '../design/tokens';
import type { EpochRecord } from '../api/types';

const WIDTH = 640;
const HEIGHT = 190;
const LEFT = 42;
const RIGHT = 628;
const TOP = 16;
const BOTTOM = 158;

/**
 * 손실 곡선. 차트 라이브러리를 쓰지 않고 SVG를 직접 그립니다.
 * 모든 점은 train이 실제로 내보낸 ``epoch_completed`` 값에서만 나옵니다.
 */
export function LossChart({
  epochs,
  totalEpochs,
  currentEpoch,
}: {
  epochs: EpochRecord[];
  totalEpochs: number | null;
  currentEpoch: number | null;
}) {
  const points = epochs.filter(
    (item) => item.train_loss !== null || item.validation_loss !== null,
  );

  if (points.length === 0) {
    return (
      <div
        style={{
          height: 190,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          font: `400 12.5px/1.6 ${font.sans}`,
          color: color.textMuted,
          textAlign: 'center',
        }}
      >
        epoch이 하나도 끝나지 않아 그릴 데이터가 없습니다.
      </div>
    );
  }

  const total = Math.max(totalEpochs ?? 0, ...points.map((item) => item.epoch), 1);
  const values = points.flatMap((item) =>
    [item.train_loss, item.validation_loss].filter((value): value is number => value !== null),
  );
  const max = Math.max(...values, 0.0001) * 1.1;

  const x = (epoch: number) => LEFT + (epoch / total) * (RIGHT - LEFT);
  const y = (value: number) => BOTTOM - (Math.min(value, max) / max) * (BOTTOM - TOP);

  const line = (pick: (item: EpochRecord) => number | null) =>
    points
      .filter((item) => pick(item) !== null)
      .map((item) => `${x(item.epoch)},${y(pick(item) as number)}`)
      .join(' ');

  const gridlines = [0.25, 0.5, 0.75].map((ratio) => BOTTOM - ratio * (BOTTOM - TOP));

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {gridlines.map((position) => (
        <line
          key={position}
          x1={LEFT}
          x2={RIGHT}
          y1={position}
          y2={position}
          stroke={color.surfaceSunken}
        />
      ))}
      <line x1={LEFT} x2={RIGHT} y1={BOTTOM} y2={BOTTOM} stroke={color.borderChart} />
      <line x1={LEFT} x2={LEFT} y1={TOP} y2={BOTTOM} stroke={color.borderChart} />

      {[0, max / 2, max].map((value, index) => (
        <text
          key={value}
          x={LEFT - 6}
          y={[BOTTOM, (TOP + BOTTOM) / 2, TOP][index]! + 3}
          textAnchor="end"
          fontFamily={font.mono}
          fontSize="9"
          fill={color.textFaint}
        >
          {value.toFixed(2)}
        </text>
      ))}
      <text x={LEFT} y={HEIGHT - 6} fontFamily={font.mono} fontSize="9" fill={color.textFaint}>
        epoch 1
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

      {currentEpoch !== null && (
        <line
          x1={x(currentEpoch)}
          x2={x(currentEpoch)}
          y1={TOP}
          y2={BOTTOM}
          stroke={chartColor.now}
          strokeWidth="1"
          strokeDasharray="3 3"
        />
      )}

      <polyline
        points={line((item) => item.train_loss)}
        fill="none"
        stroke={chartColor.train}
        strokeWidth="1.8"
      />
      <polyline
        points={line((item) => item.validation_loss)}
        fill="none"
        stroke={chartColor.validation}
        strokeWidth="1.8"
      />
    </svg>
  );
}

export function ChartLegend() {
  return (
    <div style={{ display: 'flex', gap: 14, padding: '0 16px 12px' }}>
      {[
        { label: 'train loss', tint: chartColor.train },
        { label: 'validation loss', tint: chartColor.validation },
      ].map((item) => (
        <span
          key={item.label}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            font: `400 11.5px/1 ${font.mono}`,
            color: color.textBody,
          }}
        >
          <span style={{ width: 14, height: 2, background: item.tint, display: 'inline-block' }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}
