/**
 * 곡선. 차트 라이브러리를 쓰지 않고 SVG를 직접 그립니다.
 *
 * 모든 점은 train이 실제로 내보낸 `epoch_completed` 값에서만 나옵니다. 값이 없는
 * epoch은 건너뛰고 0으로 채우지 않습니다. 채우면 손실이 0까지 떨어진 것처럼 보입니다.
 *
 * y축은 **데이터 범위**입니다. 0을 바닥으로 잡으면 후반의 0.02 차이가 한 줄로
 * 뭉개져서, 정작 견주려던 구간이 안 보입니다. 대신 축 옆에 범위를 늘 적어 둡니다.
 */

import type { CSSProperties, ReactNode } from 'react';

import { chartColor, color, font, type } from '../design/tokens';
import type { EpochRecord } from '../api/types';

/** SVG 안쪽 좌표계. 가로는 늘어나고 선 굵기는 그대로입니다(non-scaling-stroke). */
const VIEW_W = 900;
const VIEW_H = 300;
const TOP = 20;
const BOTTOM = 280;

export interface Series {
  /** 범례에 적는 이름. */
  label: string;
  color: string;
  width?: number;
  /** 점선으로 물러나게 둘 때만 줍니다. */
  dash?: string;
  points: { x: number; y: number }[];
}

function niceRange(values: number[]): { min: number; max: number } {
  if (values.length === 0) return { min: 0, max: 1 };
  const low = Math.min(...values);
  const high = Math.max(...values);
  // 값이 모두 같으면 범위가 0이라 폭이 없어집니다. 그때만 위아래로 벌립니다.
  if (high - low < 1e-9) return { min: low - 0.05, max: high + 0.05 };
  const pad = (high - low) * 0.08;
  return { min: low - pad, max: high + pad };
}

function fmtAxis(value: number): string {
  const size = Math.abs(value);
  if (size !== 0 && (size < 0.001 || size >= 10000)) return value.toExponential(1);
  return value.toFixed(size < 1 ? 3 : 2);
}

/**
 * 축과 눈금까지 갖춘 곡선 한 판. 왼쪽 52px은 y 눈금 칸입니다.
 *
 * 계열은 준 순서대로 겹칩니다. 뒤에 준 것이 위에 그려지므로 지금 보는 실행을
 * 마지막에 넘깁니다.
 */
export function Chart({
  series,
  xMax,
  height = 260,
  xLabels,
  style,
}: {
  series: Series[];
  /** x축 오른쪽 끝(보통 총 epoch 수). */
  xMax: number;
  height?: number;
  /** 아래에 적을 두 눈금. 기본은 `epoch 1`과 `epoch {xMax}`입니다. */
  xLabels?: [ReactNode, ReactNode];
  style?: CSSProperties;
}) {
  const drawable = series.filter((item) => item.points.length > 0);
  if (drawable.length === 0) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          ...type.note,
          color: color.textMuted,
          textAlign: 'center',
        }}
      >
        epoch이 하나도 끝나지 않아 그릴 데이터가 없습니다.
      </div>
    );
  }

  const range = niceRange(drawable.flatMap((item) => item.points.map((point) => point.y)));
  const span = Math.max(xMax, 1);
  const px = (x: number) => (x / span) * VIEW_W;
  const py = (y: number) =>
    BOTTOM - ((y - range.min) / (range.max - range.min)) * (BOTTOM - TOP);

  return (
    <div style={style}>
      <div style={{ display: 'grid', gridTemplateColumns: '52px minmax(0, 1fr)', gap: '0 8px' }}>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            padding: '2px 0 4px',
            textAlign: 'right',
            font: `400 11.5px/1 ${font.mono}`,
            color: color.textMuted,
          }}
        >
          <span>{fmtAxis(range.max)}</span>
          <span>{fmtAxis(range.min)}</span>
        </div>
        <svg
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          preserveAspectRatio="none"
          style={{ width: '100%', height, display: 'block' }}
          role="img"
          aria-label={drawable.map((item) => item.label).join(', ')}
        >
          {/* 색 토큰이 `var(--color-x)` 문자열이라 presentation attribute로는 풀리지
              않습니다. stroke를 style로 얹어야 테마를 따라갑니다. */}
          {[TOP, 106, 193].map((y) => (
            <line
              key={y}
              x1="0"
              x2={VIEW_W}
              y1={y}
              y2={y}
              style={{ stroke: chartColor.grid }}
              vectorEffect="non-scaling-stroke"
            />
          ))}
          <line
            x1="0"
            x2={VIEW_W}
            y1={BOTTOM}
            y2={BOTTOM}
            style={{ stroke: chartColor.axis }}
            vectorEffect="non-scaling-stroke"
          />
          {drawable.map((item) => (
            <polyline
              key={item.label}
              points={item.points.map((point) => `${px(point.x)},${py(point.y)}`).join(' ')}
              fill="none"
              style={{ stroke: item.color }}
              strokeWidth={item.width ?? 2.6}
              strokeDasharray={item.dash}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '8px 0 0 60px',
          font: `400 11.5px/1 ${font.mono}`,
          color: color.textFaint,
        }}
      >
        <span>{xLabels?.[0] ?? 'epoch 1'}</span>
        <span>{xLabels?.[1] ?? `epoch ${span}`}</span>
      </div>
    </div>
  );
}

/** 축 위에 적는 한 줄: 무엇을 그렸는지 왼쪽, y 범위를 오른쪽. */
export function ChartHead({ label, right }: { label: ReactNode; right?: ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        gap: 24,
        marginBottom: 14,
      }}
    >
      <span style={{ ...type.microLabel, color: color.textMuted }}>{label}</span>
      {/* 오른쪽 설명은 한글이라 sans입니다. mono는 한글 낱자를 전각으로 벌려 놓습니다. */}
      {right && (
        <span style={{ font: `400 12.5px/1 ${font.sans}`, color: color.textFaint }}>{right}</span>
      )}
    </div>
  );
}

/** 학습 하나의 train/validation 손실. validation이 실선, train이 점선입니다. */
export function LossChart({
  epochs,
  totalEpochs,
  height = 230,
}: {
  epochs: EpochRecord[];
  totalEpochs: number | null;
  /** 지금 지나는 epoch. 값을 받아도 세로선을 긋지 않습니다 — 곡선 끝이 곧 지금입니다. */
  currentEpoch?: number | null;
  height?: number;
}) {
  const total = Math.max(totalEpochs ?? 0, ...epochs.map((item) => item.epoch), 1);
  const pick = (get: (item: EpochRecord) => number | null | undefined) =>
    epochs
      .filter((item) => typeof get(item) === 'number')
      .map((item) => ({ x: item.epoch, y: get(item) as number }));

  return (
    <Chart
      xMax={total}
      height={height}
      series={[
        {
          label: 'train loss',
          color: chartColor.train,
          width: 1.8,
          dash: '6 5',
          points: pick((item) => item.train_loss),
        },
        {
          label: 'validation loss',
          color: chartColor.validation,
          points: pick((item) => item.validation_loss),
        },
      ]}
    />
  );
}

/** 곡선이 무엇인지 적는 줄. 색 조각 + 이름입니다. */
export function ChartLegend({
  items = [
    { label: 'validation loss', tint: chartColor.validation },
    { label: 'train loss', tint: chartColor.train },
  ],
}: {
  items?: { label: string; tint: string }[];
}) {
  return (
    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
      {items.map((item) => (
        <span
          key={item.label}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            font: `400 12.5px/1 ${font.mono}`,
            color: color.textBody,
          }}
        >
          <span style={{ width: 18, height: 2, background: item.tint, display: 'inline-block' }} />
          {item.label}
        </span>
      ))}
    </div>
  );
}
