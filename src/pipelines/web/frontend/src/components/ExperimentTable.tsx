/**
 * 등록된 실험을 고르는 표. 실험 내역과 실험 비교가 같은 것을 씁니다.
 *
 * 예전에는 실험 비교가 카드 22개를 6열 격자에 뿌렸습니다. 카드마다 높이가 달라
 * 배지가 제각각인 높이에 떠 있었고, 가장 중요한 mAP는 줄바꿈된 mono 문장 끝에
 * 있어 세로로 훑을 수 없었습니다. 값을 견주는 화면이라면 값이 한 줄에 서야 합니다.
 */

import { useMemo, useState } from 'react';

import type { ExperimentSummary } from '../api/types';
import { color, font } from '../design/tokens';
import { completionOf } from '../lib/completion';
import { loss, startedAt } from '../lib/format';
import { EmptyState } from './primitives';

const COLUMNS = '34px 1.7fr 122px .8fr .8fr .9fr';
const HEADINGS = ['', '실행 이름', '단계', 'mAP', 'VAL LOSS', '등록'];

/** 표를 세울 기준. 값이 없는 실험은 어느 기준에서도 뒤로 보냅니다. */
export type SortKey = 'map' | 'best_validation_loss' | 'created_at';

const SORTS: { key: SortKey; label: string; pick: (item: ExperimentSummary) => number | null }[] = [
  { key: 'map', label: 'mAP 높은 순', pick: (item) => item.metrics.map },
  {
    key: 'best_validation_loss',
    label: 'VAL LOSS 낮은 순',
    pick: (item) => (item.metrics.best_validation_loss === null ? null : -item.metrics.best_validation_loss),
  },
  { key: 'created_at', label: '최근 등록 순', pick: () => null },
];

function sorted(experiments: ExperimentSummary[], key: SortKey): ExperimentSummary[] {
  const rule = SORTS.find((item) => item.key === key) ?? SORTS[0]!;
  if (key === 'created_at') return experiments;
  return [...experiments].sort((left, right) => {
    const a = rule.pick(left);
    const b = rule.pick(right);
    // 값이 없는 실험이 값 있는 실험을 이기면 안 됩니다. 항상 뒤로 보냅니다.
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    return b - a;
  });
}

/** 이 목록에서 가장 높은 mAP. 하나뿐이거나 모두 같으면 표시하지 않습니다. */
function bestMap(experiments: ExperimentSummary[]): number | null {
  const scored = experiments
    .map((item) => item.metrics.map)
    .filter((value): value is number => value !== null);
  if (scored.length < 2 || new Set(scored).size === 1) return null;
  return Math.max(...scored);
}

function metric(value: number | null): string {
  return value === null ? '-' : value.toFixed(4);
}

/** 평가와 제출을 마쳤는지. 색만으로 뜻을 전하지 않도록 글자를 함께 둡니다. */
function StagePips({ experiment }: { experiment: ExperimentSummary }) {
  const completion = completionOf(experiment);
  const stages = [
    { label: '학습', done: true },
    { label: '평가', done: completion.evaluated },
    { label: '제출', done: completion.submitted },
  ];
  return (
    <span style={{ display: 'flex', gap: 7, alignItems: 'center', padding: '9px 12px' }}>
      {stages.map((stage) => (
        <span
          key={stage.label}
          title={stage.done ? `${stage.label} 끝남` : `${stage.label} 아직`}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 3,
            font: `${stage.done ? 600 : 400} 11px/1 ${font.sans}`,
            color: stage.done ? color.tealDark : color.textFaint,
          }}
        >
          <span
            aria-hidden
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: stage.done ? color.teal : 'transparent',
              border: `1px solid ${stage.done ? color.teal : color.borderControl}`,
            }}
          />
          {stage.label}
        </span>
      ))}
    </span>
  );
}

export function ExperimentTable({
  experiments,
  selectedIds,
  onToggle,
  emptyMessage,
  selectLabel = '선택',
}: {
  experiments: ExperimentSummary[];
  selectedIds: string[];
  onToggle: (experimentId: string) => void;
  emptyMessage: string;
  /** 화면마다 고르는 뜻이 달라서 checkbox 이름을 부르는 쪽이 정합니다. */
  selectLabel?: string;
}) {
  const [sortKey, setSortKey] = useState<SortKey>('map');
  const rows = useMemo(() => sorted(experiments, sortKey), [experiments, sortKey]);
  const top = useMemo(() => bestMap(experiments), [experiments]);

  if (experiments.length === 0) return <EmptyState message={emptyMessage} />;

  return (
    <div>
      <div
        style={{
          display: 'flex',
          gap: 6,
          padding: '9px 13px',
          borderBottom: `1px solid ${color.borderInner}`,
          alignItems: 'center',
          flexWrap: 'wrap',
        }}
      >
        <span style={{ font: `500 11.5px/1 ${font.sans}`, color: color.textMuted }}>정렬</span>
        {SORTS.map((item) => (
          <button
            key={item.key}
            type="button"
            aria-pressed={sortKey === item.key}
            onClick={() => setSortKey(item.key)}
            style={{
              font: `${sortKey === item.key ? 600 : 500} 11.5px/1 ${font.sans}`,
              padding: '6px 10px',
              borderRadius: 4,
              color: sortKey === item.key ? '#fff' : color.textBody,
              background: sortKey === item.key ? color.primary : color.surface,
              border: `1px solid ${sortKey === item.key ? color.primary : color.borderControl}`,
            }}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: COLUMNS,
          background: color.surfaceTableHead,
          borderBottom: `1px solid ${color.border}`,
        }}
      >
        {HEADINGS.map((heading, index) => (
          <span
            key={heading || `spacer-${index}`}
            style={{ font: `600 11.5px/1.3 ${font.sans}`, color: '#66707E', padding: '9px 12px' }}
          >
            {heading}
          </span>
        ))}
      </div>

      {rows.map((experiment) => {
        const checked = selectedIds.includes(experiment.experiment_id);
        const isBest = top !== null && experiment.metrics.map === top;
        return (
          <label
            key={experiment.experiment_id}
            style={{
              display: 'grid',
              gridTemplateColumns: COLUMNS,
              alignItems: 'center',
              borderBottom: `1px solid ${color.borderInner}`,
              cursor: 'pointer',
              ...(checked ? { background: color.primaryTint } : {}),
            }}
            // 선택된 행은 자기 배경을 쓰고, 나머지만 global.css의 hover가 칠합니다.
            {...(checked ? {} : { 'data-row-hover': '' })}
          >
            <span style={{ padding: '9px 12px' }}>
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(experiment.experiment_id)}
                aria-label={`${experiment.run_id} ${selectLabel}`}
              />
            </span>
            <span style={{ padding: '9px 12px', display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
              <span
                style={{
                  font: `600 12.5px/1.3 ${font.sans}`,
                  color: color.text,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
                title={experiment.run_id}
              >
                {experiment.run_id}
              </span>
              <span
                style={{
                  font: `400 11px/1.35 ${font.mono}`,
                  color: color.textFaint,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {[
                  experiment.dataset.label,
                  experiment.model.architecture,
                  experiment.optimizer.name,
                  experiment.training.seed === null ? null : `seed ${experiment.training.seed}`,
                ]
                  .filter((part): part is string => Boolean(part))
                  .join(' · ')}
              </span>
            </span>
            <StagePips experiment={experiment} />
            <span
              // 비교표가 data-run/data-best를 이미 다른 뜻으로 쓰므로 이름을 겹치지 않게 둡니다.
              data-map-run={experiment.run_id}
              data-map-best={isBest ? 'true' : undefined}
              style={{
                padding: '9px 12px',
                font: `${isBest ? 600 : 400} 12px/1.3 ${font.mono}`,
                color: isBest ? color.greenDark : color.textStrong,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}
            >
              {metric(experiment.metrics.map)}
              {isBest && (
                <span
                  title="이 목록에서 가장 높습니다"
                  style={{
                    font: `600 11px/1.3 ${font.mono}`,
                    color: color.greenDark,
                    border: `1px solid ${color.green}`,
                    borderRadius: 3,
                    padding: '1px 4px',
                  }}
                >
                  최고
                </span>
              )}
            </span>
            <span style={{ padding: '9px 12px', font: `400 12px/1.3 ${font.mono}`, color: color.textStrong }}>
              {loss(experiment.metrics.best_validation_loss)}
            </span>
            <span style={{ padding: '9px 12px', font: `400 12px/1.3 ${font.mono}`, color: color.textStrong }}>
              {startedAt(experiment.created_at)}
            </span>
          </label>
        );
      })}
    </div>
  );
}
