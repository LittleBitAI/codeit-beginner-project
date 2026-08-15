/**
 * 견주기 캔버스. 실행을 골라 곡선을 겹쳐 보고, 그 아래에서 값을 나란히 봅니다.
 *
 * 하나만 고르면 그 실행의 결과 한 장이 되고, 둘 이상이면 비교표가 됩니다. 화면을
 * 갈아 끼우지 않는 이유는 "하나 보다가 하나 더 얹기"가 이 화면에서 가장 흔한
 * 동작이기 때문입니다.
 */

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type {
  CapabilityValueSource,
  ExperimentHistoryCurve,
  ExperimentSummary,
} from '../api/types';
import { Chart, ChartHead, ChartLegend, type Series } from '../components/LossChart';
import {
  AlertRow,
  Button,
  EmptyState,
  LinkAction,
  Metric,
  MetricGrid,
  MicroLabel,
  StatusBadge,
  controlStyle,
} from '../components/primitives';
import { color, font, seriesColor, type } from '../design/tokens';
import { rerunSettings } from '../lib/rerunSettings';
import { datasetRelationship } from '../lib/experiments';
import { duration, loss, startedAt } from '../lib/format';
import type { RunRecord } from '../lib/records';

// evaluate가 내는 `mAP`는 IoU 0.75~0.95 평균입니다. 이름만 mAP로 적으면 흔한
// mAP@0.5로 읽혀 값이 낮아 보입니다. 다른 화면과 같은 label을 씁니다.
const MAP_LABEL = 'mAP@[0.75:0.95]';

function shown(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'boolean') return value ? '사용' : '미사용';
  return String(value);
}

function metric(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toFixed(4);
}

function capabilityValue(value: string | null, source: CapabilityValueSource): string {
  const text = shown(value);
  return source === 'legacy_fallback' && value !== null ? `${text} (호환 기본값)` : text;
}

/** 값이 클수록 좋은지 작을수록 좋은지. loss는 낮을수록, 지표는 높을수록 좋습니다. */
type Better = 'higher' | 'lower';

interface Row {
  label: string;
  values: ReactNode[];
  /** 표식을 계산할 원본 숫자입니다. 좋고 나쁨이 있는 줄만 채웁니다. */
  numbers?: (number | null)[];
  better?: Better;
}

/**
 * 이 줄에서 가장 좋은 칸의 index입니다. 다음 경우에는 아무 칸도 고르지 않습니다.
 *
 * - 실험이 하나뿐이라 비교 대상이 없을 때
 * - 값이 있는 칸이 하나뿐일 때. 없는 값이 이기는 일은 없어야 합니다.
 * - 값이 모두 같을 때. 전부 표시되면 아무 정보가 아닙니다.
 */
function bestColumns(row: Row): Set<number> {
  const chosen = new Set<number>();
  const numbers = row.numbers;
  if (!row.better || !numbers) return chosen;
  const present = numbers.filter((value): value is number => value !== null);
  if (present.length < 2 || new Set(present).size === 1) return chosen;
  const target = row.better === 'lower' ? Math.min(...present) : Math.max(...present);
  numbers.forEach((value, index) => {
    if (value === target) chosen.add(index);
  });
  return chosen;
}

type MetricPick = (metrics: ExperimentSummary['metrics']) => number | null | undefined;

function lossRow(label: string, experiments: ExperimentSummary[], pick: MetricPick): Row {
  const numbers = experiments.map((item) => pick(item.metrics) ?? null);
  return { label, values: numbers.map((value) => loss(value)), numbers, better: 'lower' };
}

function metricRow(label: string, experiments: ExperimentSummary[], pick: MetricPick): Row {
  const numbers = experiments.map((item) => pick(item.metrics) ?? null);
  return { label, values: numbers.map((value) => metric(value)), numbers, better: 'higher' };
}

/**
 * 고른 실행들이 같은 데이터로 돌았는지 한 줄.
 *
 * 이 화면은 **한 dataset 안에서만** 고를 수 있으므로 이름은 이미 같습니다. 그래서
 * `different`는 "다른 dataset이 섞였다"가 아니라 **같은 이름인데 기록된 artifact
 * URI가 다르다**는 뜻입니다. 저장 위치가 s3와 로컬로 갈렸거나, 같은 이름으로 다시
 * 만든 dataset이거나 둘 중 하나입니다. 앞의 말로 적으면 사람이 목록을 잘못
 * 골랐다고 읽어서, 실제 원인인 "다시 만든 dataset"을 놓칩니다.
 */
function relationNote(experiments: ExperimentSummary[]): string {
  if (experiments.length < 2) return '';
  const relation = datasetRelationship(experiments);
  if (relation === 'same')
    return 'data artifact URI 4개가 모두 같아 나란히 견줄 수 있습니다. 파일 내용까지 같다는 뜻은 아닙니다.';
  if (relation === 'different')
    return 'dataset 이름은 같은데 기록된 artifact URI가 서로 다릅니다. 저장 위치(s3·로컬)가 다르거나 같은 이름으로 다시 만든 dataset일 수 있어, 결과 차이에 데이터 차이가 섞여 있을 수 있습니다.';
  return '고른 기록 중 하나 이상에 data artifact 4개가 다 남아 있지 않아 같은 데이터인지 판정할 수 없습니다.';
}

function resultRows(experiments: ExperimentSummary[]): Row[] {
  return [
    metricRow('KAGGLE', experiments, (item) => item.kaggle_score),
    lossRow('BEST VAL LOSS', experiments, (item) => item.best_validation_loss),
    // BEST EPOCH는 몇 번째였는지일 뿐이라 크고 작음에 좋고 나쁨이 없습니다.
    { label: 'BEST EPOCH', values: experiments.map((item) => shown(item.metrics.best_epoch)) },
    lossRow('FINAL TRAIN LOSS', experiments, (item) => item.final_train_loss),
    lossRow('FINAL VAL LOSS', experiments, (item) => item.final_validation_loss),
    metricRow(MAP_LABEL, experiments, (item) => item.map),
    metricRow('mAP@0.5', experiments, (item) => item.map50),
    metricRow('mAP@0.75', experiments, (item) => item.map75),
    metricRow('Precision@IoU0.5', experiments, (item) => item.precision50),
    metricRow('Recall@IoU0.5', experiments, (item) => item.recall50),
  ];
}

/**
 * 중첩 설정을 한 줄로 폅니다. `cosine · warmup 1000 · min_lr_factor 0.01`처럼요.
 *
 * 안쪽 key를 골라 적지 않고 있는 대로 다 폅니다. 고르면 그때 몰랐던 값이 화면에서
 * 사라지는데, 이 화면이 고치려는 문제가 바로 그것입니다.
 */
function inline(settings: Record<string, unknown> | null | undefined): string | null {
  if (!settings) return null;
  const name = typeof settings.name === 'string' ? settings.name : null;
  const rest = Object.entries(settings)
    .filter(([key]) => key !== 'name')
    .map(([key, value]) => `${key} ${value}`);
  return [name, ...rest].filter(Boolean).join(' · ') || null;
}

function settingRows(experiments: ExperimentSummary[]): Row[] {
  return [
    {
      label: '상태',
      values: experiments.map((item) => (
        <StatusBadge key={item.experiment_id} status={item.status} label={item.status_label} />
      )),
    },
    { label: 'DATASET', values: experiments.map((item) => shown(item.dataset.label)) },
    {
      label: '모델',
      values: experiments.map((item) => capabilityValue(item.model.architecture, item.model.source)),
    },
    { label: 'PRETRAINED', values: experiments.map((item) => shown(item.model.pretrained)) },
    {
      label: 'OPTIMIZER',
      values: experiments.map((item) => capabilityValue(item.optimizer.name, item.optimizer.source)),
    },
    { label: '증강 preset', values: experiments.map((item) => shown(item.training.augmentation?.preset)) },
    { label: 'DEVICE', values: experiments.map((item) => shown(item.training.device)) },
    { label: '정밀도', values: experiments.map((item) => shown(item.training.precision)) },
    { label: 'EPOCHS', values: experiments.map((item) => shown(item.training.epochs)) },
    { label: 'BATCH SIZE', values: experiments.map((item) => shown(item.training.batch_size)) },
    {
      label: 'ACCUMULATION',
      values: experiments.map((item) => shown(item.training.gradient_accumulation_steps)),
    },
    { label: '입력 크기', values: experiments.map((item) => shown(item.training.input_size)) },
    { label: 'CHECKPOINT 주기', values: experiments.map((item) => shown(item.training.checkpoint_every)) },
    // 설정이 같아 보여도 한쪽만 남의 checkpoint에서 출발했으면 다른 실험입니다.
    { label: '이어서 학습', values: experiments.map((item) => shown(item.training.resume_from)) },
    { label: 'SEED', values: experiments.map((item) => shown(item.training.seed)) },
    { label: 'LR SCHEDULE', values: experiments.map((item) => shown(inline(item.training.lr_scheduler))) },
    { label: '조기 종료', values: experiments.map((item) => shown(inline(item.training.early_stopping))) },
    { label: 'LEARNING RATE', values: experiments.map((item) => shown(item.optimizer.learning_rate)) },
    { label: 'MOMENTUM', values: experiments.map((item) => shown(item.optimizer.momentum)) },
    { label: 'WEIGHT DECAY', values: experiments.map((item) => shown(item.optimizer.weight_decay)) },
    { label: 'BETA 1', values: experiments.map((item) => shown(item.optimizer.beta1)) },
    { label: 'BETA 2', values: experiments.map((item) => shown(item.optimizer.beta2)) },
    { label: 'EPSILON', values: experiments.map((item) => shown(item.optimizer.epsilon)) },
    { label: '경과 시간', values: experiments.map((item) => duration(item.elapsed_seconds)) },
    {
      label: '시작',
      values: experiments.map((item) => startedAt(item.started_at ?? item.created_at)),
    },
  ];
}

/**
 * 고른 실행들의 약한 class를 한 표에 나란히 놓습니다.
 *
 * class를 세로로 두고 실행을 가로로 두어, 어느 실행이 어느 알약을 개선했는지 한 줄로
 * 읽힙니다.
 *
 * 값이 없는 칸은 세 가지 뜻이 있고 **셋을 구분해서 적습니다.** evaluate가 주는 목록은
 * 상위 `top_n`개로 잘려 있어서, 목록에 없다고 "약하지 않다"고 말할 수 없기 때문입니다.
 * 그렇게 말해 버리면 실제로는 약한데 순위 밖인 class가 괜찮은 것으로 읽힙니다.
 *
 * - 숫자: 그 실행에서 약했고 AP를 쟀습니다.
 * - `미측정`: 약한 목록에 있는데 AP를 재지 못했습니다.
 * - `순위 밖`: 목록이 잘려 있어 약한지 아닌지 이 화면이 알 수 없습니다.
 * - `class 요약 없음`: 그 실행에는 class별 요약이 없어 말할 자료가 없습니다.
 *   평가 전일 수도, 이 요약이 생기기 전에 등록된 기록일 수도 있습니다.
 * - `약하지 않음`: 목록이 잘리지 않았고 거기 없으므로 약하지 않았습니다.
 *
 * 기호 대신 글자로 적습니다. `-`나 `?`는 표 위 설명을 읽지 않으면 뜻을 알 수 없고,
 * 이 표에서 가장 위험한 오해가 "말할 수 없는 것"을 "괜찮다"로 읽는 것입니다.
 *
 * 약한지 아닌지는 evaluate가 이미 정해 둔 것을 그대로 씁니다. 여기서 다시 세면 이
 * 화면과 evaluate가 서로 다른 답을 말합니다.
 */
/**
 * 한 칸에 무엇을 적을지. 값이 없는 세 경우를 서로 다르게 적습니다.
 *
 * 잘린 목록에 없는 것을 `-`로 적으면 "약하지 않다"고 단정하는 셈인데, 그 말을 할
 * 근거가 화면에 없습니다.
 */
function cell(
  values: Map<number, number | null>,
  id: number,
  { truncated, summarized }: { truncated: boolean; summarized: boolean },
): string {
  const value = values.get(id);
  if (typeof value === 'number') return value.toFixed(3);
  if (values.has(id)) return '미측정';
  // class 요약이 없으면 약한지 아닌지 말할 자료가 없습니다. "평가 없음"이라고는
  // 하지 않습니다 — 이 요약은 나중에 생긴 값이라, 평가는 했지만 그 전에 등록된
  // 기록도 여기 걸립니다. 같은 화면이 mAP를 보여 주면서 평가를 안 했다고 말하면
  // 둘 중 하나는 거짓말입니다.
  if (!summarized) return 'class 요약 없음';
  return truncated ? '순위 밖' : '약하지 않음';
}

function WeakClassTable({ experiments }: { experiments: ExperimentSummary[] }) {
  const measured = experiments.filter((item) => item.per_class_summary);

  const names = new Map<number, string>();
  for (const item of measured) {
    for (const row of item.per_class_summary?.weak ?? []) names.set(row.category_id, row.name);
  }
  /**
   * 그릴 줄이 없을 때. 표를 감추지 않고 **왜** 비었는지 적습니다.
   *
   * 요약이 있는 실행과 없는 실행이 섞이면 두 문장이 다 나와야 합니다. "약한 class가
   * 없다"만 적으면 요약이 없어 판단조차 못 한 실행이 좋은 결과로 읽힙니다.
   */
  if (names.size === 0) {
    const unsummarized = experiments.filter((item) => !item.per_class_summary);
    return (
      <div style={{ marginTop: 34 }}>
        <div style={{ ...type.subTitle, color: color.text }}>약한 class</div>
        <div style={{ ...type.note, color: color.textMuted, marginTop: 6, maxWidth: '46em' }}>
          {measured.length > 0 &&
            `정답이 ${measured[0]?.per_class_summary?.min_truth_count}개 이상인 class 중 AP가 낮은 것이 없습니다.`}
          {unsummarized.length > 0 &&
            ` ${unsummarized.map((item) => item.run_id).join(', ')}에는 class별 요약이 없어 약한지 아닌지 알 수 없습니다. 평가를 다시 실행해 등록하면 채워집니다 — 등록만으로는 생기지 않습니다.`}
        </div>
      </div>
    );
  }

  const byRun = experiments.map(
    (item) =>
      new Map((item.per_class_summary?.weak ?? []).map((row) => [row.category_id, row.ap])),
  );
  /**
   * 그 실행의 약한 class 목록이 잘려 있는지.
   *
   * `counts.weak`는 자르기 전 개수라, 목록 길이보다 크면 상위 몇 개만 온 것입니다.
   * 그때는 목록에 없다는 것이 "약하지 않다"는 뜻이 되지 못합니다.
   */
  const truncated = experiments.map((item) => {
    const summary = item.per_class_summary;
    return summary ? summary.counts.weak > summary.weak.length : false;
  });
  /** 그 실행에 class별 요약이 있는지. 없으면 약한지 아닌지 말할 자료가 없습니다. */
  const summarized = experiments.map((item) => Boolean(item.per_class_summary));
  /**
   * 모든 실행 중 가장 낮은 AP를 기준으로 세웁니다.
   *
   * 첫 실행만 보고 세우면 고른 순서를 바꿨을 뿐인데 표의 줄 순서가 달라집니다.
   * 어느 실행에서도 재지 못한 class는 뒤로 보내고, 그다음은 category_id로 갈라
   * 같은 입력이면 늘 같은 순서가 나오게 합니다.
   */
  const worst = (id: number): number => {
    const values = byRun
      .map((run) => run.get(id))
      .filter((value): value is number => typeof value === 'number');
    return values.length > 0 ? Math.min(...values) : Number.POSITIVE_INFINITY;
  };
  const ids = [...names.keys()].sort((left, right) => worst(left) - worst(right) || left - right);
  const columns = `170px repeat(${experiments.length}, minmax(150px, 1fr))`;

  return (
    <div style={{ marginTop: 34 }}>
      <div style={{ ...type.subTitle, color: color.text }}>약한 class</div>
      <div style={{ ...type.note, color: color.textMuted, marginTop: 6, maxWidth: '46em' }}>
        정답이 {measured[0]?.per_class_summary?.min_truth_count}개 이상인데 AP가 낮은 class입니다.
        표본이 적어 AP를 믿을 수 없는 class는 evaluate가 따로 세어 두므로 여기 넣지 않습니다.
        {' 잰 AP가 없으면 그 이유를 그대로 적습니다: 미측정, 순위 밖(목록이 잘려 알 수 없음), class 요약 없음, 약하지 않음.'}
      </div>
      <div style={{ overflowX: 'auto', marginTop: 14 }}>
        <div style={{ minWidth: 170 + experiments.length * 150 }}>
          {/* 어느 칸이 어느 실행인지. 가로로 밀면 이름 없이는 알 수 없습니다. */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: columns,
              gap: '0 18px',
              paddingBottom: 12,
              borderBottom: `1px solid ${color.border}`,
            }}
          >
            <span />
            {experiments.map((item, index) => (
              <span
                key={item.experiment_id}
                style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}
              >
                <span
                  style={{
                    width: 18,
                    height: 2,
                    flex: 'none',
                    background: seriesColor[index % seriesColor.length] as string,
                  }}
                />
                <span
                  style={{
                    ...type.monoSpec,
                    color: color.textMuted,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {item.run_id}
                </span>
              </span>
            ))}
          </div>
          {ids.map((id) => (
            <div
              key={id}
              style={{
                display: 'grid',
                gridTemplateColumns: columns,
                gap: '0 18px',
                padding: '12px 0',
                borderBottom: `1px solid ${color.borderRow}`,
              }}
            >
              <span style={{ ...type.note, color: color.textMuted, overflowWrap: 'break-word' }}>
                {names.get(id)}
              </span>
              {byRun.map((values, index) => (
                <span
                  key={experiments[index]?.experiment_id ?? index}
                  style={{
                    ...type.monoSpec,
                    color: typeof values.get(id) === 'number' ? color.text : color.textFaint,
                  }}
                >
                  {cell(values, id, {
                    truncated: truncated[index] ?? false,
                    summarized: summarized[index] ?? false,
                  })}
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function CompareTable({ experiments }: { experiments: ExperimentSummary[] }) {
  const rows = [...resultRows(experiments), ...settingRows(experiments)];
  const columns = `170px repeat(${experiments.length}, minmax(150px, 1fr))`;

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ minWidth: 170 + experiments.length * 150 }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: columns,
            gap: '0 18px',
            paddingBottom: 14,
            borderBottom: `1px solid ${color.border}`,
          }}
        >
          <span />
          {experiments.map((item, index) => (
            <span
              key={item.experiment_id}
              style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}
            >
              <span
                style={{
                  width: 18,
                  height: 2,
                  flex: 'none',
                  background: seriesColor[index % seriesColor.length],
                }}
              />
              <span
                style={{
                  font: `500 13px/1.4 ${font.mono}`,
                  color: color.text,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                title={item.run_id}
              >
                {item.run_id}
              </span>
            </span>
          ))}
        </div>

        {rows.map((row) => {
          const best = bestColumns(row);
          return (
            <div
              key={row.label}
              style={{
                display: 'grid',
                gridTemplateColumns: columns,
                gap: '0 18px',
                padding: '14px 0',
                borderBottom: `1px solid ${color.borderRow}`,
                alignItems: 'baseline',
              }}
            >
              <span style={{ ...type.bodySmall, color: color.textMuted, whiteSpace: 'nowrap' }}>
                {row.label}
              </span>
              {row.values.map((value, index) => {
                const isBest = best.has(index);
                return (
                  <span
                    key={`${row.label}-${experiments[index]?.experiment_id ?? index}`}
                    data-row={row.label}
                    data-run={experiments[index]?.run_id ?? ''}
                    data-best={isBest ? 'true' : undefined}
                    style={{
                      font: `${isBest ? 600 : 400} 13px/1.4 ${font.mono}`,
                      color: isBest ? color.accent : color.text,
                      fontVariantNumeric: 'tabular-nums',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      minWidth: 0,
                      overflowWrap: 'break-word',
                    }}
                  >
                    {value}
                    {/* 색만으로 뜻을 전하지 않도록 글자 표식을 함께 둡니다. */}
                    {isBest && (
                      <span
                        title={
                          row.better === 'lower'
                            ? '이 비교에서 가장 낮습니다'
                            : '이 비교에서 가장 높습니다'
                        }
                        style={{ ...type.badge, color: color.accent }}
                      >
                        최고
                      </span>
                    )}
                  </span>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Kaggle 점수를 사람이 적는 칸.
 *
 * 이 값은 자동으로 채워지지 않습니다. 제출하고 점수를 받은 사람이 직접 옮겨 적어야
 * 하고, 그래야 '제출 완료'로 셉니다. **이미 적힌 값은 잠가 둡니다** — 표를 지나가다
 * 누른 저장이 기록을 갈아치우면 그 값이 무엇이었는지 아무도 모릅니다.
 */
function KaggleScoreField({ item, onSaved }: { item: ExperimentSummary; onSaved: () => void }) {
  const recorded = item.metrics.kaggle_score ?? null;
  const [editing, setEditing] = useState(recorded === null);
  const [text, setText] = useState(recorded === null ? '' : String(recorded));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    const value = Number(text.trim());
    if (text.trim() === '' || !Number.isFinite(value)) {
      setError('숫자를 적어 주세요.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      // 이미 적힌 값을 고치는 경우에만 덮어쓰기를 요청합니다. 서버는 그 말이 없는
      // 요청으로는 기록된 점수를 바꾸지 않습니다.
      await api.saveKaggleScore(item.run_id, value, recorded !== null);
      setEditing(false);
      onSaved();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '점수를 저장하지 못했습니다.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div style={{ ...type.fieldLabel, color: color.textMuted, marginBottom: 12 }}>KAGGLE</div>
      {editing ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <input
            value={text}
            inputMode="decimal"
            placeholder="0.6123"
            aria-label="Kaggle 점수"
            onChange={(event) => setText(event.target.value)}
            style={{ ...controlStyle, width: 140 }}
          />
          <Button kind="primary" disabled={busy} onClick={() => void save()}>
            {busy ? '저장 중…' : '저장'}
          </Button>
          {recorded !== null && (
            <LinkAction
              tone="muted"
              onClick={() => {
                setText(String(recorded));
                setError(null);
                setEditing(false);
              }}
            >
              취소
            </LinkAction>
          )}
        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, flexWrap: 'wrap' }}>
          <span style={{ ...type.kpiLarge, color: color.accent }}>{metric(recorded)}</span>
          <LinkAction onClick={() => setEditing(true)}>고치기</LinkAction>
        </div>
      )}
      {error && (
        <div style={{ ...type.note, color: color.danger, marginTop: 8 }}>{error}</div>
      )}
    </div>
  );
}

/** 하나만 골랐을 때. 큰 숫자 둘과 나머지 지표를 펼칩니다. */
function SingleView({ item, onSaved }: { item: ExperimentSummary; onSaved: () => void }) {
  return (
    <div style={{ paddingTop: 26, borderTop: `1px solid ${color.border}` }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: 20,
          marginBottom: 6,
        }}
      >
        <span style={{ ...type.subTitle, color: color.text }}>
          {item.model.architecture ?? '(모델 모름)'}
        </span>
        <span style={{ ...type.monoSpec, color: color.textMuted }}>
          {startedAt(item.finished_at ?? item.started_at ?? item.created_at)}
        </span>
      </div>
      <div style={{ ...type.monoId, color: color.textMuted, marginBottom: 26, overflowWrap: 'break-word' }}>
        {item.run_id}
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 40,
          flexWrap: 'wrap',
          paddingBottom: 26,
          borderBottom: `1px solid ${color.border}`,
        }}
      >
        <KaggleScoreField item={item} onSaved={onSaved} />
        <div>
          <div style={{ ...type.fieldLabel, color: color.textMuted, marginBottom: 12 }}>
            BEST VAL LOSS
          </div>
          <div style={{ ...type.kpiLarge, color: color.text }}>
            {loss(item.metrics.best_validation_loss)}
          </div>
        </div>
        <div style={{ ...type.bodySmall, color: color.textMuted, maxWidth: '22em' }}>
          {item.status_label} · {duration(item.elapsed_seconds)} 걸렸습니다.
          {item.metrics.kaggle_score == null &&
            ' Kaggle 점수는 사람이 제출한 뒤 직접 적어야 채워집니다.'}
        </div>
      </div>

      <MetricGrid min={150} style={{ gap: '22px 26px', padding: '26px 0' }}>
        <Metric label={MAP_LABEL} value={metric(item.metrics.map)} />
        <Metric label="mAP@0.5" value={metric(item.metrics.map50)} />
        <Metric label="mAP@0.75" value={metric(item.metrics.map75)} />
        <Metric label="Precision@IoU0.5" value={metric(item.metrics.precision50)} />
        <Metric label="Recall@IoU0.5" value={metric(item.metrics.recall50)} />
        <Metric label="BEST EPOCH" value={shown(item.metrics.best_epoch)} />
        <Metric label="FINAL TRAIN LOSS" value={loss(item.metrics.final_train_loss)} />
        <Metric label="FINAL VAL LOSS" value={loss(item.metrics.final_validation_loss)} />
      </MetricGrid>

      <div style={{ ...type.body, color: color.textMuted, paddingTop: 6, maxWidth: '46em', textWrap: 'pretty' }}>
        왼쪽 목록에서 하나 더 고르면 나란히 견줍니다. 순위를 말할 수 있는 숫자는 Kaggle
        점수뿐입니다 — 로컬 mAP는 같은 dataset의 val로 잰 값이라 제출 점수와 다릅니다.
      </div>
    </div>
  );
}

export function Canvas({
  datasetKey,
  records,
  loading,
  onScoreSaved,
  onNewExperiment,
}: {
  datasetKey: string | null;
  /** 왼쪽에서 고른 dataset의 기록만 넘어옵니다. */
  records: RunRecord[];
  loading: boolean;
  /**
   * Kaggle 점수를 적은 뒤 기록 목록도 다시 읽게 합니다.
   *
   * 이 화면만 새로 읽으면 목록의 점수와 '제출 완료' 필터는 다음 polling(60초)까지
   * 옛 값으로 남습니다. 방금 적은 점수가 목록에서 `-`로 보이면 저장이 안 된 줄 압니다.
   */
  onScoreSaved: () => void;
  /**
   * '이 세팅으로 학습하기'가 그 실행의 설정을 담아 새 실험 화면을 엽니다.
   *
   * 초안은 App이 들고 있으므로 이 화면은 값을 만들어 넘기기만 합니다. 화면이
   * 초안을 직접 만지면 이 화면만 따로 그려 보는 것도 못 하게 됩니다.
   */
  onNewExperiment: (settings: Record<string, string>) => void;
}) {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [compared, setCompared] = useState<ExperimentSummary[]>([]);
  const [curves, setCurves] = useState<Record<string, ExperimentHistoryCurve>>({});
  const [error, setError] = useState<string | null>(null);
  const [showDev, setShowDev] = useState(false);

  /**
   * 견줄 수 있는 것은 registry에 등록된 실행뿐입니다.
   *
   * 비교와 곡선은 둘 다 `run_id`로 registry를 읽습니다. 등록 전 실행을 목록에
   * 올리면 눌렀을 때 조용히 빈 표가 나옵니다. 대신 몇 건이 아직 등록되지 않았는지
   * 아래에 적어, 없는 것이 아니라 아직 못 오른 것임을 말합니다.
   */
  const selectable = useMemo(() => records.filter((item) => item.registered), [records]);
  const unregistered = records.length - selectable.length;
  const picked = useMemo(() => params.getAll('run'), [params]);
  /**
   * 하나만 골랐고 그것을 이 컴퓨터가 돌렸다면 그 job의 id입니다.
   *
   * 여러 개를 겹쳐 놓았을 때는 어느 실행의 로그인지 말할 수 없어 내지 않습니다.
   */
  const localJobId = useMemo(
    () =>
      picked.length === 1
        ? (records.find((item) => item.runId === picked[0])?.jobId ?? null)
        : null,
    [picked, records],
  );

  /**
   * 선택을 **값**으로 굳혀 effect의 의존성으로 씁니다.
   *
   * 목록은 polling이라 그때마다 새 배열이 만들어집니다. 배열 자체를 의존성으로
   * 두면 내용이 그대로여도 effect가 다시 돌고, 아직 오지 않은 응답을 정리 함수가
   * 버립니다. 비교가 polling 주기보다 오래 걸리면 표가 영원히 채워지지 않습니다.
   */
  const pickedKey = picked.join('\n');
  // Kaggle 점수를 적고 나면 그 값을 다시 읽어야 화면에 반영됩니다.
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const runIds = pickedKey ? pickedKey.split('\n') : [];
    if (runIds.length === 0) {
      setCompared([]);
      setCurves({});
      setError(null);
      return;
    }
    let active = true;
    // 표와 곡선을 요청 하나로 받습니다. 실행마다 상세를 또 부르면 서버가 그때마다
    // registry index 전체를 훑고, 곡선에 쓰지 않는 평가 결과까지 함께 옵니다.
    void api.compareExperiments(runIds).then(
      (result) => {
        if (active) {
          setCompared(result.experiments);
          setCurves(result.curves ?? {});
          setError(null);
        }
      },
      (caught: unknown) => {
        if (active) {
          setCompared([]);
          setCurves({});
          setError(caught instanceof Error ? caught.message : '비교 정보를 불러오지 못했습니다.');
        }
      },
    );
    return () => {
      active = false;
    };
  }, [pickedKey, reloadKey]);

  function toggle(runId: string) {
    const next = picked.includes(runId)
      ? picked.filter((value) => value !== runId)
      : [...picked, runId];
    setParams(next.length === 0 ? {} : { run: next }, { replace: true });
  }

  const series: Series[] = compared.map((item, index) => ({
    label: item.run_id,
    color: seriesColor[index % seriesColor.length] as string,
    points: (curves[item.run_id]?.epochs ?? [])
      .filter((epoch) => epoch.validation_loss !== null)
      .map((epoch) => ({ x: epoch.epoch, y: epoch.validation_loss as number })),
  }));
  /**
   * 선이 그려지지 않은 실행마다 그 이유 한 줄입니다.
   *
   * 이유는 셋입니다: 기록 파일을 못 읽었거나, 아직 한 epoch도 안 끝났거나, epoch은
   * 끝났는데 validation loss가 없거나. 그림 하나에 하나만 적으면 **섞인 경우**를
   * 놓칩니다 — 둘 중 하나만 실패하면 그림은 나머지를 그리고, 사라진 선은 아무 말도
   * 없이 사라집니다. 그래서 실행별로 답니다.
   */
  const curveNotes = compared
    .map((item, index) => {
      if ((series[index]?.points.length ?? 0) > 0) return null;
      const curve = curves[item.run_id];
      if (!curve) return null;
      if (curve.available === false) {
        return { runId: item.run_id, note: curve.reason ?? '학습 기록을 읽지 못했습니다.' };
      }
      return {
        runId: item.run_id,
        note:
          (curve.epochs ?? []).length === 0
            ? '아직 한 epoch도 끝나지 않았습니다.'
            : 'validation loss가 기록되지 않았습니다.',
      };
    })
    .filter((note): note is { runId: string; note: string } => note !== null);
  // 하나도 못 그릴 때 그림 자리에 들어갈 말입니다. 이유는 아래 줄들이 말하므로 여기서
  // 원인을 단정하지 않습니다 — 기본 문구는 "epoch이 하나도 끝나지 않았다"입니다.
  const nothingDrawn = series.every((item) => item.points.length === 0);
  const maxEpoch = Math.max(
    1,
    ...series.flatMap((item) => item.points.map((point) => point.x)),
    ...compared.map((item) => item.training.epochs ?? 0),
  );

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 236px) minmax(0, 1fr)',
        minHeight: '100vh',
      }}
    >
      <div style={{ borderRight: `1px solid ${color.border}`, padding: '30px 0 24px', background: color.sheet }}>
        <div style={{ padding: '0 22px 20px' }}>
          <LinkAction onClick={() => navigate('/records')}>← 기록</LinkAction>
          <div style={{ ...type.subTitle, color: color.text, marginTop: 16 }}>견줄 실행</div>
          {/* 어느 dataset 안에서 고르는 중인지 늘 적습니다. 이 목록은 그 dataset의
              기록만 담습니다 — 데이터가 다른 실행을 나란히 세우면 모델 차이인지
              데이터 차이인지 구별할 수 없습니다. */}
          <div style={{ ...type.monoSpec, color: color.textMuted, marginTop: 6, overflowWrap: 'break-word' }}>
            {datasetKey ?? 'dataset 없음'}
          </div>
          <div style={{ ...type.note, color: color.textMuted, marginTop: 6 }}>
            클릭해서 겹치거나 뺍니다
          </div>
        </div>

        {selectable.length === 0 ? (
          <div style={{ padding: '0 22px', ...type.note, color: color.textFaint, textWrap: 'pretty' }}>
            {loading
              ? '기록을 불러오고 있습니다.'
              : records.length === 0
                ? '이 dataset에는 기록이 없습니다.'
                : `이 dataset의 기록 ${records.length}건이 아직 registry에 등록되지 않았습니다. 학습이 성공으로 끝나고 평가까지 돌아야 등록되고, 그때부터 여기에서 견줄 수 있습니다.`}
          </div>
        ) : (
          selectable.map((item) => {
            const index = picked.indexOf(item.runId);
            const on = index >= 0;
            return (
              <button
                key={item.runId}
                type="button"
                aria-pressed={on}
                data-row-hover={on ? undefined : ''}
                onClick={() => toggle(item.runId)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  padding: '14px 22px',
                  borderTop: `1px solid ${color.borderRow}`,
                  borderLeft: 0,
                  borderRight: 0,
                  borderBottom: 0,
                  background: on ? color.fill : 'transparent',
                }}
              >
                <span style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 5 }}>
                  <span
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      flex: 'none',
                      marginTop: 4,
                      background: on ? (seriesColor[index % seriesColor.length] as string) : 'transparent',
                      border: `1px solid ${on ? 'transparent' : color.border}`,
                    }}
                  />
                  <span
                    style={{
                      font: `500 13.5px/1.45 ${font.sans}`,
                      color: on ? color.text : color.textBody,
                      minWidth: 0,
                      textWrap: 'pretty',
                    }}
                  >
                    {item.family}
                  </span>
                </span>
                <span
                  style={{
                    display: 'block',
                    font: `400 12.5px/1.5 ${font.mono}`,
                    color: color.textMuted,
                    paddingLeft: 20,
                    marginBottom: 4,
                    overflowWrap: 'break-word',
                  }}
                >
                  {item.runId}
                </span>
                <span
                  style={{
                    display: 'block',
                    ...type.note,
                    color: color.textMuted,
                    paddingLeft: 20,
                  }}
                >
                  {[
                    item.metrics.kaggle === null ? null : `Kaggle ${metric(item.metrics.kaggle)}`,
                    item.metrics.bestValidationLoss === null
                      ? null
                      : `val ${loss(item.metrics.bestValidationLoss)}`,
                  ]
                    .filter((part): part is string => part !== null)
                    .join(' · ') || '지표 없음'}
                </span>
              </button>
            );
          })
        )}

        {/* 감춘 것이 있으면 몇 건인지 말합니다. 조용히 빼면 그만큼이 없는 줄 압니다. */}
        {selectable.length > 0 && unregistered > 0 && (
          <div
            style={{
              padding: '16px 22px 0',
              ...type.note,
              color: color.textFaint,
              textWrap: 'pretty',
            }}
          >
            아직 registry에 등록되지 않은 {unregistered}건은 목록에 없습니다. 평가까지 끝나야
            견줄 수 있습니다.
          </div>
        )}
      </div>

      <div style={{ padding: '32px 36px 48px', minWidth: 0 }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            gap: 28,
            marginBottom: 30,
          }}
        >
          <div>
            <MicroLabel style={{ marginBottom: 12 }}>
              {picked.length === 0
                ? '고른 실행 없음'
                : picked.length === 1
                  ? '실행 하나'
                  : `실행 ${picked.length}개 겹침`}
            </MicroLabel>
            {compared.length > 0 && (
              <ChartLegend
                items={compared.map((item, index) => ({
                  label: item.run_id,
                  tint: seriesColor[index % seriesColor.length] as string,
                }))}
              />
            )}
          </div>
          <div style={{ display: 'flex', gap: 10, flex: 'none', alignItems: 'center' }}>
            {/* 이 컴퓨터가 돌린 실행이면 로그와 산출물이 남아 있습니다. 기록에서
                누르면 이제 이 화면으로 오므로, 그 화면으로 가는 길을 여기 둡니다.
                주소를 직접 치게 두면 있는 것을 없는 것처럼 만듭니다. */}
            {localJobId && (
              <LinkAction onClick={() => navigate(`/monitor/${localJobId}`)}>
                로그 보기
              </LinkAction>
            )}
            {/* 하나만 골랐을 때만 냅니다. 여러 개를 겹쳐 놓고 누르면 어느 설정이
                실렸는지 화면이 말해 주지 못합니다. */}
            {compared.length === 1 && compared[0] && (
              <Button
                kind="secondary"
                onClick={() => onNewExperiment(rerunSettings(compared[0] as ExperimentSummary))}
              >
                이 세팅으로 학습하기
              </Button>
            )}
            <Button kind="ghost" onClick={() => setShowDev((value) => !value)}>
              개발자 모드
            </Button>
          </div>
        </div>

        {error && (
          <div style={{ marginBottom: 22 }}>
            <AlertRow level="error" title="비교 정보를 불러오지 못했습니다">
              {error}
            </AlertRow>
          </div>
        )}

        {picked.length === 0 ? (
          <EmptyState
            message={
              selectable.length === 0
                ? '이 dataset에는 견줄 실험이 아직 없습니다. 평가와 등록까지 끝난 학습만 여기에 올라옵니다.'
                : '왼쪽에서 실행을 하나 이상 고르면 곡선을 겹쳐 그립니다.'
            }
            action={
              selectable.length === 0 ? (
                <Button kind="secondary" onClick={() => navigate('/records')}>
                  기록 목록으로
                </Button>
              ) : undefined
            }
          />
        ) : (
          <>
            <ChartHead label="VALIDATION LOSS" right="y 데이터 범위" />
            <Chart
              series={series}
              xMax={maxEpoch}
              height={260}
              emptyMessage={
                nothingDrawn && curveNotes.length > 0
                  ? '그릴 loss 곡선이 없습니다. 실행별 이유는 아래에 있습니다.'
                  : undefined
              }
            />
            {curveNotes.length > 0 && (
              <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {curveNotes.map((item) => (
                  <span key={item.runId} style={{ ...type.note, color: color.textMuted }}>
                    <span style={{ fontFamily: font.mono }}>{item.runId}</span> — {item.note}
                  </span>
                ))}
              </div>
            )}

            {showDev && (
              <div style={{ background: color.panel, padding: '18px 20px', margin: '26px 0' }}>
                <MicroLabel style={{ marginBottom: 12 }}>RAW METRICS</MicroLabel>
                <pre
                  style={{
                    ...type.code,
                    color: color.textBody,
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    overflowWrap: 'break-word',
                  }}
                >
                  {JSON.stringify(
                    compared.map((item) => ({
                      run_id: item.run_id,
                      training: item.training,
                      optimizer: item.optimizer,
                      metrics: item.metrics,
                    })),
                    null,
                    2,
                  )}
                </pre>
              </div>
            )}

            <div style={{ marginTop: 26 }}>
              {compared.length === 0 ? (
                <EmptyState message="고른 실행의 기록을 불러오고 있습니다." />
              ) : compared.length === 1 ? (
                <>
                  <SingleView
                    item={compared[0] as ExperimentSummary}
                    onSaved={() => {
                      setReloadKey((value) => value + 1);
                      onScoreSaved();
                    }}
                  />
                  {/* 하나만 골랐을 때가 오히려 약한 class를 가장 보고 싶은 자리입니다.
                      여럿을 겹쳤을 때만 내면, 기록에서 실행 하나를 눌러 들어온 사람은
                      끝내 못 봅니다. */}
                  <WeakClassTable experiments={compared} />
                </>
              ) : (
                <div style={{ paddingTop: 26, borderTop: `1px solid ${color.border}` }}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'baseline',
                      justifyContent: 'space-between',
                      gap: 20,
                      marginBottom: 22,
                      flexWrap: 'wrap',
                    }}
                  >
                    <span style={{ ...type.subTitle, color: color.text }}>
                      견주기 {compared.length}개
                    </span>
                    <span style={{ ...type.note, color: color.textMuted, maxWidth: '34em' }}>
                      {relationNote(compared)}
                    </span>
                  </div>
                  <CompareTable experiments={compared} />
                  <div
                    style={{
                      ...type.body,
                      color: color.textMuted,
                      marginTop: 18,
                      maxWidth: '46em',
                      textWrap: 'pretty',
                    }}
                  >
                    표식이 붙은 칸은 이 비교 안에서 가장 좋은 값입니다. 값이 하나뿐이거나 모두
                    같으면 아무 칸도 고르지 않습니다 — 비교 대상이 없을 때의 1등은 뜻이 없습니다.
                  </div>
                  <WeakClassTable experiments={compared} />
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
