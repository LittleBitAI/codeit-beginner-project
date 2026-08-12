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
  EpochRecord,
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
import { datasetRelationship } from '../lib/experiments';
import { duration, loss, startedAt } from '../lib/format';
import type { RunRecord } from '../lib/records';

// evaluate가 내는 `mAP`는 IoU 0.75~0.95 평균입니다. 이름만 mAP로 적으면 흔한
// mAP@0.5로 읽혀 값이 낮아 보입니다. 다른 화면과 같은 label을 씁니다.
const MAP_LABEL = 'mAP@[0.75:0.95]';

function shown(value: string | number | boolean | null): string {
  if (value === null || value === '') return '-';
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
    { label: 'DEVICE', values: experiments.map((item) => shown(item.training.device)) },
    { label: 'EPOCHS', values: experiments.map((item) => shown(item.training.epochs)) },
    { label: 'BATCH SIZE', values: experiments.map((item) => shown(item.training.batch_size)) },
    { label: 'SEED', values: experiments.map((item) => shown(item.training.seed)) },
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
}: {
  datasetKey: string | null;
  /** 왼쪽에서 고른 dataset의 기록만 넘어옵니다. */
  records: RunRecord[];
  loading: boolean;
}) {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [compared, setCompared] = useState<ExperimentSummary[]>([]);
  const [curves, setCurves] = useState<Record<string, EpochRecord[]>>({});
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
    void api.compareExperiments(runIds).then(
      (result) => {
        if (active) {
          setCompared(result.experiments);
          setError(null);
        }
      },
      (caught: unknown) => {
        if (active) {
          setCompared([]);
          setError(caught instanceof Error ? caught.message : '비교 정보를 불러오지 못했습니다.');
        }
      },
    );
    // 곡선은 실행마다 따로 읽습니다. 하나가 없어도 나머지는 그립니다.
    for (const runId of runIds) {
      void api.experimentDetail(runId).then(
        (detail) => {
          if (!active) return;
          setCurves((current) => ({ ...current, [runId]: detail.history.epochs ?? [] }));
        },
        () => {
          if (active) setCurves((current) => ({ ...current, [runId]: [] }));
        },
      );
    }
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
    points: (curves[item.run_id] ?? [])
      .filter((epoch) => epoch.validation_loss !== null)
      .map((epoch) => ({ x: epoch.epoch, y: epoch.validation_loss as number })),
  }));
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
          <LinkAction onClick={() => navigate('/')}>← 목록</LinkAction>
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
          <Button kind="ghost" onClick={() => setShowDev((value) => !value)} style={{ flex: 'none' }}>
            개발자 모드
          </Button>
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
                <Button kind="secondary" onClick={() => navigate('/')}>
                  기록 목록으로
                </Button>
              ) : undefined
            }
          />
        ) : (
          <>
            <ChartHead label="VALIDATION LOSS" right="y 데이터 범위" />
            <Chart series={series} xMax={maxEpoch} height={260} />

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
                <SingleView
                  item={compared[0] as ExperimentSummary}
                  onSaved={() => setReloadKey((value) => value + 1)}
                />
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
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
