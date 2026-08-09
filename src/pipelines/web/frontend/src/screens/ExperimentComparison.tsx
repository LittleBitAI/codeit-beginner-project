import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { api } from '../api/client';
import type { CapabilityValueSource, ExperimentSummary } from '../api/types';
import { ExperimentTable } from '../components/ExperimentTable';
import { AlertRow, Button, EmptyState, Panel, ScreenIntro, StatusBadge } from '../components/primitives';
import { color, font, radius, type } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { datasetRelationship } from '../lib/experiments';
import { duration, loss, startedAt } from '../lib/format';

// evaluate가 내는 `mAP`는 IoU 0.75~0.95 평균입니다. 이름만 mAP로 적으면 흔한
// mAP@0.5로 읽혀 값이 낮아 보입니다. TeamActivity 화면과 같은 label을 씁니다.
const MAP_LABEL = 'mAP@[0.75:0.95]';

function shown(value: string | number | null): string {
  return value === null || value === '' ? '-' : String(value);
}

function booleanLabel(value: boolean | null): string {
  if (value === null) return '-';
  return value ? '사용' : '미사용';
}

function metric(value: number | null): string {
  return value === null ? '-' : value.toFixed(4);
}

function capabilityValue(value: string | null, source: CapabilityValueSource): string {
  const text = shown(value);
  return source === 'legacy_fallback' && value !== null ? `${text} (호환 기본값)` : text;
}

function relationLabel(experiments: ExperimentSummary[]): string {
  if (experiments.length < 2) return '실험을 2개 이상 선택하면 판정합니다.';
  const relation = datasetRelationship(experiments);
  if (relation === 'same') return '같음';
  if (relation === 'different') return '다름';
  return '판정 불가';
}

function DatasetNotice({ experiments }: { experiments: ExperimentSummary[] }) {
  if (experiments.length < 2) {
    return (
      <AlertRow level="info" title="dataset 동일 여부는 실험 2개부터 판정합니다">
        하나를 고르면 그 실험의 설정과 결과만 보여 주고, 2개 이상이면 같은 dataset 입력끼리
        비교 중인지 이 자리에 표시됩니다.
      </AlertRow>
    );
  }

  const relation = datasetRelationship(experiments);
  if (relation === 'same') {
    return (
      <AlertRow level="success" title="같은 dataset 입력으로 기록된 실험입니다">
        4개 data artifact URI의 기록이 모두 같아 설정과 결과를 나란히 비교할 수 있습니다.
        파일 내용 자체가 같다는 뜻은 아닙니다.
      </AlertRow>
    );
  }
  if (relation === 'different') {
    return (
      <AlertRow level="warning" title="서로 다른 dataset 입력이 섞여 있습니다">
        모델 결과 차이에 dataset 차이가 포함될 수 있으므로 해석할 때 주의하세요.
      </AlertRow>
    );
  }
  return (
    <AlertRow level="info" title="dataset 동일 여부를 판정할 수 없습니다">
      선택한 기록 중 하나 이상에 필요한 data artifact 4개가 모두 남아 있지 않습니다.
    </AlertRow>
  );
}

/** 값이 클수록 좋은지 작을수록 좋은지. loss는 낮을수록, 지표는 높을수록 좋습니다. */
type Better = 'higher' | 'lower';

interface Row {
  label: string;
  values: ReactNode[];
  /** 초록 표시를 계산할 원본 숫자입니다. 좋고 나쁨이 있는 줄만 채웁니다. */
  numbers?: (number | null)[];
  better?: Better;
}

/**
 * 이 줄에서 가장 좋은 칸의 index입니다. 다음 경우에는 아무 칸도 고르지 않습니다.
 *
 * - 실험이 하나뿐이라 비교 대상이 없을 때
 * - 값이 있는 칸이 하나뿐일 때. 없는 값이 이기는 일은 없어야 합니다.
 * - 값이 모두 같을 때. 전부 초록이면 아무 정보가 아닙니다.
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

type MetricPick = (metrics: ExperimentSummary['metrics']) => number | null;

/** loss 줄. 낮을수록 좋습니다. */
function lossRow(label: string, experiments: ExperimentSummary[], pick: MetricPick): Row {
  const numbers = experiments.map((experiment) => pick(experiment.metrics) ?? null);
  return { label, values: numbers.map((value) => loss(value)), numbers, better: 'lower' };
}

/** mAP·precision·recall 줄. 높을수록 좋습니다. */
function metricRow(label: string, experiments: ExperimentSummary[], pick: MetricPick): Row {
  const numbers = experiments.map((experiment) => pick(experiment.metrics) ?? null);
  return { label, values: numbers.map((value) => metric(value)), numbers, better: 'higher' };
}

function ComparisonTable({ experiments }: { experiments: ExperimentSummary[] }) {
  const [tab, setTab] = useState<'results' | 'settings'>('results');

  // 상태와 dataset 관계, 시간은 어느 탭에서도 사라지면 안 되는 배경 정보입니다.
  const fixedRows: Row[] = [
    {
      label: '상태',
      values: experiments.map((experiment) => (
        <StatusBadge
          key={experiment.experiment_id}
          status={experiment.status}
          label={experiment.status_label}
        />
      )),
    },
    {
      label: 'DATASET 관계',
      values: experiments.map(() => relationLabel(experiments)),
    },
    {
      label: '경과 시간',
      values: experiments.map((experiment) => duration(experiment.elapsed_seconds)),
    },
    {
      label: '시작',
      values: experiments.map((experiment) => startedAt(experiment.started_at ?? experiment.created_at)),
    },
  ];

  const settingRows: Row[] = [
    {
      label: '모델',
      values: experiments.map((experiment) =>
        capabilityValue(experiment.model.architecture, experiment.model.source),
      ),
    },
    {
      label: 'PRETRAINED',
      values: experiments.map((experiment) => booleanLabel(experiment.model.pretrained)),
    },
    {
      label: 'OPTIMIZER',
      values: experiments.map((experiment) =>
        capabilityValue(experiment.optimizer.name, experiment.optimizer.source),
      ),
    },
    {
      label: 'DEVICE',
      values: experiments.map((experiment) => shown(experiment.training.device)),
    },
    {
      label: 'EPOCHS',
      values: experiments.map((experiment) => shown(experiment.training.epochs)),
    },
    {
      label: 'BATCH SIZE',
      values: experiments.map((experiment) => shown(experiment.training.batch_size)),
    },
    {
      label: 'SEED',
      values: experiments.map((experiment) => shown(experiment.training.seed)),
    },
    {
      label: 'LEARNING RATE',
      values: experiments.map((experiment) => shown(experiment.optimizer.learning_rate)),
    },
    {
      label: 'MOMENTUM',
      values: experiments.map((experiment) => shown(experiment.optimizer.momentum)),
    },
    {
      label: 'WEIGHT DECAY',
      values: experiments.map((experiment) => shown(experiment.optimizer.weight_decay)),
    },
    {
      label: 'BETA 1',
      values: experiments.map((experiment) => shown(experiment.optimizer.beta1)),
    },
    {
      label: 'BETA 2',
      values: experiments.map((experiment) => shown(experiment.optimizer.beta2)),
    },
    {
      label: 'EPSILON',
      values: experiments.map((experiment) => shown(experiment.optimizer.epsilon)),
    },
  ];

  const resultRows: Row[] = [
    // BEST EPOCH는 몇 번째 epoch였는지일 뿐이라 크고 작음에 좋고 나쁨이 없습니다.
    {
      label: 'BEST EPOCH',
      values: experiments.map((experiment) => shown(experiment.metrics.best_epoch)),
    },
    lossRow('BEST VAL LOSS', experiments, (item) => item.best_validation_loss),
    lossRow('FINAL TRAIN LOSS', experiments, (item) => item.final_train_loss),
    lossRow('FINAL VAL LOSS', experiments, (item) => item.final_validation_loss),
    metricRow(MAP_LABEL, experiments, (item) => item.map),
    metricRow('mAP@0.5', experiments, (item) => item.map50),
    metricRow('mAP@0.75', experiments, (item) => item.map75),
    metricRow('Precision@IoU0.5', experiments, (item) => item.precision50),
    metricRow('Recall@IoU0.5', experiments, (item) => item.recall50),
  ];

  const rows = [...fixedRows, ...(tab === 'results' ? resultRows : settingRows)];
  const columns = `190px repeat(${experiments.length}, minmax(170px, 1fr))`;

  return (
    <div>
      <div
        role="group"
        aria-label="비교 항목 묶음"
        style={{
          display: 'flex',
          gap: 6,
          padding: '10px 13px',
          borderBottom: `1px solid ${color.border}`,
        }}
      >
        <TabButton label="결과값" active={tab === 'results'} onClick={() => setTab('results')} />
        <TabButton label="학습 세팅" active={tab === 'settings'} onClick={() => setTab('settings')} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: 190 + experiments.length * 170 }}>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: columns,
              background: color.surfaceTableHead,
              borderBottom: `1px solid ${color.border}`,
            }}
          >
            <span style={{ padding: '11px 13px', ...type.microLabel, color: color.textMuted }}>
              비교 항목
            </span>
            {experiments.map((experiment) => (
              <span
                key={experiment.experiment_id}
                style={{ padding: '10px 13px', display: 'flex', flexDirection: 'column', gap: 2 }}
              >
                <span style={{ font: `600 12.5px/1.35 ${font.sans}`, color: color.text }}>
                  {experiment.run_id}
                </span>
                {/* 예전에는 experiment_id 앞 8자였는데, registry 실험은 그것이 run_id와
                    같아서 위 이름을 잘라 놓은 것에 지나지 않았습니다. */}
                <span style={{ font: `400 11px/1.3 ${font.mono}`, color: color.textFaint }}>
                  {[experiment.dataset.label, experiment.training.seed === null ? null : `seed ${experiment.training.seed}`]
                    .filter((part): part is string => Boolean(part))
                    .join(' · ')}
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
                  borderBottom: `1px solid ${color.borderInner}`,
                }}
              >
                <span
                  style={{
                    padding: '9px 13px',
                    font: `600 11.5px/1.4 ${font.mono}`,
                    color: color.textMuted,
                    background: color.surfaceAlt,
                  }}
                >
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
                        padding: '9px 13px',
                        font: `500 12.5px/1.4 ${font.mono}`,
                        color: isBest ? color.greenDark : color.textStrong,
                        background: isBest ? color.greenTint : undefined,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      {value}
                      {/* 색만으로 뜻을 전하지 않도록 글자 표식을 함께 둡니다. */}
                      {isBest && (
                        <span
                          title={row.better === 'lower' ? '이 비교에서 가장 낮습니다' : '이 비교에서 가장 높습니다'}
                          style={{
                            ...type.badge,
                            color: color.greenDark,
                            border: `1px solid ${color.green}`,
                            borderRadius: radius.badge,
                            padding: '1px 4px',
                          }}
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
    </div>
  );
}

function TabButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      style={{
        font: `600 12.5px/1 ${font.sans}`,
        padding: '7px 12px',
        borderRadius: radius.control,
        cursor: 'pointer',
        color: active ? color.surface : color.textBody,
        background: active ? color.primary : color.surface,
        border: `1px solid ${active ? color.primary : color.borderControl}`,
      }}
    >
      {label}
    </button>
  );
}

export function ExperimentComparison() {
  const listing = usePolling(() => api.listExperiments(), 3000);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  // 고르기와 비교표를 한 화면에 함께 두지 않습니다. 22개 목록 밑에 비교표를 놓으면
  // 고른 뒤 매번 스크롤을 한참 내려야 결과가 보입니다.
  const [step, setStep] = useState<'pick' | 'compare'>('pick');
  const [compared, setCompared] = useState<ExperimentSummary[]>([]);
  const [compareError, setCompareError] = useState<string | null>(null);
  const experiments = useMemo(() => listing.data?.experiments ?? [], [listing.data]);
  const selectedRunIds = useMemo(
    () => experiments
      .filter((experiment) => selectedIds.includes(experiment.experiment_id))
      .map((experiment) => experiment.run_id),
    [experiments, selectedIds],
  );

  /**
   * 선택을 **값**으로 굳혀 effect의 의존성으로 씁니다.
   *
   * 목록은 3초마다 polling하고 그때마다 새 배열이 만들어집니다. 배열 자체를
   * 의존성으로 두면 내용이 그대로여도 effect가 다시 돌고, 정리 함수가 아직
   * 오지 않은 비교 응답을 버립니다. 비교가 polling 주기보다 오래 걸리면 표가
   * 영원히 채워지지 않습니다. 실제로 그렇게 멈춰 있었습니다.
   */
  const selectedRunKey = selectedRunIds.join('\n');

  useEffect(() => {
    // 목록은 registry index의 training 블록으로 채웁니다. 하나만 골라도 record를
    // 읽어 다시 채우며, 두 값이 다르면 record가 진실입니다.
    const runIds = selectedRunKey ? selectedRunKey.split('\n') : [];
    if (runIds.length === 0) {
      setCompared([]);
      setCompareError(null);
      return;
    }
    let active = true;
    void api.compareExperiments(runIds).then(
      (result) => {
        if (active) {
          setCompared(result.experiments);
          setCompareError(null);
        }
      },
      (error: unknown) => {
        if (active) {
          setCompared([]);
          setCompareError(error instanceof Error ? error.message : '비교 정보를 불러오지 못했습니다.');
        }
      },
    );
    return () => { active = false; };
  }, [selectedRunKey]);

  const toggle = (experimentId: string) => {
    setSelectedIds((current) =>
      current.includes(experimentId)
        ? current.filter((value) => value !== experimentId)
        : [...current, experimentId],
    );
  };

  // 고른 것이 하나도 남지 않으면 비교표에 보여 줄 것이 없으므로 고르기로 되돌립니다.
  if (step === 'compare' && selectedIds.length === 0 && experiments.length > 0) {
    setStep('pick');
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1320 }}>
      {/* 고르기와 비교표를 세로로 쌓으면 고른 뒤 한참 스크롤해야 결과가 보입니다.
          한 화면에 하나씩만 두고 갈아 끼웁니다. */}
      {step === 'pick' ? (
        <>
          <ScreenIntro
            title="견줄 실험을 고릅니다"
            terms={[
              { term: '같은 dataset', meaning: '기록된 data artifact URI 4개가 모두 같은 경우입니다' },
              { term: '판정 불가', meaning: '이전 기록에 dataset 정보가 일부 빠진 경우입니다' },
            ]}
          >
            2개 이상 고르면 나란히 견줍니다. 하나만 골라도 그 실험의 설정과 결과가 열립니다.
            다 고른 뒤 <b>선택 완료</b>를 누르면 비교표로 넘어갑니다.
          </ScreenIntro>

          {listing.error && (
            <AlertRow level="error" title="실험 목록을 불러오지 못했습니다">
              {listing.error}
            </AlertRow>
          )}

          <Panel
            title="비교할 실험"
            right={
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMuted }}>
                  {selectedIds.length}개 선택
                </span>
                <Button
                  onClick={() =>
                    setSelectedIds(experiments.slice(0, 2).map((item) => item.experiment_id))
                  }
                  disabled={experiments.length < 2}
                >
                  최근 2개 선택
                </Button>
                <Button onClick={() => setSelectedIds([])} disabled={selectedIds.length === 0}>
                  선택 해제
                </Button>
                <Button
                  kind="primary"
                  onClick={() => setStep('compare')}
                  disabled={selectedIds.length === 0}
                  title={selectedIds.length === 0 ? '먼저 실험을 고르세요' : undefined}
                >
                  선택 완료 →
                </Button>
              </div>
            }
            bodyStyle={{ padding: 0 }}
          >
            {listing.loading && experiments.length === 0 ? (
              <EmptyState message="실험 기록을 불러오고 있습니다." />
            ) : (
              <ExperimentTable
                experiments={experiments}
                selectedIds={selectedIds}
                onToggle={toggle}
                emptyMessage="비교할 학습 기록이 아직 없습니다."
                selectLabel="비교 선택"
              />
            )}
          </Panel>
        </>
      ) : (
        <>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <Button onClick={() => setStep('pick')}>← 다시 고르기</Button>
            <span style={{ font: `650 15px/1.3 ${font.sans}`, color: color.text }}>
              실험 {selectedRunIds.length}개 비교
            </span>
            <span style={{ font: `400 12px/1.4 ${font.mono}`, color: color.textMuted, overflowWrap: 'anywhere' }}>
              {selectedRunIds.join(' · ')}
            </span>
          </div>

          {compareError && (
            <AlertRow level="error" title="비교 정보를 불러오지 못했습니다">
              {compareError}
            </AlertRow>
          )}
          <DatasetNotice experiments={compared} />

          <Panel title="비교표" bodyStyle={{ padding: 0 }}>
            {compared.length === 0 ? (
              <EmptyState message="선택한 실험의 상세 기록을 불러오고 있습니다." />
            ) : (
              <ComparisonTable experiments={compared} />
            )}
          </Panel>
        </>
      )}
    </div>
  );
}
