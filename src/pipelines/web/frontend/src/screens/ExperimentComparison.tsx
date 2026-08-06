import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { api } from '../api/client';
import type { CapabilityValueSource, ExperimentSummary } from '../api/types';
import { AlertRow, Button, EmptyState, Panel, ScreenIntro, StatusBadge } from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { datasetRelationship } from '../lib/experiments';
import { duration, loss, startedAt } from '../lib/format';

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
      <AlertRow level="info" title="비교할 실험을 2개 이상 선택해 주세요">
        선택하면 같은 dataset 입력끼리 비교 중인지 이 자리에 표시됩니다.
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

function ComparisonTable({ experiments }: { experiments: ExperimentSummary[] }) {
  const rows: { label: string; values: ReactNode[] }[] = [
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
    {
      label: 'BEST EPOCH',
      values: experiments.map((experiment) => shown(experiment.metrics.best_epoch)),
    },
    {
      label: 'BEST VAL LOSS',
      values: experiments.map((experiment) => loss(experiment.metrics.best_validation_loss)),
    },
    {
      label: 'mAP',
      values: experiments.map((experiment) => metric(experiment.metrics.map)),
    },
    {
      label: 'mAP50',
      values: experiments.map((experiment) => metric(experiment.metrics.map50)),
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
  const columns = `190px repeat(${experiments.length}, minmax(170px, 1fr))`;

  return (
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
              <span style={{ font: `600 12px/1.35 ${font.sans}`, color: color.text }}>
                {experiment.run_id}
              </span>
              <span style={{ font: `400 10px/1.3 ${font.mono}`, color: color.textFaint }}>
                {experiment.experiment_id.slice(0, 8)}
              </span>
            </span>
          ))}
        </div>
        {rows.map((row) => (
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
                font: `600 10.5px/1.4 ${font.mono}`,
                color: color.textMuted,
                background: color.surfaceAlt,
              }}
            >
              {row.label}
            </span>
            {row.values.map((value, index) => (
              <span
                key={`${row.label}-${experiments[index]?.experiment_id ?? index}`}
                style={{ padding: '9px 13px', font: `500 11.5px/1.4 ${font.mono}`, color: color.textStrong }}
              >
                {value}
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function ExperimentComparison() {
  const listing = usePolling(() => api.listExperiments(), 3000);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [compared, setCompared] = useState<ExperimentSummary[]>([]);
  const [compareError, setCompareError] = useState<string | null>(null);
  const experiments = useMemo(() => listing.data?.experiments ?? [], [listing.data]);
  const selectedRunIds = useMemo(
    () => experiments
      .filter((experiment) => selectedIds.includes(experiment.experiment_id))
      .map((experiment) => experiment.run_id),
    [experiments, selectedIds],
  );

  useEffect(() => {
    if (selectedRunIds.length < 2) {
      setCompared([]);
      setCompareError(null);
      return;
    }
    let active = true;
    void api.compareExperiments(selectedRunIds).then(
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
  }, [selectedRunIds]);

  const toggle = (experimentId: string) => {
    setSelectedIds((current) =>
      current.includes(experimentId)
        ? current.filter((value) => value !== experimentId)
        : [...current, experimentId],
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1720 }}>
      <ScreenIntro
        title="실험 설정과 결과를 같은 기준으로 나란히 봅니다"
        terms={[
          { term: '같은 dataset', meaning: '기록된 data artifact URI 4개가 모두 같은 경우입니다' },
          { term: '판정 불가', meaning: '이전 기록에 dataset 정보가 일부 빠진 경우입니다' },
        ]}
      >
        비교할 실험을 2개 이상 고르세요. 기록에 없는 값은 추정하지 않고 - 로 표시합니다.
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
            <span style={{ font: `400 11px/1 ${font.mono}`, color: color.textMuted }}>
              {selectedIds.length}개 선택
            </span>
            <Button
              onClick={() => setSelectedIds(experiments.slice(0, 2).map((item) => item.experiment_id))}
              disabled={experiments.length < 2}
            >
              최근 2개 선택
            </Button>
            <Button onClick={() => setSelectedIds([])} disabled={selectedIds.length === 0}>
              선택 해제
            </Button>
          </div>
        }
        bodyStyle={{ padding: 0 }}
      >
        {listing.loading ? (
          <EmptyState message="실험 기록을 불러오고 있습니다." />
        ) : experiments.length === 0 ? (
          <EmptyState message="비교할 학습 기록이 아직 없습니다." />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
            {experiments.map((experiment) => {
              const checked = selectedIds.includes(experiment.experiment_id);
              return (
                <label
                  key={experiment.experiment_id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 9,
                    padding: '10px 13px',
                    borderBottom: `1px solid ${color.borderInner}`,
                    background: checked ? color.primaryTint : color.surface,
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(experiment.experiment_id)}
                    aria-label={`${experiment.run_id} 비교 선택`}
                  />
                  <span style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0, flex: 1 }}>
                    <span style={{ font: `600 12px/1.35 ${font.sans}`, color: color.text }}>
                      {experiment.run_id}
                    </span>
                    <span style={{ font: `400 10px/1.3 ${font.mono}`, color: color.textFaint }}>
                      {startedAt(experiment.started_at ?? experiment.created_at)}
                    </span>
                  </span>
                  <StatusBadge status={experiment.status} label={experiment.status_label} />
                </label>
              );
            })}
          </div>
        )}
      </Panel>

      {compareError && <AlertRow level="error" title="비교 정보를 불러오지 못했습니다">{compareError}</AlertRow>}
      <DatasetNotice experiments={compared} />

      <Panel title="비교표" bodyStyle={{ padding: 0 }}>
        {selectedIds.length < 2 ? (
          <EmptyState message="위에서 실험을 2개 이상 선택하면 비교표가 열립니다." />
        ) : compared.length < 2 ? (
          <EmptyState message="선택한 실험의 상세 기록을 불러오고 있습니다." />
        ) : (
          <ComparisonTable experiments={compared} />
        )}
      </Panel>
    </div>
  );
}
