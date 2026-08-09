/**
 * 등록된 실험 하나를 여는 화면. 목록에서 행을 누르면 여기로 옵니다.
 *
 * 맨 위에 결론(핵심 지표와 loss 곡선)을 두고, 숫자 표는 접어 둡니다. 딥러닝에
 * 익숙하지 않은 사람이 먼저 봐야 할 것과 깊게 파는 사람이 볼 것을 나누려는 것입니다.
 */

import { useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type { ExperimentDetail as Detail, ExperimentSummary, SweepPoint } from '../api/types';
import { LossChart, ChartLegend } from '../components/LossChart';
import { LrChart } from '../components/LrChart';
import { AlertRow, Button, KpiCard, Panel, StatusBadge } from '../components/primitives';
import { color, font, radius, type } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { duration, loss, startedAt } from '../lib/format';

// evaluate가 내는 `mAP`는 IoU 0.75~0.95 평균입니다. 다른 화면과 같은 label을 씁니다.
const MAP_LABEL = 'mAP@[0.75:0.95]';

/** 지표 9개의 사람이 읽을 이름. evaluate의 key 순서를 그대로 둡니다. */
const METRIC_LABELS: [string, string][] = [
  ['mAP', MAP_LABEL],
  ['mAP50_95', 'mAP@[0.50:0.95]'],
  ['mAP75_95', 'mAP@[0.75:0.95] (동일)'],
  ['mAP50', 'mAP@0.5'],
  ['mAP75', 'mAP@0.75'],
  ['precision50', 'Precision@IoU0.5'],
  ['recall50', 'Recall@IoU0.5'],
  ['precision75', 'Precision@IoU0.75'],
  ['recall75', 'Recall@IoU0.75'],
];

const COUNT_LABELS: [string, string][] = [
  ['image_count', '평가한 이미지'],
  ['annotation_count', '정답 상자'],
  ['prediction_count', '예측 상자'],
  ['evaluated_class_count', '측정된 class'],
];

function metric(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toFixed(4) : '-';
}

function count(value: number | null | undefined): string {
  return typeof value === 'number' ? value.toLocaleString('ko-KR') : '-';
}

function shown(value: unknown): string {
  if (value === null || value === undefined || value === '') return '-';
  return String(value);
}

/** 접었다 펴는 구역. 결론을 먼저 읽고 필요할 때만 파고들게 합니다. */
function Fold({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <details
      style={{
        border: `1px solid ${color.border}`,
        borderRadius: radius.panel,
        background: color.surface,
      }}
    >
      <summary
        style={{
          padding: '11px 15px',
          font: `600 13px/1.4 ${font.sans}`,
          color: color.text,
          cursor: 'pointer',
          display: 'flex',
          gap: 8,
          alignItems: 'baseline',
        }}
      >
        {title}
        {note && <span style={{ ...type.plainNote, color: color.textMuted }}>{note}</span>}
      </summary>
      <div style={{ padding: '4px 15px 15px' }}>{children}</div>
    </details>
  );
}

/** 이름과 값 두 칸짜리 표. 값은 mono라 소수점이 세로로 맞습니다. */
function ValueTable({ rows }: { rows: [string, string][] }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(150px, 240px) 1fr' }}>
      {rows.map(([label, value]) => (
        <div key={label} style={{ display: 'contents' }}>
          <span
            style={{
              padding: '8px 12px',
              font: `500 12px/1.4 ${font.sans}`,
              color: color.textMuted,
              borderBottom: `1px solid ${color.borderInner}`,
              background: color.surfaceAlt,
            }}
          >
            {label}
          </span>
          <span
            style={{
              padding: '8px 12px',
              font: `400 12px/1.4 ${font.mono}`,
              color: color.textStrong,
              borderBottom: `1px solid ${color.borderInner}`,
              overflowWrap: 'anywhere',
            }}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

function SettingsTab({ experiment }: { experiment: ExperimentSummary }) {
  const { model, optimizer, training, dataset } = experiment;
  const optimizerRows: [string, string][] = [
    ['Optimizer', shown(optimizer.name)],
    ['Learning rate', shown(optimizer.learning_rate)],
    ['Weight decay', shown(optimizer.weight_decay)],
    ...(optimizer.name === 'SGD'
      ? ([['Momentum', shown(optimizer.momentum)]] as [string, string][])
      : ([
          ['Beta 1', shown(optimizer.beta1)],
          ['Beta 2', shown(optimizer.beta2)],
          ['Epsilon', shown(optimizer.epsilon)],
        ] as [string, string][])),
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Panel title="모델과 데이터" bodyStyle={{ padding: 0 }}>
        <ValueTable
          rows={[
            ['모델', shown(model.architecture)],
            ['Pretrained 가중치', model.pretrained === null ? '-' : model.pretrained ? '사용' : '미사용'],
            ['데이터셋', shown(dataset.label)],
            [
              'dataset artifact',
              dataset.artifacts_complete
                ? '4개 모두 기록됨'
                : '일부가 기록에 없어 같은 데이터인지 판정할 수 없습니다',
            ],
          ]}
        />
      </Panel>
      <Panel title="학습 설정" bodyStyle={{ padding: 0 }}>
        <ValueTable
          rows={[
            ['Device', shown(training.device)],
            ['Epochs', shown(training.epochs)],
            ['Batch size', shown(training.batch_size)],
            ['Num workers', shown(training.num_workers)],
            ['Seed', shown(training.seed)],
          ]}
        />
      </Panel>
      <Panel title="Optimizer" bodyStyle={{ padding: 0 }}>
        <ValueTable rows={optimizerRows} />
      </Panel>
    </div>
  );
}

/** score threshold를 옮겨 가며 잰 값. 표가 아니라 곡선이라야 어디가 꼭대기인지 보입니다. */
function SweepTable({ points }: { points: SweepPoint[] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${points.length + 1}, minmax(62px, 1fr))`, minWidth: 420 }}>
        {[
          ['score 기준', ...points.map((point) => point.threshold.toFixed(2))],
          ['Precision', ...points.map((point) => metric(point.precision))],
          ['Recall', ...points.map((point) => metric(point.recall))],
          ['F1', ...points.map((point) => metric(point.f1))],
        ].map((row, rowIndex) =>
          row.map((cell, index) => (
            <span
              key={`${rowIndex}-${index}`}
              style={{
                padding: '6px 8px',
                font: `${index === 0 ? 500 : 400} 11.5px/1.3 ${index === 0 ? font.sans : font.mono}`,
                color: index === 0 ? color.textMuted : color.textStrong,
                background: rowIndex === 0 ? color.surfaceAlt : undefined,
                borderBottom: `1px solid ${color.borderInner}`,
                whiteSpace: 'nowrap',
              }}
            >
              {cell}
            </span>
          )),
        )}
      </div>
    </div>
  );
}

function EvaluationTab({ detail }: { detail: Detail }) {
  const { evaluation, history, experiment } = detail;
  const epochs = history.epochs ?? [];
  const weak = evaluation.per_class_summary ?? null;
  // 점수가 가장 낮은 class. evaluate가 이미 AP 오름차순으로 정렬해 둡니다.
  const lowest = weak?.weak?.[0] ?? null;
  // 평가를 못 읽은 응답에는 이 blocks가 통째로 없을 수 있습니다. 값을 만지기 전에
  // 빈 것으로 받아 둡니다. 예전에는 여기서 곧바로 파고들어 상세 화면이 흰 채로
  // 멈췄습니다. IoU 0.5 기준을 먼저 보여 주고, 없으면 있는 것 중 첫 번째를 씁니다.
  const sweeps = evaluation.score_sweep ?? {};
  const bests = evaluation.best_f1 ?? {};
  const sweepLabel = '0.50' in sweeps ? '0.50' : Object.keys(sweeps)[0];
  const best = sweepLabel ? bests[sweepLabel] ?? null : null;
  const metrics = evaluation.metrics ?? {};

  if (!evaluation.available && !history.available) {
    return (
      <AlertRow level="info" title="이 실험에는 볼 수 있는 결과 파일이 없습니다">
        {evaluation.reason ?? '평가 결과를 읽지 못했습니다.'} 설정은 세팅 탭에서 볼 수 있습니다.
      </AlertRow>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* 결론. 이 네 칸이 "얼마나 잘 나왔나"의 답입니다. */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10 }}>
        <KpiCard
          label={MAP_LABEL}
          value={metric(metrics.mAP ?? experiment.metrics.map)}
          valueColor={color.tealDark}
          note="대회가 보는 값입니다. 높을수록 좋습니다."
        />
        <KpiCard label="mAP@0.5" value={metric(metrics.mAP50 ?? experiment.metrics.map50)} note="상자를 느슨하게 맞춰도 되는 기준입니다." />
        <KpiCard
          label="BEST VAL LOSS"
          value={loss(experiment.metrics.best_validation_loss)}
          note={`가장 좋았던 epoch ${shown(experiment.metrics.best_epoch)}`}
        />
        {/* `counts.weak`는 "약한 class 수"가 아니라 **점수를 매길 만큼 정답이 있었던
            class 수**입니다. 그 숫자를 약한 class로 읽으면 57개 전부가 약해 보입니다.
            사람이 다음에 손댈 곳은 그중 가장 낮은 class이므로 그것을 보여 줍니다. */}
        <KpiCard
          label="가장 낮은 CLASS"
          value={lowest ? metric(lowest.ap) : '-'}
          valueColor={lowest && typeof lowest.ap === 'number' && lowest.ap < 0.8 ? color.amber : undefined}
          note={lowest ? lowest.name : '이 평가에는 class별 요약이 없습니다.'}
        />
      </div>

      {evaluation.available && !history.available && (
        <AlertRow level="info" title="loss 곡선은 볼 수 없습니다">
          {history.reason ?? '학습 기록 파일을 읽지 못했습니다.'}
        </AlertRow>
      )}

      {epochs.length > 0 && (
        <Panel title="epoch별 loss" bodyStyle={{ padding: '12px 16px 0' }}>
          <LossChart epochs={epochs} totalEpochs={epochs.length} currentEpoch={epochs.length} />
          <ChartLegend />
          <div
            style={{
              display: 'flex',
              gap: 18,
              flexWrap: 'wrap',
              padding: '10px 0 14px',
              font: `400 12px/1.5 ${font.mono}`,
              color: color.textStrong,
            }}
          >
            <span>final train {loss(experiment.metrics.final_train_loss)}</span>
            <span>final val {loss(experiment.metrics.final_validation_loss)}</span>
            <span>best val {loss(experiment.metrics.best_validation_loss)}</span>
            <span>epoch {epochs.length}회</span>
          </div>
        </Panel>
      )}

      {/* schedule을 쓴 학습만 값이 있습니다. 없으면 그렇다고 말하고 곡선을 그리지 않습니다. */}
      {epochs.some((item) => typeof item.learning_rate === 'number') && (
        <Panel title="epoch별 learning rate" bodyStyle={{ padding: '12px 16px 0' }}>
          <LrChart epochs={epochs} totalEpochs={epochs.length} />
        </Panel>
      )}

      {evaluation.available && (
        <>
          <Fold title="지표 전체" note="9개">
            <ValueTable
              rows={METRIC_LABELS.map(([key, label]) => [label, metric(metrics[key])])}
            />
            <div style={{ marginTop: 12 }}>
              <ValueTable
                rows={[
                  ...COUNT_LABELS.map(
                    ([key, label]) => [label, count((evaluation.counts ?? {})[key])] as [string, string],
                  ),
                  ['분석 score 기준', shown(evaluation.score_threshold)],
                  ['이미지당 최대 예측', shown(evaluation.max_detections_per_image)],
                ]}
              />
            </div>
          </Fold>

          {sweepLabel && (sweeps[sweepLabel] ?? []).length > 0 && (
            <Fold
              title="점수 기준을 얼마로 둘까"
              note={
                best
                  ? `IoU ${sweepLabel} · ${best.threshold.toFixed(2)}에서 F1 ${metric(best.f1)}로 가장 높습니다`
                  : `IoU ${sweepLabel}`
              }
            >
              <p style={{ ...type.body, color: color.textBody, margin: '0 0 10px' }}>
                예측을 몇 점부터 믿을지 정하는 값입니다. 낮추면 많이 잡아 recall이 오르고 precision이
                떨어집니다. 제출 전에 F1이 가장 높은 지점을 쓰면 대개 무난합니다.
              </p>
              <SweepTable points={sweeps[sweepLabel] ?? []} />
            </Fold>
          )}

          {weak && (
            <Fold
              title="점수가 낮은 class"
              note={`정답 ${weak.min_truth_count}개 이상인 ${weak.counts.weak}개 중 아래 ${weak.weak.length}개`}
            >
              <p style={{ ...type.body, color: color.textBody, margin: '0 0 10px' }}>
                정답이 <b>{weak.min_truth_count}개 이상</b>이라 점수를 믿을 수 있는 class{' '}
                {weak.counts.weak}개를 AP가 낮은 순으로 세운 것입니다. 정답이 그보다 적은 class는
                점수가 실력이 아니라 표본 수에 흔들려서 아래에 따로 셉니다.
              </p>
              {weak.weak.length === 0 ? (
                <span style={{ ...type.body, color: color.textBody }}>
                  점수를 매길 만큼 정답이 있는 class가 없습니다.
                </span>
              ) : (
                <ValueTable
                  rows={weak.weak.map((row) => [
                    `${row.name} (id ${row.category_id})`,
                    `AP ${metric(row.ap)}${row.truth_count === undefined ? '' : ` · 정답 ${count(row.truth_count)}개`}`,
                  ])}
                />
              )}
              {weak.sparse.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <span style={{ ...type.fieldLabel, color: color.textStrong }}>
                    정답이 적어 점수를 믿기 어려운 class {weak.counts.sparse}개
                  </span>
                  <div style={{ marginTop: 6 }}>
                    <ValueTable
                      rows={weak.sparse.map((row) => [
                        `${row.name} (id ${row.category_id})`,
                        `AP ${metric(row.ap)} · 정답 ${count(row.truth_count)}개`,
                      ])}
                    />
                  </div>
                </div>
              )}
            </Fold>
          )}
        </>
      )}
    </div>
  );
}

export function ExperimentDetail() {
  const navigate = useNavigate();
  const params = useParams<{ runId?: string }>();
  const runId = params.runId ?? '';
  const [tab, setTab] = useState<'settings' | 'evaluation'>('evaluation');
  // 등록된 실험은 더 변하지 않으므로 한 번만 읽습니다.
  const detail = usePolling<Detail>(() => api.experimentDetail(runId), 0, runId !== '');
  const experiment = detail.data?.experiment;

  const tabs = useMemo(
    () =>
      [
        { key: 'settings' as const, label: '세팅' },
        { key: 'evaluation' as const, label: '평가 결과' },
      ],
    [],
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1320 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button onClick={() => navigate('/history')}>← 실험 내역</Button>
        <span style={{ font: `700 16px/1.2 ${font.mono}`, color: color.text, overflowWrap: 'anywhere' }}>
          {runId}
        </span>
        {experiment && <StatusBadge status={experiment.status} label={experiment.status_label} />}
        {experiment && (
          <span style={{ ...type.plainNote, color: color.textMuted }}>
            {startedAt(experiment.created_at)} 등록
            {/* registry 기록에는 학습 시간이 없는 경우가 많습니다. "학습 -"은 정보가 아닙니다. */}
            {experiment.elapsed_seconds !== null && ` · 학습 ${duration(experiment.elapsed_seconds)}`}
          </span>
        )}
      </div>

      {detail.error && (
        <AlertRow
          level="error"
          title="실험을 불러오지 못했습니다"
          action={<Button onClick={() => detail.refresh()}>다시 읽기</Button>}
        >
          {detail.error}
        </AlertRow>
      )}

      {!detail.data && !detail.error && (
        <Panel>실험 기록과 평가 결과를 읽고 있습니다.</Panel>
      )}

      {detail.data && (
        <>
          <div role="tablist" aria-label="실험 상세 묶음" style={{ display: 'flex', gap: 6 }}>
            {tabs.map((item) => {
              const active = item.key === tab;
              return (
                <button
                  key={item.key}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setTab(item.key)}
                  style={{
                    font: `600 12.5px/1 ${font.sans}`,
                    padding: '9px 16px',
                    borderRadius: radius.control,
                    color: active ? color.surface : color.textBody,
                    background: active ? color.primary : color.surface,
                    border: `1px solid ${active ? color.primary : color.borderControl}`,
                  }}
                >
                  {item.label}
                </button>
              );
            })}
          </div>

          {tab === 'settings' ? (
            <SettingsTab experiment={detail.data.experiment} />
          ) : (
            <EvaluationTab detail={detail.data} />
          )}
        </>
      )}
    </div>
  );
}

/** 화면 밖에서도 오류 문구를 같게 쓰려고 남겨 둡니다. */
export const DETAIL_LOAD_ERROR = (error: unknown): string =>
  error instanceof ApiError ? error.message : '실험을 불러오지 못했습니다.';
