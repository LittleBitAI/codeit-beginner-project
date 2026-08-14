/**
 * 화면 하나가 쓰는 "기록" 한 줄을 만듭니다.
 *
 * 같은 학습이 두 곳에 있습니다. **registry**(`/experiments`)에는 등록까지 끝난
 * 실험이 팀원 것까지 들어 있고 Kaggle 점수와 mAP가 거기에만 있습니다. **job
 * 목록**(`/jobs`)에는 이 컴퓨터가 시작한 실행만 있지만, 실패·취소·진행 중처럼
 * 등록되지 않은 것까지 들어 있고 로그와 중지 같은 조작이 붙습니다.
 *
 * 둘 중 하나만 보여 주면 각각 "팀원 기록이 없다"와 "실패한 학습이 사라졌다"가
 * 됩니다. 그래서 `run_id`로 합칩니다. 겹치면 지표는 registry, 조작은 job에서
 * 가져옵니다 — 지표는 등록된 쪽이 정본이고, 조작은 이 컴퓨터만 할 수 있습니다.
 */

import type { ExperimentSummary, JobRecord, JobStatus } from '../api/types';
import { completionOf } from './completion';
import { datasetLabel } from './runSpec';

/** dataset을 알 수 없는 기록이 모이는 자리입니다. */
export const UNKNOWN_DATASET = '(dataset 모름)';

export interface RunRecord {
  runId: string;
  /** 모델 계열. 목록에서 사람이 먼저 읽는 이름입니다. */
  family: string;
  /** 어느 dataset으로 돌렸는지. 왼쪽 목록의 key와 같습니다. */
  datasetKey: string;
  /** 이름 아래 한 줄: epoch·batch·learning rate·seed. */
  spec: string;
  status: JobStatus;
  statusLabel: string;
  /** registry에 등록된 시각. 등록 전이면 학습을 시작한 시각입니다. */
  at: string | null;
  /** 이 컴퓨터에서 시작한 실행이면 job id. 모니터와 삭제가 여기에 달립니다. */
  jobId: string | null;
  /** registry에 등록됐는지. 등록 전에는 Kaggle 점수를 적을 수 없습니다. */
  registered: boolean;
  evaluated: boolean;
  submitted: boolean;
  metrics: {
    kaggle: number | null;
    map: number | null;
    map50: number | null;
    map75: number | null;
    precision50: number | null;
    recall50: number | null;
    bestValidationLoss: number | null;
    bestEpoch: number | null;
    epochs: number | null;
    elapsedSeconds: number | null;
  };
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function text(value: unknown): string | null {
  if (typeof value === 'string' && value.trim() !== '') return value.trim();
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return null;
}

/** `e15 · b4 · lr 0.006 · seed 42`. 모르는 값은 지어내지 않고 빼기만 합니다. */
function specOf(parts: {
  epochs: unknown;
  batchSize: unknown;
  learningRate: unknown;
  seed: unknown;
}): string {
  return [
    text(parts.epochs) === null ? null : `e${text(parts.epochs)}`,
    text(parts.batchSize) === null ? null : `b${text(parts.batchSize)}`,
    text(parts.learningRate) === null ? null : `lr ${text(parts.learningRate)}`,
    text(parts.seed) === null ? null : `seed ${text(parts.seed)}`,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');
}

function fromExperiment(item: ExperimentSummary): RunRecord {
  const completion = completionOf(item);
  return {
    runId: item.run_id,
    family: item.model.architecture ?? '(모델 모름)',
    datasetKey: item.dataset.label ?? item.dataset.identity ?? UNKNOWN_DATASET,
    spec: specOf({
      epochs: item.training.epochs,
      batchSize: item.training.batch_size,
      learningRate: item.optimizer.learning_rate,
      seed: item.training.seed,
    }),
    status: item.status,
    statusLabel: item.status_label,
    at: item.finished_at ?? item.started_at ?? item.created_at,
    jobId: null,
    registered: true,
    evaluated: completion.evaluated,
    submitted: completion.submitted,
    metrics: {
      kaggle: num(item.metrics.kaggle_score),
      map: num(item.metrics.map),
      map50: num(item.metrics.map50),
      map75: num(item.metrics.map75),
      precision50: num(item.metrics.precision50),
      recall50: num(item.metrics.recall50),
      bestValidationLoss: num(item.metrics.best_validation_loss),
      bestEpoch: num(item.metrics.best_epoch),
      epochs: num(item.training.epochs),
      elapsedSeconds: num(item.elapsed_seconds),
    },
  };
}

function fromJob(job: JobRecord): RunRecord {
  const settings = job.settings ?? {};
  const evaluation = job.evaluation?.summary?.metrics;
  return {
    runId: job.run_id,
    family: text(settings.architecture) ?? '(모델 모름)',
    datasetKey: datasetLabel(job.data_inputs) ?? UNKNOWN_DATASET,
    spec: specOf({
      epochs: settings.epochs,
      batchSize: settings.batch_size,
      learningRate: settings.learning_rate,
      seed: settings.seed,
    }),
    status: job.status,
    statusLabel: job.status_label,
    at: job.finished_at ?? job.started_at ?? job.created_at,
    jobId: job.job_id,
    registered: false,
    evaluated: job.evaluation?.status === 'succeeded',
    submitted: false,
    metrics: {
      kaggle: null,
      map: num(evaluation?.mAP),
      map50: num(evaluation?.mAP50),
      map75: num(evaluation?.mAP75),
      precision50: num(evaluation?.precision50),
      recall50: num(evaluation?.recall50),
      bestValidationLoss: num(job.summary?.best_validation_loss),
      bestEpoch: num(job.progress?.best?.epoch),
      epochs: num(job.progress?.total_epochs) ?? num(settings.epochs),
      elapsedSeconds: num(job.elapsed_seconds),
    },
  };
}

/**
 * 두 목록을 `run_id`로 합칩니다. 새 기록이 위로 오도록 시각 내림차순입니다.
 *
 * 겹친 줄에서 지표는 registry 것을 쓰되, registry가 아직 못 읽어 비어 있는 칸은
 * job이 들고 있는 값으로 메웁니다. 로컬에서 방금 평가를 돌렸는데 등록 index가
 * 아직 갱신되지 않은 사이에 mAP가 사라지는 일을 막습니다.
 */
export function mergeRecords(
  experiments: ExperimentSummary[],
  jobs: JobRecord[],
): RunRecord[] {
  const byRunId = new Map<string, RunRecord>();
  for (const item of experiments) byRunId.set(item.run_id, fromExperiment(item));

  /**
   * 같은 `run_id`의 job이 여럿일 수 있습니다. 설정과 seed가 같으면 이름도 같게
   * 지어지는데, 그것이 곧 "같은 실험을 또 돌렸다"는 신호입니다. `/jobs`는 최신순
   * 이므로 **먼저 만난 것이 최신**이고, 뒤에 오는 옛 실행이 그것을 덮으면 안 됩니다.
   */
  const merged = new Set<string>();

  for (const job of jobs) {
    if (merged.has(job.run_id)) continue;
    merged.add(job.run_id);
    const local = fromJob(job);
    const registered = byRunId.get(job.run_id);
    if (!registered) {
      byRunId.set(job.run_id, local);
      continue;
    }
    byRunId.set(job.run_id, {
      ...registered,
      // 조작은 이 컴퓨터만 할 수 있습니다. 상태도 job이 더 최신입니다.
      jobId: local.jobId,
      status: local.status,
      statusLabel: local.statusLabel,
      metrics: {
        ...registered.metrics,
        map: registered.metrics.map ?? local.metrics.map,
        map50: registered.metrics.map50 ?? local.metrics.map50,
        map75: registered.metrics.map75 ?? local.metrics.map75,
        precision50: registered.metrics.precision50 ?? local.metrics.precision50,
        recall50: registered.metrics.recall50 ?? local.metrics.recall50,
        bestValidationLoss:
          registered.metrics.bestValidationLoss ?? local.metrics.bestValidationLoss,
        bestEpoch: registered.metrics.bestEpoch ?? local.metrics.bestEpoch,
        elapsedSeconds: registered.metrics.elapsedSeconds ?? local.metrics.elapsedSeconds,
      },
    });
  }

  return [...byRunId.values()].sort((left, right) => (right.at ?? '').localeCompare(left.at ?? ''));
}

export interface DatasetGroup {
  key: string;
  count: number;
}

/** 어떤 dataset으로 돌렸는지 이름을 댈 수 있는 기록인지. */
export function namesDataset(record: RunRecord): boolean {
  return record.datasetKey !== UNKNOWN_DATASET;
}

/**
 * 왼쪽 목록에 세울 dataset들. 기록이 많은 순, 같으면 이름 순입니다.
 *
 * 이름을 댈 수 없는 기록은 줄을 만들지 않습니다. dataset을 고르는 목록이라
 * `(dataset 모름)`은 고를 것이 아니라 데이터가 없다는 말이고, 실제로 옛 smoke
 * test와 pytest 찌꺼기가 `data`·`fixtures`라는 이름으로 맨 위를 차지했습니다.
 * 감춘 건수는 목록 화면이 따로 말합니다 — 조용히 빼면 그만큼이 없는 줄 압니다.
 */
export function groupByDataset(records: RunRecord[]): DatasetGroup[] {
  const counts = new Map<string, number>();
  for (const record of records.filter(namesDataset)) {
    counts.set(record.datasetKey, (counts.get(record.datasetKey) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([key, count]) => ({ key, count }))
    .sort((left, right) => right.count - left.count || left.key.localeCompare(right.key));
}

export type RecordFilter = 'all' | 'evaluated' | 'submitted' | 'running' | 'unregistered';

export const FILTER_LABEL: Record<RecordFilter, string> = {
  all: '전체',
  evaluated: '평가 완료',
  submitted: '제출 완료',
  running: '학습 중',
  unregistered: '미등록·실패',
};

/** 아직 돌고 있는 학습입니다. 대기열에 줄만 서 있는 것은 여기 들어오지 않습니다. */
export function isRunning(record: RunRecord): boolean {
  return record.status === 'running' || record.status === 'starting';
}

export function matchesFilter(record: RunRecord, filter: RecordFilter): boolean {
  if (filter === 'all') return true;
  if (filter === 'evaluated') return record.evaluated;
  if (filter === 'submitted') return record.submitted;
  if (filter === 'running') return isRunning(record);
  return !record.registered;
}

/**
 * 목록에 그대로 세울 기록인지.
 *
 * 실패와 취소가 성공한 학습과 한 줄에 섞이면 눈으로 골라내야 합니다. 실제로 35건
 * 중 32건이 결과 없이 끝난 기록이라 볼 것 3건이 가운데 묻혀 있었습니다.
 *
 * 중단(interrupted)은 접지 않습니다. epoch마다 저장한 checkpoint가 남아 있어 이어서
 * 학습할 수 있으므로, 사람이 아직 판단할 것이 있는 기록입니다. 실패·취소여도 검증
 * 오차가 남았으면 결과가 있는 것이므로 함께 세웁니다.
 */
export function hasResult(record: RunRecord): boolean {
  if (record.status !== 'failed' && record.status !== 'cancelled') return true;
  return record.metrics.bestValidationLoss !== null;
}

/** 접어 둔 구역의 머리글에 쓸 내역입니다. 몇 건을 감췄는지 항상 말해 줍니다. */
export function countLabel(records: RunRecord[]): string {
  const failed = records.filter((record) => record.status === 'failed').length;
  const cancelled = records.length - failed;
  const parts = [
    failed > 0 ? `실패 ${failed}` : null,
    cancelled > 0 ? `취소·중단 ${cancelled}` : null,
  ].filter((part): part is string => part !== null);
  return parts.length > 0 ? `${records.length}건 (${parts.join(' · ')})` : `${records.length}건`;
}

export type RecordSort = 'recent' | 'kaggle' | 'loss';

export const SORT_LABEL: Record<RecordSort, string> = {
  recent: '최근',
  kaggle: 'Kaggle',
  loss: 'val loss',
};

/**
 * 값이 없는 줄은 방향과 상관없이 언제나 뒤로 보냅니다. `null`을 0으로 두면
 * Kaggle 정렬에서 점수를 아직 안 적은 실험이 1등이 됩니다.
 */
export function sortRecords(records: RunRecord[], sort: RecordSort): RunRecord[] {
  const copy = [...records];
  if (sort === 'recent') return copy;
  if (sort === 'kaggle') {
    return copy.sort((left, right) => {
      if (left.metrics.kaggle === null) return right.metrics.kaggle === null ? 0 : 1;
      if (right.metrics.kaggle === null) return -1;
      return right.metrics.kaggle - left.metrics.kaggle;
    });
  }
  return copy.sort((left, right) => {
    if (left.metrics.bestValidationLoss === null) return right.metrics.bestValidationLoss === null ? 0 : 1;
    if (right.metrics.bestValidationLoss === null) return -1;
    return left.metrics.bestValidationLoss - right.metrics.bestValidationLoss;
  });
}
