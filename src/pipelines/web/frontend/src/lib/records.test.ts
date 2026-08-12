import { describe, expect, it } from 'vitest';

import type { ExperimentSummary, JobRecord } from '../api/types';
import { groupByDataset, mergeRecords, sortRecords, UNKNOWN_DATASET } from './records';

function experiment(overrides: Partial<ExperimentSummary> = {}): ExperimentSummary {
  return {
    experiment_id: 'exp-1',
    run_id: 'retina-e15-b4-a7f3',
    status: 'succeeded',
    status_label: '완료',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T02:00:00Z',
    elapsed_seconds: 7200,
    dataset: {
      identity: 'sha:abc',
      identity_source: 'dataset_id',
      artifacts_complete: true,
      label: 'v5-118cls',
    },
    model: { architecture: 'retinanet_resnet50_fpn_v2', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.006,
      momentum: null,
      weight_decay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    training: {
      device: 'cuda',
      epochs: 15,
      batch_size: 4,
      num_workers: 4,
      gradient_accumulation_steps: 1,
      input_size: null,
      seed: 42,
    },
    metrics: {
      best_epoch: 12,
      best_validation_loss: 0.41,
      final_train_loss: 0.3,
      final_validation_loss: 0.44,
      map: 0.52,
      map50: 0.8,
      map75: 0.55,
      precision50: 0.7,
      recall50: 0.72,
      kaggle_score: 0.61,
    },
    completion: {
      evaluated: true,
      submission_generated: true,
      submitted: true,
      submission_checked: true,
      submission_rows: 100,
    },
    ...overrides,
  };
}

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    job_id: 'job-1',
    config_id: 'cfg-1',
    run_id: 'retina-e15-b4-a7f3',
    status: 'running',
    status_label: '학습 중',
    created_at: '2026-01-02T00:00:00Z',
    started_at: '2026-01-02T00:00:00Z',
    finished_at: null,
    elapsed_seconds: 600,
    exit_code: null,
    message: null,
    artifacts: {},
    summary: {},
    settings: { architecture: 'retinanet_resnet50_fpn_v2', epochs: 15, batch_size: 4, seed: 42 },
    data_inputs: { train_manifest_uri: 'artifacts/data/v5-118cls/train_manifest.json' },
    progress: {
      available: true,
      reason: null,
      message: null,
      total_epochs: 15,
      current_epoch: 3,
      eta_seconds: null,
      epochs: [],
    },
    log_lines: 10,
    orphan_note: null,
    ...overrides,
  };
}

describe('mergeRecords', () => {
  it('같은 run_id는 한 줄로 합치고 조작은 job, 지표는 registry에서 가져온다', () => {
    const [record, ...rest] = mergeRecords([experiment()], [job()]);

    expect(rest).toHaveLength(0);
    expect(record?.jobId).toBe('job-1');
    expect(record?.status).toBe('running');
    expect(record?.registered).toBe(true);
    expect(record?.metrics.kaggle).toBe(0.61);
  });

  it('registry에 아직 없는 실패한 실행도 목록에 남는다', () => {
    const records = mergeRecords(
      [],
      [job({ run_id: 'retina-oom', job_id: 'job-2', status: 'failed', status_label: '실패' })],
    );

    expect(records).toHaveLength(1);
    expect(records[0]?.registered).toBe(false);
    expect(records[0]?.status).toBe('failed');
  });

  it('registry가 아직 못 읽은 지표는 로컬 평가 결과로 메운다', () => {
    const [record] = mergeRecords(
      [experiment({ metrics: { ...experiment().metrics, map: null } })],
      [
        job({
          evaluation: { status: 'succeeded', summary: { metrics: { mAP: 0.49 } } },
        }),
      ],
    );

    expect(record?.metrics.map).toBe(0.49);
  });

  it('같은 run_id의 job이 여럿이면 최신 것이 이긴다', () => {
    // 설정과 seed가 같으면 이름도 같게 지어진다. `/jobs`는 최신순이라 앞이 최신이다.
    const records = mergeRecords(
      [],
      [
        job({ job_id: 'new', status: 'failed', status_label: '실패', finished_at: '2026-01-03T00:00:00Z' }),
        job({ job_id: 'old', status: 'succeeded', status_label: '완료', finished_at: '2026-01-01T00:00:00Z' }),
      ],
    );

    expect(records).toHaveLength(1);
    expect(records[0]?.jobId).toBe('new');
    expect(records[0]?.status).toBe('failed');
  });

  it('dataset을 알 수 없는 기록도 버리지 않는다', () => {
    const [record] = mergeRecords([], [job({ data_inputs: {} })]);

    expect(record?.datasetKey).toBe(UNKNOWN_DATASET);
  });
});

describe('sortRecords', () => {
  it('값이 없는 줄은 정렬 방향과 상관없이 뒤로 간다', () => {
    const scored = mergeRecords([experiment()], []);
    const unscored = mergeRecords(
      [
        experiment({
          run_id: 'no-score',
          metrics: { ...experiment().metrics, kaggle_score: null },
        }),
      ],
      [],
    );

    const sorted = sortRecords([...unscored, ...scored], 'kaggle');

    expect(sorted[0]?.runId).toBe('retina-e15-b4-a7f3');
    expect(sorted[1]?.runId).toBe('no-score');
  });
});

describe('groupByDataset', () => {
  it('dataset 이름을 댈 수 없는 기록은 줄을 만들지 않는다', () => {
    const records = mergeRecords(
      [],
      [
        job({ run_id: 'a', data_inputs: { train_manifest_uri: 'artifacts/data/train_manifest_uri.json' } }),
        job({ run_id: 'b', data_inputs: {} }),
      ],
    );

    expect(records).toHaveLength(2);
    expect(groupByDataset(records)).toEqual([]);
  });

  it('기록이 많은 dataset을 위에 세운다', () => {
    const records = mergeRecords(
      [
        experiment({ run_id: 'a' }),
        experiment({ run_id: 'b' }),
        experiment({
          run_id: 'c',
          dataset: { ...experiment().dataset, label: 'v4-57cls' },
        }),
      ],
      [],
    );

    expect(groupByDataset(records)).toEqual([
      { key: 'v5-118cls', count: 2 },
      { key: 'v4-57cls', count: 1 },
    ]);
  });
});
