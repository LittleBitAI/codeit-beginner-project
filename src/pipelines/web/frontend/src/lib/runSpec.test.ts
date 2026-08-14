import type { JobRecord } from '../api/types';
import { datasetLabel, specLine, stagesOf } from './runSpec';

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    job_id: 'a'.repeat(32),
    config_id: 'c',
    run_id: 'run',
    status: 'succeeded',
    status_label: '성공',
    created_at: '2026-08-09T00:00:00Z',
    started_at: null,
    finished_at: null,
    elapsed_seconds: null,
    exit_code: 0,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: { available: false, reason: null, message: null, total_epochs: null, current_epoch: null, eta_seconds: null, epochs: [] },
    log_lines: 0,
    orphan_note: null,
    ...overrides,
  };
}

describe('datasetLabel', () => {
  it('s3 URI에서 manifest가 든 폴더 이름만 꺼낸다', () => {
    const label = datasetLabel({
      train_manifest_uri:
        's3://pill-detection-team/datasets/pill_detection/processed/v3-seed42-8020-group/train_manifest.json',
    });

    expect(label).toBe('v3-seed42-8020-group');
  });

  it('저장소 상대 경로도 같은 규칙으로 읽는다', () => {
    expect(datasetLabel({ train_manifest_uri: 'artifacts/data/v4-8020/train_manifest.json' })).toBe(
      'v4-8020',
    );
  });

  it('폴더가 없거나 값이 없으면 지어내지 않는다', () => {
    expect(datasetLabel({ train_manifest_uri: 'train_manifest.json' })).toBeNull();
    expect(datasetLabel({})).toBeNull();
    expect(datasetLabel(null)).toBeNull();
  });

  it('manifest가 아닌 파일을 가리키면 담고 있던 폴더 이름을 데이터셋으로 삼지 않는다', () => {
    // 값 대신 field 이름이 적힌 옛 기록. 예전에는 `data`가 데이터셋으로 잡혔습니다.
    expect(datasetLabel({ train_manifest_uri: 'artifacts/data/train_manifest_uri.json' })).toBeNull();
    // pytest 임시 폴더가 남긴 기록. 예전에는 `fixtures`가 데이터셋으로 잡혔습니다.
    expect(
      datasetLabel({ train_manifest_uri: 'artifacts/pytest-main/test_run0/fixtures/train.json' }),
    ).toBeNull();
  });
});

describe('specLine', () => {
  it('데이터셋과 학습 설정을 한 줄로 잇는다', () => {
    const line = specLine(
      job({
        data_inputs: { train_manifest_uri: 'artifacts/data/v3-8020/train_manifest.json' },
        settings: { device: 'cuda', optimizer: 'AdamW', seed: 42 },
      }),
    );

    expect(line).toBe('v3-8020 · cuda · AdamW · seed 42');
  });

  it('모르는 값은 빼고 잇는다', () => {
    expect(specLine(job({ settings: { optimizer: 'SGD' } }))).toBe('SGD');
    expect(specLine(job())).toBe('');
  });
});

describe('stagesOf', () => {
  it('평가와 등록까지 끝나야 제출이 끝난 것으로 본다', () => {
    const done = stagesOf(
      job({
        evaluation: { status: 'succeeded', artifacts: { submission_uri: 's3://b/submission.csv' } },
        registration: { status: 'succeeded' },
      }),
    );

    expect(done.map((stage) => stage.done)).toEqual([true, true, true]);
  });

  it('submission_requested가 없는 예전 기록도 결과물이 있으면 제출로 센다', () => {
    // 그 field는 나중에 생겨서, 이미 제출을 만든 기록에는 아예 없습니다.
    const stages = stagesOf(
      job({
        evaluation: {
          status: 'succeeded',
          artifacts: { metrics_uri: 'a.json', submission_uri: 's3://b/submission.csv' },
        },
        registration: { status: 'succeeded' },
      }),
    );

    expect(stages[2]!.done).toBe(true);
  });

  it('submission을 만들지 않은 평가는 제출로 세지 않는다', () => {
    const stages = stagesOf(
      job({
        evaluation: { status: 'succeeded', artifacts: { metrics_uri: 'a.json' } },
        registration: { status: 'succeeded' },
      }),
    );

    expect(stages.map((stage) => stage.done)).toEqual([true, true, false]);
  });

  it('등록이 index_failed면 실험 목록에 안 나오므로 제출도 끝나지 않았다', () => {
    const stages = stagesOf(
      job({
        evaluation: { status: 'succeeded', artifacts: { submission_uri: 's3://b/s.csv' } },
        registration: { status: 'index_failed' },
      }),
    );

    expect(stages[2]!.done).toBe(false);
  });

  it('학습이 성공하지 않으면 뒤 단계는 모두 남아 있다', () => {
    expect(stagesOf(job({ status: 'failed' })).map((stage) => stage.done)).toEqual([
      false,
      false,
      false,
    ]);
  });
});

