import { datasetLabel } from './runSpec';

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
