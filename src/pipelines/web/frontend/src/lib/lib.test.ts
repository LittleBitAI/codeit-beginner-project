import { describe, expect, it } from 'vitest';

import { color, radius, type } from '../design/tokens';
import { describeRun, diffAgainstDefaults } from './describeRun';
import { toPayload, messageFor } from './draftPayload';
import { duration, loss, megabytes, percent } from './format';
import type { RuntimeConfig, FieldSpec } from '../api/types';

const FIELDS: FieldSpec[] = [
  { name: 'epochs', type: 'integer', default: 1, label: 'Epochs', hint: '' },
  { name: 'learning_rate', type: 'number', default: 0.005, label: 'LR', hint: '' },
  { name: 'pretrained', type: 'boolean', default: false, label: 'Pretrained', hint: '' },
  { name: 'run_id', type: 'string', label: '실행 이름', hint: '' },
];

describe('디자인 토큰', () => {
  it('반경이 6px를 넘지 않는다', () => {
    // 디자인 하드 제약: badge 3~4, control 4, panel 5~6. 더 둥근 값은 금지입니다.
    for (const value of Object.values(radius)) {
      expect(value).toBeLessThanOrEqual(6);
    }
  });

  it('모든 색이 6자리 hex이다', () => {
    for (const value of Object.values(color)) {
      expect(value).toMatch(/^#[0-9A-F]{6}$/i);
    }
  });

  it('숫자를 다루는 타입은 mono 글꼴을 쓴다', () => {
    // 소수점 정렬이 실행 간 비교의 전제라 숫자는 예외 없이 mono입니다.
    for (const key of ['tableCell', 'kpiLarge', 'kpiCompact', 'logLine', 'code'] as const) {
      expect(type[key].font).toContain('IBM Plex Mono');
    }
  });
});

describe('toPayload', () => {
  it('빈 값은 보내지 않아 backend 기본값을 쓴다', () => {
    const payload = toPayload({ train: { epochs: '', run_id: '  ' }, data: {} }, FIELDS);

    expect(payload.train).toEqual({});
  });

  it('정수와 실수를 알맞은 타입으로 바꾼다', () => {
    const payload = toPayload({ train: { epochs: '12', learning_rate: '0.001' }, data: {} }, FIELDS);

    expect(payload.train.epochs).toBe(12);
    expect(payload.train.learning_rate).toBe(0.001);
  });

  it('숫자가 아닌 값은 바꾸지 않고 그대로 보내 서버가 거부하게 한다', () => {
    // parseInt('3abc')를 3으로 통과시키면 train이 거부할 값을 GUI가 받아들이게 됩니다.
    const payload = toPayload({ train: { epochs: '3abc', learning_rate: '빠르게' }, data: {} }, FIELDS);

    expect(payload.train.epochs).toBe('3abc');
    expect(payload.train.learning_rate).toBe('빠르게');
  });

  it('boolean은 실제 true/false로 보낸다', () => {
    expect(toPayload({ train: { pretrained: 'true' }, data: {} }, FIELDS).train.pretrained).toBe(true);
    expect(toPayload({ train: { pretrained: 'false' }, data: {} }, FIELDS).train.pretrained).toBe(false);
  });

  it('data 입력의 공백을 정리하고 빈 값은 뺀다', () => {
    const payload = toPayload(
      { train: {}, data: { class_map_uri: '  artifacts/a.json ', dataset_summary_uri: '' } },
      FIELDS,
    );

    expect(payload.inputs.data).toEqual({ class_map_uri: 'artifacts/a.json' });
  });
});

describe('messageFor', () => {
  it('해당 field의 메시지를 찾는다', () => {
    const messages = [{ field: 'train.epochs', message: '1 이상' }];

    expect(messageFor(messages, 'train.epochs')).toBe('1 이상');
    expect(messageFor(messages, 'train.seed')).toBeUndefined();
  });
});

describe('describeRun', () => {
  const config: RuntimeConfig = {
    project: { name: 'pill-object-detection' },
    execution: { mode: 'real' },
    storage: { backend: 'local', local: { root: 'artifacts' } },
    train: {
      run_id: 'exp-1',
      epochs: 10,
      batch_size: 2,
      learning_rate: 0.005,
      momentum: 0.9,
      weight_decay: 0.0005,
      seed: 42,
      num_workers: 0,
      device: 'cuda',
      pretrained: true,
      output_dir: 'artifacts/experiments/completed',
    },
    inputs: {
      data: { a: '1', b: '2', c: '3', d: '4' },
    },
  };

  it('설정 값을 문장에 그대로 담는다', () => {
    const text = describeRun(config);

    expect(text).toContain('10 epoch');
    expect(text).toContain('batch 2');
    expect(text).toContain('CUDA GPU');
    expect(text).toContain('COCO 사전학습 가중치');
    expect(text).toContain('exp-1');
    expect(text).toContain('로컬 디스크');
  });

  it('설정이 없으면 지어내지 않는다', () => {
    expect(describeRun(null)).toBe('설정이 아직 준비되지 않았습니다.');
  });
});

describe('diffAgainstDefaults', () => {
  it('기본값과 다른 항목만 뽑는다', () => {
    const rows = diffAgainstDefaults({ epochs: 10, seed: 42 }, { epochs: 1, seed: 42 });

    expect(rows).toEqual([{ key: 'train.epochs', before: '1', after: '10' }]);
  });
});

describe('format', () => {
  it('값이 없으면 지어내지 않고 "-"를 쓴다', () => {
    expect(loss(null)).toBe('-');
    expect(duration(undefined)).toBe('-');
    expect(megabytes(null)).toBe('-');
    expect(percent(null)).toBe('-');
  });

  it('손실은 소수점 4자리로 정렬한다', () => {
    expect(loss(0.5)).toBe('0.5000');
  });

  it('시간을 사람이 읽는 단위로 바꾼다', () => {
    expect(duration(45)).toBe('45초');
    expect(duration(125)).toBe('2분 5초');
    expect(duration(3700)).toBe('1시간 1분');
  });

  it('0인 자리는 붙이지 않는다', () => {
    expect(duration(720)).toBe('12분');
    expect(duration(3600)).toBe('1시간');
  });

  it('메모리를 1024MB부터 GB로 표시한다', () => {
    expect(megabytes(512)).toBe('512 MB');
    expect(megabytes(2048)).toBe('2.0 GB');
  });
});
