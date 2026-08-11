import { describe, expect, it } from 'vitest';

import { color, radius, type } from '../design/tokens';
import { describeRun, diffAgainstDefaults } from './describeRun';
import { toPayload, messageFor } from './draftPayload';
import { duration, loss, megabytes, percent } from './format';
import type { DataSource, RuntimeConfig, FieldSpec } from '../api/types';
import { DATA_KEYS } from './dataKeys';
import { dataMatchesSource, sourceKeyOf } from './dataSource';

const FIELDS: FieldSpec[] = [
  {
    name: 'architecture',
    type: 'enum',
    default: 'mobile',
    choices: ['mobile', 'resnet'],
    label: 'Model',
    hint: '',
  },
  {
    name: 'optimizer',
    type: 'enum',
    default: 'AdamW',
    choices: ['AdamW', 'SGD', 'Adam'],
    label: 'Optimizer',
    hint: '',
  },
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
  it('빈 수치는 보내지 않고 enum과 boolean 기본값만 명시한다', () => {
    const payload = toPayload({ train: { epochs: '', run_id: '  ' }, data: {} }, FIELDS);

    expect(payload.train).toEqual({ architecture: 'mobile', optimizer: 'AdamW', pretrained: false });
  });

  it('손대지 않은 boolean은 화면이 알려 준 기본값을 그대로 실어 보낸다', () => {
    // 서버 fallback은 train 기본값(False)을 따릅니다. GUI 기본값을 바꿔도 명시해
    // 보내지 않으면 그 fallback이 이깁니다.
    const fields = FIELDS.map((spec) =>
      spec.name === 'pretrained' ? { ...spec, default: true } : spec,
    );

    expect(toPayload({ train: {}, data: {} }, fields).train.pretrained).toBe(true);
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

  it('조기 종료를 끄면 patience와 min delta를 보내지 않는다', () => {
    // 화면에서 숨긴 값을 payload에 남기면 서버가 "쓰지 않는 값"이라고 거부합니다.
    const fields: FieldSpec[] = [
      ...FIELDS,
      { name: 'early_stopping', type: 'boolean', default: false, label: '조기 종료', hint: '' },
      { name: 'early_stopping_patience', type: 'integer', default: 5, label: 'Patience', hint: '' },
      { name: 'early_stopping_min_delta', type: 'number', default: 0, label: 'Min delta', hint: '' },
    ];

    const off = toPayload(
      { train: { early_stopping: 'false', early_stopping_patience: '5' }, data: {} },
      fields,
    );
    expect(off.train.early_stopping_patience).toBeUndefined();
    expect(off.train.early_stopping_min_delta).toBeUndefined();

    const on = toPayload(
      { train: { early_stopping: 'true', early_stopping_patience: '5' }, data: {} },
      fields,
    );
    expect(on.train.early_stopping_patience).toBe(5);
  });

  it('고른 schedule이 쓰지 않는 칸은 보내지 않는다', () => {
    // 화면에서 감춘 값을 payload에 남기면 서버가 "그 schedule에서 쓰지 않는 값"이라며
    // 저장을 막습니다. 숨김 규칙과 제외 규칙은 같은 함수에서 나와야 합니다.
    const fields: FieldSpec[] = [
      ...FIELDS,
      {
        name: 'lr_scheduler',
        type: 'enum',
        default: 'none',
        choices: ['none', 'cosine', 'step', 'linear'],
        label: 'Schedule',
        hint: '',
      },
      { name: 'lr_min_factor', type: 'number', default: 0.01, label: '최저 배율', hint: '' },
      { name: 'lr_step_size', type: 'integer', default: 3, label: '간격', hint: '' },
      { name: 'lr_gamma', type: 'number', default: 0.1, label: '배율', hint: '' },
    ];
    const draft = {
      train: { lr_scheduler: 'cosine', lr_min_factor: '0.05', lr_step_size: '3', lr_gamma: '0.1' },
      data: {},
    };

    const payload = toPayload(draft, fields);

    expect(payload.train.lr_min_factor).toBe(0.05);
    expect(payload.train.lr_step_size).toBeUndefined();
    expect(payload.train.lr_gamma).toBeUndefined();
  });

  it('손대지 않은 device도 화면에 보이는 기본값 그대로 실어 보낸다', () => {
    // GPU가 있는 PC에서 서버는 device 기본값을 cuda로, precision을 amp로 내려 줍니다.
    // 둘은 짝이라 device만 빼고 보내면 서버 fallback인 cpu가 이깁니다. 그러면 화면에는
    // cuda가 보이는데 "amp 정밀도는 device가 cuda일 때만" 오류가 폼을 열자마자 떴습니다.
    const fields: FieldSpec[] = [
      ...FIELDS,
      { name: 'device', type: 'enum', default: 'cuda', choices: ['cpu', 'cuda'], label: 'Device', hint: '' },
      { name: 'precision', type: 'enum', default: 'amp', choices: ['fp32', 'amp'], label: '연산 정밀도', hint: '' },
    ];

    const payload = toPayload({ train: {}, data: {} }, fields);

    expect(payload.train.device).toBe('cuda');
    expect(payload.train.precision).toBe('amp');
  });

  it('data 입력의 공백을 정리하고 빈 값은 뺀다', () => {
    const payload = toPayload(
      { train: {}, data: { class_map_uri: '  artifacts/a.json ', dataset_summary_uri: '' } },
      FIELDS,
    );

    expect(payload.inputs.data).toEqual({ class_map_uri: 'artifacts/a.json' });
  });

  it('새 enum 기본값을 명시하고 optimizer와 무관한 수치는 보내지 않는다', () => {
    const fields: FieldSpec[] = [
      ...FIELDS,
      { name: 'momentum', type: 'number', default: 0.9, label: 'Momentum', hint: '' },
      { name: 'beta1', type: 'number', default: 0.9, label: 'Beta 1', hint: '' },
      { name: 'epsilon', type: 'number', default: 1e-8, label: 'Epsilon', hint: '' },
    ];

    const adam = toPayload(
      { train: { momentum: '0.7', beta1: '0.8' }, data: {} },
      fields,
    );
    expect(adam.train).toMatchObject({ architecture: 'mobile', optimizer: 'AdamW', beta1: 0.8 });
    expect(adam.train).not.toHaveProperty('momentum');

    const sgd = toPayload(
      { train: { optimizer: 'SGD', momentum: '0.7', beta1: '0.8' }, data: {} },
      fields,
    );
    expect(sgd.train).toMatchObject({ architecture: 'mobile', optimizer: 'SGD', momentum: 0.7 });
    expect(sgd.train).not.toHaveProperty('beta1');
    expect(sgd.train).not.toHaveProperty('epsilon');
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

  it('조기 종료를 쓰면 언제 멈추는지도 말한다', () => {
    const text = describeRun({
      ...config,
      train: { ...config.train, early_stopping: { patience: 5, min_delta: 0.01 } },
    });

    expect(text).toContain('5 epoch');
    expect(text).toContain('0.01');
  });

  it('조기 종료를 쓰지 않으면 그 말을 꺼내지 않는다', () => {
    expect(describeRun(config)).not.toContain('조기 종료');
  });

  it('고른 모델과 optimizer를 그대로 말한다', () => {
    // 이름을 단정해 두면 다른 모델을 골라도 화면은 늘 같은 이름을 말합니다.
    // 사용자는 무엇으로 학습하는지 여기서만 확인하므로 기록이 조용히 틀어집니다.
    const text = describeRun({
      ...config,
      train: { ...config.train, architecture: 'dino_r50_4scale', optimizer: 'AdamW' },
    });

    expect(text).toContain('dino_r50_4scale');
    expect(text).toContain('AdamW');
    expect(text).not.toContain('torchvision Faster R-CNN');
  });

  it('모아서 갱신하면 유효 batch를 알려 준다', () => {
    const text = describeRun({
      ...config,
      train: { ...config.train, batch_size: 1, gradient_accumulation_steps: 8 },
    });

    expect(text).toContain('유효 batch는 8');
  });

  it('모으지 않으면 그 말을 꺼내지 않는다', () => {
    expect(describeRun(config)).not.toContain('유효 batch');
  });

  it('MMDetection 모델이면 입력 크기를 말한다', () => {
    const text = describeRun({
      ...config,
      train: { ...config.train, architecture: 'dino_r50_4scale', input_size: 640 },
    });

    expect(text).toContain('긴 변이 640');
    // 쓰지 않는 실행에는 그 말을 꺼내지 않습니다.
    expect(describeRun(config)).not.toContain('긴 변이');
  });

  it('learning rate schedule을 쓰면 어떻게 변하는지 말한다', () => {
    const text = describeRun({
      ...config,
      train: {
        ...config.train,
        lr_scheduler: { name: 'cosine', warmup_steps: 500, warmup_start_factor: 0.001 },
      },
    });

    expect(text).toContain('곡선');
    // warmup은 batch가 아니라 optimizer 갱신을 셉니다. 모아서 갱신하면 둘이 달라져,
    // batch라고 적어 두면 실제보다 짧은 구간을 말하게 됩니다.
    expect(text).toContain('500번의 갱신');
  });

  it('schedule을 쓰지 않으면 learning rate 이야기를 꺼내지 않는다', () => {
    expect(describeRun(config)).not.toContain('batch 동안');
  });
});

describe('diffAgainstDefaults', () => {
  it('기본값과 다른 항목만 뽑는다', () => {
    const rows = diffAgainstDefaults({ epochs: 10, seed: 42 }, { epochs: 1, seed: 42 });

    expect(rows).toEqual([{ key: 'train.epochs', before: '1', after: '10' }]);
  });
});

describe('데이터셋 바뀜 감지', () => {
  const base = 's3://bucket/processed/v1-seed42-8020/';
  const other = 'artifacts/datasets/processed/v1-seed42-8020/';

  function makeSource(prefix: string, complete = true): DataSource {
    return {
      directory: prefix,
      complete,
      data: Object.fromEntries(DATA_KEYS.map((key) => [key, prefix + key + '.json'])),
      matched: {},
      labels: {},
      missing: [],
      problems: [],
      examined: [],
    };
  }

  it('읽을 파일이 바뀌면 다른 값이 된다', () => {
    expect(sourceKeyOf(makeSource(base))).not.toBe(sourceKeyOf(makeSource(other)));
  });

  it('같은 데이터셋이면 같은 값이 된다', () => {
    expect(sourceKeyOf(makeSource(base))).toBe(sourceKeyOf(makeSource(base)));
  });

  it('아직 완전하지 않은 데이터셋은 값이 없다', () => {
    expect(sourceKeyOf(makeSource(base, false))).toBeNull();
    expect(sourceKeyOf(null)).toBeNull();
  });

  it('칸의 값이 고른 데이터셋과 같으면 일치로 본다', () => {
    const source = makeSource(base);

    expect(dataMatchesSource(source.data, source)).toBe(true);
  });

  it('데이터셋을 바꿨는데 칸에 예전 값이 남아 있으면 잡아낸다', () => {
    // 실제로 이것 때문에 화면에는 S3 데이터가 보이는데 로컬 표본으로 학습된 적이 있습니다.
    const stale = makeSource(other).data;

    expect(dataMatchesSource(stale, makeSource(base))).toBe(false);
  });

  it('한 칸만 달라도 잡아낸다', () => {
    const source = makeSource(base);
    const edited = { ...source.data, class_map_uri: 'artifacts/다른곳/class_map.json' };

    expect(dataMatchesSource(edited, source)).toBe(false);
  });

  it('칸이 비어 있으면 일치가 아니다', () => {
    expect(dataMatchesSource({}, makeSource(base))).toBe(false);
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
