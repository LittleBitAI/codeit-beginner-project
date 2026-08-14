import { describe, expect, it } from 'vitest';

import { color, palette, radius, type } from '../design/tokens';
import { toPayload, messageFor } from './draftPayload';
import { duration, loss, megabytes, percent } from './format';
import type { DataSource, FieldSpec } from '../api/types';
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

/** WCAG 상대 휘도. 두 색이 얼마나 갈라져 보이는지 재는 데 씁니다. */
function luminance(hex: string): number {
  const channels = [1, 3, 5].map((start) => {
    const value = parseInt(hex.slice(start, start + 2), 16) / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrast(front: string, back: string): number {
  const [bright, dim] = [luminance(front), luminance(back)].sort((a, b) => b - a) as [number, number];
  return (bright + 0.05) / (dim + 0.05);
}

describe('디자인 토큰', () => {
  it('반경이 6px를 넘지 않는다', () => {
    // 디자인 하드 제약: badge 3~4, control 4, panel 5~6. 더 둥근 값은 금지입니다.
    for (const value of Object.values(radius)) {
      expect(value).toBeLessThanOrEqual(6);
    }
  });

  it('화면 코드가 쓰는 색은 CSS 변수 참조다', () => {
    // hex를 직접 돌려주면 테마를 바꿀 때 350군데를 다시 그려야 합니다.
    for (const value of Object.values(color)) {
      expect(value).toMatch(/^var\(--color-[a-z-]+\)$/);
    }
  });

  it('두 판이 같은 색 이름을 모두 채운다', () => {
    // 한쪽에만 있는 이름이 생기면 그 테마에서 그 자리가 통째로 비어 보입니다.
    expect(Object.keys(palette.light).sort()).toEqual(Object.keys(palette.dark).sort());
    for (const value of [...Object.values(palette.dark), ...Object.values(palette.light)]) {
      expect(value).toMatch(/^#[0-9A-F]{6}$/i);
    }
  });

  it('두 판 모두 글자가 읽히는 대비를 지킨다', () => {
    // 밝은 판에서 amber를 그대로 쓰면 1.9:1이라 글자가 안 보입니다. 색을 바꿀 때
    // 이 선을 넘지 않았는지 여기서 걸립니다.
    for (const theme of ['dark', 'light'] as const) {
      const shade = palette[theme];
      expect(contrast(shade.text, shade.page)).toBeGreaterThanOrEqual(4.5);
      expect(contrast(shade.accent, shade.page)).toBeGreaterThanOrEqual(4.5);
      expect(contrast(shade.onAccent, shade.accent)).toBeGreaterThanOrEqual(4.5);
      // 보조 글자는 원본 디자인 값 그대로(두 판 모두 약 4.3:1)라 큰 글자 기준을 씁니다.
      expect(contrast(shade.textMuted, shade.page)).toBeGreaterThanOrEqual(3);
      // 강조색 위에 흰색을 박아 두면 어두운 판에서 2.1:1이 됩니다. onAccent를 씁니다.
      expect(contrast('#FFFFFF', shade.accent)).toBeGreaterThanOrEqual(
        theme === 'light' ? 4.5 : 0,
      );
    }
  });

  it('강조색 위에는 흰색을 박아 쓰지 않는다', () => {
    // 어두운 판의 amber 위 흰 글자는 2.1:1이라 읽히지 않습니다. 실제로 dataset 준비
    // 화면의 선택된 비율 버튼이 그랬습니다.
    expect(contrast('#FFFFFF', palette.dark.accent)).toBeLessThan(4.5);
    expect(contrast(palette.dark.onAccent, palette.dark.accent)).toBeGreaterThanOrEqual(4.5);
  });

  it('숫자를 다루는 타입은 mono 글꼴을 쓴다', () => {
    // 소수점 정렬이 실행 간 비교의 전제라 숫자는 예외 없이 mono입니다.
    for (const key of ['tableCell', 'kpiLarge', 'kpiSmall', 'logLine', 'code'] as const) {
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

  it('고른 모델이 쓰지 않는 칸은 값이 남아 있어도 보내지 않는다', () => {
    // MMDetection을 고르고 입력 크기를 적었다가 모델을 되돌리면 화면에서는 칸이
    // 사라지지만 draft에는 값이 남습니다. 그대로 실어 보내면 서버가 거부하는데,
    // 사용자에게는 그 칸이 보이지 않아 지울 수도 없습니다.
    const fields: FieldSpec[] = [
      ...FIELDS,
      {
        name: 'input_size',
        type: 'integer',
        default: 640,
        only_for_architectures: ['resnet'],
        label: '입력 크기',
        hint: '',
      },
    ];
    const draft = { train: { input_size: '800' }, data: {} };

    expect(toPayload({ ...draft, train: { ...draft.train, architecture: 'mobile' } }, fields).train)
      .not.toHaveProperty('input_size');
    expect(toPayload({ ...draft, train: { ...draft.train, architecture: 'resnet' } }, fields).train.input_size)
      .toBe(800);
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
