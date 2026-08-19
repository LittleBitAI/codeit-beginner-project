import { describe, expect, it } from 'vitest';

import {
  architectureForFamily,
  architectureOf,
  backboneOf,
  displayChoices,
  familyOf,
} from './architectureBackbone';

const TABLE = {
  dino: {
    resnet50: 'dino_r50_4scale',
    swin_t: 'dino_swin_t_4scale',
    swin_b: 'dino_swin_b_4scale',
  },
};
const DEFAULTS = { dino: 'resnet50' };

describe('architectureBackbone', () => {
  it('접힌 이름과 backbone을 architecture 하나로 오간다', () => {
    expect(familyOf(TABLE, 'dino_swin_b_4scale')).toBe('dino');
    expect(backboneOf(TABLE, 'dino_swin_b_4scale')).toBe('swin_b');
    expect(architectureOf(TABLE, 'dino', 'swin_b')).toBe('dino_swin_b_4scale');
  });

  it('모델 목록을 갈래 하나로 접되 순서와 나머지는 그대로 둔다', () => {
    // 서버가 주는 choices는 계약의 진짜 이름입니다. 접는 일은 여기서만 합니다 —
    // 서버 목록에 접힌 이름을 실으면 그것을 그대로 보내는 소비자가 거절당합니다.
    expect(
      displayChoices(TABLE, [
        'fasterrcnn_resnet50_fpn_v2',
        'dino_r50_4scale',
        'dino_swin_t_4scale',
        'dino_swin_b_4scale',
        'cascade_rcnn_swin_t_fpn',
      ]),
    ).toEqual(['fasterrcnn_resnet50_fpn_v2', 'dino', 'cascade_rcnn_swin_t_fpn']);
  });

  it('표에 없는 이름은 접지 않고 그대로 보여 준다', () => {
    // 계약에 이름이 늘었는데 표에 넣지 않아도 화면에서 사라지지 않습니다.
    expect(displayChoices(TABLE, ['dino_swin_l_5scale'])).toEqual(['dino_swin_l_5scale']);
  });

  it('접히지 않는 모델은 그대로 둔다', () => {
    expect(familyOf(TABLE, 'cascade_rcnn_swin_t_fpn')).toBeUndefined();
    expect(backboneOf(TABLE, 'cascade_rcnn_swin_t_fpn')).toBeUndefined();
    expect(architectureForFamily(TABLE, DEFAULTS, 'cascade_rcnn_swin_t_fpn', 'x')).toBe(
      'cascade_rcnn_swin_t_fpn',
    );
  });

  it('같은 갈래를 다시 고르면 고른 backbone을 지킨다', () => {
    // 모델 칸을 건드리지 않았는데 backbone이 기본값으로 되돌아가면, 사람은 swin_b를
    // 골라 둔 채로 resnet50을 학습합니다.
    expect(architectureForFamily(TABLE, DEFAULTS, 'dino', 'dino_swin_b_4scale')).toBe(
      'dino_swin_b_4scale',
    );
  });

  it('다른 모델에서 갈래로 옮기면 기본 backbone으로 시작한다', () => {
    expect(architectureForFamily(TABLE, DEFAULTS, 'dino', 'retinanet_resnet50_fpn_v2')).toBe(
      'dino_r50_4scale',
    );
  });
});
