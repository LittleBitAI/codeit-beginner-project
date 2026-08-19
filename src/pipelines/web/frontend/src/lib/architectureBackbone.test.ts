import { describe, expect, it } from 'vitest';

import {
  architectureForFamily,
  architectureOf,
  backboneOf,
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
