import { describe, expect, it } from 'vitest';

import type { Defaults, TrainCapability } from '../api/types';
import { resolveTrainCapability } from './trainCapabilities';

function defaults(capability?: TrainCapability): Defaults {
  return {
    architecture: 'legacy_detector',
    architecture_note: '고정 모델',
    train_capability: capability,
    fields: [],
    data_fields: [],
    devices: [],
  };
}

describe('resolveTrainCapability', () => {
  it('구버전 defaults에는 현재 architecture와 SGD fallback을 붙인다', () => {
    const resolved = resolveTrainCapability(defaults());

    expect(resolved.source).toBe('legacy_fallback');
    expect(resolved.model.default).toBe('legacy_detector');
    expect(resolved.optimizer.default).toBe('SGD');
  });

  it('backend가 보낸 유효한 capability는 그대로 사용한다', () => {
    const capability: TrainCapability = {
      schema_version: 1,
      source: 'train',
      fallback_reason: null,
      model: { default: 'detector_a', choices: ['detector_a'], selection_supported: false },
      optimizer: { default: 'AdamW', choices: ['AdamW'], selection_supported: false },
    };

    expect(resolveTrainCapability(defaults(capability))).toBe(capability);
  });

  it('깨진 capability가 오면 구버전과 같은 fallback을 쓴다', () => {
    const broken = {
      schema_version: 1,
      source: 'train',
      fallback_reason: null,
      model: { default: 'detector_a', choices: [], selection_supported: true },
      optimizer: { default: 'AdamW', choices: ['AdamW'], selection_supported: false },
    } as TrainCapability;

    expect(resolveTrainCapability(defaults(broken)).source).toBe('legacy_fallback');
  });
});
