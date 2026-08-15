import { describe, expect, it } from 'vitest';

import type { ExperimentSummary } from '../api/types';
import { rerunSettings } from './rerunSettings';

function summary(overrides: Partial<ExperimentSummary['training']> = {}): ExperimentSummary {
  return {
    model: { architecture: 'dino_r50_4scale', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.0001,
      momentum: null,
      weight_decay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    training: {
      device: 'cuda',
      epochs: 24,
      batch_size: 1,
      num_workers: 2,
      gradient_accumulation_steps: 8,
      input_size: 640,
      precision: 'amp',
      checkpoint_every: 1,
      augmentation: { preset: 'pill_geometric' },
      lr_scheduler: { name: 'cosine', warmup_steps: 1000, min_lr_factor: 0.01 },
      early_stopping: { patience: 4, min_delta: 0 },
      seed: 42,
      ...overrides,
    },
  } as unknown as ExperimentSummary;
}

describe('rerunSettings', () => {
  it('중첩 설정을 화면의 평평한 칸 이름으로 되돌린다', () => {
    const values = rerunSettings(summary());

    expect(values.augmentation).toBe('pill_geometric');
    expect(values.lr_scheduler).toBe('cosine');
    expect(values.lr_warmup_steps).toBe('1000');
    expect(values.lr_min_factor).toBe('0.01');
    expect(values.early_stopping).toBe('true');
    expect(values.early_stopping_patience).toBe('4');
    // 0은 값입니다. 비어 있는 것과 다릅니다.
    expect(values.early_stopping_min_delta).toBe('0');
    expect(values.precision).toBe('amp');
    expect(values.pretrained).toBe('true');
  });

  it('dataset은 담지 않는다', () => {
    const values = rerunSettings(summary());

    for (const name of Object.keys(values)) {
      expect(name).not.toMatch(/manifest|class_map|dataset|artifact/);
    }
  });

  it('이어서 학습한 실행이어도 그 checkpoint를 물려주지 않는다', () => {
    // registry는 `resume_from`을 담습니다. 무엇에서 이어 학습했는지가 설정이기
    // 때문입니다. 그러나 그 값을 새 실험에 채우면 남의 checkpoint에서 출발하는
    // 학습이 만들어집니다. 이어서 하기는 화면에 따로 있는 버튼입니다.
    const values = rerunSettings(
      summary({
        resume_from: 'artifacts/experiments/exp-0000/last_checkpoint.pt',
      } as Partial<ExperimentSummary['training']>),
    );

    expect('resume_from' in values).toBe(false);
    expect(values.epochs).toBe('24');
  });

  it('그 실행이 쓰지 않은 설정은 칸을 비워 둔다', () => {
    const values = rerunSettings(
      summary({ lr_scheduler: null, early_stopping: null, augmentation: null, input_size: null }),
    );

    expect('lr_scheduler' in values).toBe(false);
    expect('lr_warmup_steps' in values).toBe(false);
    expect('early_stopping' in values).toBe(false);
    expect('augmentation' in values).toBe(false);
    expect('input_size' in values).toBe(false);
    // 쓰지 않은 것만 빠지고 나머지는 그대로 채워집니다.
    expect(values.epochs).toBe('24');
  });
});
