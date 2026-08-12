import { describe, expect, it } from 'vitest';

import type { Progress } from '../api/types';
import { epochsDone, progressRatio } from './progress';

function progress(overrides: Partial<Progress> = {}): Progress {
  return {
    available: true,
    reason: null,
    message: null,
    total_epochs: 12,
    current_epoch: null,
    eta_seconds: null,
    epochs: [],
    ...overrides,
  };
}

describe('진행 epoch 읽기', () => {
  it('이어서 학습한 실행을 0 epoch으로 읽지 않는다', () => {
    // 옛 backend가 남긴 기록입니다. 앞선 10 epoch를 세지 못해 0으로 적혀 있습니다.
    const resumed = progress({ current_epoch: 11, completed_epochs: 0 });

    expect(epochsDone(resumed)).toBe(10);
    expect(progressRatio(resumed)).toBeCloseTo(10 / 12);
  });

  it('첫 epoch을 도는 중에는 아직 끝난 것이 없다', () => {
    expect(epochsDone(progress({ current_epoch: 1, completed_epochs: 0 }))).toBe(0);
  });

  it('끝난 학습은 조기 종료로 계획 epoch가 남아 있어도 100%다', () => {
    expect(progressRatio(progress({ finished: true, current_epoch: 5, completed_epochs: 5 }))).toBe(1);
  });

  it('계획 epoch를 모르면 진행률을 지어내지 않는다', () => {
    expect(progressRatio(progress({ total_epochs: null, current_epoch: 3 }))).toBeNull();
  });
});
