/**
 * 재순위 embedding 칸 test입니다.
 *
 * 여기가 조용히 틀리면 제출이 나빠집니다. 아직 checkpoint가 없는 학습을 고를 수 있는
 * 것이 그렇습니다 — 화면에서는 성공처럼 보이고, 합치기를 다 끝낸 뒤에 실패합니다.
 *
 * 학습 칸은 이제 여기 없습니다(`screens/EmbeddingTrainSheet.tsx`).
 */

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { EmbeddingRun } from '../api/types';
import { EmbeddingPanel } from './EmbeddingPanel';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function run(overrides: Partial<EmbeddingRun> = {}): EmbeddingRun {
  return {
    run_id: 'emb-r18',
    job_id: 'job-1',
    status: 'succeeded',
    backbone: 'resnet18',
    epochs: 30,
    checkpoint_uri: 'artifacts/embeddings/emb-r18/best_checkpoint.pt',
    crop_bank_uri: 'datasets/pill_detection/processed/v5/crop_bank.tar',
    created_at: '2026-08-17T00:00:00Z',
    ready: true,
    ...overrides,
  };
}

describe('재순위 embedding 칸', () => {
  it('checkpoint가 없는 학습은 고를 수 없다', async () => {
    // 고를 수 있게 두면 합치기를 끝낸 뒤에야 실패합니다.
    vi.spyOn(api, 'embeddingRuns').mockResolvedValue({
      runs: [run(), run({ run_id: 'emb-r34', status: 'running', checkpoint_uri: null, ready: false })],
    });

    render(<EmbeddingPanel selected={[]} onToggle={() => undefined} />);
    await screen.findByText('emb-r34');

    const boxes = screen.getAllByRole('checkbox');
    expect((boxes[0] as HTMLInputElement).disabled).toBe(false);
    expect((boxes[1] as HTMLInputElement).disabled).toBe(true);
  });
});
