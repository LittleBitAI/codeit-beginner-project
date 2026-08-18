/**
 * 재순위 embedding 칸 test입니다.
 *
 * 두 가지가 조용히 틀리면 제출이 나빠집니다. 아직 checkpoint가 없는 학습을 고를 수
 * 있는 것, 그리고 참조 crop을 엉뚱한 폴더에서 집는 것입니다. 둘 다 화면에서는
 * 성공처럼 보이므로 여기서 잡습니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { EmbeddingRun, ProcessedDataset } from '../api/types';
import { EmbeddingPanel } from './EmbeddingPanel';

// 로그인한 사람으로 그립니다. 이 학습도 학습 대기열을 지나므로 token 없이 넣으면
// 꺼내 시작할 때 팀 기록을 못 만들어 거절당하고, 그 항목에서 대기열이 멈춥니다.
vi.mock('../team/TeamContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../team/TeamContext')>();
  return {
    ...actual,
    useTeam: () => ({ ...actual.useTeam(), getAccessToken: async () => 'login-token' }),
  };
});

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

function dataset(overrides: Partial<ProcessedDataset> = {}): ProcessedDataset {
  return {
    name: 'v5-seed42-8020-group',
    directory: 'datasets/pill_detection/processed/v5-seed42-8020-group/',
    complete: true,
    missing: [],
    has_test_manifest: true,
    has_eda_report: false,
    has_crop_bank: true,
    ...overrides,
  };
}

function stub(runs: EmbeddingRun[], datasets: ProcessedDataset[] = [dataset()]) {
  vi.spyOn(api, 'embeddingRuns').mockResolvedValue({ runs });
  vi.spyOn(api, 'embeddingDefaults').mockResolvedValue({
    backbones: ['resnet18', 'resnet34', 'resnet50'],
    devices: ['cpu', 'cuda'],
    run_id_pattern: '.*',
    defaults: {
      backbone: 'resnet18',
      epochs: 30,
      batch_size: 32,
      learning_rate: 0.0003,
      weight_decay: 0.0001,
      seed: 42,
      pretrained: true,
      device: 'cpu',
    },
  });
  vi.spyOn(api, 'listDatasets').mockResolvedValue({
    backend: 'local',
    root: 'datasets/pill_detection/processed/',
    datasets,
    problems: [],
  });
}

describe('재순위 embedding 칸', () => {
  it('checkpoint가 없는 학습은 고를 수 없다', async () => {
    // 고를 수 있게 두면 합치기를 끝낸 뒤에야 실패합니다.
    stub([run(), run({ run_id: 'emb-r34', status: 'running', checkpoint_uri: null, ready: false })]);

    render(<EmbeddingPanel selected={[]} onToggle={() => undefined} />);
    await screen.findByText('emb-r34');

    const boxes = screen.getAllByRole('checkbox');
    expect((boxes[0] as HTMLInputElement).disabled).toBe(false);
    expect((boxes[1] as HTMLInputElement).disabled).toBe(true);
  });

  it('고른 전처리 폴더에서 참조 crop과 class map을 함께 집는다', async () => {
    stub([]);
    const start = vi
      .spyOn(api, 'startEmbedding')
      .mockResolvedValue({ config_id: 'c1', run_id: 'web-emb' });

    render(<EmbeddingPanel selected={[]} onToggle={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: '새 embedding 학습' }));
    const picker = await screen.findByLabelText('crop 은행');
    fireEvent.change(picker, {
      target: { value: 'datasets/pill_detection/processed/v5-seed42-8020-group/' },
    });
    fireEvent.click(screen.getByRole('button', { name: '학습 걸기' }));

    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[0]).toMatchObject({
      crop_bank_uri: 'datasets/pill_detection/processed/v5-seed42-8020-group/crop_bank.tar',
      class_map_uri: 'datasets/pill_detection/processed/v5-seed42-8020-group/class_map.json',
    });
  });

  it('학습 화면과 같은 login token을 함께 보낸다', async () => {
    // 없으면 대기열이 그 항목을 꺼낼 때 거절당하고 거기서 멈춥니다.
    stub([]);
    const start = vi
      .spyOn(api, 'startEmbedding')
      .mockResolvedValue({ config_id: 'c1', run_id: 'web-emb' });

    render(<EmbeddingPanel selected={[]} onToggle={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: '새 embedding 학습' }));
    fireEvent.change(await screen.findByLabelText('crop 은행'), {
      target: { value: 'datasets/pill_detection/processed/v5-seed42-8020-group/' },
    });
    fireEvent.click(screen.getByRole('button', { name: '학습 걸기' }));

    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[1]).toBe('login-token');
  });

  it('은행이 없는 전처리 폴더는 고르는 자리에 두지 않는다', async () => {
    // 은행 없는 폴더로 걸면 대기열에 들어간 뒤 자기 차례에 실패합니다.
    stub([], [dataset(), dataset({ name: 'v4-old', has_crop_bank: false })]);

    render(<EmbeddingPanel selected={[]} onToggle={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: '새 embedding 학습' }));
    await screen.findByText('v5-seed42-8020-group');

    expect(screen.queryByText('v4-old')).toBeNull();
  });
});
