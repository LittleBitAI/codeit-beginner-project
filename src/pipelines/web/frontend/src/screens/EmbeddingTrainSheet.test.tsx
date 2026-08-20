/**
 * embedding 학습 시트 test입니다. (앙상블 화면에 있던 학습 칸이 여기로 왔습니다.)
 *
 * 두 가지가 조용히 틀리면 밤이 버려집니다. 참조 crop을 엉뚱한 폴더에서 집는 것과,
 * login token 없이 대기열에 넣는 것입니다. 둘 다 화면에서는 성공처럼 보이고, 실패는
 * 자기 차례가 왔을 때 옵니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { ProcessedDataset } from '../api/types';
import { EmbeddingTrainSheet } from './EmbeddingTrainSheet';

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

function stub(datasets: ProcessedDataset[] = [dataset()], banks: ProcessedDataset[] = []) {
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
  vi.spyOn(api, 'listCropBanks').mockResolvedValue({
    backend: 'local',
    root: 'datasets/pill_detection/crop-bank/',
    datasets: banks,
    problems: [],
  });
}

/** 전처리 폴더 밖에 손으로 올린 은행. manifest가 없으므로 `complete`가 false입니다. */
function handMadeBank(): ProcessedDataset {
  return dataset({
    name: '20260817',
    directory: 'datasets/pill_detection/crop-bank/20260817/',
    complete: false,
    missing: ['train_manifest_uri', 'validation_manifest_uri', 'dataset_summary_uri'],
    has_test_manifest: false,
  });
}

describe('embedding 학습 시트', () => {
  it('고른 전처리 폴더에서 참조 crop과 class map을 함께 집는다', async () => {
    stub();
    const start = vi
      .spyOn(api, 'startEmbedding')
      .mockResolvedValue({ config_id: 'c1', run_id: 'web-emb' });

    render(<EmbeddingTrainSheet onClose={() => undefined} />);
    fireEvent.change(await screen.findByLabelText('crop 은행'), {
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
    stub();
    const start = vi
      .spyOn(api, 'startEmbedding')
      .mockResolvedValue({ config_id: 'c1', run_id: 'web-emb' });

    render(<EmbeddingTrainSheet onClose={() => undefined} />);
    fireEvent.change(await screen.findByLabelText('crop 은행'), {
      target: { value: 'datasets/pill_detection/processed/v5-seed42-8020-group/' },
    });
    fireEvent.click(screen.getByRole('button', { name: '학습 걸기' }));

    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[1]).toBe('login-token');
  });

  it('전처리 폴더 밖에 손으로 올린 은행도 같은 목록에서 고른다', async () => {
    // 0.63594를 만든 은행은 준비가 만든 것이 아니라 `crop-bank/<날짜>/`에 있습니다.
    // 전처리 폴더만 보여 주면 그 은행으로는 학습을 걸 방법이 없습니다.
    stub([dataset()], [handMadeBank()]);
    const start = vi
      .spyOn(api, 'startEmbedding')
      .mockResolvedValue({ config_id: 'c1', run_id: 'web-emb' });

    render(<EmbeddingTrainSheet onClose={() => undefined} />);
    fireEvent.change(await screen.findByLabelText('crop 은행'), {
      target: { value: 'datasets/pill_detection/crop-bank/20260817/' },
    });
    fireEvent.click(screen.getByRole('button', { name: '학습 걸기' }));

    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[0]).toMatchObject({
      crop_bank_uri: 'datasets/pill_detection/crop-bank/20260817/crop_bank.tar',
      class_map_uri: 'datasets/pill_detection/crop-bank/20260817/class_map.json',
    });
  });

  it('은행이 없는 전처리 폴더는 고르는 자리에 두지 않는다', async () => {
    // 은행 없는 폴더로 걸면 대기열에 들어간 뒤 자기 차례에 실패합니다.
    stub([dataset(), dataset({ name: 'v4-old', has_crop_bank: false })]);

    render(<EmbeddingTrainSheet onClose={() => undefined} />);
    await screen.findByText('v5-seed42-8020-group');

    expect(screen.queryByText('v4-old')).toBeNull();
  });
});
