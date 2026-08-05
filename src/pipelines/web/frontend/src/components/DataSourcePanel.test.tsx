import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { DataSource } from '../api/types';

const prepareStatus = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    prepareStatus: (...args: unknown[]) => prepareStatus(...args),
    startPreparation: vi.fn(),
    verifyDataSource: vi.fn(),
    inspectDirectory: vi.fn(),
    setDataSource: vi.fn(),
  },
}));

const { DataSourcePanel } = await import('./DataSourcePanel');

const source: DataSource = {
  directory: 'artifacts/data/v1',
  complete: true,
  available: true,
  origin: 'prepared',
  data: {
    train_manifest_uri: 'artifacts/data/v1/train_manifest.json',
    validation_manifest_uri: 'artifacts/data/v1/validation_manifest.json',
    class_map_uri: 'artifacts/data/v1/class_map.json',
    dataset_summary_uri: 'artifacts/data/v1/dataset_summary.json',
    test_manifest_uri: 'artifacts/data/v1/test_manifest.json',
  },
  matched: {
    train_manifest_uri: { name: 'train_manifest.json', uri: 'artifacts/data/v1/train_manifest.json' },
    validation_manifest_uri: { name: 'validation_manifest.json', uri: 'artifacts/data/v1/validation_manifest.json' },
    class_map_uri: { name: 'class_map.json', uri: 'artifacts/data/v1/class_map.json' },
    dataset_summary_uri: { name: 'dataset_summary.json', uri: 'artifacts/data/v1/dataset_summary.json' },
    test_manifest_uri: { name: 'test_manifest.json', uri: 'artifacts/data/v1/test_manifest.json' },
  },
  labels: {
    train_manifest_uri: '학습 manifest',
    validation_manifest_uri: '검증 manifest',
    class_map_uri: '클래스 맵',
    dataset_summary_uri: '데이터셋 요약',
    test_manifest_uri: '테스트 manifest',
  },
  missing: [],
  problems: [],
  examined: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  prepareStatus.mockResolvedValue({
    split_ratios: ['8:2', '9:1'],
    backends: ['auto', 'local'],
    storage: {
      bucket: null,
      bucket_configured: false,
      profile_configured: false,
      region: null,
      forced_backend: null,
      default_backend: 'local',
    },
    preparation: { status: 'idle' },
  });
});

describe('DataSourcePanel · test manifest', () => {
  it('선택 입력으로 인식한 다섯 번째 artifact를 표시한다', async () => {
    render(<DataSourcePanel source={source} onSelected={vi.fn()} onPrepared={vi.fn()} />);

    expect(await screen.findByText('5개 인식됨')).toBeInTheDocument();
    expect(screen.getByText('테스트 manifest')).toBeInTheDocument();
    expect(screen.getByText('test_manifest.json')).toBeInTheDocument();
  });

  it('준비 결과에서 test가 학습에 섞이지 않았음을 수치로 보여 준다', async () => {
    prepareStatus.mockResolvedValue({
      split_ratios: ['8:2', '9:1'],
      backends: ['auto', 'local'],
      storage: {
        bucket: null,
        bucket_configured: false,
        profile_configured: false,
        region: null,
        forced_backend: null,
        default_backend: 'local',
      },
      preparation: {
        status: 'succeeded',
        split_ratio: '8:2',
        finished_at: '2026-08-05T00:00:00Z',
        message: '준비 완료',
        selected: true,
        summary: { test_manifest_images: 842, test_images_used: 0 },
      },
    });

    render(<DataSourcePanel source={null} onSelected={vi.fn()} onPrepared={vi.fn()} />);

    expect(await screen.findByText('테스트 이미지')).toBeInTheDocument();
    expect(screen.getByText('842')).toBeInTheDocument();
    expect(screen.getByText('test 이미지 842장을 학습과 검증에 사용하지 않았습니다.')).toBeInTheDocument();
  });
});
