import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { DataSource } from '../api/types';

const prepareStatus = vi.fn();
const listDatasets = vi.fn();
const setDataSource = vi.fn();

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    prepareStatus: (...args: unknown[]) => prepareStatus(...args),
    listDatasets: (...args: unknown[]) => listDatasets(...args),
    setDataSource: (...args: unknown[]) => setDataSource(...args),
    startPreparation: vi.fn(),
    verifyDataSource: vi.fn(),
    inspectDirectory: vi.fn(),
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

function listing(names: { name: string; complete?: boolean }[] = []) {
  return {
    backend: 's3' as const,
    root: 's3://bucket/datasets/pill_detection/processed/',
    datasets: names.map(({ name, complete = true }) => ({
      name,
      directory: `s3://bucket/datasets/pill_detection/processed/${name}/`,
      complete,
      missing: complete ? [] : ['class_map_uri'],
      has_test_manifest: complete,
      has_eda_report: name.endsWith('angle'),
    })),
    problems: [],
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  listDatasets.mockResolvedValue(listing());
  setDataSource.mockImplementation(async (directory: string) => ({
    source: { ...source, directory },
  }));
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

describe('DataSourcePanel · 전처리 dataset 고르기', () => {
  it('있는 폴더를 눌러서 고른다', async () => {
    listDatasets.mockResolvedValue(
      listing([{ name: 'v5-seed42-8020-group' }, { name: 'v5-seed42-8020-group-angle' }]),
    );
    const onSelected = vi.fn();

    render(<DataSourcePanel source={null} onSelected={onSelected} onPrepared={vi.fn()} />);
    fireEvent.click(await screen.findByText(/v5-seed42-8020-group-angle/));

    await waitFor(() =>
      expect(setDataSource).toHaveBeenCalledWith(
        's3://bucket/datasets/pill_detection/processed/v5-seed42-8020-group-angle/',
      ),
    );
    await waitFor(() => expect(onSelected).toHaveBeenCalled());
  });

  it('필수 artifact가 없는 폴더는 고를 수 없다', async () => {
    listDatasets.mockResolvedValue(listing([{ name: 'v9-broken', complete: false }]));

    render(<DataSourcePanel source={null} onSelected={vi.fn()} onPrepared={vi.fn()} />);

    expect(((await screen.findByText(/v9-broken/)) as HTMLButtonElement).disabled).toBe(true);
  });

  it('목록이 비어도 경로를 붙여넣는 길은 남는다', async () => {
    render(<DataSourcePanel source={null} onSelected={vi.fn()} onPrepared={vi.fn()} />);

    expect(await screen.findByPlaceholderText(/s3:\/\/bucket/)).toBeInTheDocument();
    expect(screen.queryByText(/아래에서 찾은 폴더/)).toBeNull();
  });
});
