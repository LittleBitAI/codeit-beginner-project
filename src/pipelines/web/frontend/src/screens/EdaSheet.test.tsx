import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EdaSheet } from './EdaSheet';

function report(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: '1.0',
    dataset_directory: 'artifacts/p',
    shape: {
      train: { images: 8396, annotations: 32196, objects_per_image: { '4': 8000 }, images_with_a_repeated_class: 0 },
      validation: { images: 2099, annotations: 8052, objects_per_image: { '4': 1756 }, images_with_a_repeated_class: 0 },
    },
    classes: {
      class_count: 118,
      train_images_per_class: { count: 118, min: 36, p10: 115, median: 283, p90: 500, max: 714 },
      imbalance_ratio: 19.8,
      classes_missing_from_train: [],
      classes_missing_from_validation: [],
      per_class: [],
    },
    combinations: {
      train: { groups: 2882, images_per_group: null },
      validation: { groups: 719, images_per_group: null },
      groups_in_both_splits: 0,
      leaked_group_sample: [],
      capture_conditions: { a: 1, b: 2 },
    },
    object_size: {
      train_annotation_fraction: null,
      validation_annotation_fraction: null,
      calibration: { images: 120, measured_over_annotation: 0.589, limits: [0.5, 1.5], trustworthy: true },
      train_foreground_fraction: { count: 120, min: 0.1, p10: 0.13, median: 0.177, p90: 0.22, max: 0.3 },
      test_foreground_fraction: { count: 842, min: 0.1, p10: 0.12, median: 0.168, p90: 0.21, max: 0.3 },
      test_over_train: { area_ratio: 0.92, length_ratio: 0.959 },
      ...(overrides.object_size as object),
    },
    appearance: {
      train_background_color: [108.5, 131, 153],
      test_background_color: [102, 115, 149.5],
      train_foreground_color: [140.9, 137.2, 123.5],
      test_foreground_color: [128.2, 117.3, 114.2],
      background_color_distance: 17.6,
      foreground_color_distance: 25.4,
    },
    sources: { test_annotations_read: false },
  };
}

function stubFetch(state: Record<string, unknown>) {
  const calls: { path: string; method: string }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ path: String(input), method: init?.method ?? 'GET' });
      return new Response(JSON.stringify({ eda: state }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
  return calls;
}

beforeEach(() => vi.useRealTimers());
afterEach(() => vi.unstubAllGlobals());

describe('EDA 시트', () => {
  it('리포트가 없으면 실행하라고만 말하고 숫자를 지어내지 않는다', async () => {
    stubFetch({ status: 'idle' });

    render(<EdaSheet onClose={() => {}} />);

    expect(await screen.findByText(/아직 리포트가 없습니다/)).toBeTruthy();
  });

  it('리포트가 오면 크기·색·누수를 한 판에 보여 준다', async () => {
    stubFetch({ status: 'succeeded', message: '완료', report: report() });

    render(<EdaSheet onClose={() => {}} />);

    expect(await screen.findByText('118')).toBeTruthy();
    expect(screen.getByText(/변 길이 0.959배/)).toBeTruthy();
    expect(screen.getByText('17.6')).toBeTruthy();
    expect(screen.getByText('누수 없음')).toBeTruthy();
  });

  it('자를 못 믿으면 크기 비교 대신 재지 못했다고 적는다', async () => {
    stubFetch({
      status: 'succeeded',
      report: report({
        object_size: {
          calibration: { images: 120, measured_over_annotation: 0.2, limits: [0.5, 1.5], trustworthy: false },
          train_foreground_fraction: null,
          test_foreground_fraction: null,
          test_over_train: null,
          train_annotation_fraction: null,
          validation_annotation_fraction: null,
        },
      }),
    });

    render(<EdaSheet onClose={() => {}} />);

    expect(await screen.findByText('재지 못했습니다')).toBeTruthy();
    expect(screen.getByText(/구간 밖/)).toBeTruthy();
  });

  it('자를 못 믿으면 비율이 남아 있어도 그리지 않는다', async () => {
    // 손상되거나 옛 schema인 report가 둘을 동시에 들고 올 수 있습니다.
    const broken = report();
    broken.object_size.calibration.trustworthy = false;
    stubFetch({ status: 'succeeded', report: broken });

    render(<EdaSheet onClose={() => {}} />);

    expect(await screen.findByText('재지 못했습니다')).toBeTruthy();
    expect(screen.queryByText(/변 길이 0.959배/)).toBeNull();
  });

  it('다른 dataset의 결과는 이 dataset의 숫자처럼 보여 주지 않는다', async () => {
    stubFetch({ status: 'succeeded', stale: true, report: null });

    render(<EdaSheet onClose={() => {}} />);

    expect(await screen.findByText(/다른 dataset을 분석한 결과/)).toBeTruthy();
  });

  it('리포트를 못 읽어도 요청을 되풀이하지 않는다', async () => {
    const calls = stubFetch({ status: 'succeeded', report: null, stale: false });

    render(<EdaSheet onClose={() => {}} />);
    await screen.findByText(/아직 리포트가 없습니다/);
    const first = calls.length;
    await new Promise((resolve) => setTimeout(resolve, 200));

    expect(calls.length).toBe(first);
  });

  it('실행 버튼이 표본 수를 실어 보낸다', async () => {
    const calls = stubFetch({ status: 'idle' });

    render(<EdaSheet onClose={() => {}} />);
    fireEvent.click(await screen.findByText('EDA 실행'));

    await waitFor(() =>
      expect(calls.some((call) => call.method === 'POST' && call.path.includes('/api/data/eda'))).toBe(true),
    );
  });
});
