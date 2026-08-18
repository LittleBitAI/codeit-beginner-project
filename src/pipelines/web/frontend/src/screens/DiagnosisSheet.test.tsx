/**
 * 오검출 진단 시트 test입니다.
 *
 * 이 화면의 값은 전부 "이미 잰 것을 읽은 것"이라 틀려도 오류가 나지 않습니다.
 * 잘린 목록을 전부로 읽는 것, 없는 것을 0으로 읽는 것, background를 class 이름으로
 * 읽는 것 — 셋 다 화면에서는 멀쩡해 보이므로 여기서 잡습니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { ExperimentDetail } from '../api/types';
import { DiagnosisSheet } from './DiagnosisSheet';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function detail(overrides: Record<string, unknown> = {}): ExperimentDetail {
  return {
    experiment: { run_id: 'dino-e12' },
    history: { available: false, reason: null, epochs: [] },
    evaluation: {
      available: true,
      reason: null,
      score_sweep: {
        '0.50': [
          { threshold: 0.05, precision: 0.7, recall: 1.0, f1: 0.82 },
          { threshold: 0.5, precision: 0.95, recall: 0.93, f1: 0.94 },
        ],
        '0.75': [{ threshold: 0.5, precision: 0.6, recall: 0.5, f1: 0.55 }],
      },
      best_f1: {
        '0.50': { threshold: 0.5, precision: 0.95, recall: 0.93, f1: 0.94 },
        '0.75': null,
      },
      confusions: {
        '0.50': [
          { truth_id: 1, truth: '타이레놀500', predicted_id: 2, predicted: '타이레놀650', count: 41 },
          { truth_id: null, truth: 'background', predicted_id: 5, predicted: '리피토20', count: 22 },
          { truth_id: 6, truth: '게보린', predicted_id: null, predicted: 'background', count: 19 },
        ],
        '0.75': [],
      },
      confusion_counts: {
        '0.50': { pairs: 214, shown: 3 },
        '0.75': { pairs: 0, shown: 0 },
      },
      error_breakdown: {
        '0.50': { localization: 12, classification: 34, background: 5, duplicate: 7 },
        '0.75': null,
      },
      ...overrides,
    },
  } as unknown as ExperimentDetail;
}

describe('오검출 진단 시트', () => {
  it('자르는 기준 탐색에서 F1이 가장 높은 줄을 짚어 준다', async () => {
    // 응답에는 진작 담겨 있었는데 그리는 화면이 없어 아무도 보지 못했습니다.
    vi.spyOn(api, 'experimentDetail').mockResolvedValue(detail());

    render(<DiagnosisSheet runId="dino-e12" onClose={() => undefined} />);

    expect(await screen.findByText('F1 최고')).toBeTruthy();
    expect(screen.getByText('0.05')).toBeTruthy();
  });

  it('background는 class 이름이 아니라 무슨 일이 일어났는지로 적는다', async () => {
    // "background → 리피토20"이라고 적으면 background라는 알약이 있는 줄 읽습니다.
    vi.spyOn(api, 'experimentDetail').mockResolvedValue(detail());

    render(<DiagnosisSheet runId="dino-e12" onClose={() => undefined} />);

    expect(await screen.findByText('없음 → 리피토20')).toBeTruthy();
    expect(screen.getByText('게보린 → 없음')).toBeTruthy();
    expect(screen.getByText('타이레놀500 → 타이레놀650')).toBeTruthy();
  });

  it('목록이 잘렸으면 전체가 몇 개인지 말한다', async () => {
    // 말하지 않으면 상위 몇 개가 전부로 읽혀, 헷갈리는 class가 셋뿐인 줄 압니다.
    vi.spyOn(api, 'experimentDetail').mockResolvedValue(detail());

    render(<DiagnosisSheet runId="dino-e12" onClose={() => undefined} />);

    expect(await screen.findByText(/헷갈린 쌍은 모두 214개/)).toBeTruthy();
  });

  it('원인 분류가 없는 옛 평가를 0건으로 그리지 않는다', async () => {
    // 0건과 "안 쟀다"는 다른 말입니다. 0으로 그리면 오검출이 없는 실행으로 읽힙니다.
    vi.spyOn(api, 'experimentDetail').mockResolvedValue(detail());

    render(<DiagnosisSheet runId="dino-e12" onClose={() => undefined} />);
    fireEvent.click(await screen.findByRole('button', { name: 'IoU 0.75' }));

    await waitFor(() => expect(screen.getByText(/원인 분류가 없습니다/)).toBeTruthy());
    expect(screen.queryByText('이름을 틀림')).toBeNull();
  });

  it('평가를 못 읽은 실행에서는 빈 표를 지어내지 않는다', async () => {
    vi.spyOn(api, 'experimentDetail').mockResolvedValue({
      experiment: { run_id: 'dino-e12' },
      history: { available: false, reason: null, epochs: [] },
      evaluation: { available: false, reason: '평가 결과 파일이 없습니다.' },
    } as unknown as ExperimentDetail);

    render(<DiagnosisSheet runId="dino-e12" onClose={() => undefined} />);

    expect(await screen.findByText('평가 결과 파일이 없습니다.')).toBeTruthy();
    expect(screen.queryByText('헷갈린 쌍')).toBeNull();
  });
});
