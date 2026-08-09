import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import { App } from '../App';
import type { ExperimentListing, ExperimentSummary } from '../api/types';
import scenario from './fixtures/multiExperiment.json';

function experimentListing(): ExperimentListing {
  const experiments: ExperimentSummary[] = scenario.experiments.map((experiment, index) => ({
    experiment_id: String(index + 1).repeat(32),
    run_id: experiment.run_id,
    status: 'succeeded',
    status_label: '성공',
    created_at: `2026-08-05T00:0${index}:00Z`,
    started_at: `2026-08-05T00:0${index}:00Z`,
    finished_at: `2026-08-05T00:0${index + 1}:00Z`,
    elapsed_seconds: 60,
    dataset: {
      identity: 'fixture-same-dataset',
      identity_source: 'artifact_set',
      artifacts_complete: true,
      label: 'fixture-same-dataset',
    },
    completion: {
      evaluated: true,
      submitted: true,
      submission_checked: true,
      submission_rows: 100,
    },
    model: {
      architecture: experiment.summary.architecture,
      pretrained: experiment.train.pretrained,
      source: 'record',
    },
    optimizer: {
      name: 'SGD',
      source: 'legacy_fallback',
      learning_rate: experiment.train.learning_rate,
      momentum: experiment.train.momentum,
      weight_decay: experiment.train.weight_decay,
      beta1: null,
      beta2: null,
      epsilon: null,
    },
    training: {
      device: experiment.train.device,
      epochs: experiment.train.epochs,
      batch_size: experiment.train.batch_size,
      num_workers: experiment.train.num_workers,
      seed: experiment.train.seed,
    },
    metrics: {
      best_epoch: experiment.summary.best_epoch,
      best_validation_loss: experiment.summary.best_validation_loss,
      final_train_loss: null,
      final_validation_loss: null,
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
    },
  }));
  return { experiments };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path =
        typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (path === '/api/train/jobs') return jsonResponse({ jobs: [], active_job_id: null });
      if (path === '/api/data/source') return jsonResponse({ source: null });
      if (path === '/api/train/experiments') return jsonResponse(experimentListing());
      if (path === '/api/train/experiments/compare') {
        return jsonResponse({ experiments: experimentListing().experiments, missing: [] });
      }
      if (path === '/api/train/defaults') {
        return jsonResponse({
          architecture: 'fasterrcnn_mobilenet_v3_large_320_fpn',
          architecture_note: 'E2E fixture',
          fields: [],
          data_fields: [],
          devices: [],
        });
      }
      throw new Error(`E2E fixture가 처리하지 않는 요청입니다: ${path}`);
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Web multi-experiment E2E', () => {
  it('비교 route에서 최근 두 실험의 dataset과 학습 결과를 나란히 보여 준다', async () => {
    const listing = experimentListing();
    const best = listing.experiments.reduce((current, candidate) =>
      (candidate.metrics.best_validation_loss ?? Number.POSITIVE_INFINITY) <
      (current.metrics.best_validation_loss ?? Number.POSITIVE_INFINITY)
        ? candidate
        : current,
    );
    expect(best.run_id).toBe(scenario.expectation.best_run_id);

    render(
      <MemoryRouter initialEntries={['/compare']}>
        <App />
      </MemoryRouter>,
    );

    const selectRecent = await screen.findByRole('button', { name: '최근 2개 선택' });
    await waitFor(() => expect(selectRecent).toBeEnabled());
    fireEvent.click(selectRecent);

    expect(
      await screen.findByText('같은 dataset 입력으로 기록된 실험입니다'),
    ).toBeInTheDocument();
    expect(screen.getAllByText('e2e-baseline').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('e2e-tuned').length).toBeGreaterThanOrEqual(1);
    // 기본 탭인 결과값에는 loss가, 학습 세팅 탭에는 optimizer가 있습니다.
    // 고르는 표에도 같은 값이 있으므로 비교표 칸(data-run이 붙은 곳)만 봅니다.
    // 가장 좋은 칸에는 값 뒤에 "최고" 표식이 붙으므로 앞부분으로 확인합니다.
    const hasValue = (runId: string, value: string) =>
      [...document.querySelectorAll(`[data-run="${runId}"]`)].some((cell) =>
        (cell.textContent ?? '').startsWith(value),
      );
    expect(hasValue('e2e-baseline', '0.7200')).toBe(true);
    expect(hasValue('e2e-tuned', '0.4800')).toBe(true);

    fireEvent.click(screen.getByRole('button', { name: '학습 세팅' }));

    expect(screen.getAllByText('SGD (호환 기본값)')).toHaveLength(2);
  });
});
