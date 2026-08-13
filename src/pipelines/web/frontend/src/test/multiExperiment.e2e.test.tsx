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
      gradient_accumulation_steps: 1,
      input_size: null,
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
      if (path === '/api/train/queue') return jsonResponse({ entries: [], paused: false });
      if (path === '/api/gpu/status') {
        return jsonResponse({
          torch: { cuda_available: false, device_count: 0, reason: 'E2E fixture' },
          telemetry: { source: 'none', reason: 'E2E fixture', message: null, devices: [] },
          queried_at: '2026-08-05T00:00:00Z',
        });
      }
      if (path === '/api/train/experiments') return jsonResponse(experimentListing());
      if (path === '/api/train/experiments/compare') {
        return jsonResponse({ experiments: experimentListing().experiments, missing: [] });
      }
      // 캔버스는 곡선을 실행마다 따로 읽습니다. 이 fixture에는 epoch 기록이 없습니다.
      if (path.startsWith('/api/train/experiments/')) {
        const runId = decodeURIComponent(path.split('/').pop() ?? '');
        const found = experimentListing().experiments.find((item) => item.run_id === runId);
        if (!found) throw new Error(`E2E fixture에 없는 실험입니다: ${runId}`);
        return jsonResponse({
          experiment: found,
          evaluation: { available: false, reason: 'E2E fixture' },
          history: { available: false, reason: 'E2E fixture', epochs: [] },
        });
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
      if (path === '/api/data/datasets') {
        return jsonResponse({
          backend: 'local',
          root: 'datasets/pill_detection/processed/',
          datasets: [
            {
              name: 'e2e-prepared',
              directory: 'datasets/pill_detection/processed/e2e-prepared/',
              complete: true,
              missing: [],
              has_test_manifest: true,
              has_eda_report: false,
            },
          ],
          problems: [],
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
  it('캔버스에서 두 실행을 겹치면 결과와 세팅을 나란히 보여 준다', async () => {
    const listing = experimentListing();
    const best = listing.experiments.reduce((current, candidate) =>
      (candidate.metrics.best_validation_loss ?? Number.POSITIVE_INFINITY) <
      (current.metrics.best_validation_loss ?? Number.POSITIVE_INFINITY)
        ? candidate
        : current,
    );
    expect(best.run_id).toBe(scenario.expectation.best_run_id);

    render(
      <MemoryRouter initialEntries={['/canvas']}>
        <App />
      </MemoryRouter>,
    );

    // 왼쪽 목록에서 눌러서 겹칩니다. 화면을 갈아 끼우는 단계가 없습니다.
    for (const runId of ['e2e-baseline', 'e2e-tuned']) {
      const pick = await screen.findByRole('button', { name: new RegExp(runId) });
      fireEvent.click(pick);
    }

    await waitFor(() => expect(document.querySelector('[data-run="e2e-tuned"]')).not.toBeNull());

    // 비교표 칸(data-run이 붙은 곳)만 봅니다. 왼쪽 고르기 목록에도 같은 글자가 있습니다.
    // 가장 좋은 칸에는 값 뒤에 "최고" 표식이 붙으므로 앞부분으로 확인합니다.
    const hasValue = (runId: string, value: string) =>
      [...document.querySelectorAll(`[data-run="${runId}"]`)].some((cell) =>
        (cell.textContent ?? '').startsWith(value),
      );
    expect(hasValue('e2e-baseline', '0.7200')).toBe(true);
    expect(hasValue('e2e-tuned', '0.4800')).toBe(true);

    // 결과와 세팅이 한 표에 함께 있습니다. 탭으로 갈아 끼우지 않습니다.
    expect(screen.getAllByText('SGD (호환 기본값)')).toHaveLength(2);
    expect(screen.getByText(/data artifact URI 4개가 모두 같아/)).toBeInTheDocument();
  });
});

describe('왼쪽 dataset 목록', () => {
  it('전처리는 끝났지만 아직 학습하지 않은 판도 보여 준다', async () => {
    // 기록에서만 목록을 만들면 방금 만든 판이 보이지 않아, 그것으로 학습하려면
    // 어디로 가야 할지 알 수 없습니다.
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    );

    expect(await screen.findByText('e2e-prepared')).toBeInTheDocument();
    expect(screen.getByText('기록 없음 · 학습 전')).toBeInTheDocument();
  });
});
