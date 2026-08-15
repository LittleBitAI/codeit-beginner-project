/**
 * 기록 화면에서 고르는 dataset은 **보는 대상**일 뿐입니다.
 *
 * 예전에는 왼쪽 목록에서 고른 이름이 새 실험 시트의 부제목으로 찍혔습니다. 그래서
 * 화면에는 A가 보이는데 학습은 dataset 준비에서 고른 B로 도는 상태가 가능했습니다.
 * 이 test는 그 분리를 **App 수준에서** 잽니다 — 화면 하나씩만 재면, App이 고르기를
 * 다시 학습 입력에 이어 붙여도 둘 다 초록이기 때문입니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import { App } from '../App';
import type { DataSource, ExperimentSummary } from '../api/types';

/** dataset 준비에서 고른 데이터. 학습에 **실제로 실려 갈** 값입니다. */
const PREPARED = 'v9-prepared';
const MANIFEST = `datasets/pill_detection/processed/${PREPARED}/train_manifest.json`;

/** 기록 화면에서 고를 수 있는, 그러나 학습과는 상관없는 dataset들입니다. */
const WATCHED = ['v5-118cls', 'v4-57cls'];

let calls: { path: string; method: string; body: unknown }[] = [];

function source(): DataSource {
  return {
    directory: `datasets/pill_detection/processed/${PREPARED}`,
    complete: true,
    data: {
      train_manifest_uri: MANIFEST,
      validation_manifest_uri: MANIFEST.replace('train_', 'validation_'),
      class_map_uri: MANIFEST.replace('train_manifest', 'class_map'),
      dataset_summary_uri: MANIFEST.replace('train_manifest', 'dataset_summary'),
    },
    matched: {},
    labels: {},
    missing: [],
    problems: [],
    examined: [],
  };
}

function experiment(label: string, index: number): ExperimentSummary {
  return {
    experiment_id: `exp-${index}`,
    run_id: `run-${index}`,
    status: 'succeeded',
    status_label: '완료',
    created_at: `2026-08-05T00:0${index}:00Z`,
    started_at: `2026-08-05T00:0${index}:00Z`,
    finished_at: `2026-08-05T00:0${index + 1}:00Z`,
    elapsed_seconds: 60,
    dataset: {
      identity: label,
      identity_source: 'artifact_set',
      artifacts_complete: true,
      label,
    },
    model: { architecture: 'retinanet_resnet50_fpn_v2', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.0001,
      momentum: null,
      weight_decay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    training: {
      device: 'cuda',
      epochs: 15,
      batch_size: 4,
      num_workers: 4,
      gradient_accumulation_steps: 1,
      input_size: null,
      seed: 42,
      // '이 세팅으로 학습하기'가 되살려야 하는 값들입니다.
      precision: 'amp',
      checkpoint_every: 2,
      augmentation: { preset: 'pill_geometric' },
      lr_scheduler: { name: 'cosine', warmup_steps: 500 },
      early_stopping: null,
    },
    metrics: {
      best_epoch: 3,
      best_validation_loss: 0.4,
      final_train_loss: 0.3,
      final_validation_loss: 0.44,
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      kaggle_score: null,
    },
    completion: {
      evaluated: false,
      submission_generated: false,
      submitted: false,
      submission_checked: false,
      submission_rows: null,
    },
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path =
        typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      calls.push({
        path,
        method: init?.method ?? 'GET',
        body: init?.body ? JSON.parse(String(init.body)) : null,
      });

      if (path === '/api/data/source') return jsonResponse({ source: source() });
      if (path === '/api/train/jobs') return jsonResponse({ jobs: [], active_job_id: null });
      if (path === '/api/train/queue') return jsonResponse({ entries: [], paused: false });
      if (path === '/api/train/experiments/compare') {
        return jsonResponse({
          experiments: [experiment(PREPARED, 0)],
          missing: [],
          curves: {},
        });
      }
      if (path === '/api/train/experiments') {
        return jsonResponse({
          experiments: WATCHED.map((label, index) => experiment(label, index)),
          scope: { backend: 'local', shared: false },
        });
      }
      if (path === '/api/gpu/status') {
        return jsonResponse({
          torch: { cuda_available: false, device_count: 0, reason: 'E2E fixture' },
          telemetry: { source: 'none', reason: 'E2E fixture', message: null, devices: [] },
          queried_at: '2026-08-05T00:00:00Z',
        });
      }
      if (path === '/api/train/defaults') {
        return jsonResponse({
          architecture: 'fasterrcnn_mobilenet_v3_large_320_fpn',
          architecture_note: 'E2E fixture',
          // 초안이 payload에 실리려면 그 칸이 선언되어 있어야 합니다. '이 세팅으로
          // 학습하기'가 되살리는 값 중 대표만 둡니다.
          fields: [
            { name: 'architecture', type: 'enum', label: '모델', hint: '', choices: [] },
            { name: 'epochs', type: 'integer', label: 'Epochs', hint: '' },
            { name: 'precision', type: 'enum', label: '정밀도', hint: '', choices: [] },
            { name: 'augmentation', type: 'enum', label: '증강', hint: '', choices: [] },
            { name: 'lr_scheduler', type: 'enum', label: 'LR schedule', hint: '', choices: [] },
            { name: 'lr_warmup_steps', type: 'integer', label: 'Warmup', hint: '' },
          ],
          data_fields: [],
          devices: [],
        });
      }
      if (path === '/api/train/validate') {
        return jsonResponse({ valid: true, errors: [], warnings: [], normalized: null });
      }
      if (path === '/api/settings') return jsonResponse({ evaluation_mode: null });
      if (path === '/api/team/config') {
        return jsonResponse({
          enabled: false,
          team_id: null,
          appsync_url: null,
          region: 'ap-northeast-2',
          user_pool_id: null,
          user_pool_client_id: null,
          cognito_domain: null,
          actor: null,
          unattended_actor: null,
        });
      }
      if (path === '/api/data/datasets') {
        return jsonResponse({
          backend: 'local',
          root: 'datasets/pill_detection/processed/',
          datasets: [],
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

describe('보는 dataset과 학습에 쓰는 dataset', () => {
  it('기록에서 dataset을 바꿔도 새 실험은 준비된 데이터를 쓴다', async () => {
    render(
      <MemoryRouter initialEntries={['/records']}>
        <App />
      </MemoryRouter>,
    );

    // 기록 화면에서 보는 대상을 바꿉니다.
    const picker = await screen.findByLabelText('DATASET');
    fireEvent.change(picker, { target: { value: 'v4-57cls' } });
    await waitFor(() => expect((picker as HTMLSelectElement).value).toBe('v4-57cls'));

    // 그 상태로 새 실험을 엽니다.
    fireEvent.click(screen.getAllByRole('button', { name: '새 실험' })[0] as HTMLElement);

    // 시트가 말하는 데이터는 **준비된 것**이어야 합니다. 제목과 본문 두 자리에 나옵니다.
    await waitFor(() => expect(screen.getAllByText(PREPARED).length).toBeGreaterThan(0));
    // 보는 대상으로 고른 이름은 시트에 나타나지 않습니다.
    expect(screen.queryByText('v4-57cls')).toBeNull();

    // 그리고 고르기가 학습 입력을 바꾸지 않았어야 합니다.
    expect(calls.filter((call) => call.path === '/api/data/source' && call.method !== 'GET')).toEqual(
      [],
    );

    /**
     * 이름과 요청 method만 보면 부족합니다. 고르기가 draft의 artifact URI 일부만
     * 바꿔 놓아도 위 두 검사는 그대로 통과하고, 정작 서버로 가는 설정에는 다른
     * dataset이 실립니다. **실제로 실려 가는 값**을 봅니다.
     */
    // 검증 요청은 입력이 멈춘 뒤(250ms) 나갑니다. 그것이 실제로 서버로 갈 설정입니다.
    await waitFor(() =>
      expect(calls.some((call) => call.path === '/api/train/validate')).toBe(true),
    );
    for (const call of calls.filter((call) => call.path === '/api/train/validate')) {
      const data = (call.body as { inputs?: { data?: Record<string, string> } }).inputs?.data ?? {};
      expect(data).toEqual(source().data);
    }
  });

  it("'이 세팅으로 학습하기'는 설정만 물려주고 dataset은 준비된 것을 쓴다", async () => {
    render(
      <MemoryRouter initialEntries={['/canvas?run=run-0']}>
        <App />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByText('이 세팅으로 학습하기'));

    // 새 실험 시트가 열리고, 서버로 갈 설정에 그 실행의 값이 실려야 합니다.
    await waitFor(() =>
      expect(calls.some((call) => call.path === '/api/train/validate')).toBe(true),
    );
    const sent = calls.filter((call) => call.path === '/api/train/validate');
    for (const call of sent) {
      const body = call.body as { train?: Record<string, unknown>; inputs?: { data?: unknown } };
      const train = body.train ?? {};
      expect(train.architecture).toBe('retinanet_resnet50_fpn_v2');
      expect(train.epochs).toBe(15);
      expect(train.precision).toBe('amp');
      /**
       * 이 계층은 화면의 **평평한** 칸 값을 보냅니다. 중첩으로 바꾸는 것은
       * `train_config.py`이고, 그 왕복(중첩 -> 평평 -> 중첩)이 맞는지가 이
       * 기능의 핵심입니다. 여기서는 평평한 쪽이 제대로 채워졌는지 봅니다.
       */
      expect(train.augmentation).toBe('pill_geometric');
      expect(train.lr_scheduler).toBe('cosine');
      expect(String(train.lr_warmup_steps)).toBe('500');
      // dataset은 물려받지 않고 **준비된 것**을 그대로 씁니다.
      expect(body.inputs?.data).toEqual(source().data);
    }
  });
});
