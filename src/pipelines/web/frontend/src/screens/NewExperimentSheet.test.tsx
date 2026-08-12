import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { Defaults } from '../api/types';
import { DraftProvider } from '../state/DraftContext';
import { NewExperimentSheet } from './NewExperimentSheet';

const DEFAULTS: Defaults = {
  architecture: 'retinanet_resnet50_fpn_v2',
  architecture_note: '',
  fields: [
    {
      name: 'architecture',
      type: 'enum',
      default: 'retinanet_resnet50_fpn_v2',
      choices: ['retinanet_resnet50_fpn_v2'],
      label: '모델',
      hint: '',
    },
    {
      name: 'optimizer',
      type: 'enum',
      default: 'AdamW',
      choices: ['AdamW', 'SGD'],
      label: 'Optimizer',
      hint: '',
    },
    { name: 'epochs', type: 'integer', default: 15, label: 'Epochs', hint: '' },
    { name: 'run_id', type: 'string', label: '실행 이름', hint: '' },
  ],
  data_fields: [
    { name: 'train_manifest_uri', type: 'uri', label: '학습 manifest', hint: '', required: true },
  ],
  devices: [{ value: 'cpu', available: true, reason: null }],
};

let posted: { path: string; body: unknown }[] = [];

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  posted = [];
  window.sessionStorage.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path =
        typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (init?.method === 'POST') {
        posted.push({ path, body: JSON.parse(String(init.body)) });
      }
      if (path === '/api/train/validate') {
        return jsonResponse({
          valid: true,
          errors: [],
          warnings: [],
          normalized: {
            project: { name: 'pill' },
            execution: { mode: 'local' },
            storage: {},
            train: { run_id: 'retina-basic-e15-a7f3', architecture: 'retinanet_resnet50_fpn_v2' },
            inputs: { data: { train_manifest_uri: 'artifacts/data/v5/train_manifest.json' } },
          },
        });
      }
      if (path === '/api/train/configs') {
        return jsonResponse({
          config_id: 'cfg-1',
          run_id: 'retina-basic-e15-a7f3',
          config: { train: {}, inputs: { data: {} } },
          warnings: [],
        });
      }
      if (path === '/api/train/queue') {
        return jsonResponse({ entries: [], paused: false, started: null });
      }
      throw new Error(`fixture가 처리하지 않는 요청입니다: ${path}`);
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function show(props: Partial<Parameters<typeof NewExperimentSheet>[0]> = {}) {
  return render(
    <MemoryRouter>
      <DraftProvider>
        <NewExperimentSheet
          defaults={DEFAULTS}
          source={null}
          datasetKey="v5-118cls"
          queuedCount={0}
          busy={false}
          onClose={() => {}}
          onStarted={() => {}}
          {...props}
        />
      </DraftProvider>
    </MemoryRouter>,
  );
}

describe('NewExperimentSheet', () => {
  it('data artifact 칸이 비어 있으면 시작할 수 없다', async () => {
    show();

    expect(await screen.findByText('data artifact 위치가 비어 있습니다')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeDisabled();
  });

  it('설정을 만든 뒤에 대기열에 넣는다 — 만들기가 먼저다', async () => {
    const onStarted = vi.fn();
    show({ onStarted });

    fireEvent.change(screen.getByRole('textbox', { name: /학습 manifest/ }), {
      target: { value: 'artifacts/data/v5/train_manifest.json' },
    });

    const queueButton = screen.getByRole('button', { name: '대기열에 추가' });
    await waitFor(() => expect(queueButton).toBeEnabled());
    fireEvent.click(queueButton);

    await waitFor(() => expect(onStarted).toHaveBeenCalled());
    expect(posted.map((item) => item.path)).toEqual([
      '/api/train/validate',
      '/api/train/configs',
      '/api/train/queue',
    ]);
    expect(posted[2]?.body).toEqual({ config_id: 'cfg-1' });
  });

  it('다른 학습이 도는 중에는 바로 시작만 막고 대기열은 열어 둔다', async () => {
    show({ busy: true, queuedCount: 2 });

    fireEvent.change(screen.getByRole('textbox', { name: /학습 manifest/ }), {
      target: { value: 'artifacts/data/v5/train_manifest.json' },
    });

    await waitFor(() =>
      expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeEnabled(),
    );
    expect(screen.getByRole('button', { name: '바로 시작' })).toBeDisabled();
    expect(screen.getByText(/대기열 3번/)).toBeInTheDocument();
  });

  it('이름을 비워 두면 서버가 지어 줄 이름을 미리 보여 준다', async () => {
    show();

    expect(await screen.findByText('retina-basic-e15-a7f3')).toBeInTheDocument();
  });
});
