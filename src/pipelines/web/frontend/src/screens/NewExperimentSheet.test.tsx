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
            train: {
              run_id: 'retina-basic-e15-a7f3',
              architecture: 'retinanet_resnet50_fpn_v2',
              batch_size: 2,
              gradient_accumulation_steps: 8,
            },
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
      if (path === '/api/train/jobs') {
        return jsonResponse({ job_id: 'job-1', run_id: 'retina-basic-e15-a7f3' });
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

  /** 시작할 수 있는 상태까지 채웁니다. */
  async function fillAndReady() {
    fireEvent.change(screen.getByRole('textbox', { name: /학습 manifest/ }), {
      target: { value: 'artifacts/data/v5/train_manifest.json' },
    });
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeEnabled(),
    );
  }

  // 시트를 닫고 나면 무엇으로 돌고 있는지 다시 볼 자리가 없었습니다. 시작하기 전에
  // 실제로 보낼 값을 한 번 더 펼쳐 보여 줍니다.
  it('바로 시작은 보낼 설정을 펼쳐 보여 주고 확인을 받는다', async () => {
    show();
    await fillAndReady();

    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));

    expect(screen.getByText('이 설정으로 시작할까요?')).toBeInTheDocument();
    expect(screen.getByText('architecture')).toBeInTheDocument();
    // 아직 아무것도 만들지 않았습니다. 검증만 오갔습니다.
    expect(posted.map((item) => item.path)).not.toContain('/api/train/configs');

    fireEvent.click(screen.getByRole('button', { name: '시작' }));

    await waitFor(() => expect(posted.map((item) => item.path)).toContain('/api/train/jobs'));
  });

  // 확인 창은 마지막 검증 결과를 보여 주고, 만들기는 지금 draft를 보냅니다. 값을
  // 바꾸고 검증이 돌아오기 전에 눌러 버리면 확인한 것과 다른 설정으로 학습이 돕니다.
  it('바꾼 값이 아직 검증되지 않았으면 시작할 수 없다', async () => {
    show();
    await fillAndReady();

    fireEvent.change(screen.getByRole('textbox', { name: /실행 이름/ }), {
      target: { value: 'my-run' },
    });

    expect(screen.getByRole('button', { name: '바로 시작' })).toBeDisabled();
    await waitFor(() => expect(screen.getByRole('button', { name: '바로 시작' })).toBeEnabled());
  });

  // 창이 떠 있는 동안에도 뒤쪽 칸은 살아 있습니다. 열어 둔 채로 값이 바뀌면 창은
  // 옛 값을 보여 주는데 만들기는 바뀐 값을 보냅니다.
  it('확인 창을 연 뒤 설정이 바뀌면 창을 닫아 다시 확인하게 한다', async () => {
    show();
    await fillAndReady();
    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));
    expect(screen.getByText('이 설정으로 시작할까요?')).toBeInTheDocument();

    fireEvent.change(screen.getByRole('textbox', { name: /실행 이름/ }), {
      target: { value: 'changed' },
    });

    expect(screen.queryByText('이 설정으로 시작할까요?')).toBeNull();
  });

  it('확인 창은 포커스를 받아 ESC로 닫히고, 닫으면 포커스를 돌려준다', async () => {
    show();
    await fillAndReady();

    const startButton = screen.getByRole('button', { name: '바로 시작' });
    startButton.focus();
    fireEvent.click(startButton);

    const dialog = screen.getByRole('dialog', { name: /시작 확인/ });
    expect(dialog).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'Escape' });

    expect(screen.queryByText('이 설정으로 시작할까요?')).toBeNull();
    expect(startButton).toHaveFocus();
  });

  // 지운 설명 문단이 계산해 주던 값입니다. 표에 원시 값 둘만 두면 실제 갱신 규모가
  // 보이지 않습니다.
  it('모아서 갱신하면 유효 batch를 확인 창에서 알려 준다', async () => {
    show();
    await fillAndReady();

    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));

    expect(screen.getByText(/유효 batch 16/)).toBeInTheDocument();
  });

  it('다시 고치기를 누르면 아무것도 만들지 않는다', async () => {
    show();
    await fillAndReady();

    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));
    fireEvent.click(screen.getByRole('button', { name: '다시 고치기' }));

    expect(screen.queryByText('이 설정으로 시작할까요?')).toBeNull();
    expect(posted.map((item) => item.path)).not.toContain('/api/train/configs');
  });

  it('설정을 만든 뒤에 대기열에 넣는다 — 만들기가 먼저다', async () => {
    const onStarted = vi.fn();
    show({ onStarted });
    await fillAndReady();

    fireEvent.click(screen.getByRole('button', { name: '대기열에 추가' }));
    fireEvent.click(screen.getByRole('button', { name: '대기열에 넣습니다' }));

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
