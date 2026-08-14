import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { DataSource, Defaults } from '../api/types';
import { DraftProvider } from '../state/DraftContext';
import { NewExperimentSheet } from './NewExperimentSheet';

const MANIFEST = 'artifacts/data/processed/v5-118cls/train_manifest.json';

const SOURCE: DataSource = {
  directory: 'artifacts/data/processed/v5-118cls',
  complete: true,
  data: { train_manifest_uri: MANIFEST },
  matched: {},
  labels: {},
  missing: [],
  problems: [],
  examined: [],
};

/**
 * 데이터셋을 이미 고른 상태를 만듭니다.
 *
 * 이 시트는 artifact 위치를 직접 받지 않습니다. dataset 준비에서 고른 값이 draft로
 * 들어오는 것이 유일한 경로라, 테스트도 같은 자리에 넣습니다.
 */
function seedData(data: Record<string, string> = { train_manifest_uri: MANIFEST }): void {
  window.sessionStorage.setItem('pill-training-draft', JSON.stringify({ train: {}, data }));
}

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
/** 서버가 돌려줄 오류. 비어 있으면 검증을 통과한 것으로 답합니다. */
let serverErrors: { field: string; message: string }[] = [];

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  posted = [];
  serverErrors = [];
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
          valid: serverErrors.length === 0,
          errors: serverErrors,
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
  it('데이터셋을 고르지 않았으면 시작할 수 없다', async () => {
    show();

    expect(await screen.findByText('학습에 쓸 데이터셋을 아직 고르지 않았습니다')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeDisabled();
  });

  // 칸을 없앤 뒤로 draft에 남은 예전 값을 화면에서 지울 방법이 없습니다. 서버가 고른
  // 것이 없다고 말하는데도 시작되면, 아무도 모르는 예전 데이터로 밤새 학습이 돕니다.
  it('고른 데이터셋이 없으면 예전 값이 남아 있어도 시작할 수 없다', async () => {
    seedData();
    show({ source: null });

    expect(await screen.findByText('학습에 쓸 데이터셋을 아직 고르지 않았습니다')).toBeInTheDocument();
    // 검증이 끝난 **뒤에도** 잠겨 있어야 합니다. 응답 전에 재면 아직 안 온 검증 때문에
    // 잠긴 것을 보고 통과해 버립니다.
    await screen.findByText('retina-basic-e15-a7f3');
    expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '바로 시작' })).toBeDisabled();
  });

  // 판단 기준은 언제나 서버입니다. 화면이 통과시켜도 서버가 거부하면 시작할 수 없어야
  // 합니다.
  it('서버가 설정을 거부하면 시작할 수 없다', async () => {
    serverErrors = [{ field: 'train.epochs', message: '1 이상이어야 합니다.' }];
    seedData();
    show({ source: SOURCE });

    expect(await screen.findByText('1 이상이어야 합니다.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeDisabled();
  });

  it('고른 데이터셋을 그대로 쓰고, artifact 위치를 고치는 칸은 두지 않는다', async () => {
    seedData();
    // 이름은 시트 제목이 아니라 실려 갈 값에서 나와야 합니다.
    show({ source: SOURCE, datasetKey: null });

    expect(screen.queryByRole('textbox', { name: /학습 manifest/ })).toBeNull();
    expect(screen.getByText('v5-118cls')).toBeInTheDocument();
    expect(screen.getByText(MANIFEST)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeEnabled(),
    );
  });

  // 화면에는 새 데이터셋이 보이는데 예전 값으로 학습된 적이 있습니다. 칸이 사라져도
  // 실려 갈 값이 고른 것과 다를 수 있습니다 — 새 데이터셋에 없는 선택 artifact는
  // 예전 값이 그대로 남기 때문입니다. GPU를 밤새 쓰는 일이라 조용히 넘기지 않습니다.
  it('실려 갈 값이 고른 데이터셋과 다르면 알리고, 맞추면 남은 값까지 지운다', async () => {
    seedData({ train_manifest_uri: MANIFEST, test_manifest_uri: 'artifacts/data/old/test.json' });
    show({ source: SOURCE });

    expect(await screen.findByText('고른 데이터셋과 실려 갈 값이 다릅니다')).toBeInTheDocument();
    // 검증이 끝난 **뒤에도** 잠겨 있어야 합니다. 응답 전에 재면 아직 안 온 검증 때문에
    // 잠긴 것을 보고 통과해 버립니다.
    await screen.findByText('retina-basic-e15-a7f3');
    // 경고만으로는 부족합니다. 칸을 없앤 뒤로 일부러 다른 데이터로 돌릴 이유가 없으므로
    // 맞추기 전에는 시작 자체를 막습니다.
    expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: '맞추기' }));

    await waitFor(() =>
      expect(screen.queryByText('고른 데이터셋과 실려 갈 값이 다릅니다')).toBeNull(),
    );
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeEnabled(),
    );
  });

  it('설정을 만든 뒤에 대기열에 넣는다 — 만들기가 먼저다', async () => {
    const onStarted = vi.fn();
    seedData();
    show({ onStarted, source: SOURCE });

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
    seedData();
    show({ busy: true, queuedCount: 2, source: SOURCE });

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
