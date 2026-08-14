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
    { name: 'batch_size', type: 'integer', default: 4, label: 'Batch size', hint: '' },
    { name: 'seed', type: 'integer', default: 42, label: 'Random seed', hint: '' },
    { name: 'learning_rate', type: 'number', default: 0.006, label: 'Learning rate', hint: '' },
    { name: 'weight_decay', type: 'number', default: 0.01, label: 'Weight decay', hint: '' },
    {
      name: 'checkpoint_every',
      type: 'integer',
      default: 1,
      label: 'Checkpoint 주기',
      hint: '',
    },
    {
      name: 'precision',
      type: 'enum',
      default: 'fp32',
      choices: ['fp32', 'amp'],
      label: '연산 정밀도',
      hint: '',
    },
    { name: 'device', type: 'enum', default: 'cpu', choices: ['cpu', 'cuda'], label: 'Device', hint: '' },
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

/** 화면에 그려진 칸을 위에서부터 이름만 뽑습니다. 순서를 그대로 읽습니다. */
function fieldLabels(): string[] {
  return Array.from(document.querySelectorAll('label > span:first-child')).map(
    (node) => node.textContent ?? '',
  );
}

describe('NewExperimentSheet', () => {
  // 기본과 고급을 가르는 기준은 "얼마나 자주 바꾸는가" 하나입니다. 기준이 보이지
  // 않으면 사람이 매번 두 표를 다 열어 찾습니다. 실제로 그 불평이 나왔습니다.
  it('자주 바꾸는 칸을 기본에, 잘 안 바꾸는 칸을 고급에 순서대로 둔다', async () => {
    show();

    expect(fieldLabels().slice(0, 6)).toEqual([
      '모델',
      'Optimizer',
      'Random seed',
      'Epochs',
      'Batch size',
      'Learning rate',
    ]);

    fireEvent.click(screen.getByText('고급'));

    // 서버가 주는 칸은 하나도 빠뜨리지 않습니다. 표에 자리가 없으면 그 칸은 화면에
    // 아예 나타나지 않고, 사람은 기본값으로 돈다는 것조차 모릅니다.
    expect(fieldLabels()).toEqual([
      '실행 이름',
      'Weight decay',
      'Checkpoint 주기',
      '연산 정밀도',
      'Device',
    ]);
  });

  it('데이터셋을 고르지 않았으면 시작할 수 없다', async () => {
    show();

    expect(await screen.findByText('학습에 쓸 데이터셋을 아직 고르지 않았습니다')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeDisabled();
  });

  /**
   * 시작할 수 있는 상태로 시트를 엽니다.
   *
   * artifact 위치를 치는 칸이 없어졌으므로 dataset 준비에서 고른 것과 같은 자리에
   * 넣고 그 데이터셋을 함께 넘깁니다.
   */
  async function showReady(props: Partial<Parameters<typeof NewExperimentSheet>[0]> = {}) {
    seedData();
    show({ source: SOURCE, ...props });
    await waitFor(() =>
      expect(screen.getByRole('button', { name: '대기열에 추가' })).toBeEnabled(),
    );
  }

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
    // 이름은 다른 화면에서 고른 값이 아니라 **실려 갈** 값에서 나와야 합니다.
    show({ source: SOURCE });

    expect(screen.queryByRole('textbox', { name: /학습 manifest/ })).toBeNull();
    // 제목과 본문 둘 다 실려 갈 값에서 이름을 뽑습니다.
    expect(screen.getAllByText('v5-118cls')).toHaveLength(2);
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


  // 시트를 닫고 나면 무엇으로 돌고 있는지 다시 볼 자리가 없었습니다. 시작하기 전에
  // 실제로 보낼 값을 한 번 더 펼쳐 보여 줍니다.
  it('바로 시작은 보낼 설정을 펼쳐 보여 주고 확인을 받는다', async () => {
    await showReady();

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
    await showReady();

    fireEvent.change(screen.getByRole('textbox', { name: /Random seed/ }), {
      target: { value: '7' },
    });

    expect(screen.getByRole('button', { name: '바로 시작' })).toBeDisabled();
    await waitFor(() => expect(screen.getByRole('button', { name: '바로 시작' })).toBeEnabled());
  });

  // 창이 떠 있는 동안에도 뒤쪽 칸은 살아 있습니다. 열어 둔 채로 값이 바뀌면 창은
  // 옛 값을 보여 주는데 만들기는 바뀐 값을 보냅니다.
  it('확인 창을 연 뒤 설정이 바뀌면 창을 닫아 다시 확인하게 한다', async () => {
    await showReady();
    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));
    expect(screen.getByText('이 설정으로 시작할까요?')).toBeInTheDocument();

    fireEvent.change(screen.getByRole('textbox', { name: /Random seed/ }), {
      target: { value: '7' },
    });

    expect(screen.queryByText('이 설정으로 시작할까요?')).toBeNull();
  });

  // 이미 떠난 검증 요청은 취소할 수 없습니다. 옛 설정의 답이 새 설정의 답을 덮으면
  // 화면의 값은 멀쩡한데 시작이 잠긴 채로 남고, 다시 무언가를 고치기 전까지 풀리지
  // 않습니다. 응답을 손으로 풀어 순서를 뒤집으므로 시간에 기대지 않습니다.
  it('늦게 온 옛 검증 응답이 최신 설정을 잠그지 않는다', async () => {
    const release: (() => void)[] = [];
    seedData();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const path =
          typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
        if (path !== '/api/train/validate') throw new Error(`예상 밖 요청입니다: ${path}`);
        await new Promise<void>((resolve) => release.push(resolve));
        return jsonResponse({
          valid: true,
          errors: [],
          warnings: [],
          normalized: {
            project: { name: 'pill' },
            execution: { mode: 'local' },
            storage: {},
            train: { run_id: 'retina-basic-e15-a7f3' },
            inputs: { data: {} },
          },
        });
      }),
    );

    show({ source: SOURCE });
    await waitFor(() => expect(release).toHaveLength(1));

    fireEvent.change(screen.getByRole('textbox', { name: /Random seed/ }), {
      target: { value: '7' },
    });
    await waitFor(() => expect(release).toHaveLength(2));

    release[1]!();
    await waitFor(() => expect(screen.getByRole('button', { name: '바로 시작' })).toBeEnabled());

    release[0]!();
    await waitFor(() => expect(release).toHaveLength(2));

    expect(screen.getByRole('button', { name: '바로 시작' })).toBeEnabled();
  });

  it('확인 창은 포커스를 받아 ESC로 닫히고, 닫으면 포커스를 돌려준다', async () => {
    await showReady();

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
    await showReady();

    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));

    expect(screen.getByText(/유효 batch 16/)).toBeInTheDocument();
  });

  it('다시 고치기를 누르면 아무것도 만들지 않는다', async () => {
    await showReady();

    fireEvent.click(screen.getByRole('button', { name: '바로 시작' }));
    fireEvent.click(screen.getByRole('button', { name: '다시 고치기' }));

    expect(screen.queryByText('이 설정으로 시작할까요?')).toBeNull();
    expect(posted.map((item) => item.path)).not.toContain('/api/train/configs');
  });

  it('설정을 만든 뒤에 대기열에 넣는다 — 만들기가 먼저다', async () => {
    const onStarted = vi.fn();
    await showReady({ onStarted });

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
