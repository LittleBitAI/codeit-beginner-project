import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import type { Defaults, TeamRun } from '../api/types';

const state = vi.hoisted(() => ({
  latestEvent: null as TeamRun | null,
  actor: null as string | null,
  user: { username: 'a' } as { username: string } | null,
}));
const cloud = vi.hoisted(() => ({
  listRuns: vi.fn(),
  listLogs: vi.fn(),
  subscribeLogs: vi.fn(),
}));

vi.mock('../team/cloud', () => cloud);
vi.mock('../team/TeamContext', () => ({
  useTeam: () => ({
    config: { enabled: true, team_id: 'pill-team', actor: state.actor },
    user: state.user,
    latestEvent: state.latestEvent,
  }),
}));

const ARCHITECTURE = 'retinanet_resnet50_fpn_v2';

const defaults = {
  architecture: ARCHITECTURE,
  architecture_note: '',
  fields: [
    { name: 'architecture', type: 'enum', label: '모델', hint: '' },
    { name: 'optimizer', type: 'enum', label: 'Optimizer', hint: '' },
    { name: 'epochs', type: 'integer', label: 'Epochs', hint: '' },
    { name: 'batch_size', type: 'integer', label: 'Batch size', hint: '' },
    { name: 'learning_rate', type: 'number', label: 'Learning rate', hint: '' },
  ],
  data_fields: [],
  devices: [],
} as unknown as Defaults;

function run(actor: string, id: string, status: TeamRun['status'] = 'running'): TeamRun {
  const done = status === 'succeeded';
  return {
    teamId: 'pill-team',
    cloudRunId: id,
    localJobId: id,
    runId: `${actor}-run`,
    actorSub: actor,
    actorName: actor,
    actorSource: 'cognito',
    status,
    settings: {
      architecture: ARCHITECTURE,
      optimizer: 'AdamW',
      epochs: 10,
      batch_size: 4,
      learning_rate: 0.001,
    },
    dataInputs: {},
    progress: { current_epoch: 2 },
    summary: done
      ? {
          architecture: ARCHITECTURE,
          optimizer: 'AdamW',
          epochs: 10,
          best_epoch: 8,
          best_validation_loss: 0.12,
          class_count: 56,
        }
      : {},
    artifacts: done ? { checkpoint: 's3://pill-team/a.pt' } : {},
    evaluation: done
      ? {
          status: 'succeeded',
          metrics: {
            mAP: 0.73489,
            mAP50: null,
            mAP75: 0.97268,
            precision50: null,
            recall50: null,
          },
          registration_status: 'succeeded',
        }
      : {},
    message: null,
    createdAt: `2026-08-05T00:00:0${id}.000Z`,
    startedAt: '2026-08-05T00:00:00.000Z',
    finishedAt: done ? '2026-08-05T00:10:00.000Z' : null,
    heartbeatAt: new Date().toISOString(),
    revision: 1,
  };
}

const { TeamActivity } = await import('./TeamActivity');

beforeEach(() => {
  state.latestEvent = null;
  state.actor = null;
  state.user = { username: 'a' };
  cloud.listRuns.mockResolvedValue([run('a', '1')]);
  cloud.listLogs.mockResolvedValue([
    {
      lines: [{ seq: 1, stream: 'system', level: 'info', text: 'a 학습 시작', ts: 'now' }],
      endSeq: 1,
    },
  ]);
  cloud.subscribeLogs.mockReturnValue({ unsubscribe: vi.fn() });
});

afterEach(() => vi.clearAllMocks());

test('a와 b 어느 쪽이 시작해도 팀 목록과 설정, 로그에 나타난다', async () => {
  const view = render(<TeamActivity defaults={defaults} />);
  expect(await screen.findByText('a-run')).toBeInTheDocument();
  expect(await screen.findByText('a 학습 시작')).toBeInTheDocument();
  expect(screen.getByText('Learning rate')).toBeInTheDocument();

  state.latestEvent = run('b', '2');
  view.rerender(<TeamActivity defaults={defaults} />);
  expect(await screen.findByText('b-run')).toBeInTheDocument();
  fireEvent.click(screen.getByText('b-run'));
  expect(screen.getByText('b · b-run')).toBeInTheDocument();
});

test('목록 한 줄에 모델명과 mAP가 그 순서로 보인다', async () => {
  cloud.listRuns.mockResolvedValue([run('a', '1', 'succeeded')]);
  render(<TeamActivity defaults={defaults} />);
  expect(
    await screen.findByText(`${ARCHITECTURE} · mAP@[0.75:0.95] 0.7349`),
  ).toBeInTheDocument();
});

test('평가 전에는 mAP 자리가 사라지지 않고 -로 남는다', async () => {
  render(<TeamActivity defaults={defaults} />);
  expect(await screen.findByText(`${ARCHITECTURE} · mAP@[0.75:0.95] -`)).toBeInTheDocument();
});

test('로그인이 확인해 준 이름과 직접 적은 이름을 구분해 보여준다', async () => {
  const headless = { ...run('지현 (Colab)', '1'), actorSource: 'iam' as const };
  cloud.listRuns.mockResolvedValue([headless]);
  render(<TeamActivity defaults={defaults} />);

  expect(await screen.findByText('이름 직접 입력')).toBeInTheDocument();

  cloud.listRuns.mockResolvedValue([run('a', '1')]);
  render(<TeamActivity defaults={defaults} />);
  await waitFor(() => expect(screen.getAllByText('a-run').length).toBeGreaterThan(0));
  expect(screen.getAllByText('이름 직접 입력')).toHaveLength(1);
});

test('로그인할 수 없는 환경에서는 왜 팀 기록이 안 보이는지 말해 준다', async () => {
  // Colab입니다. 쓰기는 되지만 읽기는 로그인이 필요합니다. 빈 목록을 보여 주면
  // 설정이 잘못된 줄 알고 헤매게 됩니다.
  state.user = null;
  state.actor = '지현 (Colab)';
  render(<TeamActivity defaults={defaults} />);

  expect(await screen.findByText('이 환경에서는 팀 기록을 볼 수 없습니다')).toBeInTheDocument();
  expect(screen.getByText('지현 (Colab)')).toBeInTheDocument();
  expect(cloud.listRuns).not.toHaveBeenCalled();
});

test('진행 중·성공·실패를 구역으로 나누고 실패 구역은 접어 둔다', async () => {
  // 예전에는 셋이 시간순으로 섞여 실패와 취소가 목록 여기저기에 흩어졌습니다.
  cloud.listRuns.mockResolvedValue([
    run('a', '1', 'failed'),
    run('b', '2', 'succeeded'),
    run('c', '3', 'running'),
  ]);
  render(<TeamActivity defaults={defaults} />);

  expect(await screen.findByText('진행 중 1건')).toBeInTheDocument();
  expect(screen.getByText('성공 1건')).toBeInTheDocument();
  const closed = screen.getByText('실패·취소 1건').closest('details');
  expect(closed).not.toBeNull();
  expect(closed!.open).toBe(false);

  // 실패가 가장 최근이어도 진행 중인 학습이 먼저 열립니다.
  expect(screen.getByText('c · c-run')).toBeInTheDocument();
});

test('비어 있는 구역은 머리글조차 만들지 않는다', async () => {
  cloud.listRuns.mockResolvedValue([run('a', '1', 'succeeded')]);
  render(<TeamActivity defaults={defaults} />);

  expect(await screen.findByText('성공 1건')).toBeInTheDocument();
  expect(screen.queryByText(/진행 중/)).toBeNull();
  expect(screen.queryByText(/실패·취소/)).toBeNull();
});

test('완료 결과를 JSON이 아니라 한글 label로 보여준다', async () => {
  cloud.listRuns.mockResolvedValue([run('a', '1', 'succeeded')]);
  render(<TeamActivity defaults={defaults} />);
  expect(await screen.findByText('완료 결과와 산출물')).toBeInTheDocument();
  expect(screen.getByText('Best validation loss')).toBeInTheDocument();
  expect(screen.getByText('클래스 수')).toBeInTheDocument();
  // 예전에는 summary를 통째로 JSON.stringify해서 보여 줬습니다.
  expect(screen.queryByText(/"best_validation_loss":/)).toBeNull();
  expect(screen.getByText(/팀 공유 가능/)).toBeInTheDocument();
  await waitFor(() => expect(cloud.subscribeLogs).toHaveBeenCalled());
});
