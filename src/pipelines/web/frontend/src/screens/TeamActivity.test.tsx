import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import type { Defaults, TeamRun } from '../api/types';

const state = vi.hoisted(() => ({ latestEvent: null as TeamRun | null }));
const cloud = vi.hoisted(() => ({
  listRuns: vi.fn(),
  listLogs: vi.fn(),
  subscribeLogs: vi.fn(),
}));

vi.mock('../team/cloud', () => cloud);
vi.mock('../team/TeamContext', () => ({
  useTeam: () => ({
    config: { enabled: true, team_id: 'pill-team' },
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
