import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import type { TeamRun } from '../api/types';

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

function run(actor: string, id: string, status: TeamRun['status'] = 'running'): TeamRun {
  return {
    teamId: 'pill-team',
    cloudRunId: id,
    localJobId: id,
    runId: `${actor}-run`,
    actorSub: actor,
    actorName: actor,
    status,
    settings: { epochs: 10, learning_rate: 0.001 },
    dataInputs: {},
    progress: { current_epoch: 2 },
    summary: status === 'succeeded' ? { best_epoch: 8, best_validation_loss: 0.12 } : {},
    artifacts: status === 'succeeded' ? { checkpoint: 's3://pill-team/a.pt' } : {},
    message: null,
    createdAt: `2026-08-05T00:00:0${id}.000Z`,
    startedAt: '2026-08-05T00:00:00.000Z',
    finishedAt: status === 'succeeded' ? '2026-08-05T00:10:00.000Z' : null,
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
  const view = render(<TeamActivity />);
  expect(await screen.findByText('a-run')).toBeInTheDocument();
  expect(await screen.findByText('a 학습 시작')).toBeInTheDocument();
  expect(screen.getByText('learning_rate')).toBeInTheDocument();

  state.latestEvent = run('b', '2');
  view.rerender(<TeamActivity />);
  expect(await screen.findByText('b-run')).toBeInTheDocument();
  fireEvent.click(screen.getByText('b-run'));
  expect(screen.getByText('b · b-run')).toBeInTheDocument();
});

test('완료 결과와 S3 산출물의 팀 공유 가능 여부를 보여준다', async () => {
  cloud.listRuns.mockResolvedValue([run('a', '1', 'succeeded')]);
  render(<TeamActivity />);
  expect(await screen.findByText('완료 결과와 산출물')).toBeInTheDocument();
  expect(screen.getByText(/best_validation_loss/)).toBeInTheDocument();
  expect(screen.getByText(/팀 공유 가능/)).toBeInTheDocument();
  await waitFor(() => expect(cloud.subscribeLogs).toHaveBeenCalled());
});
