import { render, screen } from '@testing-library/react';

import type { TeamConfig } from '../api/types';

const state = vi.hoisted(() => ({
  config: null as TeamConfig | null,
  user: null as { username: string } | null,
}));

vi.mock('./TeamContext', () => ({
  useTeam: () => ({
    ready: true,
    config: state.config,
    user: state.user,
    error: null,
    login: vi.fn(),
  }),
}));

const { TeamGate } = await import('./TeamGate');

function config(overrides: Partial<TeamConfig> = {}): TeamConfig {
  return {
    enabled: true,
    team_id: 'pill-team',
    appsync_url: 'https://example/graphql',
    region: 'ap-northeast-2',
    user_pool_id: 'pool',
    user_pool_client_id: 'client',
    cognito_domain: 'example.auth',
    actor: null,
    unattended_actor: null,
    ...overrides,
  };
}

afterEach(() => {
  state.config = null;
  state.user = null;
});

test('팀 동기화가 켜져 있고 로그인이 없으면 로그인 화면으로 막는다', () => {
  state.config = config();
  render(
    <TeamGate>
      <p>학습 화면</p>
    </TeamGate>,
  );

  expect(screen.getByText('팀 학습 관제판 로그인')).toBeInTheDocument();
  expect(screen.queryByText('학습 화면')).toBeNull();
});

test('로그인할 수 없는 환경이라고 서버가 알려 주면 그대로 통과시킨다', () => {
  // Colab처럼 Cognito redirect 주소를 등록할 수 없는 곳입니다. 서버가
  // PILL_TEAM_ACTOR로 그 사실을 알려 주면 화면을 막을 이유가 없습니다.
  state.config = config({ actor: '지현 (Colab)' });
  render(
    <TeamGate>
      <p>학습 화면</p>
    </TeamGate>,
  );

  expect(screen.getByText('학습 화면')).toBeInTheDocument();
  expect(screen.queryByText('팀 학습 관제판 로그인')).toBeNull();
});

test('무인 대기열 이름만으로는 로그인을 건너뛰지 않는다', () => {
  // PILL_TEAM_UNATTENDED_ACTOR는 로그인이 **되는** PC에서 밤샘 대기열을 위해 켭니다.
  // 이것까지 관문을 열어 버리면 사람이 로그인을 잊은 채로 쓰다가, 학습이 본인 이름
  // 대신 그 이름으로 팀에 기록됩니다. Colab용 actor와 나눠 둔 이유가 이것입니다.
  state.config = config({ unattended_actor: '다솔 야간 대기열' });
  render(
    <TeamGate>
      <p>학습 화면</p>
    </TeamGate>,
  );

  expect(screen.getByText('팀 학습 관제판 로그인')).toBeInTheDocument();
  expect(screen.queryByText('학습 화면')).toBeNull();
});
