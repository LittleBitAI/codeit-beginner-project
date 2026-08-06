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
