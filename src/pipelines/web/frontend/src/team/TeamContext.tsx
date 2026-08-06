import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { AuthUser } from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';

import { api } from '../api/client';
import type { TeamConfig, TeamRun } from '../api/types';
import * as cloud from './cloud';

interface TeamState {
  ready: boolean;
  config: TeamConfig;
  user: AuthUser | null;
  error: string | null;
  latestEvent: TeamRun | null;
  getAccessToken: () => Promise<string | null>;
  login: () => Promise<void>;
  logout: () => Promise<void>;
}

const DISABLED: TeamConfig = {
  enabled: false,
  team_id: null,
  appsync_url: null,
  region: 'ap-northeast-2',
  user_pool_id: null,
  user_pool_client_id: null,
  cognito_domain: null,
  actor: null,
};

const TeamContext = createContext<TeamState>({
  ready: true,
  config: DISABLED,
  user: null,
  error: null,
  latestEvent: null,
  getAccessToken: async () => null,
  login: cloud.login,
  logout: cloud.logout,
});

export function TeamProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [config, setConfig] = useState<TeamConfig>(DISABLED);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [latestEvent, setLatestEvent] = useState<TeamRun | null>(null);

  useEffect(
    () =>
      Hub.listen('auth', ({ payload }) => {
        if (payload.event === 'signInWithRedirect' || payload.event === 'signedIn') {
          void cloud.currentUser().then(setUser);
        } else if (payload.event === 'signedOut') {
          setUser(null);
        } else if (payload.event === 'signInWithRedirect_failure') {
          setError('Cognito 로그인 redirect를 완료하지 못했습니다.');
        }
      }),
    [],
  );

  useEffect(() => {
    let active = true;
    void api
      .teamConfig()
      .then(async (loaded) => {
        if (!active) return;
        setConfig(loaded);
        if (loaded.enabled) {
          cloud.configureCloud(loaded);
          setUser(await cloud.currentUser());
        }
      })
      .catch((problem) => {
        if (active) setError(problem instanceof Error ? problem.message : '팀 설정을 읽지 못했습니다.');
      })
      .finally(() => {
        if (active) setReady(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!config.enabled || !config.team_id || !user) return;
    const subscription = cloud.subscribeRuns(
      config.team_id,
      (run) => setLatestEvent(run),
      () => setError('실시간 연결이 끊겼습니다. 팀 화면에서 기록을 다시 불러옵니다.'),
    );
    return () => subscription.unsubscribe();
  }, [config, user]);

  const getAccessToken = useCallback(
    async () => (config.enabled ? cloud.accessToken() : null),
    [config.enabled],
  );
  const value = useMemo(
    () => ({
      ready,
      config,
      user,
      error,
      latestEvent,
      getAccessToken,
      login: cloud.login,
      logout: cloud.logout,
    }),
    [ready, config, user, error, latestEvent, getAccessToken],
  );
  return <TeamContext.Provider value={value}>{children}</TeamContext.Provider>;
}

export function useTeam(): TeamState {
  return useContext(TeamContext);
}
