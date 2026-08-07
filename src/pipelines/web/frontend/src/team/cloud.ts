import { Amplify } from 'aws-amplify';
import { generateClient } from 'aws-amplify/api';
import 'aws-amplify/auth/enable-oauth-listener';
import {
  fetchAuthSession,
  getCurrentUser,
  signInWithRedirect,
  signOut,
  type AuthUser,
} from 'aws-amplify/auth';

import type { TeamConfig, TeamLogBatch, TeamRun } from '../api/types';
import { decodeJson, decodeLines } from './decode';

const RUN_FIELDS = `
  teamId cloudRunId localJobId runId actorSub actorName actorSource status settings dataInputs
  progress summary artifacts evaluation message createdAt startedAt finishedAt heartbeatAt
  revision
`;
const LOG_FIELDS = `teamId cloudRunId startSeq endSeq lines createdAt`;

let configured = false;
interface GraphQLClient {
  graphql: (options: Record<string, unknown>) => any;
}

let client: GraphQLClient | null = null;

function required(value: string | null, label: string): string {
  if (!value) throw new Error(`팀 동기화 설정 ${label}이 없습니다.`);
  return value;
}

export function configureCloud(config: TeamConfig): void {
  if (!config.enabled || configured) return;
  const origin = window.location.origin;
  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId: required(config.user_pool_id, 'user_pool_id'),
        userPoolClientId: required(config.user_pool_client_id, 'user_pool_client_id'),
        loginWith: {
          oauth: {
            domain: required(config.cognito_domain, 'cognito_domain'),
            scopes: ['openid', 'email', 'profile'],
            redirectSignIn: [`${origin}/`],
            redirectSignOut: [`${origin}/`],
            responseType: 'code',
          },
        },
      },
    },
    API: {
      GraphQL: {
        endpoint: required(config.appsync_url, 'appsync_url'),
        region: config.region,
        defaultAuthMode: 'userPool',
      },
    },
  });
  client = generateClient() as unknown as GraphQLClient;
  configured = true;
}

export async function currentUser(): Promise<AuthUser | null> {
  try {
    return await getCurrentUser();
  } catch {
    return null;
  }
}

export async function login(): Promise<void> {
  await signInWithRedirect();
}

export async function logout(): Promise<void> {
  await signOut();
}

export async function accessToken(): Promise<string | null> {
  const session = await fetchAuthSession();
  return session.tokens?.accessToken.toString() ?? null;
}

function normalizeRun(raw: Record<string, unknown>): TeamRun {
  return {
    ...(raw as unknown as TeamRun),
    settings: decodeJson(raw.settings),
    dataInputs: decodeJson(raw.dataInputs),
    progress: decodeJson(raw.progress),
    summary: decodeJson(raw.summary),
    artifacts: decodeJson(raw.artifacts),
    evaluation: decodeJson(raw.evaluation),
  };
}

function normalizeBatch(raw: Record<string, unknown>): TeamLogBatch {
  return { ...(raw as unknown as TeamLogBatch), lines: decodeLines(raw.lines) };
}

function graphClient(): GraphQLClient {
  if (!client) throw new Error('팀 cloud client가 초기화되지 않았습니다.');
  return client;
}

export async function listRuns(teamId: string): Promise<TeamRun[]> {
  const result = await graphClient().graphql({
    query: `query TeamRuns($teamId: ID!) { teamRuns(teamId: $teamId) { ${RUN_FIELDS} } }`,
    variables: { teamId },
    authMode: 'userPool',
  });
  const data = 'data' in result ? result.data : undefined;
  const rows = (data as { teamRuns?: Record<string, unknown>[] } | undefined)?.teamRuns ?? [];
  return rows.map(normalizeRun);
}

export async function listLogs(
  teamId: string,
  cloudRunId: string,
  afterSeq: number,
): Promise<TeamLogBatch[]> {
  const result = await graphClient().graphql({
    query: `query RunLogs($teamId: ID!, $cloudRunId: ID!, $afterSeq: Int!) {
      runLogs(teamId: $teamId, cloudRunId: $cloudRunId, afterSeq: $afterSeq) { ${LOG_FIELDS} }
    }`,
    variables: { teamId, cloudRunId, afterSeq },
    authMode: 'userPool',
  });
  const data = 'data' in result ? result.data : undefined;
  const rows = (data as { runLogs?: Record<string, unknown>[] } | undefined)?.runLogs ?? [];
  return rows.map(normalizeBatch);
}

interface Subscription {
  unsubscribe: () => void;
}

export function subscribeRuns(
  teamId: string,
  next: (run: TeamRun) => void,
  error: (problem: unknown) => void,
): Subscription {
  const operation = graphClient().graphql({
    query: `subscription RunChanged($teamId: ID!) {
      onTeamRunChanged(teamId: $teamId) { ${RUN_FIELDS} }
    }`,
    variables: { teamId },
    authMode: 'userPool',
  });
  if (!('subscribe' in operation)) throw new Error('AppSync subscription을 시작하지 못했습니다.');
  return operation.subscribe({
    next: ({ data }: { data: Record<string, unknown> }) => {
      const raw = (data as { onTeamRunChanged?: Record<string, unknown> }).onTeamRunChanged;
      if (raw) next(normalizeRun(raw));
    },
    error,
  });
}

export function subscribeLogs(
  teamId: string,
  cloudRunId: string,
  next: (batch: TeamLogBatch) => void,
  error: (problem: unknown) => void,
): Subscription {
  const operation = graphClient().graphql({
    query: `subscription Logs($teamId: ID!, $cloudRunId: ID!) {
      onRunLogBatch(teamId: $teamId, cloudRunId: $cloudRunId) { ${LOG_FIELDS} }
    }`,
    variables: { teamId, cloudRunId },
    authMode: 'userPool',
  });
  if (!('subscribe' in operation)) throw new Error('AppSync log subscription을 시작하지 못했습니다.');
  return operation.subscribe({
    next: ({ data }: { data: Record<string, unknown> }) => {
      const raw = (data as { onRunLogBatch?: Record<string, unknown> }).onRunLogBatch;
      if (raw) next(normalizeBatch(raw));
    },
    error,
  });
}
