/**
 * job을 옮겼는데 이전 응답이 늦게 오는 경우.
 *
 * 응답의 주인을 확인하지 않으면 화면이 옛 job으로 되돌아갑니다. 주소는 B인데
 * 화면과 삭제 버튼은 A를 가리키게 되어, 지우면 엉뚱한 기록이 사라집니다.
 */

import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useJobStream } from './useJobStream';

/** job별로 응답을 붙잡아 두었다가 원할 때 풀어 줍니다. */
const gates = new Map<string, () => void>();

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

function jobBody(jobId: string) {
  return {
    job_id: jobId,
    config_id: 'cfg',
    run_id: `run-${jobId}`,
    status: 'succeeded',
    status_label: '완료',
    created_at: '2026-01-01T00:00:00Z',
    started_at: '2026-01-01T00:00:00Z',
    finished_at: '2026-01-01T01:00:00Z',
    elapsed_seconds: 60,
    exit_code: 0,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: {
      available: false,
      reason: null,
      message: null,
      total_epochs: null,
      current_epoch: null,
      eta_seconds: null,
      epochs: [],
    },
    log_lines: 0,
    orphan_note: null,
  };
}

beforeEach(() => {
  gates.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      const match = /\/api\/train\/jobs\/([^/?]+)/.exec(path);
      const jobId = match?.[1] ?? '';
      if (path.includes('/logs')) return jsonResponse({ lines: [], next: 0, complete: true });
      // A는 붙잡아 둡니다. 테스트가 풀어 줄 때까지 응답하지 않습니다.
      if (jobId === 'A') {
        await new Promise<void>((resolve) => gates.set('A', resolve));
      }
      return jsonResponse(jobBody(jobId));
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useJobStream', () => {
  it('job을 옮긴 뒤 늦게 온 이전 응답을 화면에 넣지 않는다', async () => {
    const { result, rerender } = renderHook(({ id }) => useJobStream(id), {
      initialProps: { id: 'A' },
    });

    // A 응답이 붙잡혀 있는 동안 B로 옮깁니다.
    await waitFor(() => expect(gates.has('A')).toBe(true));
    rerender({ id: 'B' });

    // B는 A의 요청이 아직 끝나지 않았어도 조회를 건너뛰지 않아야 합니다.
    await waitFor(() => expect(result.current.job?.job_id).toBe('B'));

    // 이제 A를 풀어 줍니다. 늦게 도착해도 화면은 B로 남아야 합니다.
    gates.get('A')?.();
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(result.current.job?.job_id).toBe('B');
  });
});
