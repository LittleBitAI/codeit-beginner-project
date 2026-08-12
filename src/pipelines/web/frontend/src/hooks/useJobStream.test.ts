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
/** A의 응답을 붙잡을지. 두 번째 A는 곧바로 답하게 두려고 끕니다. */
let holdA = true;
/**
 * 같은 A라도 몇 번째 요청인지 응답에 새깁니다.
 *
 * 두 A가 똑같은 값을 돌려주면 첫 A의 낡은 응답을 잘못 적용해도 관찰값이 같아
 * 테스트가 통과합니다. 그러면 회귀를 막지 못하는 테스트가 됩니다.
 */
let aRequests = 0;

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

function jobBody(jobId: string, mark = '') {
  return {
    job_id: jobId,
    config_id: 'cfg',
    run_id: `run-${jobId}${mark}`,
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
  holdA = true;
  aRequests = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      const match = /\/api\/train\/jobs\/([^/?]+)/.exec(path);
      const jobId = match?.[1] ?? '';
      if (path.includes('/logs')) return jsonResponse({ lines: [], next: 0, complete: true });
      if (jobId !== 'A') return jsonResponse(jobBody(jobId));
      // 몇 번째 A인지 응답에 새깁니다. 어느 응답이 화면에 남았는지 구별하려는 것입니다.
      const mark = `-${(aRequests += 1)}`;
      // 첫 A는 붙잡아 둡니다. 테스트가 풀어 줄 때까지 응답하지 않습니다.
      if (holdA) {
        await new Promise<void>((resolve) => gates.set('A', resolve));
      }
      return jsonResponse(jobBody('A', mark));
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useJobStream', () => {
  it('A에서 B를 거쳐 다시 A로 와도 첫 A의 늦은 응답을 받아들이지 않는다', async () => {
    // job 이름만 비교하면 이 경로가 통과합니다. 첫 A의 응답이 두 번째 A의 것인 양
    // 새 상태를 덮고, 두 번째 A의 잠금까지 풀어 조회가 겹칩니다.
    const { result, rerender } = renderHook(({ id }) => useJobStream(id), {
      initialProps: { id: 'A' },
    });

    await waitFor(() => expect(gates.has('A')).toBe(true));
    const firstA = gates.get('A');
    gates.delete('A');

    rerender({ id: 'B' });
    await waitFor(() => expect(result.current.job?.job_id).toBe('B'));

    // 다시 A로. 이번 A는 붙잡지 않고 곧바로 응답합니다.
    holdA = false;
    rerender({ id: 'A' });
    // 두 번째 A의 응답에는 -2가 새겨져 있습니다.
    await waitFor(() => expect(result.current.job?.run_id).toBe('run-A-2'));

    // 이제 첫 A(-1)를 풀어 줍니다. 같은 이름이지만 지난 세대라 버려야 합니다.
    firstA?.();
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(result.current.job?.run_id).toBe('run-A-2');
    expect(result.current.error).toBeNull();
  });

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
