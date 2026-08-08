import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from './client';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('resumeJob', () => {
  it('브라우저 로그인 token을 Authorization header로 보낸다', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ run_id: 'resumed' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetch);

    await api.resumeJob('job-id', undefined, 'browser-token');

    expect(fetch).toHaveBeenCalledWith(
      '/api/train/jobs/job-id/resume',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer browser-token',
        },
      }),
    );
  });
});
