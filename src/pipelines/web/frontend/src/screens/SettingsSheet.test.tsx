import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { AppSettings, GpuStatus } from '../api/types';
import { SettingsSheet } from './SettingsSheet';

let put: { path: string; body: unknown }[] = [];

beforeEach(() => {
  put = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = typeof input === 'string' ? input : input instanceof URL ? input.pathname : input.url;
      if (init?.method === 'PUT') put.push({ path, body: JSON.parse(String(init.body)) });
      return new Response(JSON.stringify({ evaluation_mode: 'serial' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

/** 8GB 카드에서 학습이 6GB를 쓰고 있는 상태. 평가 몫 1.8GB가 안 남습니다. */
function gpu(usedMb: number): GpuStatus {
  return {
    torch: { cuda_available: true, device_count: 1, reason: null },
    telemetry: {
      source: 'nvidia-smi',
      reason: null,
      message: null,
      devices: [
        {
          index: 0,
          name: 'RTX 4060 Ti',
          utilization_percent: 63,
          memory_used_mb: usedMb,
          memory_total_mb: 8192,
          temperature_c: 70,
        },
      ],
    },
    queried_at: '2026-08-12T00:00:00Z',
  };
}

function show(settings: AppSettings | null, used = 2000, onSaved = () => {}) {
  return render(
    <SettingsSheet
      gpu={gpu(used)}
      scope={{ backend: 'local', shared: false }}
      settings={settings}
      onClose={() => {}}
      onSaved={onSaved}
    />,
  );
}

describe('SettingsSheet', () => {
  it('고른 적이 없으면 자동 평가가 꺼져 있다고 말하고 저장을 막는다', () => {
    show({ evaluation_mode: null });

    expect(screen.getByText(/아직 고르지 않아 자동 평가가 꺼져 있습니다/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '저장' })).toBeDisabled();
  });

  it('고르면 저장이 열리고 그 값만 보낸다', async () => {
    const onSaved = vi.fn();
    show({ evaluation_mode: null }, 2000, onSaved);

    fireEvent.click(screen.getByRole('button', { name: /학습과 함께/ }));

    const save = screen.getByRole('button', { name: '저장' });
    expect(save).toBeEnabled();
    fireEvent.click(save);

    await waitFor(() => expect(onSaved).toHaveBeenCalled());
    expect(put).toEqual([{ path: '/api/settings', body: { evaluation_mode: 'parallel' } }]);
  });

  it('이미 고른 값이 눌려 있다', () => {
    show({ evaluation_mode: 'serial' });

    expect(screen.getByRole('button', { name: /학습이 끝난 뒤/ })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });

  it('VRAM이 모자라면 병렬을 고르지 말라고 말한다', () => {
    // 8192 - 6800 = 1392MB. 평가 몫 1800MB가 안 들어갑니다.
    show({ evaluation_mode: null }, 6800);

    expect(screen.getByText(/병렬로 두면 둘 다 out of memory/)).toBeInTheDocument();
  });
});
