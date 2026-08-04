import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { usePolling } from './usePolling';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('usePolling', () => {
  it('처음 한 번 즉시 부르고 주기마다 다시 부른다', async () => {
    const fetcher = vi.fn().mockResolvedValue('ok');

    renderHook(() => usePolling(fetcher, 1000));
    await act(async () => {});

    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('이전 요청이 끝나기 전에는 새 요청을 보내지 않는다', async () => {
    let release: (value: string) => void = () => {};
    const fetcher = vi.fn(
      () =>
        new Promise<string>((resolve) => {
          release = resolve;
        }),
    );

    renderHook(() => usePolling(fetcher, 100));
    await act(async () => {});

    await act(async () => {
      vi.advanceTimersByTime(500); // 여러 주기가 지나도
    });

    expect(fetcher).toHaveBeenCalledTimes(1); // 요청은 겹치지 않습니다

    await act(async () => {
      release('ok');
    });
  });

  it('enabled가 false면 아무것도 부르지 않는다', async () => {
    const fetcher = vi.fn().mockResolvedValue('ok');

    renderHook(() => usePolling(fetcher, 1000, false));
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });

    expect(fetcher).not.toHaveBeenCalled();
  });

  it('실패하면 오류 메시지를 담고 예외를 밖으로 던지지 않는다', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('backend 연결 실패'));

    const { result } = renderHook(() => usePolling(fetcher, 0));
    // fake timer를 쓰는 중이라 waitFor 대신 act로 microtask를 비웁니다.
    await act(async () => {});

    expect(result.current.error).toBe('backend 연결 실패');
    expect(result.current.data).toBeNull();
  });

  it('성공하면 결과를 담고 오류를 지운다', async () => {
    const fetcher = vi.fn().mockResolvedValue({ value: 1 });

    const { result } = renderHook(() => usePolling(fetcher, 0));
    await act(async () => {});

    expect(result.current.data).toEqual({ value: 1 });
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it('interval이 0이면 반복하지 않는다', async () => {
    const fetcher = vi.fn().mockResolvedValue('ok');

    renderHook(() => usePolling(fetcher, 0));
    await act(async () => {
      vi.advanceTimersByTime(10000);
    });

    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
