import { useCallback, useEffect, useRef, useState } from 'react';

export interface PollingState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
}

/**
 * 주기적으로 같은 요청을 반복합니다.
 *
 * 요청이 겹치지 않도록 이전 요청이 끝나기 전에는 새 요청을 보내지 않습니다.
 * 라이브러리를 쓰지 않는 이유는 이 화면이 필요로 하는 것이 이게 전부이기 때문입니다.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled = true,
): PollingState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(enabled);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const result = await fetcherRef.current();
      if (!mounted.current) return;
      setData(result);
      setError(null);
    } catch (caught) {
      if (!mounted.current) return;
      setError(caught instanceof Error ? caught.message : '알 수 없는 오류가 발생했습니다.');
    } finally {
      inFlight.current = false;
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    void run();
    if (intervalMs <= 0) return;
    const timer = window.setInterval(() => void run(), intervalMs);
    return () => window.clearInterval(timer);
  }, [enabled, intervalMs, run]);

  return { data, error, loading, refresh: run };
}
