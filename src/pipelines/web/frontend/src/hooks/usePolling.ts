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
  /**
   * 도는 중에 들어온 `refresh()` 요청. 끝나면 한 번 더 돕니다.
   *
   * 그냥 버리면 방금 바꾼 값이 화면에 안 옵니다. 진행 중이던 응답은 바꾸기 **전**
   * 상태라, 그것을 넣고 끝내면 다음 주기(최대 60초)까지 옛 값이 남습니다. 실제로
   * Kaggle 점수를 저장해도 목록이 `-`인 채로 있었습니다.
   */
  const again = useRef(false);
  const mounted = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(async () => {
    if (inFlight.current) {
      again.current = true;
      return;
    }
    inFlight.current = true;
    try {
      do {
        again.current = false;
        try {
          const result = await fetcherRef.current();
          if (!mounted.current) return;
          setData(result);
          setError(null);
        } catch (caught) {
          if (!mounted.current) return;
          setError(caught instanceof Error ? caught.message : '알 수 없는 오류가 발생했습니다.');
        }
      } while (again.current);
    } finally {
      inFlight.current = false;
      again.current = false;
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
