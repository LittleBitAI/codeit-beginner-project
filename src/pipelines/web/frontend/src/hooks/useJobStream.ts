import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../api/client';
import type { JobRecord, LogLine } from '../api/types';

const ACTIVE_INTERVAL_MS = 1000;
const IDLE_INTERVAL_MS = 3000;
const MAX_LINES = 2000;

export interface JobStream {
  job: JobRecord | null;
  lines: LogLine[];
  error: string | null;
  streaming: boolean;
  refresh: () => void;
}

/**
 * 한 job의 상태와 log를 cursor 방식으로 따라갑니다.
 *
 * ``?after=<seq>``로 이어 받기 때문에 빠뜨리는 줄이 없습니다. 실행 중에는 1초,
 * 끝난 뒤에는 3초 간격으로 확인하고, 끝난 job의 log를 다 받으면 멈춥니다.
 */
export function useJobStream(jobId: string | undefined): JobStream {
  const [job, setJob] = useState<JobRecord | null>(null);
  const [lines, setLines] = useState<LogLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const cursor = useRef(0);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  /**
   * 지금 보고 있는 job. 응답을 반영하기 전에 이것과 같은지 확인합니다.
   *
   * 확인하지 않으면 job을 옮긴 뒤 늦게 도착한 이전 응답이 화면을 옛 job으로
   * 되돌려 놓습니다. 주소는 B인데 화면과 삭제 버튼은 A를 가리키게 되어, 지우면
   * 엉뚱한 기록이 사라집니다.
   */
  const watching = useRef(jobId);

  const reset = useCallback(() => {
    cursor.current = 0;
    setLines([]);
    setJob(null);
    setError(null);
  }, []);

  const tick = useCallback(async () => {
    if (!jobId || inFlight.current) return;
    inFlight.current = true;
    try {
      const record = await api.getJob(jobId);
      const page = await api.logs(jobId, cursor.current);
      if (!mounted.current || watching.current !== jobId) return;
      setJob(record);
      if (page.lines.length > 0) {
        cursor.current = page.next;
        setLines((previous) => [...previous, ...page.lines].slice(-MAX_LINES));
      }
      setError(null);
      setStreaming(record.status === 'running' || record.status === 'queued');
    } catch (caught) {
      if (!mounted.current || watching.current !== jobId) return;
      setError(caught instanceof Error ? caught.message : '알 수 없는 오류가 발생했습니다.');
      setStreaming(false);
    } finally {
      // 지금 보고 있는 job의 요청일 때만 잠금을 풉니다. 옮긴 뒤 늦게 끝난 이전
      // 요청이 새 job의 잠금을 풀어 버리면 요청이 겹칩니다.
      if (watching.current === jobId) inFlight.current = false;
    }
  }, [jobId]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    // 보는 대상을 먼저 바꿉니다. 이전 job의 응답은 이 시점부터 버려지고, 아직
    // 끝나지 않은 그 요청 때문에 새 job의 첫 조회를 거르지도 않습니다.
    watching.current = jobId;
    inFlight.current = false;
    reset();
    if (!jobId) return;
    void tick();
  }, [jobId, reset, tick]);

  const active = job === null || job.status === 'running' || job.status === 'queued';

  useEffect(() => {
    if (!jobId) return;
    const timer = window.setInterval(
      () => void tick(),
      active ? ACTIVE_INTERVAL_MS : IDLE_INTERVAL_MS,
    );
    return () => window.clearInterval(timer);
  }, [jobId, active, tick]);

  return { job, lines, error, streaming, refresh: tick };
}
