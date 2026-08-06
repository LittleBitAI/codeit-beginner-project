import { useCallback, useEffect, useMemo } from 'react';
import { Route, Routes } from 'react-router-dom';

import { api } from './api/client';
import type { DataSource, Defaults, JobListing } from './api/types';
import { AppShell } from './components/AppShell';
import { AlertRow } from './components/primitives';
import { usePolling } from './hooks/usePolling';
import { sourceKeyOf } from './lib/dataSource';
import { ConfigReview } from './screens/ConfigReview';
import { ExperimentComparison } from './screens/ExperimentComparison';
import { LiveMonitor } from './screens/LiveMonitor';
import { NewExperiment } from './screens/NewExperiment';
import { TrainingOverview } from './screens/TrainingOverview';
import { TeamActivity } from './screens/TeamActivity';
import { DraftProvider, useDraft } from './state/DraftContext';
import { TeamGate } from './team/TeamGate';
import { TeamProvider } from './team/TeamContext';

export function App() {
  return (
    <TeamProvider>
      <TeamGate>
        <DraftProvider>
          <Shell />
        </DraftProvider>
      </TeamGate>
    </TeamProvider>
  );
}

function Shell() {
  // 목록은 3초마다, 설정 정의와 전처리 선택은 필요할 때만 읽습니다.
  const listing = usePolling<JobListing>(() => api.listJobs(), 3000);
  const defaults = usePolling<Defaults>(() => api.defaults(), 0);
  const source = usePolling<{ source: DataSource | null }>(() => api.getDataSource(), 0);
  const { draft, setDataFields } = useDraft();

  const active =
    listing.data?.jobs.find((job) => job.job_id === listing.data?.active_job_id) ?? null;

  const selected = source.data?.source ?? null;

  // 고른 데이터셋이 바뀌면 필수 4개와 숨은 선택 test manifest를 그 값으로 맞춥니다.
  //
  // 예전에는 빈 칸만 채웠습니다. 그래서 데이터셋을 바꿔도 이전 값이 그대로 남아,
  // 화면에는 새 데이터셋이 보이는데 실제로는 예전 데이터로 학습되는 일이 있었습니다.
  // 데이터셋 선택은 명시적인 행동이므로 그 값이 이깁니다.
  const sourceKey = useMemo(() => sourceKeyOf(selected), [selected]);

  useEffect(() => {
    if (!sourceKey || !selected || draft.sourceKey === sourceKey) return;
    setDataFields({ ...selected.data }, sourceKey);
  }, [sourceKey, selected, draft.sourceKey, setDataFields]);

  const handleSourceSelected = useCallback(() => source.refresh(), [source]);
  const handlePrepared = useCallback(() => source.refresh(), [source]);

  return (
    <AppShell activeJob={active}>
      {listing.error && (
        <div style={{ marginBottom: 12 }}>
          <AlertRow level="error" title="backend에 연결하지 못했습니다">
            {listing.error} 서버를 실행하려면 저장소 root에서{' '}
            <code>python -m src.pipelines.web.server</code>를 실행하세요.
          </AlertRow>
        </div>
      )}
      <Routes>
        <Route
          path="/"
          element={
            <TrainingOverview
              listing={listing.data}
              source={source.data?.source ?? null}
              onSourceSelected={handleSourceSelected}
              onPrepared={handlePrepared}
            />
          }
        />
        <Route
          path="/new"
          element={
            <NewExperiment defaults={defaults.data} source={source.data?.source ?? null} />
          }
        />
        <Route
          path="/review"
          element={
            <ConfigReview
              defaults={defaults.data}
              listing={listing.data}
              onStarted={listing.refresh}
            />
          }
        />
        <Route path="/monitor" element={<LiveMonitor listing={listing.data} />} />
        <Route path="/monitor/:jobId" element={<LiveMonitor listing={listing.data} />} />
        <Route path="/compare" element={<ExperimentComparison />} />
        <Route path="/team" element={<TeamActivity defaults={defaults.data} />} />
      </Routes>
    </AppShell>
  );
}
