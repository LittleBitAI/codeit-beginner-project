import { useCallback } from 'react';
import { Route, Routes } from 'react-router-dom';

import { api } from './api/client';
import type { DataSource, Defaults, JobListing } from './api/types';
import { AppShell } from './components/AppShell';
import { AlertRow } from './components/primitives';
import { usePolling } from './hooks/usePolling';
import { ConfigReview } from './screens/ConfigReview';
import { LiveMonitor } from './screens/LiveMonitor';
import { NewExperiment } from './screens/NewExperiment';
import { TrainingOverview } from './screens/TrainingOverview';
import { DraftProvider, useDraft } from './state/DraftContext';

export function App() {
  return (
    <DraftProvider>
      <Shell />
    </DraftProvider>
  );
}

function Shell() {
  // 목록은 3초마다, 설정 정의와 전처리 선택은 필요할 때만 읽습니다.
  const listing = usePolling<JobListing>(() => api.listJobs(), 3000);
  const defaults = usePolling<Defaults>(() => api.defaults(), 0);
  const source = usePolling<{ source: DataSource | null }>(() => api.getDataSource(), 0);
  const { setDataFields } = useDraft();

  const active =
    listing.data?.jobs.find((job) => job.job_id === listing.data?.active_job_id) ?? null;

  // 전처리 데이터셋을 고르면 새 실험의 artifact 4칸을 곧바로 채웁니다.
  const handleSourceSelected = useCallback(
    (selected: DataSource) => {
      setDataFields(selected.data);
      source.refresh();
    },
    [setDataFields, source],
  );

  // 준비가 끝나면 backend가 그 결과를 이미 골라 두었으므로 다시 읽어 옵니다.
  const handlePrepared = useCallback(() => {
    source.refresh();
  }, [source]);

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
      </Routes>
    </AppShell>
  );
}
