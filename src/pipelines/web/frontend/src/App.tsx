import { Route, Routes } from 'react-router-dom';

import { api } from './api/client';
import type { Defaults, JobListing } from './api/types';
import { AppShell } from './components/AppShell';
import { AlertRow } from './components/primitives';
import { usePolling } from './hooks/usePolling';
import { ConfigReview } from './screens/ConfigReview';
import { LiveMonitor } from './screens/LiveMonitor';
import { NewExperiment } from './screens/NewExperiment';
import { TrainingOverview } from './screens/TrainingOverview';
import { DraftProvider } from './state/DraftContext';

export function App() {
  // 목록은 3초마다, 설정 정의는 한 번만 읽습니다.
  const listing = usePolling<JobListing>(() => api.listJobs(), 3000);
  const defaults = usePolling<Defaults>(() => api.defaults(), 0);

  const active =
    listing.data?.jobs.find((job) => job.job_id === listing.data?.active_job_id) ?? null;

  return (
    <DraftProvider>
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
          <Route path="/" element={<TrainingOverview listing={listing.data} />} />
          <Route path="/new" element={<NewExperiment defaults={defaults.data} />} />
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
    </DraftProvider>
  );
}
