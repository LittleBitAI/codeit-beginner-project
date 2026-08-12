import { useCallback, useEffect, useMemo, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';

import { api } from './api/client';
import type {
  AppSettings,
  DataSource,
  Defaults,
  ExperimentListing,
  GpuStatus,
  JobListing,
  QueueState,
} from './api/types';
import { AppShell, type DatasetOption } from './components/AppShell';
import { usePolling } from './hooks/usePolling';
import { sourceKeyOf } from './lib/dataSource';
import { groupByDataset, mergeRecords, namesDataset } from './lib/records';
import { datasetLabel } from './lib/runSpec';
import { Canvas } from './screens/Canvas';
import { Live } from './screens/Live';
import { NewExperimentSheet } from './screens/NewExperimentSheet';
import { PrepareSheet } from './screens/PrepareSheet';
import { Runs } from './screens/Runs';
import { SettingsSheet } from './screens/SettingsSheet';
import { DraftProvider, useDraft } from './state/DraftContext';
import { TeamGate } from './team/TeamGate';
import { TeamProvider } from './team/TeamContext';
import { useTeamRuns } from './team/useTeamRuns';

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

type SheetKey = 'new' | 'settings' | 'prepare' | null;

function Shell() {
  // 목록은 3초마다, 설정 정의와 전처리 선택은 필요할 때만 읽습니다.
  const listing = usePolling<JobListing>(() => api.listJobs(), 3000);
  // registry는 훨씬 느리게 읽습니다. 이 응답은 index 전체를 훑기 때문에 기록이 쌓인
  // 저장소에서는 수십 초가 걸리고, 내용은 학습 하나가 평가·등록까지 끝나야 바뀝니다.
  // 3초 주기로 두면 앞 요청이 끝나기도 전에 다음 주기가 와서 backend가 쉬지 못합니다.
  const experiments = usePolling<ExperimentListing>(() => api.listExperiments(), 60000);
  const queue = usePolling<QueueState>(() => api.readQueue(), 3000);
  const gpu = usePolling<GpuStatus>(() => api.gpu(), 5000);
  const defaults = usePolling<Defaults>(() => api.defaults(), 0);
  const source = usePolling<{ source: DataSource | null }>(() => api.getDataSource(), 0);
  const settings = usePolling<AppSettings>(() => api.settings(), 0);
  // 팀원이 지금 돌리는 학습. 목록을 한 번 읽고 그 뒤 변화는 구독으로 받습니다.
  const team = useTeamRuns();
  const { draft, setDataFields } = useDraft();

  const [sheet, setSheet] = useState<SheetKey>(null);
  // 사람이 고른 dataset. 아직 안 골랐으면 기록이 가장 많은 것을 씁니다.
  const [pickedDataset, setPickedDataset] = useState<string | null>(null);

  const selected = source.data?.source ?? null;
  const active =
    listing.data?.jobs.find((job) => job.job_id === listing.data?.active_job_id) ?? null;

  const records = useMemo(
    () => mergeRecords(experiments.data?.experiments ?? [], listing.data?.jobs ?? []),
    [experiments.data, listing.data],
  );
  const groups = useMemo(() => groupByDataset(records), [records]);

  const datasets: DatasetOption[] = useMemo(
    () =>
      groups.map((group) => ({
        key: group.key,
        short: group.key,
        sub: `${group.count}건의 기록`,
        count: group.count,
      })),
    [groups],
  );

  const datasetKey = pickedDataset ?? groups[0]?.key ?? null;
  const shownRecords = useMemo(
    () => records.filter((item) => item.datasetKey === datasetKey),
    [records, datasetKey],
  );
  // 어떤 dataset으로 돌렸는지 이름을 댈 수 없는 기록. 왼쪽 목록에 줄이 없으므로
  // 어느 화면에도 나오지 않습니다. 몇 건인지는 목록 화면이 말해 줍니다.
  const unnamedCount = records.length - records.filter(namesDataset).length;
  // 지금 도는 학습이 어느 dataset의 것인지. 왼쪽 목록에 점을 다는 데 씁니다.
  const runningDataset = active ? (datasetLabel(active.data_inputs) ?? null) : null;

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

  const refreshJobs = useCallback(() => {
    listing.refresh();
    queue.refresh();
  }, [listing, queue]);

  const handleSourceChanged = useCallback(() => source.refresh(), [source]);

  return (
    <AppShell
      datasets={datasets}
      activeDataset={datasetKey}
      onPickDataset={setPickedDataset}
      gpu={gpu.data}
      running={runningDataset}
      onOpenPrepare={() => setSheet('prepare')}
      onOpenSettings={() => setSheet('settings')}
    >
      <Routes>
        <Route
          path="/"
          element={
            <Runs
              datasetKey={datasetKey}
              records={shownRecords}
              liveJob={active}
              queue={queue.data}
              scope={experiments.data?.scope}
              unnamedCount={unnamedCount}
              teamRuns={team.runs}
              teamAvailable={team.available}
              error={listing.error}
              onNewExperiment={() => setSheet('new')}
              onRemoveFromQueue={(entryId) => {
                void api.removeFromQueue(entryId).then(() => queue.refresh());
              }}
              onResumeQueue={() => {
                void api.resumeQueue().then(() => queue.refresh());
              }}
              onCancelJob={(jobId) => {
                void api.cancelJob(jobId).then(refreshJobs);
              }}
            />
          }
        />
        {/* 캔버스도 왼쪽에서 고른 dataset 안에서만 고릅니다. 데이터가 다른 실행을
            나란히 세우면 모델 차이인지 데이터 차이인지 구별할 수 없습니다. */}
        <Route
          path="/canvas"
          element={
            <Canvas
              datasetKey={datasetKey}
              records={shownRecords}
              loading={experiments.loading}
            />
          }
        />
        <Route path="/monitor" element={<Live listing={listing.data} onNewExperiment={() => setSheet('new')} />} />
        <Route
          path="/monitor/:jobId"
          element={<Live listing={listing.data} onNewExperiment={() => setSheet('new')} />}
        />
        {/* 없어진 주소(`/team` 같은)를 눌러도 빈 화면에 갇히지 않습니다. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      {sheet === 'new' && (
        <NewExperimentSheet
          defaults={defaults.data}
          source={selected}
          datasetKey={datasetKey}
          queuedCount={queue.data?.entries.length ?? 0}
          busy={Boolean(listing.data?.active_job_id)}
          onClose={() => setSheet(null)}
          onStarted={refreshJobs}
        />
      )}
      {sheet === 'settings' && (
        <SettingsSheet
          gpu={gpu.data}
          scope={experiments.data?.scope}
          settings={settings.data}
          onClose={() => setSheet(null)}
          onSaved={settings.refresh}
        />
      )}
      {sheet === 'prepare' && (
        <PrepareSheet
          source={selected}
          onSelected={handleSourceChanged}
          onPrepared={handleSourceChanged}
          onClose={() => setSheet(null)}
        />
      )}
    </AppShell>
  );
}
