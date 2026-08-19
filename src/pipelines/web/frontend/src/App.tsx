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
import { AppShell } from './components/AppShell';
import { usePolling } from './hooks/usePolling';
import { sourceKeyOf } from './lib/dataSource';
import { groupByDataset, mergeRecords, namesDataset } from './lib/records';
import { Board } from './screens/Board';
import { Canvas } from './screens/Canvas';
import { Home } from './screens/Home';
import { Live } from './screens/Live';
import { NewExperimentSheet } from './screens/NewExperimentSheet';
import { DiagnosisSheet } from './screens/DiagnosisSheet';
import { EdaSheet } from './screens/EdaSheet';
import { EmbeddingTrainSheet } from './screens/EmbeddingTrainSheet';
import { Ensemble } from './screens/Ensemble';
import { PrepareSheet } from './screens/PrepareSheet';
import { Records, type DatasetOption } from './screens/Records';
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

type SheetKey = 'new' | 'settings' | 'prepare' | 'eda' | 'embedding' | 'diagnosis' | null;

function Shell() {
  // 목록은 3초마다, 설정 정의와 전처리 선택은 필요할 때만 읽습니다.
  const listing = usePolling<JobListing>(() => api.listJobs(), 3000);
  // registry는 훨씬 느리게 읽습니다. 이 응답은 index 전체를 훑기 때문에 기록이 쌓인
  // 저장소에서는 오래 걸리고, 내용은 학습 하나가 평가·등록까지 끝나야 바뀝니다.
  // 3초 주기로 두면 앞 요청이 끝나기도 전에 다음 주기가 와서 backend가 쉬지 못합니다.
  const experiments = usePolling<ExperimentListing>(() => api.listExperiments(), 60000);
  const queue = usePolling<QueueState>(() => api.readQueue(), 3000);
  const gpu = usePolling<GpuStatus>(() => api.gpu(), 5000);
  const defaults = usePolling<Defaults>(() => api.defaults(), 0);
  const source = usePolling<{ source: DataSource | null }>(() => api.getDataSource(), 0);
  const settings = usePolling<AppSettings>(() => api.settings(), 0);
  // 팀원이 지금 돌리는 학습. 목록을 한 번 읽고 그 뒤 변화는 구독으로 받습니다.
  const team = useTeamRuns();
  const { draft, setDataFields, setTrainFields } = useDraft();

  const [sheet, setSheet] = useState<SheetKey>(null);
  // 진단은 실행 하나에 대한 것이라 어느 실행인지 함께 들고 있어야 합니다.
  const [diagnosisRun, setDiagnosisRun] = useState<string | null>(null);
  // 기록 화면에서 사람이 고른 dataset. 아직 안 골랐으면 기록이 가장 많은 것을 씁니다.
  // **보는 대상일 뿐입니다** — 학습에 실려 갈 데이터는 아래 `selected`가 정합니다.
  const [pickedDataset, setPickedDataset] = useState<string | null>(null);

  const selected = source.data?.source ?? null;
  const active =
    listing.data?.jobs.find((job) => job.job_id === listing.data?.active_job_id) ?? null;

  const records = useMemo(
    () => mergeRecords(experiments.data?.experiments ?? [], listing.data?.jobs ?? []),
    [experiments.data, listing.data],
  );
  const groups = useMemo(() => groupByDataset(records), [records]);

  // 전처리는 끝났지만 아직 학습한 적 없는 dataset도 목록에 둡니다. 기록에서만
  // 목록을 만들면 방금 만든 판이 보이지 않아, 그것으로 학습하려면 어디로 가야 할지
  // 알 수 없습니다. 준비된 판이 다 보여야 다음에 무엇을 돌릴지 고를 수 있습니다.
  const prepared = usePolling(() => api.listDatasets(), 0);

  const datasets: DatasetOption[] = useMemo(() => {
    const withRecords = groups.map((group) => ({
      key: group.key,
      sub: `${group.count}건의 기록`,
      count: group.count,
    }));
    const known = new Set(withRecords.map((item) => item.key));
    const untouched = (prepared.data?.datasets ?? [])
      .filter((item) => item.complete && !known.has(item.name))
      .map((item) => ({ key: item.name, sub: '기록 없음 · 학습 전', count: 0 }));
    return [...withRecords, ...untouched];
  }, [groups, prepared.data]);

  // 아직 아무것도 고르지 않았으면 기록이 가장 많은 dataset, 그것도 없으면 준비만 된
  // 것 중 첫 번째입니다. 고른 것이 없다고 두면 기록 화면의 고르기가 빈 채로 보입니다.
  const datasetKey = pickedDataset ?? groups[0]?.key ?? datasets[0]?.key ?? null;
  const shownRecords = useMemo(
    () => records.filter((item) => item.datasetKey === datasetKey),
    [records, datasetKey],
  );
  // 어떤 dataset으로 돌렸는지 이름을 댈 수 없는 기록. 목록에 줄이 없으므로 어느
  // 화면에도 나오지 않습니다. 몇 건인지는 기록 화면이 말해 줍니다.
  const unnamedCount = records.length - records.filter(namesDataset).length;

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
      gpu={gpu.data}
      running={Boolean(active)}
      onOpenPrepare={() => setSheet('prepare')}
      onOpenEda={() => setSheet('eda')}
      onOpenEmbedding={() => setSheet('embedding')}
      onOpenSettings={() => setSheet('settings')}
    >
      <Routes>
        {/* 첫 화면은 내 학습 하나만 답합니다. 기록과 팀은 각자의 화면이 맡습니다. */}
        <Route
          path="/"
          element={
            <Home
              liveJob={active}
              queue={queue.data}
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
        <Route
          path="/records"
          element={
            <Records
              datasets={datasets}
              datasetKey={datasetKey}
              onPickDataset={setPickedDataset}
              records={shownRecords}
              scope={experiments.data?.scope}
              unnamedCount={unnamedCount}
              error={listing.error}
              onNewExperiment={() => setSheet('new')}
            />
          }
        />
        <Route
          path="/board"
          element={
            <Board
              liveJob={active}
              records={records}
              teamRuns={team.runs}
              teamAvailable={team.available}
              teamLoaded={team.loaded}
              teamError={team.error}
            />
          }
        />
        {/* 캔버스도 기록 화면에서 고른 dataset 안에서만 고릅니다. 데이터가 다른 실행을
            나란히 세우면 모델 차이인지 데이터 차이인지 구별할 수 없습니다. */}
        <Route
          path="/canvas"
          element={
            <Canvas
              datasetKey={datasetKey}
              records={shownRecords}
              loading={experiments.loading}
              onScoreSaved={experiments.refresh}
              onNewExperiment={(settings) => {
                setTrainFields(settings);
                setSheet('new');
              }}
              onOpenDiagnosis={(runId) => {
                setDiagnosisRun(runId);
                setSheet('diagnosis');
              }}
            />
          }
        />
        {/* 기록과 달리 dataset으로 거르지 않습니다. 합쳐도 되는 조합인지는 화면이
            직접 재서 알려 주므로, 미리 좁혀 두면 그 판단을 가립니다. */}
        <Route path="/ensemble" element={<Ensemble />} />
        <Route path="/monitor" element={<Live
              listing={listing.data}
              onNewExperiment={() => setSheet('new')}
              onJobsChanged={refreshJobs}
            />} />
        <Route
          path="/monitor/:jobId"
          element={<Live
              listing={listing.data}
              onNewExperiment={() => setSheet('new')}
              onJobsChanged={refreshJobs}
            />}
        />
        {/* 없어진 주소(`/team` 같은)를 눌러도 빈 화면에 갇히지 않습니다. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      {sheet === 'new' && (
        <NewExperimentSheet
          defaults={defaults.data}
          source={selected}
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
      {sheet === 'eda' && <EdaSheet onClose={() => setSheet(null)} />}
      {sheet === 'embedding' && <EmbeddingTrainSheet onClose={() => setSheet(null)} />}
      {sheet === 'diagnosis' && diagnosisRun !== null && (
        <DiagnosisSheet runId={diagnosisRun} onClose={() => setSheet(null)} />
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
