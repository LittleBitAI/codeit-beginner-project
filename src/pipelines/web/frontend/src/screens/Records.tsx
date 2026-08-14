/**
 * 기록 화면. **어떤 dataset의, 어떤 모델이 얼마나 나왔나**에 답합니다.
 *
 * 맨 위에서 dataset 하나를 고르고, 그 안의 기록을 모델별로 묶어 세웁니다. 한 번에
 * 한 dataset만 보는 것은 데이터가 다른 실행을 나란히 세우면 모델 차이인지 데이터
 * 차이인지 구별할 수 없기 때문입니다. 여기서 고르는 값은 **보는 대상**일 뿐,
 * 학습에 실제로 쓰이는 데이터는 dataset 준비에서만 바뀝니다.
 */

import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import type { RegistryScope } from '../api/types';
import {
  AlertRow,
  Badge,
  Button,
  Chip,
  EmptyState,
  LinkAction,
  LiveDot,
  Metric,
  MetricGrid,
  SortToggle,
  controlStyle,
} from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { duration, loss, startedAt } from '../lib/format';
import {
  FILTER_LABEL,
  SORT_LABEL,
  countLabel,
  groupByModel,
  hasResult,
  isRunning,
  matchesFilter,
  sortRecords,
  type RecordFilter,
  type RecordSort,
  type RunRecord,
} from '../lib/records';

const FILTERS: RecordFilter[] = ['all', 'evaluated', 'submitted', 'unregistered'];
const SORTS: RecordSort[] = ['recent', 'kaggle', 'loss'];

/** 기록 화면 맨 위에서 고를 수 있는 dataset 하나입니다. */
export interface DatasetOption {
  /** 고르기·비교에 쓰는 값. dataset 이름 그대로입니다. */
  key: string;
  /** 이름 옆에 적는 한 줄 설명. */
  sub: string;
  /** 이 dataset으로 남은 기록 수. */
  count: number;
}

function score(value: number | null): string {
  return value === null ? '-' : value.toFixed(4);
}

export function Records({
  datasets,
  datasetKey,
  onPickDataset,
  records,
  scope,
  unnamedCount,
  error,
  onNewExperiment,
}: {
  /** 고를 수 있는 dataset. 기록이 있는 것과 준비만 된 것이 함께 옵니다. */
  datasets: DatasetOption[];
  datasetKey: string | null;
  onPickDataset: (key: string) => void;
  /** 이미 dataset으로 걸러진 기록들입니다. */
  records: RunRecord[];
  scope: RegistryScope | undefined;
  /** 어떤 dataset으로 돌렸는지 이름을 댈 수 없어 목록에 세우지 못한 기록 수. */
  unnamedCount: number;
  error: string | null;
  onNewExperiment: () => void;
}) {
  const navigate = useNavigate();
  const [filter, setFilter] = useState<RecordFilter>('all');
  const [sort, setSort] = useState<RecordSort>('recent');

  const shown = useMemo(
    () => sortRecords(records.filter((record) => matchesFilter(record, filter)), sort),
    [records, filter, sort],
  );

  /**
   * 결과 없이 끝난 기록은 목록 맨 아래로 접습니다.
   *
   * 전체를 볼 때만 접습니다. 다른 표는 사람이 이미 좁혀 놓은 것이라, 그 안에서 또
   * 접으면 "12건이라는데 아무것도 안 보인다"가 됩니다.
   */
  const folded = filter === 'all' ? shown.filter((record) => !hasResult(record)) : [];
  const listed = folded.length > 0 ? shown.filter(hasResult) : shown;
  const groups = useMemo(() => groupByModel(listed), [listed]);

  const openRecord = (record: RunRecord) =>
    record.jobId
      ? navigate(`/monitor/${record.jobId}`)
      : navigate(`/canvas?run=${encodeURIComponent(record.runId)}`);

  const bestKaggle = records
    .map((record) => record.metrics.kaggle)
    .filter((value): value is number => value !== null);
  const bestLoss = records
    .map((record) => record.metrics.bestValidationLoss)
    .filter((value): value is number => value !== null);

  const stats = [
    `기록 ${records.length}건`,
    bestKaggle.length > 0 ? `최고 Kaggle ${Math.max(...bestKaggle).toFixed(4)}` : null,
    bestLoss.length > 0 ? `최저 val loss ${Math.min(...bestLoss).toFixed(4)}` : null,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');

  return (
    <div style={{ padding: '36px 40px 60px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: 28,
          marginBottom: 16,
        }}
      >
        <h1 style={{ ...type.pageTitle, margin: 0, color: color.textStrong, minWidth: 0 }}>기록</h1>
        <Button kind="primary" onClick={onNewExperiment} style={{ flex: 'none' }}>
          새 실험
        </Button>
      </div>

      {/* 어떤 dataset의 기록을 보는 중인지. 고르는 것은 보는 대상뿐이고 학습 입력은
          바뀌지 않습니다 — 그 말을 옆에 적어 둡니다. */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px 14px',
          flexWrap: 'wrap',
          marginBottom: 12,
        }}
      >
        {datasets.length === 0 ? (
          <span style={{ ...type.body, color: color.textFaint }}>
            아직 dataset이 없습니다. 왼쪽 <b style={{ color: color.textMuted }}>dataset 준비</b>로
            전처리를 먼저 돌리세요.
          </span>
        ) : (
          <>
            <label
              htmlFor="records-dataset"
              style={{ ...type.fieldLabel, color: color.textMuted, flex: 'none' }}
            >
              DATASET
            </label>
            <select
              id="records-dataset"
              value={datasetKey ?? ''}
              onChange={(event) => onPickDataset(event.target.value)}
              style={{ ...controlStyle, width: 'auto', maxWidth: '100%' }}
            >
              {datasets.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.key} — {item.sub}
                </option>
              ))}
            </select>
            <span style={{ ...type.note, color: color.textFaint }}>
              보는 대상만 바꿉니다. 학습에 쓰는 데이터는 dataset 준비에서 고릅니다.
            </span>
          </>
        )}
      </div>

      <div style={{ ...type.body, color: color.textBody, marginBottom: 16 }}>{stats}</div>

      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 10,
          paddingBottom: 22,
          borderBottom: `1px solid ${color.border}`,
          maxWidth: '62em',
        }}
      >
        {/* registry를 아직 못 읽었으면 "이 컴퓨터"라고 단정하지 않습니다. index 전체를
            훑는 응답이라 시간이 걸리는데, 그동안 팀 기록이 없다고 말해 버리면
            사실이 아닌 것을 화면이 먼저 주장하는 셈입니다. */}
        <Badge tone={scope ? 'accent' : 'muted'}>
          {scope ? (scope.shared ? '팀 공유' : '이 컴퓨터') : '읽는 중'}
        </Badge>
        <span style={{ ...type.body, color: color.textBody, textWrap: 'pretty' }}>
          {!scope
            ? '등록된 실험 목록을 읽고 있습니다. 아래는 이 컴퓨터가 시작한 학습이고, 다 읽으면 팀 기록과 Kaggle 점수가 합쳐집니다.'
            : scope.shared
              ? '팀이 같은 S3 저장소를 쓰고 있어 팀원이 등록한 실험도 함께 나옵니다. 순위를 말할 수 있는 숫자는 Kaggle 점수뿐입니다 — 로컬 mAP는 참고용입니다.'
              : `지금 backend가 ${scope.backend}이라 이 컴퓨터에 등록된 실험만 보입니다. 팀원 것까지 보려면 PILL_STORAGE_S3_BUCKET을 설정한 뒤 서버를 다시 시작하세요.`}
        </span>
      </div>

      {error && (
        <div style={{ marginTop: 22 }}>
          <AlertRow level="error" title="backend에 연결하지 못했습니다">
            {error} 서버를 실행하려면 저장소 root에서{' '}
            <code style={{ fontFamily: font.mono }}>python -m src.pipelines.web.server</code>를
            실행하세요.
          </AlertRow>
        </div>
      )}

      {/* 조용히 빼면 그만큼이 없는 줄 압니다. 몇 건을 왜 뺐는지 늘 말합니다. */}
      {unnamedCount > 0 && (
        <div style={{ ...type.note, color: color.textFaint, marginTop: 22 }}>
          어떤 dataset으로 돌렸는지 알 수 없는 기록 {unnamedCount}건은 위 목록에 세우지
          않았습니다. data artifact 위치가{' '}
          <code style={{ fontFamily: font.mono }}>…/&lt;dataset&gt;/train_manifest.json</code> 모양이
          아닌 옛 실행입니다.
        </div>
      )}

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '20px 24px',
          flexWrap: 'wrap',
          margin: '28px 0 4px',
        }}
      >
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {FILTERS.map((key) => (
            <Chip
              key={key}
              active={filter === key}
              count={records.filter((record) => matchesFilter(record, key)).length}
              onClick={() => setFilter(key)}
            >
              {FILTER_LABEL[key]}
            </Chip>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
          <span
            style={{ font: `400 11px/1.4 ${font.mono}`, letterSpacing: '0.08em', color: color.textFaint }}
          >
            SORT
          </span>
          {SORTS.map((key) => (
            <SortToggle key={key} active={sort === key} onClick={() => setSort(key)}>
              {SORT_LABEL[key]}
            </SortToggle>
          ))}
          <LinkAction onClick={() => navigate('/canvas')}>캔버스에서 견주기 →</LinkAction>
        </div>
      </div>

      {shown.length === 0 ? (
        <EmptyState
          message={
            records.length === 0
              ? '이 dataset에는 아직 기록이 없습니다. 오른쪽 위 새 실험으로 첫 학습을 걸어 보세요.'
              : '고른 조건에 맞는 기록이 없습니다. 위 표를 바꾸면 나머지가 보입니다.'
          }
          action={
            records.length === 0 ? (
              <Button kind="primary" onClick={onNewExperiment}>
                새 실험
              </Button>
            ) : undefined
          }
        />
      ) : (
        <>
          {groups.map((group) => (
            <ModelSection key={group.model} group={group} onOpen={openRecord} />
          ))}
          {folded.length > 0 && <FoldedRecords records={folded} onOpen={openRecord} />}
        </>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- 모델 묶음 */

/**
 * 모델 하나의 기록을 접었다 펼치는 구역입니다.
 *
 * 머리글에 건수와 가장 좋은 값을 적습니다. 모델을 견주러 온 사람이 줄을 세어 보지
 * 않고도 답을 얻는 자리라, 여기가 비면 묶은 뜻이 없습니다. 처음에는 펼쳐 둡니다 —
 * 접힌 채로 시작하면 기록을 보러 온 사람이 매번 같은 클릭을 먼저 해야 합니다.
 */
function ModelSection({
  group,
  onOpen,
}: {
  group: ReturnType<typeof groupByModel>[number];
  onOpen: (record: RunRecord) => void;
}) {
  const [open, setOpen] = useState(true);
  const summary = [
    `${group.records.length}건`,
    group.bestKaggle === null ? null : `최고 Kaggle ${group.bestKaggle.toFixed(4)}`,
    group.bestValidationLoss === null ? null : `최저 val ${group.bestValidationLoss.toFixed(4)}`,
  ]
    .filter((part): part is string => part !== null)
    .join(' · ');

  return (
    <div style={{ marginTop: 26 }}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 12,
          width: '100%',
          padding: '0 0 12px',
          background: 'transparent',
          border: 0,
          borderBottom: `1px solid ${color.border}`,
          textAlign: 'left',
          flexWrap: 'wrap',
        }}
      >
        <span style={{ ...type.sectionTitle, color: color.text }}>
          {open ? '▾' : '▸'} {group.model}
        </span>
        <span style={{ ...type.monoSpec, color: color.textMuted }}>{summary}</span>
      </button>
      {open &&
        group.records.map((record) => (
          <RecordRow key={record.runId} record={record} onOpen={() => onOpen(record)} />
        ))}
    </div>
  );
}

/* ------------------------------------------------- 결과 없이 끝난 기록 구역 */

/**
 * 결과 없이 끝난 기록을 접어 두는 구역입니다.
 *
 * 지우지 않고 접습니다. 왜 실패했는지는 로그를 봐야 알 수 있고, 그 로그로 가는
 * 길이 이 줄뿐입니다. 대신 몇 건을 무슨 이유로 접었는지 머리글이 늘 말합니다 —
 * 조용히 빼면 그만큼이 없는 줄 압니다. 모델별로 나누지 않는 것은, 여기 오는 기록
 * 대부분이 설정 하나 잘못 넣어 시작조차 못 한 것이라 모델이 답이 아니기 때문입니다.
 */
function FoldedRecords({
  records,
  onOpen,
}: {
  records: RunRecord[];
  onOpen: (record: RunRecord) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ marginTop: 26, borderTop: `1px solid ${color.border}` }}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 10,
          width: '100%',
          padding: '18px 0',
          background: 'transparent',
          border: 0,
          textAlign: 'left',
        }}
      >
        <span style={{ ...type.body, color: color.textMuted }}>
          {open ? '▾' : '▸'} 결과 없이 끝남
        </span>
        <span style={{ font: `400 13px/1.4 ${font.mono}`, color: color.textFaint }}>
          {countLabel(records)}
        </span>
      </button>
      {open &&
        records.map((record) => (
          <RecordRow
            key={record.runId}
            record={record}
            showModel
            onOpen={() => onOpen(record)}
          />
        ))}
    </div>
  );
}

/* -------------------------------------------------------------- 기록 한 줄 */

/** 기록 한 줄. 식별자 → 설정 → 지표 순으로 내려갑니다. */
function RecordRow({
  record,
  showModel,
  onOpen,
}: {
  record: RunRecord;
  /** 모델 묶음 밖에 세울 때만 켭니다. 묶음 안에서는 머리글이 이미 말했습니다. */
  showModel?: boolean;
  onOpen: () => void;
}) {
  const running = isRunning(record);
  return (
    <div
      role="button"
      tabIndex={0}
      data-row-hover=""
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') onOpen();
      }}
      style={{ padding: '20px 0', borderTop: `1px solid ${color.border}`, cursor: 'pointer' }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: 18,
          marginBottom: 5,
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 9, minWidth: 0 }}>
          <span style={{ ...type.listName, color: color.text, minWidth: 0 }}>{record.runId}</span>
          {record.submitted && <Badge>제출</Badge>}
          {/* 실패·취소한 줄에 미등록까지 붙이면 배지 둘이 같은 말을 합니다. 등록될
              수 있었는데 아직 안 된 것, 곧 성공으로 끝난 학습에만 붙입니다. */}
          {record.status === 'succeeded' && !record.registered && <Badge tone="muted">미등록</Badge>}
          {/* 끝난 이유는 반드시 적습니다. 미등록을 성공에만 붙이기로 하면서 취소·중단
              줄에는 아무 표시도 남지 않았습니다. 특히 중단은 이어서 학습할 대상이라
              성공한 기록과 눈으로 구별되어야 합니다. */}
          {record.status === 'failed' && <Badge tone="danger">{record.statusLabel}</Badge>}
          {(record.status === 'cancelled' || record.status === 'interrupted') && (
            <Badge tone="muted">{record.statusLabel}</Badge>
          )}
          {running && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 7, flex: 'none' }}>
              <LiveDot size={6} pulse />
              <span style={{ font: `500 12px/1.4 ${font.mono}`, color: color.accent }}>
                {record.statusLabel}
              </span>
            </span>
          )}
        </span>
        <span style={{ ...type.monoSpec, color: color.textMuted, flex: 'none', whiteSpace: 'nowrap' }}>
          {startedAt(record.at)}
        </span>
      </div>
      <div style={{ ...type.monoSpec, color: color.textMuted, marginBottom: 16 }}>
        {[showModel ? record.family : null, record.spec === '' ? null : record.spec]
          .filter((part): part is string => part !== null)
          .join(' · ')}
      </div>
      <MetricGrid>
        <Metric
          label="KAGGLE"
          value={score(record.metrics.kaggle)}
          strong
          tone={record.metrics.kaggle === null ? 'muted' : 'accent'}
        />
        <Metric label="BEST VAL LOSS" value={loss(record.metrics.bestValidationLoss)} strong />
        <Metric label="mAP" value={score(record.metrics.map)} tone="muted" />
        <Metric label="mAP50" value={score(record.metrics.map50)} tone="muted" />
        <Metric label="mAP75" value={score(record.metrics.map75)} tone="muted" />
        <Metric label="PRECISION50" value={score(record.metrics.precision50)} tone="muted" />
        <Metric label="RECALL50" value={score(record.metrics.recall50)} tone="muted" />
        <Metric label="BEST EPOCH" value={record.metrics.bestEpoch?.toString() ?? '-'} tone="muted" />
        <Metric label="경과" value={duration(record.metrics.elapsedSeconds)} tone="muted" />
      </MetricGrid>
    </div>
  );
}
