import { useState } from 'react';

import { api } from '../api/client';
import type { DataSource, DataVerification, ProcessedDataset } from '../api/types';
import { DATA_KEYS, OPTIONAL_DATA_KEYS } from '../lib/dataKeys';
import { describeError } from '../lib/describeError';
import { color, font, radius } from '../design/tokens';
import { IconCheck, IconError, IconWarning } from './Icon';
import { PreparePanel } from './PreparePanel';
import { usePolling } from '../hooks/usePolling';
import { AlertRow, Button, MicroLabel, Panel, controlStyle, invalidControlStyle } from './primitives';

/**
 * 전처리 결과 폴더를 한 번 고르면 필수 artifact와 선택 test manifest를 찾아 기억합니다.
 *
 * 실험마다 경로 4개를 손으로 넣지 않도록, 데이터셋을 실험 단위가 아니라 프로젝트
 * 단위 설정으로 둡니다. 고르고 나면 새 실험 화면의 4칸이 자동으로 채워집니다.
 */
/**
 * 고를 수 있는 전처리 폴더를 눌러서 고릅니다.
 *
 * 목록이 없을 때는 경로를 손으로 붙여넣는 수밖에 없어서, 어떤 판이 있는지 알려면
 * 저장소를 직접 뒤져야 했고 v3와 v5처럼 이름이 비슷한 판을 잘못 고르기 쉬웠습니다.
 * 아래 입력칸은 그대로 둡니다 — 목록에 없는 위치도 여전히 고를 수 있어야 합니다.
 */
function DatasetList({
  current,
  busy,
  onPick,
}: {
  current: string | null;
  busy: boolean;
  onPick: (dataset: ProcessedDataset) => void;
}) {
  const listing = usePolling(() => api.listDatasets(), 0);
  const datasets = listing.data?.datasets ?? [];
  if (!datasets.length) return null;

  const same = (directory: string) =>
    current !== null && current.replace(/\/$/, '') === directory.replace(/\/$/, '');

  return (
    <div>
      <MicroLabel style={{ marginBottom: 8 }}>
        {listing.data?.root ?? ''} 아래에서 찾은 폴더
      </MicroLabel>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {datasets.map((dataset) => (
          <button
            key={dataset.directory}
            type="button"
            disabled={busy || !dataset.complete}
            onClick={() => onPick(dataset)}
            title={
              dataset.complete
                ? dataset.directory
                : `필수 artifact가 없습니다: ${dataset.missing.join(', ')}`
            }
            style={{
              font: `400 12px/1.5 ${font.mono}`,
              padding: '6px 10px',
              cursor: dataset.complete ? 'pointer' : 'not-allowed',
              background: same(dataset.directory) ? color.accent : color.panel,
              color: same(dataset.directory)
                ? color.onAccent
                : dataset.complete
                  ? color.textStrong
                  : color.textFaint,
              border: `1px solid ${same(dataset.directory) ? color.accent : color.border}`,
              borderRadius: radius.control,
            }}
          >
            {dataset.name}
            {dataset.has_eda_report && ' · EDA'}
          </button>
        ))}
      </div>
    </div>
  );
}

export function DataSourcePanel({
  source,
  onSelected,
  onPrepared,
}: {
  source: DataSource | null;
  onSelected: (source: DataSource) => void;
  onPrepared: () => void;
}) {
  const [directory, setDirectory] = useState('');
  const [preview, setPreview] = useState<DataSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);
  const [verification, setVerification] = useState<DataVerification | null>(null);
  const [verifying, setVerifying] = useState(false);

  async function select(target: string) {
    setBusy(true);
    setError(null);
    setVerification(null);
    try {
      const result = await api.setDataSource(target);
      setPreview(null);
      setDirectory('');
      setEditing(false);
      onSelected(result.source);
    } catch (caught) {
      setPreview(null);
      setError(describeError(caught, '위치를 확인하지 못했습니다.'));
    } finally {
      setBusy(false);
    }
  }

  async function run(action: 'inspect' | 'select') {
    const target = directory.trim();
    if (!target) {
      setError('전처리 결과가 들어 있는 폴더 경로를 넣어 주세요.');
      return;
    }
    if (action === 'select') {
      await select(target);
      return;
    }
    setBusy(true);
    setError(null);
    setVerification(null);
    try {
      setPreview(await api.inspectDirectory(target));
    } catch (caught) {
      setPreview(null);
      setError(describeError(caught, '위치를 확인하지 못했습니다.'));
    } finally {
      setBusy(false);
    }
  }

  /** 실제 data pipeline을 불러 계약이 성립하는지 확인합니다. */
  async function verify(target: { data?: Record<string, string>; directory?: string }) {
    setVerifying(true);
    setError(null);
    try {
      const result = await api.verifyDataSource(target);
      setVerification(result.verification);
    } catch (caught) {
      setVerification(null);
      setError(describeError(caught, 'data pipeline을 부르지 못했습니다.'));
    } finally {
      setVerifying(false);
    }
  }

  return (
    <Panel
      title="전처리 데이터셋"
      right={
        source && !editing ? (
          <Button onClick={() => setEditing(true)}>다른 폴더 고르기</Button>
        ) : source ? (
          <Button
            onClick={() => {
              setEditing(false);
              setPreview(null);
              setError(null);
            }}
          >
            취소
          </Button>
        ) : null
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <span style={{ font: `400 12.5px/1.7 ${font.sans}`, color: color.textBody }}>
          data pipeline이 만든 결과가 있는 위치를 한 번만 고르면, 그 안에서 학습에 필요한
          JSON 4개와 선택 test manifest를 찾아 기억합니다. 새 실험을 만들 때 자동으로 채워집니다.
          로컬 폴더와 S3 위치를 모두 받고, 파일 이름이 달라도 내용을 보고 찾습니다.
        </span>

        {source && !editing ? (
          <>
            <SelectedSummary source={source} />
            {source.available !== false && source.complete && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <Button onClick={() => void verify({ data: source.data })} disabled={verifying}>
                  {verifying ? 'data pipeline 실행 중…' : 'data pipeline으로 검증'}
                </Button>
                <span style={{ font: `400 11.5px/1.5 ${font.sans}`, color: color.textMuted }}>
                  python -m src.main_pipeline --only data 를 실제로 실행해, 필수 4개와 선택 test
                  manifest가 다음 단계로 넘어갈 수 있는지 확인합니다.
                </span>
              </div>
            )}
          </>
        ) : (
          <>
            <DatasetList
              current={source?.directory ?? null}
              busy={busy}
              onPick={(dataset) => {
                setDirectory(dataset.directory);
                void select(dataset.directory);
              }}
            />
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'flex-start' }}>
              <input
                value={directory}
                placeholder="예: artifacts/data/v1 또는 s3://bucket/datasets/.../processed/v1/"
                onChange={(event) => setDirectory(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void run('inspect');
                }}
                style={{
                  ...(error ? invalidControlStyle : controlStyle),
                  flex: '1 1 320px',
                  width: 'auto',
                }}
              />
              <Button onClick={() => void run('inspect')} disabled={busy}>
                {busy ? '확인 중…' : '폴더 확인'}
              </Button>
              <Button
                kind="primary"
                onClick={() => void run('select')}
                disabled={busy || (preview !== null && !preview.complete)}
                title={
                  preview !== null && !preview.complete
                    ? '4개를 모두 찾은 폴더만 고를 수 있습니다.'
                    : undefined
                }
              >
                이 폴더 사용
              </Button>
            </div>
            <span style={{ font: `400 11.5px/1.5 ${font.sans}`, color: color.textMuted }}>
              저장소 기준 상대 경로 또는 <code style={{ fontFamily: font.mono }}>s3://bucket/prefix/</code>
              를 받습니다. 이미 S3에 준비해 둔 산출물이 있으면 그 위치를 그대로 넣으세요.
              절대 경로와 ..는 거부됩니다.
            </span>
          </>
        )}

        {error && (
          <AlertRow level="error" title="폴더를 쓸 수 없습니다">
            {error}
          </AlertRow>
        )}

        {preview && <MatchTable source={preview} heading="찾은 파일" />}

        <PreparePanel onPrepared={onPrepared} />

        {verification && (
          <AlertRow
            level={verification.ok ? 'success' : 'error'}
            title={
              verification.ok
                ? 'data pipeline 검증 통과'
                : 'data pipeline이 이 데이터를 받아들이지 않았습니다'
            }
          >
            {verification.message}
            {verification.ok && (
              <>
                {' '}
                필수 artifact 4개가 train으로 넘어갈 수 있습니다. data pipeline은 파일을 만들지 않고
                넘긴 위치를 검증해 그대로 넘겨줍니다.
              </>
            )}
            {verification.exit_code !== null && verification.exit_code !== 0 && (
              <> (exit code {verification.exit_code})</>
            )}
          </AlertRow>
        )}
      </div>
    </Panel>
  );
}

function SelectedSummary({ source }: { source: DataSource }) {
  if (source.available === false) {
    return (
      <AlertRow level="warning" title="고른 폴더를 지금은 읽을 수 없습니다">
        {source.directory} — {source.problems[0] ?? '폴더가 사라졌거나 접근할 수 없습니다.'}
      </AlertRow>
    );
  }
  const recognizedCount =
    DATA_KEYS.length + OPTIONAL_DATA_KEYS.filter((key) => Boolean(source.data[key])).length;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <IconCheck size={13} color={color.ok} />
        <span
          style={{
            font: `500 12.5px/1.4 ${font.mono}`,
            color: color.text,
            overflowWrap: 'anywhere',
          }}
        >
          {source.directory}
        </span>
        <span
          style={{
            font: `600 11px/1.3 ${font.mono}`,
            color: color.accent,
            background: color.fill,
            borderRadius: radius.badge,
            padding: '4px 6px',
          }}
        >
          {recognizedCount}개 인식됨
        </span>
      </div>
      <MatchTable source={source} heading={null} />
    </div>
  );
}

function MatchTable({ source, heading }: { source: DataSource; heading: string | null }) {
  const displayedKeys = [
    ...DATA_KEYS,
    ...OPTIONAL_DATA_KEYS.filter(
      (key) => Boolean(source.data[key]) || Boolean(source.matched[key]),
    ),
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {heading && (
        <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.textStrong }}>{heading}</span>
      )}
      <div
        style={{
          border: `1px solid ${color.borderRow}`,
          borderRadius: 5,
          overflow: 'hidden',
        }}
      >
        {displayedKeys.map((key, index) => {
          const entry = source.matched[key];
          return (
            <div
              key={key}
              style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(120px, 1fr) 2fr',
                gap: 10,
                padding: '7px 12px',
                borderTop: index === 0 ? undefined : `1px solid ${color.borderRow}`,
                background: entry ? color.panel : color.sheet,
                alignItems: 'center',
              }}
            >
              <span style={{ font: `500 12px/1.4 ${font.sans}`, color: color.textStrong }}>
                {source.labels[key] ?? key}
              </span>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  font: `400 12px/1.4 ${font.mono}`,
                  color: entry ? color.text : color.danger,
                  overflowWrap: 'anywhere',
                }}
              >
                {entry ? (
                  <IconCheck size={11} color={color.ok} />
                ) : (
                  <IconError size={11} color={color.danger} />
                )}
                {entry ? entry.name : '찾지 못했습니다'}
              </span>
            </div>
          );
        })}
      </div>
      {source.problems.map((problem) => (
        <span
          key={problem}
          style={{
            display: 'flex',
            gap: 6,
            alignItems: 'flex-start',
            font: `400 12px/1.5 ${font.sans}`,
            color: color.warn,
          }}
        >
          <span style={{ marginTop: 2 }}>
            <IconWarning size={11} color={color.warn} />
          </span>
          {problem}
        </span>
      ))}
    </div>
  );
}
