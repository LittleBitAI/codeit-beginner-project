import { useState } from 'react';

import { api, ApiError } from '../api/client';
import type { DataSource } from '../api/types';
import { DATA_KEYS } from '../lib/dataKeys';
import { color, font, radius } from '../design/tokens';
import { IconCheck, IconError, IconWarning } from './Icon';
import { AlertRow, Button, Panel, controlStyle, invalidControlStyle } from './primitives';

/**
 * 전처리 결과 폴더를 한 번 고르면 artifact 4개를 찾아 기억합니다.
 *
 * 실험마다 경로 4개를 손으로 넣지 않도록, 데이터셋을 실험 단위가 아니라 프로젝트
 * 단위 설정으로 둡니다. 고르고 나면 새 실험 화면의 4칸이 자동으로 채워집니다.
 */
export function DataSourcePanel({
  source,
  onSelected,
}: {
  source: DataSource | null;
  onSelected: (source: DataSource) => void;
}) {
  const [directory, setDirectory] = useState('');
  const [preview, setPreview] = useState<DataSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState(false);

  async function run(action: 'inspect' | 'select') {
    const target = directory.trim();
    if (!target) {
      setError('전처리 결과가 들어 있는 폴더 경로를 넣어 주세요.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      if (action === 'inspect') {
        setPreview(await api.inspectDirectory(target));
      } else {
        const result = await api.setDataSource(target);
        setPreview(null);
        setDirectory('');
        setEditing(false);
        onSelected(result.source);
      }
    } catch (caught) {
      setPreview(null);
      setError(caught instanceof ApiError ? caught.message : '폴더를 확인하지 못했습니다.');
    } finally {
      setBusy(false);
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
        <span style={{ font: `400 12px/1.7 ${font.sans}`, color: color.textBody }}>
          data pipeline이 만든 결과가 들어 있는 폴더를 한 번만 고르면, 그 안에서 학습에 필요한
          JSON 4개를 찾아 기억합니다. 새 실험을 만들 때 자동으로 채워집니다. 파일 이름이 달라도
          내용을 보고 찾습니다.
        </span>

        {source && !editing ? (
          <SelectedSummary source={source} />
        ) : (
          <>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'flex-start' }}>
              <input
                value={directory}
                placeholder="예: artifacts/data/v1"
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
            <span style={{ font: `400 10.5px/1.5 ${font.sans}`, color: color.textMuted }}>
              저장소 기준 상대 경로만 받습니다. 절대 경로와 ..는 거부됩니다.
            </span>
          </>
        )}

        {error && (
          <AlertRow level="error" title="폴더를 쓸 수 없습니다">
            {error}
          </AlertRow>
        )}

        {preview && <MatchTable source={preview} heading="찾은 파일" />}
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
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <IconCheck size={13} color={color.green} />
        <span
          style={{
            font: `500 12px/1.4 ${font.mono}`,
            color: color.text,
            overflowWrap: 'anywhere',
          }}
        >
          {source.directory}
        </span>
        <span
          style={{
            font: `600 10px/1.3 ${font.mono}`,
            color: color.tealDark,
            background: color.tealTint,
            borderRadius: radius.badge,
            padding: '4px 6px',
          }}
        >
          4개 인식됨
        </span>
      </div>
      <MatchTable source={source} heading={null} />
    </div>
  );
}

function MatchTable({ source, heading }: { source: DataSource; heading: string | null }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {heading && (
        <span style={{ font: `600 11.5px/1 ${font.sans}`, color: color.textStrong }}>{heading}</span>
      )}
      <div
        style={{
          border: `1px solid ${color.borderInner}`,
          borderRadius: 5,
          overflow: 'hidden',
        }}
      >
        {DATA_KEYS.map((key, index) => {
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
                background: entry ? color.surface : color.surfaceAlt,
                alignItems: 'center',
              }}
            >
              <span style={{ font: `500 11px/1.4 ${font.sans}`, color: color.textStrong }}>
                {source.labels[key] ?? key}
              </span>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  font: `400 11px/1.4 ${font.mono}`,
                  color: entry ? color.text : color.red,
                  overflowWrap: 'anywhere',
                }}
              >
                {entry ? (
                  <IconCheck size={11} color={color.green} />
                ) : (
                  <IconError size={11} color={color.red} />
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
            font: `400 11px/1.5 ${font.sans}`,
            color: color.amber,
          }}
        >
          <span style={{ marginTop: 2 }}>
            <IconWarning size={11} color={color.amber} />
          </span>
          {problem}
        </span>
      ))}
    </div>
  );
}
