import { useEffect, useState } from 'react';

import { api, ApiError } from '../api/client';
import type { PreparationProgress, PreparationState, StorageEnvironment } from '../api/types';
import { color, font, radius } from '../design/tokens';
import { formatDuration, useElapsedSeconds } from '../hooks/useElapsedSeconds';
import { usePolling } from '../hooks/usePolling';
import { AlertRow, Button, Field, controlStyle } from './primitives';

const RUNNING_INTERVAL_MS = 2000;
const IDLE_INTERVAL_MS = 10000;

/**
 * 원본에서 필수 artifact 4개와 선택 test manifest를 만들도록 data pipeline을 부릅니다.
 *
 * 원본을 다 읽어야 해서 오래 걸릴 수 있으므로 시작만 시키고 상태를 주기적으로
 * 확인합니다. 성공하면 그 결과가 곧바로 현재 전처리 데이터셋이 됩니다.
 */
export function PreparePanel({ onPrepared }: { onPrepared: () => void }) {
  const [ratio, setRatio] = useState('8:2');
  const [seed, setSeed] = useState('42');
  const [overwrite, setOverwrite] = useState(false);
  // 은행은 판마다 따로 만들어지고 나중에 덧붙일 수 없습니다. 임베딩 학습과 재순위를
  // 하려면 여기서 켜는 수밖에 없습니다.
  const [cropBank, setCropBank] = useState(false);
  const [rawPrefix, setRawPrefix] = useState('');
  const [backend, setBackend] = useState('auto');
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [lastFinished, setLastFinished] = useState<string | null>(null);

  const status = usePolling(() => api.prepareStatus(), RUNNING_INTERVAL_MS);
  const state: PreparationState | undefined = status.data?.preparation;
  const ratios = status.data?.split_ratios ?? ['8:2', '9:1'];
  const running = state?.status === 'running';

  // 준비가 끝나는 순간 한 번만 상위에 알려 선택 상태를 새로 읽게 합니다.
  useEffect(() => {
    if (!state || state.status === 'running' || state.status === 'idle') return;
    const marker = state.finished_at ?? null;
    if (marker && marker !== lastFinished) {
      setLastFinished(marker);
      if (state.status === 'succeeded') onPrepared();
    }
  }, [state, lastFinished, onPrepared]);

  async function start() {
    setStarting(true);
    setError(null);
    try {
      await api.startPreparation({
        split_ratio: ratio,
        seed: Number.parseInt(seed, 10),
        overwrite,
        backend,
        crop_bank: cropBank,
        // 비운 칸은 보내지 않습니다. 빈 문자열을 보내면 서버가 기본값 대신 그것을
        // 경로로 읽습니다.
        ...(rawPrefix.trim() ? { raw_prefix: rawPrefix.trim() } : {}),
      });
      status.refresh();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '준비를 시작하지 못했습니다.');
    } finally {
      setStarting(false);
    }
  }

  const seedInvalid = !/^\d+$/.test(seed.trim());

  return (
    <div
      style={{
        borderTop: `1px solid ${color.borderRow}`,
        paddingTop: 14,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.text }}>
          원본에서 직접 준비하기
        </span>
        <span style={{ font: `400 12.5px/1.6 ${font.sans}`, color: color.textBody }}>
          이미 만들어 둔 폴더를 고르는 대신, data pipeline이 원본을 읽어 학습용 JSON 4개와
          대회용 test manifest를 새로 만들게 합니다. 학습과 검증을 몇 대 몇으로 나눌지 여기서 정합니다.
        </span>
      </div>

      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, minWidth: 190 }}>
          <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.textStrong }}>
            학습 : 검증 비율
          </span>
          <div
            style={{
              display: 'flex',
              border: `1px solid ${color.border}`,
              borderRadius: radius.control,
              overflow: 'hidden',
            }}
          >
            {ratios.map((option, index) => {
              const active = option === ratio;
              return (
                <button
                  key={option}
                  type="button"
                  disabled={running}
                  onClick={() => setRatio(option)}
                  style={{
                    flex: 1,
                    padding: '8px 0',
                    font: `600 12.5px/1 ${font.mono}`,
                    // 강조색 위에 얹는 글자는 판마다 다릅니다. 흰색을 박아 두면
                    // 어두운 판에서 #FFF on #E0A96D가 되어 대비가 2.1:1입니다.
                    color: active ? color.onAccent : color.textBody,
                    background: active ? color.accent : color.panel,
                    border: 'none',
                    borderRight:
                      index === ratios.length - 1 ? undefined : `1px solid ${color.borderRow}`,
                    opacity: running ? 0.6 : 1,
                  }}
                >
                  {option}
                </button>
              );
            })}
          </div>
          <span style={{ font: `400 11.5px/1.45 ${font.sans}`, color: color.textMuted }}>
            8:2는 검증에 20%, 9:1은 10%를 씁니다.
          </span>
        </div>

        {/* 이 칸이 없으면 data pipeline의 기본 원본 하나만 준비할 수 있습니다. 판이
            여럿인 저장소에서 화면으로 고를 길이 아예 없었습니다. */}
        <div style={{ flex: 1, minWidth: 240 }}>
          <Field label="원본 경로" hint="비워 두면 data pipeline의 기본 원본을 씁니다">
            <input
              value={rawPrefix}
              disabled={running}
              placeholder="datasets/pill_detection/raw/<판>/"
              onChange={(event) => setRawPrefix(event.target.value)}
              style={controlStyle}
            />
          </Field>
        </div>

        <div style={{ width: 110 }}>
          <Field
            label="Seed"
            hint="같으면 같은 분할"
            error={seedInvalid ? '0 이상의 정수' : undefined}
          >
            <input
              value={seed}
              disabled={running}
              onChange={(event) => setSeed(event.target.value)}
              style={controlStyle}
            />
          </Field>
        </div>

        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            font: `400 12.5px/1 ${font.sans}`,
            color: color.textStrong,
            paddingBottom: 9,
          }}
        >
          <input
            type="checkbox"
            checked={overwrite}
            disabled={running}
            onChange={(event) => setOverwrite(event.target.checked)}
          />
          이미 있으면 덮어쓰기
        </label>

        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            font: `400 12.5px/1 ${font.sans}`,
            color: color.textStrong,
            paddingBottom: 9,
          }}
          title="학습 이미지에서 알약을 한 개씩 잘라 모읍니다. 임베딩 학습과 재순위가 이것을 읽습니다."
        >
          <input
            type="checkbox"
            checked={cropBank}
            disabled={running}
            onChange={(event) => setCropBank(event.target.checked)}
          />
          crop 은행 함께 만들기
        </label>

        <Button
          kind="primary"
          disabled={running || starting || seedInvalid}
          onClick={() => void start()}
          style={{ padding: '9px 14px', marginBottom: 9 }}
        >
          {running ? '준비 중…' : starting ? '시작하는 중…' : '데이터 준비 실행'}
        </Button>
      </div>

      <StorageChoice
        storage={status.data?.storage}
        backend={backend}
        onChange={setBackend}
        disabled={running}
      />

      {error && (
        <AlertRow level="error" title="시작하지 못했습니다">
          {error}
        </AlertRow>
      )}

      {state && state.status !== 'idle' && <PreparationResult state={state} />}
    </div>
  );
}

/** 어느 저장소의 원본을 읽을지 고릅니다. */
function StorageChoice({
  storage,
  backend,
  onChange,
  disabled,
}: {
  storage: StorageEnvironment | undefined;
  backend: string;
  onChange: (value: string) => void;
  disabled: boolean;
}) {
  if (!storage) return null;

  const options: { value: string; label: string; hint: string; usable: boolean }[] = [
    {
      value: 'auto',
      label: `자동 (${storage.default_backend})`,
      hint: '환경 설정을 보고 정합니다',
      usable: true,
    },
    {
      value: 's3',
      label: 'AWS S3',
      hint: storage.bucket_configured
        ? `bucket ${storage.bucket}${storage.region ? ` · ${storage.region}` : ''}`
        : 'PILL_STORAGE_S3_BUCKET 환경 변수가 없습니다',
      usable: storage.bucket_configured,
    },
    { value: 'local', label: '로컬 디스크', hint: 'artifacts/ 아래', usable: true },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.textStrong }}>
        원본을 읽을 곳
      </span>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {options.map((option) => {
          const active = option.value === backend;
          return (
            <button
              key={option.value}
              type="button"
              disabled={disabled || !option.usable}
              onClick={() => onChange(option.value)}
              style={{
                textAlign: 'left',
                padding: '8px 11px',
                borderRadius: radius.control,
                border: `1px solid ${active ? color.accent : color.border}`,
                background: active ? color.fill : color.panel,
                opacity: option.usable ? 1 : 0.5,
                display: 'flex',
                flexDirection: 'column',
                gap: 3,
                minWidth: 150,
              }}
            >
              <span
                style={{
                  font: `600 12.5px/1 ${font.sans}`,
                  color: active ? color.accent : color.textStrong,
                }}
              >
                {option.label}
              </span>
              <span
                style={{
                  font: `400 11px/1.35 ${font.mono}`,
                  color: color.textMuted,
                  overflowWrap: 'anywhere',
                }}
              >
                {option.hint}
              </span>
            </button>
          );
        })}
      </div>
      {storage.forced_backend && (
        <span style={{ font: `400 11.5px/1.5 ${font.sans}`, color: color.warn }}>
          환경 변수 PILL_STORAGE_BACKEND={storage.forced_backend} 가 설정돼 있어 여기 선택보다
          우선합니다.
        </span>
      )}
    </div>
  );
}

/** 지금 어느 단계인지, 읽기라면 얼마나 읽었는지 보여 줍니다. */
function PreparationProgressView({
  progress,
  fallback,
}: {
  progress: PreparationProgress | undefined;
  fallback: string;
}) {
  // 진행 줄이 한 번도 없으면 지금까지와 같은 고정 안내 문구만 둡니다.
  // 여기서 가짜 진행률을 그리면 없는 정보를 지어내는 것입니다.
  if (!progress?.available) {
    return (
      <span style={{ font: `400 12.5px/1.6 ${font.sans}`, color: color.textBody }}>
        {fallback} 원본이 크면 몇 분 걸릴 수 있습니다.
      </span>
    );
  }

  const read = progress.read ?? null;
  const percent = read?.percent ?? null;
  const eta = progress.eta_seconds;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', flexWrap: 'wrap' }}>
        {progress.stage_label && (
          <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.text }}>
            {progress.stage_label}
          </span>
        )}
        {read && read.done !== null && (
          <span style={{ font: `500 12.5px/1 ${font.mono}`, color: color.textBody }}>
            {read.total !== null && read.total !== undefined
              ? `${read.done} / ${read.total}`
              : `${read.done}개`}
          </span>
        )}
        {eta !== null && eta !== undefined && (
          <span style={{ font: `400 11.5px/1 ${font.sans}`, color: color.textMuted }}>
            남은 시간 약 {formatDuration(eta)}
          </span>
        )}
      </div>

      {percent !== null && percent !== undefined && (
        <div
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={progress.stage_label ?? '진행률'}
          style={{
            height: 6,
            borderRadius: radius.control,
            background: color.borderRow,
            overflow: 'hidden',
          }}
        >
          <div style={{ width: `${percent}%`, height: '100%', background: color.accent }} />
        </div>
      )}

      {progress.sources && (
        <span style={{ font: `400 11.5px/1.45 ${font.mono}`, color: color.textMuted }}>
          원본 · 학습 이미지 {progress.sources.train_images ?? '-'} · annotation{' '}
          {progress.sources.annotations ?? '-'} · test 이미지 {progress.sources.test_images ?? '-'}
        </span>
      )}
    </div>
  );
}

function PreparationResult({ state }: { state: PreparationState }) {
  const elapsed = useElapsedSeconds(state.started_at, state.status === 'running');

  if (state.status === 'running') {
    return (
      <AlertRow level="info" title={`데이터 준비 중 · ${state.split_ratio ?? ''}`}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {elapsed !== null && (
            <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
              <span style={{ font: `500 11px/1.3 ${font.mono}`, color: color.textMuted }}>
                경과 시간
              </span>
              <span style={{ font: `600 12.5px/1 ${font.mono}`, color: color.text }}>
                {formatDuration(elapsed)}
              </span>
            </div>
          )}
          <PreparationProgressView
            progress={state.progress}
            fallback={state.message ?? '원본을 읽고 있습니다.'}
          />
        </div>
      </AlertRow>
    );
  }

  if (state.status === 'failed') {
    return (
      <AlertRow
        level={state.supported === false ? 'warning' : 'error'}
        title={
          state.supported === false
            ? 'data pipeline이 아직 이 기능을 지원하지 않습니다'
            : '데이터 준비에 실패했습니다'
        }
      >
        {state.message}
        {state.exit_code !== null && state.exit_code !== undefined && state.exit_code !== 0 && (
          <> (exit code {state.exit_code})</>
        )}
      </AlertRow>
    );
  }

  const summary = state.summary ?? {};
  const numbers: [string, unknown][] = [
    ['학습 이미지', summary.train_images],
    ['검증 이미지', summary.validation_images],
    ['클래스', summary.category_count],
    ['제외된 이미지', summary.excluded_images],
    ['테스트 이미지', summary.test_manifest_images],
  ];
  const testImages = summary.test_manifest_images;
  const testImagesUsed = summary.test_images_used;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <AlertRow level="success" title={`데이터 준비 완료 · ${state.split_ratio ?? ''}`}>
        {state.message}
        {state.selected && <> 이 결과를 현재 전처리 데이터셋으로 골랐습니다.</>}
      </AlertRow>
      <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
        {numbers
          .filter(([, value]) => value !== undefined && value !== null)
          .map(([label, value]) => (
            <span key={label} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
              <span style={{ font: `500 11px/1.3 ${font.mono}`, color: color.textMuted }}>
                {label}
              </span>
              <span style={{ font: `600 12.5px/1 ${font.mono}`, color: color.text }}>
                {String(value)}
              </span>
            </span>
          ))}
      </div>
      {testImages !== undefined && testImages !== null && testImagesUsed === 0 && (
        <AlertRow level="success" title="test 분리 확인">
          test 이미지 {String(testImages)}장을 학습과 검증에 사용하지 않았습니다.
        </AlertRow>
      )}
    </div>
  );
}

/** 실행 중일 때는 더 자주 확인합니다. */
export const PREPARE_POLL_INTERVALS = {
  running: RUNNING_INTERVAL_MS,
  idle: IDLE_INTERVAL_MS,
};
