/**
 * 제출 하나를 여러 실행에서 만들어 내는 화면입니다. 두 갈래가 있습니다.
 *
 * **모델 앙상블**은 실행 여럿의 test 예측을 합쳐 새 상자를 만듭니다.
 * **임베딩 앙상블**은 실행 하나의 상자를 그대로 두고 crop embedding으로 점수만 다시
 * 매깁니다. 예전에는 뒤쪽이 앞쪽에 딸린 선택지여서, 융합하지 않고 재순위만 한 제출은
 * 다시 만들 길이 아예 없었습니다 — 그렇게 만든 제출이 실제로 있는데도 그랬습니다.
 *
 * 앙상블의 이득은 **합쳐 보기 전에는 모릅니다.** 확인하는 방법이 Kaggle 제출뿐이라
 * 잘못 고르면 하루치 제출이 사라집니다. 그래서 이 화면의 절반은 고르는 자리가 아니라
 * **합치기 전에 알 수 있는 것을 보여 주는 자리**입니다.
 *
 * 경고가 떠도 실행 버튼은 살아 있습니다. 예측이 틀릴 때가 있고, 막아 버리면 반증할
 * 길까지 막히기 때문입니다.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api, ApiError } from '../api/client';
import type { EnsembleCandidate, EnsembleDiagnosis, EnsembleJob } from '../api/types';
import { EmbeddingPanel } from '../components/EmbeddingPanel';
import {
  AlertRow,
  Badge,
  Button,
  Chip,
  EmptyState,
  MicroLabel,
  Panel,
  controlStyle,
} from '../components/primitives';
import { color, font, type } from '../design/tokens';

/** 화면이 스스로 이름을 만들 때 쓰는 앞머리입니다. 목록에서 결과를 알아보려는 것입니다. */
const NAME_PREFIX = { model: 'fusion', embedding: 'rerank' } as const;

type Mode = keyof typeof NAME_PREFIX;

function scoreText(value: number | null): string {
  return value === null ? '점수 없음' : value.toFixed(5);
}

export function Ensemble() {
  const [mode, setMode] = useState<Mode>('model');
  const [candidates, setCandidates] = useState<EnsembleCandidate[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [diagnosis, setDiagnosis] = useState<EnsembleDiagnosis | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [job, setJob] = useState<EnsembleJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [allowCopied, setAllowCopied] = useState(false);
  // 합친 뒤(또는 그대로) 점수를 다시 매기는 데 쓸 embedding입니다.
  const [embeddings, setEmbeddings] = useState<string[]>([]);
  // 응답이 늦게 도착해 지난 조합의 진단을 덮어쓰는 것을 막습니다.
  const requestId = useRef(0);

  // 화면을 닫은 뒤 응답이 도착하면 사라진 화면의 상태를 건드립니다. 목록을 읽는 데
  // 몇 초가 걸릴 수 있어서 실제로 일어납니다.
  useEffect(() => {
    let alive = true;
    api
      .ensembleCandidates()
      .then((result) => {
        if (alive) setCandidates(result.candidates);
      })
      .catch((cause) => {
        if (alive) setError(cause instanceof Error ? cause.message : '후보를 읽지 못했습니다.');
      });
    api
      .ensembleStatus()
      .then((state) => {
        if (alive) setJob(state);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  // 고를 때마다 자동으로 잽니다. 재 본 쌍은 서버가 저장해 두므로 두 번째부터 빠릅니다.
  // 임베딩 앙상블은 합치지 않으므로 잴 쌍이 없습니다.
  useEffect(() => {
    if (selected.length < 2) {
      setDiagnosis(null);
      return;
    }
    const ticket = ++requestId.current;
    setDiagnosing(true);
    api
      .diagnoseEnsemble(selected)
      .then((result) => {
        if (ticket === requestId.current) setDiagnosis(result);
      })
      .catch((cause) => {
        if (ticket === requestId.current) {
          setError(cause instanceof ApiError ? cause.message : '진단하지 못했습니다.');
        }
      })
      .finally(() => {
        if (ticket === requestId.current) setDiagnosing(false);
      });
  }, [selected]);

  const toggle = useCallback(
    (runId: string) => {
      // 임베딩 앙상블의 기준은 **하나**입니다. 고르면 앞의 것을 대신합니다.
      setSelected((current) => {
        if (mode === 'embedding') return [runId];
        return current.includes(runId)
          ? current.filter((item) => item !== runId)
          : [...current, runId];
      });
      // **고른 것이 바뀌면 지난 동의는 무효입니다.** A와 B가 같은 사진이라고 확인했다고
      // 해서 C까지 확인한 것이 아닙니다. 남겨 두면 새 진단이 오기 전에 실행할 때 C까지
      // 위치 검사를 면제해, 같은 id·크기의 다른 사진을 조용히 합칩니다.
      setAllowCopied(false);
    },
    [mode],
  );

  const pickMode = useCallback((next: Mode) => {
    setMode(next);
    // 갈래를 바꾸면 고른 것도 그 갈래의 규칙을 따릅니다. 셋을 고른 채로 임베딩
    // 앙상블에 들어가면 버튼이 이유 없이 죽은 것처럼 보입니다.
    setSelected((current) => (next === 'embedding' ? current.slice(0, 1) : current));
    setAllowCopied(false);
  }, []);

  const toggleEmbedding = useCallback((runId: string) => {
    setEmbeddings((current) =>
      current.includes(runId) ? current.filter((item) => item !== runId) : [...current, runId],
    );
  }, []);

  const warnings = useMemo(
    () => (diagnosis?.checks ?? []).filter((check) => check.level === 'warn'),
    [diagnosis],
  );
  // 시험지가 다르다는 경고가 있을 때만 물어봅니다. 늘 보이면 뜻 없이 켜집니다.
  const needsCopyConsent = mode === 'model' && warnings.some((check) => check.id === 'test_set');
  const running = job?.status === 'running';
  // 진단이 지금 고른 것에 대한 답인지입니다. 아직 안 왔거나 옛 조합의 답이면,
  // 그 위에서 실행을 결정하면 안 됩니다.
  const fresh =
    diagnosis !== null &&
    !diagnosing &&
    diagnosis.run_ids.length === selected.length &&
    diagnosis.run_ids.every((item, index) => item === selected[index]);

  const blocked =
    running ||
    (mode === 'model'
      ? selected.length < 2 || !fresh
      : selected.length !== 1 || embeddings.length === 0);

  const start = useCallback(() => {
    const runId = name.trim() || `${NAME_PREFIX[mode]}-${selected.length}-${Date.now()}`;
    setError(null);
    api
      .startEnsemble({
        run_ids: selected,
        run_id: runId,
        mode,
        allow_copied_images: needsCopyConsent && allowCopied,
        embedding_run_ids: embeddings,
      })
      .then(setJob)
      .catch((cause) => setError(cause instanceof Error ? cause.message : '시작하지 못했습니다.'));
  }, [allowCopied, embeddings, mode, name, needsCopyConsent, selected]);

  // 도는 동안만 상태를 물어봅니다. 융합은 몇 분이라 짧게 잡아도 됩니다.
  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => {
      api.ensembleStatus().then(setJob).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [running]);

  const pairs = diagnosis?.diversity?.pairs ?? [];
  const picker = (
    <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
      {candidates.map((item) => (
        <li key={item.run_id} style={{ borderTop: `1px solid ${color.borderRow}` }}>
          <label
            style={{
              display: 'flex',
              gap: 10,
              alignItems: 'baseline',
              cursor: 'pointer',
              padding: '10px 2px',
            }}
          >
            <input
              type={mode === 'model' ? 'checkbox' : 'radio'}
              name={mode === 'model' ? undefined : 'ensemble-base'}
              checked={selected.includes(item.run_id)}
              onChange={() => toggle(item.run_id)}
              disabled={running}
            />
            <span style={{ display: 'grid', gap: 2, flex: 1, minWidth: 0 }}>
              <span style={{ ...type.body, color: color.text }}>{item.run_id}</span>
              {item.dataset_label ? (
                <span style={{ ...type.note, color: color.textFaint }}>{item.dataset_label}</span>
              ) : null}
            </span>
            {/* 예측이 없는 실행도 고를 수 있습니다. 다만 GPU를 쓰게 되므로
                누르기 전에 보이는 자리에 표시합니다. 임베딩 앙상블은 어느 쪽이든
                다시 추론하므로 이 표시가 뜻이 없습니다. */}
            {mode === 'model' && !item.ready ? (
              <span style={{ ...type.note, color: color.warn }}>추론 필요</span>
            ) : null}
            <span style={{ ...type.monoId, color: color.textMid }}>
              {scoreText(item.kaggle_score)}
            </span>
          </label>
        </li>
      ))}
    </ul>
  );

  const embeddingPicker = (
    <EmbeddingPanel
      selected={embeddings}
      onToggle={toggleEmbedding}
      disabled={running}
      hint={
        mode === 'model'
          ? '고르면 합친 뒤 그 embedding으로 점수만 다시 매깁니다. 고르지 않으면 지금까지와 똑같은 제출이 나갑니다.'
          : '고른 embedding들의 margin을 평균 내 제출 행의 점수를 다시 매깁니다. 하나 이상 골라야 합니다.'
      }
    />
  );

  return (
    <div style={{ padding: '36px 40px 60px', display: 'grid', gap: 22 }}>
      <header>
        <h1 style={{ ...type.pageTitle, margin: 0, color: color.textStrong }}>앙상블</h1>
        <p
          style={{
            ...type.body,
            color: color.textBody,
            margin: '10px 0 0',
            maxWidth: '62em',
            textWrap: 'pretty',
          }}
        >
          끝난 실행들로 제출 하나를 만듭니다. 이득은 <b>Kaggle에 내 보기 전에는 알 수 없으므로</b>,
          고른 조합이 값어치가 있는지 먼저 재서 보여 줍니다.
        </p>
      </header>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <Chip active={mode === 'model'} onClick={() => pickMode('model')}>
          모델 앙상블
        </Chip>
        <Chip active={mode === 'embedding'} onClick={() => pickMode('embedding')}>
          임베딩 앙상블
        </Chip>
        <span style={{ ...type.note, color: color.textMuted, flex: '1 1 20em' }}>
          {mode === 'model'
            ? '실행 여럿의 예측을 합쳐 새 상자를 만듭니다. embedding 재순위를 함께 걸 수도 있습니다.'
            : '실행 하나의 상자는 그대로 두고 점수만 다시 매깁니다. test 추론을 다시 돌리므로 GPU를 씁니다.'}
        </span>
      </div>

      {error ? (
        <AlertRow level="error" title="앙상블">
          {error}
        </AlertRow>
      ) : null}

      <div
        style={{
          display: 'grid',
          gap: 16,
          gridTemplateColumns: 'minmax(280px, 1fr) minmax(320px, 1.1fr)',
          alignItems: 'start',
        }}
      >
        <Panel
          title={mode === 'model' ? '후보' : '기준 실행'}
          right={
            <span style={{ ...type.monoId, color: color.textMuted }}>
              {candidates.length}개 중 {selected.length}개
            </span>
          }
          bodyStyle={{ padding: '4px 20px 14px' }}
        >
          {candidates.length === 0 ? (
            <EmptyState message="합칠 수 있는 예측이 없습니다. 체크포인트를 남긴 실행만 후보가 됩니다." />
          ) : (
            picker
          )}
        </Panel>

        {mode === 'model' ? (
          <Panel title="진단" right={<Badge tone={warnings.length > 0 ? 'danger' : 'muted'}>{`경고 ${warnings.length}`}</Badge>}>
            {selected.length < 2 ? (
              <p style={{ ...type.body, color: color.textMuted, margin: 0 }}>
                둘 이상 고르면 잽니다.
              </p>
            ) : null}
            {diagnosing ? (
              <p style={{ ...type.body, color: color.textMuted, margin: 0 }}>예측을 읽는 중…</p>
            ) : null}
            <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 12 }}>
              {(diagnosis?.checks ?? []).map((check) => (
                <li key={check.id}>
                  <span style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                    <Badge tone={check.level === 'warn' ? 'danger' : 'muted'}>
                      {check.level === 'warn' ? '경고' : '통과'}
                    </Badge>
                    <span style={{ ...type.body, color: color.text }}>{check.title}</span>
                  </span>
                  <p style={{ ...type.note, color: color.textMid, margin: '4px 0 0 4px' }}>
                    {check.detail}
                  </p>
                </li>
              ))}
            </ul>

            {/* 평균만 보면 어느 짝이 붙어 있는지 알 수 없어, 무엇을 뺄지 못 정합니다. */}
            {pairs.length > 1 ? (
              <div style={{ marginTop: 16, display: 'grid', gap: 5 }}>
                <MicroLabel>쌍마다</MicroLabel>
                {pairs.map((pair) => (
                  <div
                    key={pair.runs.join('|')}
                    style={{ display: 'flex', gap: 12, alignItems: 'baseline' }}
                  >
                    <span
                      style={{
                        font: `400 11.5px/1.5 ${font.mono}`,
                        color: color.textMuted,
                        flex: 1,
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {pair.runs[0]} · {pair.runs[1]}
                    </span>
                    <span style={{ ...type.note, color: color.textMid }}>
                      일치 {(pair.agreement * 100).toFixed(1)}%
                    </span>
                    <span style={{ ...type.note, color: color.textFaint }}>
                      IoU {pair.box_iou.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}

            {diagnosis?.expected?.floor !== undefined && diagnosis.expected.ceiling !== undefined ? (
              <p style={{ ...type.note, color: color.textMid, margin: '16px 0 0' }}>
                예상 구간 {diagnosis.expected.floor.toFixed(5)} ~{' '}
                {diagnosis.expected.ceiling.toFixed(5)}. 지난 융합들은 이 구간의{' '}
                {((diagnosis.expected.observed_ratio ?? 0.82) * 100).toFixed(0)}% 지점에 떨어졌고,{' '}
                <strong>최고 실행을 넘지 못했습니다.</strong>
              </p>
            ) : null}
          </Panel>
        ) : (
          <Panel
            title="재순위 embedding"
            right={
              <span style={{ ...type.monoId, color: color.textMuted }}>
                {embeddings.length}개 선택
              </span>
            }
          >
            {embeddingPicker}
          </Panel>
        )}
      </div>

      {mode === 'model' ? (
        <Panel
          title="재순위 embedding"
          right={
            <span style={{ ...type.monoId, color: color.textMuted }}>
              {embeddings.length}개 선택 · 선택 사항
            </span>
          }
        >
          {embeddingPicker}
        </Panel>
      ) : null}

      <Panel title="실행">
        <div style={{ display: 'grid', gap: 12, maxWidth: '46em' }}>
          <input
            style={controlStyle}
            placeholder="결과 이름 (비우면 자동)"
            value={name}
            onChange={(event) => setName(event.target.value)}
            disabled={running}
            aria-label="결과 이름"
          />
          {needsCopyConsent ? (
            <label style={{ ...type.note, color: color.textMid, display: 'flex', gap: 8 }}>
              <input
                type="checkbox"
                checked={allowCopied}
                onChange={(event) => setAllowCopied(event.target.checked)}
              />
              사진이 같은데 위치만 다른 것을 확인했습니다 (fusion_allow_copied_images)
            </label>
          ) : null}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <Button kind="primary" onClick={start} disabled={blocked}>
              {running
                ? '도는 중…'
                : mode === 'embedding'
                  ? `embedding ${embeddings.length}개로 재순위`
                  : embeddings.length > 0
                    ? `${selected.length}개 합치고 embedding ${embeddings.length}개로 재순위`
                    : `${selected.length}개 합치기`}
            </Button>
            {warnings.length > 0 ? (
              <span style={{ ...type.note, color: color.textMuted }}>
                경고 {warnings.length}개 — 막지는 않습니다.
              </span>
            ) : null}
          </div>
        </div>
      </Panel>

      {job && job.status !== 'idle' ? (
        <AlertRow
          level={job.status === 'failed' ? 'error' : job.status === 'succeeded' ? 'success' : 'info'}
          title={String(job.run_id ?? '앙상블')}
        >
          {job.status !== 'running'
            ? job.message ?? job.status
            : job.stage === 'harvest'
              ? `test 예측을 만드는 중 ${job.harvesting ?? ''} (${job.harvest_progress?.[0] ?? 0}/${job.harvest_progress?.[1] ?? 0})`
              : job.stage === 'rerank'
                ? 'test 추론과 재순위를 도는 중…'
                : '합치는 중…'}
        </AlertRow>
      ) : null}
    </div>
  );
}

export default Ensemble;
