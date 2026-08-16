/**
 * 여러 실행의 test 예측을 합치는 화면입니다.
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
import { AlertRow, Button, Chip, MicroLabel, controlStyle } from '../components/primitives';
import { color, type } from '../design/tokens';

/** 화면이 스스로 이름을 만들 때 쓰는 앞머리입니다. 목록에서 융합 결과를 알아보려는 것입니다. */
const NAME_PREFIX = 'fusion';

function scoreText(value: number | null): string {
  return value === null ? '점수 없음' : value.toFixed(5);
}

export function Ensemble() {
  const [candidates, setCandidates] = useState<EnsembleCandidate[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [diagnosis, setDiagnosis] = useState<EnsembleDiagnosis | null>(null);
  const [diagnosing, setDiagnosing] = useState(false);
  const [job, setJob] = useState<EnsembleJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState('');
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

  const toggle = useCallback((runId: string) => {
    setSelected((current) =>
      current.includes(runId) ? current.filter((item) => item !== runId) : [...current, runId],
    );
  }, []);

  const warnings = useMemo(
    () => (diagnosis?.checks ?? []).filter((check) => check.level === 'warn'),
    [diagnosis],
  );
  // 시험지가 다르다는 경고가 있을 때만 물어봅니다. 늘 보이면 뜻 없이 켜집니다.
  const needsCopyConsent = warnings.some((check) => check.id === 'test_set');
  const [allowCopied, setAllowCopied] = useState(false);
  const running = job?.status === 'running';

  const start = useCallback(() => {
    const runId = name.trim() || `${NAME_PREFIX}-${selected.length}-${Date.now()}`;
    setError(null);
    api
      .startEnsemble({
        run_ids: selected,
        run_id: runId,
        allow_copied_images: needsCopyConsent && allowCopied,
      })
      .then(setJob)
      .catch((cause) => setError(cause instanceof Error ? cause.message : '시작하지 못했습니다.'));
  }, [allowCopied, name, needsCopyConsent, selected]);

  // 도는 동안만 상태를 물어봅니다. 융합은 몇 분이라 짧게 잡아도 됩니다.
  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => {
      api.ensembleStatus().then(setJob).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [running]);

  return (
    <div style={{ display: 'grid', gap: 16, padding: 20 }}>
      <header>
        <h1 style={{ ...type.pageTitle, color: color.text, margin: 0 }}>앙상블</h1>
        <p style={{ ...type.body, color: color.textMid, margin: '4px 0 0' }}>
          여러 실행의 test 예측을 합쳐 제출 하나를 만듭니다. 이득은 합쳐 보기 전에는 알 수 없으므로,
          <strong> 고른 조합이 합칠 값어치가 있는지 먼저 재서 보여 줍니다.</strong>
        </p>
      </header>

      {error ? (
        <AlertRow level="error" title="앙상블">
          {error}
        </AlertRow>
      ) : null}

      <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'minmax(280px, 1fr) minmax(320px, 1.2fr)' }}>
        <section>
          <MicroLabel>후보 ({candidates.length})</MicroLabel>
          {candidates.length === 0 ? (
            <p style={{ ...type.body, color: color.textMid }}>
              합칠 수 있는 예측이 없습니다. evaluate가 <code>test_predictions.json</code>을 남긴
              실행만 후보가 됩니다.
            </p>
          ) : null}
          <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 6 }}>
            {candidates.map((item) => (
              <li key={item.run_id}>
                <label style={{ display: 'flex', gap: 10, alignItems: 'baseline', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={selected.includes(item.run_id)}
                    onChange={() => toggle(item.run_id)}
                    disabled={running}
                  />
                  <span style={{ ...type.body, color: color.text, flex: 1 }}>{item.run_id}</span>
                  {/* 예측이 없는 실행도 고를 수 있습니다. 다만 GPU를 쓰게 되므로
                      누르기 전에 보이는 자리에 표시합니다. */}
                  {item.ready ? null : (
                    <span style={{ ...type.note, color: color.warn }}>추론 필요</span>
                  )}
                  <span style={{ ...type.monoId, color: color.textMid }}>{scoreText(item.kaggle_score)}</span>
                </label>
              </li>
            ))}
          </ul>
        </section>

        <section>
          <MicroLabel>진단</MicroLabel>
          {selected.length < 2 ? (
            <p style={{ ...type.body, color: color.textMid }}>둘 이상 고르면 잽니다.</p>
          ) : null}
          {diagnosing ? <p style={{ ...type.body, color: color.textMid }}>예측을 읽는 중…</p> : null}
          <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 10 }}>
            {(diagnosis?.checks ?? []).map((check) => (
              <li key={check.id}>
                <Chip active={check.level === 'warn'}>
                  {check.level === 'warn' ? '경고' : '통과'}
                </Chip>
                <span style={{ ...type.body, color: color.text, marginLeft: 8 }}>{check.title}</span>
                <p style={{ ...type.note, color: color.textMid, margin: '2px 0 0' }}>{check.detail}</p>
              </li>
            ))}
          </ul>

          {diagnosis?.expected?.floor !== undefined ? (
            <p style={{ ...type.note, color: color.textMid, marginTop: 12 }}>
              예상 구간 {diagnosis.expected.floor?.toFixed(5)} ~ {diagnosis.expected.ceiling?.toFixed(5)}.
              지난 일곱 개 융합은 이 구간의 82% 지점에 떨어졌고, <strong>최고 실행을 넘지 못했습니다.</strong>
            </p>
          ) : null}
        </section>
      </div>

      <section style={{ display: 'grid', gap: 8 }}>
        <MicroLabel>실행</MicroLabel>
        <input
          style={controlStyle}
          placeholder="결과 이름 (비우면 자동)"
          value={name}
          onChange={(event) => setName(event.target.value)}
          disabled={running}
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
        <div>
          <Button onClick={start} disabled={selected.length < 2 || running}>
            {running ? '합치는 중…' : `${selected.length}개 합치기`}
          </Button>
          {warnings.length > 0 ? (
            <span style={{ ...type.note, color: color.textMid, marginLeft: 10 }}>
              경고 {warnings.length}개 — 막지는 않습니다.
            </span>
          ) : null}
        </div>
        {job && job.status !== 'idle' ? (
          <AlertRow
            level={job.status === 'failed' ? 'error' : job.status === 'succeeded' ? 'success' : 'info'}
            title={String(job.run_id ?? '융합')}
          >
            {job.status !== 'running'
              ? job.message ?? job.status
              : job.stage === 'harvest'
                ? `test 예측을 만드는 중 ${job.harvesting ?? ''} (${job.harvest_progress?.[0] ?? 0}/${job.harvest_progress?.[1] ?? 0})`
                : '합치는 중…'}
          </AlertRow>
        ) : null}
      </section>
    </div>
  );
}

export default Ensemble;
