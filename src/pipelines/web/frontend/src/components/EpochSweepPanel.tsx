/**
 * 보관해 둔 epoch checkpoint를 훑어 **제일 잘 맞히는 epoch**을 고릅니다.
 *
 * 지금까지 best epoch은 validation loss가 정했습니다. 그 loss가 정말 상자를 잘
 * 맞히는 epoch을 고르는지는 아무도 재 보지 않았고, 로컬 지표와 Kaggle 순위가
 * 뒤집히는 것은 이미 관측됐습니다. 그래서 이 판은 재 보고 고릅니다.
 *
 * 후보마다 표본 평가 → 고른 지표로 순위 → 이긴 하나만 전수 평가 → 제출까지.
 */

import { useState } from 'react';

import { api } from '../api/client';
import type { EpochSweepCandidate, JobRecord } from '../api/types';
import { color, font, radius } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { describeError } from '../lib/describeError';
import { AlertRow, Button, Field, Panel, controlStyle } from './primitives';

const POLL_MS = 3000;

function metricText(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toFixed(4);
}

function Cell({ children, dim }: { children: React.ReactNode; dim?: boolean }) {
  return (
    <td
      style={{
        padding: '7px 10px',
        font: `400 12px/1.4 ${font.mono}`,
        color: dim ? color.textMuted : color.text,
        whiteSpace: 'nowrap',
        fontVariantNumeric: 'tabular-nums',
      }}
    >
      {children}
    </td>
  );
}

export function EpochSweepPanel({ job }: { job: JobRecord }) {
  const [sampleSize, setSampleSize] = useState('300');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const status = usePolling(() => api.epochSweepStatus(job.job_id), POLL_MS);
  const state = status.data?.epoch_sweep;
  const candidates = status.data?.candidates ?? [];
  const metrics = status.data?.metrics ?? null;
  const running = state?.status === 'running';
  const busyElsewhere = Boolean(state?.busy_with && state.busy_with !== job.job_id);
  const rows: EpochSweepCandidate[] = state?.candidates?.length ? state.candidates : candidates;
  const shownMetrics = state?.metrics ?? metrics ?? [];
  const invalidSample = !/^\d+$/.test(sampleSize.trim()) || Number(sampleSize) < 1;

  // 보관한 epoch이 없으면 훑을 것이 없습니다. 판 자체를 두지 않습니다 — 켜지 않은
  // 학습에 "시작" 단추를 세워 두면 눌러 보고서야 안 됩니다.
  if (candidates.length === 0 && !state?.candidates?.length) return null;

  async function start() {
    setStarting(true);
    setError(null);
    try {
      await api.startEpochSweep(job.job_id, { sample_size: Number(sampleSize) });
      status.refresh();
    } catch (caught) {
      setError(describeError(caught, '훑기를 시작하지 못했습니다.'));
    } finally {
      setStarting(false);
    }
  }

  return (
    <Panel title="epoch 훑기">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <span style={{ font: `400 12.5px/1.7 ${font.sans}`, color: color.textBody }}>
          보관한 checkpoint {candidates.length}개를 표본으로 재어 순위를 매기고, 이긴 epoch
          하나만 전수로 다시 재어 제출까지 만듭니다. 이긴 epoch은{' '}
          <code style={{ fontFamily: font.mono }}>{job.run_id}-e번호</code> 이름의 별개
          실행으로 남고, 이 학습의 결과는 그대로 있습니다.
        </span>

        {metrics === null && (
          <AlertRow level="warning" title="순위를 매길 지표를 먼저 고르세요">
            무엇이 Kaggle 점수를 예측하는지 아직 모르기 때문에 기본값을 두지 않았습니다.
            설정 화면에서 지표 3개를 순서대로 고르면 시작할 수 있습니다.
          </AlertRow>
        )}

        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div style={{ width: 150 }}>
            <Field
              label="표본 이미지"
              hint="후보마다 이만큼만 봅니다"
              error={invalidSample ? '1 이상의 정수' : undefined}
            >
              <input
                value={sampleSize}
                disabled={running}
                onChange={(event) => setSampleSize(event.target.value)}
                style={controlStyle}
              />
            </Field>
          </div>
          <Button
            kind="primary"
            disabled={running || starting || invalidSample || metrics === null || busyElsewhere}
            onClick={() => void start()}
            style={{ flex: 'none' }}
          >
            {running ? '훑는 중…' : starting ? '시작하는 중…' : '훑기 시작'}
          </Button>
          {shownMetrics.length > 0 && (
            <span style={{ font: `400 12px/1.6 ${font.mono}`, color: color.textMuted }}>
              {shownMetrics.map((name, index) => `${index + 1}순위 ${name}`).join(' · ')}
            </span>
          )}
        </div>

        {busyElsewhere && (
          <AlertRow level="warning" title="다른 학습을 훑고 있습니다">
            한 번에 하나만 돕니다. 그 훑기가 끝난 뒤 다시 눌러 주세요.
          </AlertRow>
        )}

        {state?.message && (
          <div style={{ font: `400 12.5px/1.6 ${font.sans}`, color: color.textBody }}>
            {running && state.total
              ? `${state.done ?? 0}/${state.total} · ${state.message}`
              : state.message}
          </div>
        )}

        {error && (
          <AlertRow level="error" title="시작하지 못했습니다">
            {error}
          </AlertRow>
        )}

        {rows.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr>
                  {['EPOCH', '점수', ...shownMetrics].map((label) => (
                    <th
                      key={label}
                      style={{
                        textAlign: 'left',
                        padding: '7px 10px',
                        font: `500 11px/1.4 ${font.mono}`,
                        color: color.textMuted,
                        borderBottom: `1px solid ${color.border}`,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const won = state?.winner?.epoch === row.epoch;
                  return (
                    <tr
                      key={row.epoch}
                      style={{
                        background: won ? color.fill : 'transparent',
                        borderRadius: radius.control,
                      }}
                    >
                      <Cell>
                        {won ? '★ ' : ''}
                        {row.epoch}
                      </Cell>
                      <Cell dim={row.failed}>
                        {row.failed ? '실패' : row.score === undefined ? '-' : row.score.toFixed(3)}
                      </Cell>
                      {shownMetrics.map((name) => (
                        <Cell key={name} dim={row.failed}>
                          {metricText(row.metrics?.[name])}
                        </Cell>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {state?.status === 'succeeded' && state.winner && (
          <AlertRow level="info" title={`epoch ${state.winner.epoch}이 이겼습니다`}>
            <code style={{ fontFamily: font.mono, overflowWrap: 'anywhere' }}>
              {state.winner.run_id}
            </code>{' '}
            이름으로 전수 평가와 제출을 만들었습니다.
            {state.artifacts?.submission_uri ? ` 제출: ${state.artifacts.submission_uri}` : ''}
            {state.registration?.status === 'succeeded'
              ? ' 기록에도 등록했습니다.'
              : state.registration?.message
                ? ` 등록은 실패했습니다: ${state.registration.message}`
                : ''}
          </AlertRow>
        )}

        {state?.status === 'failed' && (
          <AlertRow level="error" title="훑기가 실패했습니다">
            {state.message ?? '원인을 알 수 없습니다.'}
          </AlertRow>
        )}
      </div>
    </Panel>
  );
}
