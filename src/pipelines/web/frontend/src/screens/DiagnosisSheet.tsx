/**
 * 오검출 진단 시트.
 *
 * 점수가 낮을 때 **왜** 낮은지를 봅니다. 지금까지 이 셋은 `metrics.json`을 직접
 * 열어야만 볼 수 있었고, score threshold 탐색은 응답에 실려 오는데도 그리는 화면이
 * 없었습니다.
 *
 * 세 가지가 서로 다른 질문에 답합니다.
 * - threshold 탐색: 자르는 기준을 어디에 두면 가장 나은가.
 * - 헷갈린 쌍: 무엇을 무엇으로 착각하는가.
 * - 오검출 원인: 틀린 상자가 왜 틀렸는가 — 위치인가, 이름인가, 헛것인가, 중복인가.
 *
 * 실행 하나에 대한 진단이라 견주기 화면(Canvas)이 아니라 시트입니다. 여럿을 겹치면
 * 어느 실행의 착각인지 흐려집니다.
 */

import { useEffect, useState } from 'react';

import { api, ApiError } from '../api/client';
import type {
  ConfusionPair,
  ExperimentDetail,
  ExperimentEvaluation,
  FalsePositiveCauses,
  SweepPoint,
} from '../api/types';
import { AlertRow, MicroLabel, Sheet } from '../components/primitives';
import { color, type } from '../design/tokens';

/** evaluate가 내는 IoU label입니다. 어느 엄격도로 볼지 사람이 고릅니다. */
const IOU_LABELS = ['0.50', '0.75'] as const;

/**
 * 원인별로 무엇을 고쳐야 하는지가 다릅니다. 이름만 보면 무슨 뜻인지 알기 어려워
 * 한 줄씩 붙입니다.
 */
const CAUSE_NOTES: { key: keyof FalsePositiveCauses; label: string; note: string }[] = [
  { key: 'classification', label: '이름을 틀림', note: '위치는 맞는데 다른 알약으로 읽었습니다' },
  { key: 'localization', label: '위치가 어긋남', note: '같은 알약을 가리키지만 상자가 덜 맞습니다' },
  { key: 'duplicate', label: '같은 것을 두 번', note: '이미 맞힌 알약에 상자를 하나 더 그렸습니다' },
  { key: 'background', label: '없는 것을 찾음', note: '알약이 없는 자리에 상자를 그렸습니다' },
];

const percent = (value: number | null | undefined) =>
  value === null || value === undefined ? '-' : `${(value * 100).toFixed(1)}%`;

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 30 }}>
      <MicroLabel style={{ marginBottom: 8 }}>{title}</MicroLabel>
      {note && (
        <div style={{ ...type.note, color: color.textMuted, marginBottom: 12, maxWidth: '46em' }}>
          {note}
        </div>
      )}
      {children}
    </div>
  );
}

/**
 * threshold를 바꿔 가며 잰 precision·recall·F1입니다.
 *
 * F1이 가장 높은 줄을 표시합니다. 그 값을 그대로 쓰라는 뜻은 아닙니다 — 대회 채점은
 * 순위로 하므로 자르는 기준을 올리면 맞힌 것까지 함께 사라집니다. 어디서 무엇을
 * 잃는지 보라는 표입니다.
 */
function SweepTable({ rows, best }: { rows: SweepPoint[]; best: SweepPoint | null }) {
  if (rows.length === 0) {
    return <div style={{ ...type.note, color: color.textMuted }}>이 실행에는 탐색 결과가 없습니다.</div>;
  }
  return (
    <table style={{ borderCollapse: 'collapse', width: '100%', maxWidth: 520 }}>
      <thead>
        <tr>
          {['기준', 'precision', 'recall', 'F1'].map((head) => (
            <th
              key={head}
              style={{
                ...type.monoSpec,
                color: color.textMuted,
                textAlign: head === '기준' ? 'left' : 'right',
                padding: '5px 10px 5px 0',
                fontWeight: 400,
              }}
            >
              {head}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const peak = best !== null && row.threshold === best.threshold;
          return (
            <tr key={row.threshold} style={{ borderTop: `1px solid ${color.border}` }}>
              <td style={{ ...type.monoValue, color: peak ? color.accent : color.textStrong, padding: '6px 10px 6px 0' }}>
                {row.threshold.toFixed(2)}
                {peak && <span style={{ ...type.note, marginLeft: 8 }}>F1 최고</span>}
              </td>
              {[row.precision, row.recall, row.f1].map((value, index) => (
                <td
                  key={index}
                  style={{
                    ...type.monoValue,
                    color: peak ? color.accent : color.textStrong,
                    textAlign: 'right',
                    padding: '6px 10px 6px 0',
                  }}
                >
                  {percent(value)}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** background는 class가 아니라 "없음"이라, 이름 대신 무슨 일이 일어났는지 적습니다. */
function pairLabel(pair: ConfusionPair): { text: string; note: string } {
  if (pair.truth === 'background') {
    return { text: `없음 → ${pair.predicted}`, note: '없는 자리에 그렸습니다' };
  }
  if (pair.predicted === 'background') {
    return { text: `${pair.truth} → 없음`, note: '놓쳤습니다' };
  }
  return { text: `${pair.truth} → ${pair.predicted}`, note: '' };
}

function ConfusionList({
  pairs,
  counts,
}: {
  pairs: ConfusionPair[];
  counts: { pairs: number; shown: number } | undefined;
}) {
  if (pairs.length === 0) {
    return <div style={{ ...type.note, color: color.textMuted }}>이 실행에는 혼동 기록이 없습니다.</div>;
  }
  return (
    <>
      <table style={{ borderCollapse: 'collapse', width: '100%', maxWidth: 560 }}>
        <tbody>
          {pairs.map((pair) => {
            const shown = pairLabel(pair);
            return (
              <tr
                key={`${pair.truth}-${pair.predicted}`}
                style={{ borderTop: `1px solid ${color.border}` }}
              >
                <td style={{ ...type.monoValue, color: color.textStrong, padding: '6px 12px 6px 0' }}>
                  {shown.text}
                </td>
                <td style={{ ...type.note, color: color.textMuted, padding: '6px 12px 6px 0' }}>
                  {shown.note}
                </td>
                <td
                  style={{
                    ...type.monoValue,
                    color: color.textStrong,
                    textAlign: 'right',
                    padding: '6px 0',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {pair.count}건
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {/* 잘렸다는 말을 안 하면 이 목록이 전부로 읽힙니다. */}
      {counts !== undefined && counts.pairs > counts.shown && (
        <div style={{ ...type.note, color: color.textMuted, marginTop: 10 }}>
          잦은 {counts.shown}개만 보여 줍니다. 헷갈린 쌍은 모두 {counts.pairs}개입니다.
        </div>
      )}
    </>
  );
}

function CauseTable({ causes }: { causes: FalsePositiveCauses | null | undefined }) {
  if (!causes) {
    return (
      <div style={{ ...type.note, color: color.textMuted }}>
        이 실행에는 원인 분류가 없습니다. 이 기능 이전에 돌린 평가입니다.
      </div>
    );
  }
  const total = CAUSE_NOTES.reduce((sum, item) => sum + causes[item.key], 0);
  return (
    <table style={{ borderCollapse: 'collapse', width: '100%', maxWidth: 560 }}>
      <tbody>
        {CAUSE_NOTES.map((item) => (
          <tr key={item.key} style={{ borderTop: `1px solid ${color.border}` }}>
            <td style={{ ...type.monoValue, color: color.textStrong, padding: '6px 12px 6px 0' }}>
              {item.label}
            </td>
            <td style={{ ...type.note, color: color.textMuted, padding: '6px 12px 6px 0' }}>
              {item.note}
            </td>
            <td
              style={{
                ...type.monoValue,
                color: color.textStrong,
                textAlign: 'right',
                padding: '6px 0',
                whiteSpace: 'nowrap',
              }}
            >
              {causes[item.key]}건
              {/* 0으로 나누지 않습니다. 틀린 상자가 하나도 없으면 비율이 없습니다. */}
              {total > 0 && (
                <span style={{ ...type.note, color: color.textMuted, marginLeft: 8 }}>
                  {((causes[item.key] / total) * 100).toFixed(0)}%
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Body({ evaluation, iou }: { evaluation: ExperimentEvaluation; iou: string }) {
  if (!evaluation.available) {
    return (
      <AlertRow level="info" title="진단할 평가 결과가 없습니다">
        {evaluation.reason ?? '이 실험에는 평가 결과 파일이 기록돼 있지 않습니다.'}
      </AlertRow>
    );
  }
  return (
    <>
      <Section
        title="자르는 기준 탐색"
        note="점수가 이 기준보다 낮은 상자를 버렸다면 어떻게 됐을지 다시 셉니다. 평가를 다시 돌린 것이 아니라 이미 잰 매칭을 잘라 본 것이라, 기준을 낮춰도 이미 잘려 나간 예측은 되살아나지 않습니다."
      >
        <SweepTable
          rows={evaluation.score_sweep?.[iou] ?? []}
          best={evaluation.best_f1?.[iou] ?? null}
        />
      </Section>

      <Section
        title="헷갈린 쌍"
        note="정답이 무엇인데 무엇으로 봤는지입니다. class를 무시하고 매칭한 결과라 같은 자리를 다른 이름으로 읽은 것이 보입니다. 이름이 닮은 알약끼리 몰려 있으면 crop 재순위가 들 자리입니다."
      >
        <ConfusionList
          pairs={evaluation.confusions?.[iou] ?? []}
          counts={evaluation.confusion_counts?.[iou]}
        />
      </Section>

      <Section
        title="틀린 상자의 원인"
        note="맞지 않은 상자를 원인별로 나눈 것입니다. 이름을 틀린 것이 많으면 분류를, 위치가 어긋난 것이 많으면 상자를, 없는 것을 찾은 것이 많으면 자르는 기준을 봐야 합니다."
      >
        <CauseTable causes={evaluation.error_breakdown?.[iou]} />
      </Section>
    </>
  );
}

export function DiagnosisSheet({ runId, onClose }: { runId: string; onClose: () => void }) {
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [iou, setIou] = useState<string>(IOU_LABELS[0]);

  // 한 번만 읽습니다. 끝난 실행의 평가 결과는 바뀌지 않으므로 폴링할 것이 없습니다.
  useEffect(() => {
    let alive = true;
    setDetail(null);
    setError(null);
    api
      .experimentDetail(runId)
      .then((result) => {
        if (alive) setDetail(result);
      })
      .catch((caught) => {
        if (alive) {
          setError(caught instanceof ApiError ? caught.message : '진단을 불러오지 못했습니다.');
        }
      });
    return () => {
      alive = false;
    };
  }, [runId]);

  return (
    <Sheet title={`진단 · ${runId}`} onClose={onClose}>
      <div style={{ ...type.body, color: color.textBody, marginBottom: 22, textWrap: 'pretty' }}>
        점수가 낮을 때 <strong>왜</strong> 낮은지 봅니다. 여기 숫자는 평가를 다시 돌린 것이
        아니라 그때 남긴 <code>metrics.json</code>을 읽은 것이라, 실행마다 값이 고정입니다.
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <MicroLabel style={{ alignSelf: 'center', marginRight: 4 }}>겹침 기준</MicroLabel>
        {IOU_LABELS.map((label) => (
          <button
            key={label}
            type="button"
            onClick={() => setIou(label)}
            style={{
              ...type.monoValue,
              padding: '5px 12px',
              background: 'transparent',
              color: iou === label ? color.accent : color.textMuted,
              border: `1px solid ${iou === label ? color.accent : color.border}`,
              cursor: 'pointer',
            }}
          >
            IoU {label}
          </button>
        ))}
      </div>

      {error && (
        <AlertRow level="error" title="진단을 불러오지 못했습니다">
          {error}
        </AlertRow>
      )}
      {!error && detail === null && (
        <div style={{ ...type.note, color: color.textMuted }}>불러오는 중…</div>
      )}
      {detail !== null && <Body evaluation={detail.evaluation} iou={iou} />}
    </Sheet>
  );
}
