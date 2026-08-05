import { useEffect, useState } from 'react';

import { api } from '../api/client';
import type { DetectionMetrics, EvaluationState, JobRecord } from '../api/types';
import { color, font } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { describeError } from '../lib/describeError';
import { AlertRow, Button, Field, KpiCard, Panel, controlStyle } from './primitives';

const POLL_MS = 2000;
const COMPETITION_IOU_THRESHOLDS = [0.75, 0.8, 0.85, 0.9, 0.95];

/** evaluate가 내는 지표. 계산하지 않은 값은 null로 오므로 지어내지 않고 "-"로 둡니다. */
const METRICS: { key: keyof DetectionMetrics; label: string; note: string }[] = [
  { key: 'mAP', label: 'mAP@0.5:0.95', note: '여러 엄격도로 잰 종합 점수. 높을수록 좋습니다' },
  { key: 'mAP50', label: 'mAP@0.5', note: '절반쯤 겹치면 맞다고 볼 때의 점수' },
  { key: 'mAP75', label: 'mAP@0.75', note: '더 정확히 겹쳐야 맞다고 볼 때의 점수' },
  { key: 'precision50', label: 'PRECISION@0.5', note: "'약이다'라고 한 것 중 진짜 비율" },
  { key: 'recall50', label: 'RECALL@0.5', note: '실제 약 중에 찾아낸 비율' },
];

function metricText(value: number | null | undefined): string {
  return value === null || value === undefined ? '-' : value.toFixed(4);
}

/**
 * 끝난 학습의 checkpoint로 evaluate pipeline을 돌립니다.
 *
 * 학습이 만드는 값은 loss뿐입니다. mAP 같은 detection metric은 이 단계에서 처음
 * 나오므로, 학습이 성공한 뒤에만 보여 줍니다.
 */
export function EvaluatePanel({ job }: { job: JobRecord }) {
  const recordedTestManifest = job.data_inputs.test_manifest_uri ?? '';
  const [threshold, setThreshold] = useState('0.0');
  const [overwrite, setOverwrite] = useState(false);
  const [testManifestUri, setTestManifestUri] = useState(recordedTestManifest);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    setTestManifestUri(recordedTestManifest);
  }, [job.job_id, recordedTestManifest]);

  const status = usePolling(() => api.evaluationStatus(job.job_id), POLL_MS);
  const state: EvaluationState | undefined = status.data?.evaluation;
  const running = state?.status === 'running';
  const busyElsewhere = Boolean(state?.busy_with && state.busy_with !== job.job_id);
  const attachedTestManifest = testManifestUri.trim();
  const formRequestsSubmission = Boolean(attachedTestManifest);
  const submissionRequested = running
    ? (state?.submission_requested ?? formRequestsSubmission)
    : formRequestsSubmission;
  const resultUsedSubmission =
    Boolean(state?.submission_requested) || Boolean(state?.artifacts?.submission_uri);
  const thresholdInvalid =
    !/^\d*\.?\d+$/.test(threshold.trim()) || Number(threshold) < 0 || Number(threshold) > 1;

  async function start() {
    setStarting(true);
    setError(null);
    try {
      await api.startEvaluation(job.job_id, {
        score_threshold: Number(threshold),
        overwrite,
        ...(recordedTestManifest || !attachedTestManifest
          ? {}
          : { test_manifest_uri: attachedTestManifest }),
      });
      status.refresh();
    } catch (caught) {
      setError(describeError(caught, '평가를 시작하지 못했습니다.'));
    } finally {
      setStarting(false);
    }
  }

  return (
    <Panel title="평가">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <span style={{ font: `400 11.5px/1.7 ${font.sans}`, color: color.textBody }}>
          학습이 남긴 checkpoint로 검증 이미지를 다시 추론해 성능을 잽니다.
          {submissionRequested
            ? ' 대회 test 이미지도 정답 label 없이 추론해 submission.csv를 함께 만듭니다.'
            : ''}{' '}
          학습을 다시 하지는 않습니다.
        </span>

        {recordedTestManifest ? (
          <AlertRow level="info" title="Test manifest 연결됨">
            <code style={{ fontFamily: font.mono, overflowWrap: 'anywhere' }}>
              {recordedTestManifest}
            </code>
          </AlertRow>
        ) : (
          <Field
            label="Test manifest URI"
            hint="기존 checkpoint에 test_manifest.json을 연결합니다. 비워 두면 validation만 평가합니다."
          >
            <input
              value={testManifestUri}
              disabled={running}
              placeholder="s3://bucket/path/test_manifest.json"
              onChange={(event) => setTestManifestUri(event.target.value)}
              style={controlStyle}
            />
          </Field>
        )}

        {!recordedTestManifest &&
          attachedTestManifest &&
          state?.status === 'succeeded' &&
          !overwrite && (
          <AlertRow level="warning" title="기존 평가 파일이 있습니다">
            같은 run의 metrics와 predictions가 이미 있다면 ‘이미 있으면 덮어쓰기’를 체크해야
            submission 평가를 다시 실행할 수 있습니다.
          </AlertRow>
        )}

        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div style={{ width: 150 }}>
            <Field
              label="Confidence 기준"
              hint="이 값 미만은 제외"
              error={thresholdInvalid ? '0 이상 1 이하' : undefined}
            >
              <input
                value={threshold}
                disabled={running}
                onChange={(event) => setThreshold(event.target.value)}
                style={controlStyle}
              />
            </Field>
          </div>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              font: `400 11.5px/1 ${font.sans}`,
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
          <Button
            kind="primary"
            disabled={running || starting || thresholdInvalid || busyElsewhere}
            onClick={() => void start()}
            style={{ padding: '9px 14px', marginBottom: 9 }}
            title={busyElsewhere ? '다른 학습의 평가가 실행 중입니다.' : undefined}
          >
            {running
              ? submissionRequested
                ? '평가 및 submission 생성 중…'
                : '평가 중…'
              : starting
                ? '시작하는 중…'
                : submissionRequested
                  ? '평가 및 submission 생성'
                  : '평가 실행'}
          </Button>
        </div>

        {busyElsewhere && (
          <AlertRow level="warning" title="다른 학습의 평가가 실행 중입니다">
            한 번에 하나만 실행할 수 있어 지금은 시작할 수 없습니다.
          </AlertRow>
        )}

        {error && (
          <AlertRow level="error" title="시작하지 못했습니다">
            {error}
          </AlertRow>
        )}

        {state && state.status !== 'idle' && (
          <EvaluationResult state={state} submissionRequested={resultUsedSubmission} />
        )}
      </div>
    </Panel>
  );
}

function EvaluationResult({
  state,
  submissionRequested,
}: {
  state: EvaluationState;
  submissionRequested: boolean;
}) {
  if (state.status === 'running') {
    return (
      <AlertRow level="info" title="평가 중">
        {state.message} 검증 이미지와 test 이미지 수만큼 추론하므로 잠시 걸립니다.
      </AlertRow>
    );
  }

  if (state.status === 'failed') {
    return (
      <AlertRow level="error" title="평가에 실패했습니다">
        {state.message}
        {state.exit_code !== null && state.exit_code !== undefined && state.exit_code !== 0 && (
          <> (exit code {state.exit_code})</>
        )}
      </AlertRow>
    );
  }

  const summary = state.summary ?? {};
  const metrics = summary.metrics ?? {};
  const competitionMetric =
    summary.iou_thresholds?.length === COMPETITION_IOU_THRESHOLDS.length &&
    summary.iou_thresholds.every(
      (value, index) => value === COMPETITION_IOU_THRESHOLDS[index],
    );
  const submissionUri = state.artifacts?.submission_uri;
  const otherArtifacts = Object.entries(state.artifacts ?? {}).filter(
    ([key]) => key !== 'submission_uri',
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <AlertRow level="success" title="평가 완료">
        {state.message}
      </AlertRow>

      {submissionUri && (
        <AlertRow level="success" title="대회 제출 파일 생성 완료">
          <code style={{ fontFamily: font.mono, overflowWrap: 'anywhere' }}>{submissionUri}</code>
        </AlertRow>
      )}

      {submissionRequested && !submissionUri && (
        <AlertRow level="warning" title="submission 파일을 확인하지 못했습니다">
          test manifest를 사용한 실행이지만 결과에 submission_uri가 없습니다.
        </AlertRow>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: 8,
        }}
      >
        {METRICS.map((item) => (
          <KpiCard
            key={item.key}
            label={
              item.key === 'mAP' && competitionMetric ? 'mAP@[0.75:0.95]' : item.label
            }
            value={metricText(metrics[item.key])}
            compact
            valueColor={item.key === 'mAP50' ? color.tealDark : undefined}
            note={metrics[item.key] === null ? '이 기준으로는 계산하지 않았습니다' : item.note}
          />
        ))}
      </div>

      <div
        style={{
          display: 'flex',
          gap: 18,
          flexWrap: 'wrap',
          borderTop: `1px solid ${color.borderInner}`,
          paddingTop: 10,
        }}
      >
        {[
          ['평가 이미지', summary.image_count],
          ['정답 개수', summary.annotation_count],
          ['예측 개수', summary.prediction_count],
          ['평가한 클래스', summary.evaluated_class_count],
        ]
          .filter(([, value]) => value !== undefined && value !== null)
          .map(([label, value]) => (
            <span key={String(label)} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
              <span style={{ font: `500 10px/1.3 ${font.mono}`, color: color.textMuted }}>
                {String(label)}
              </span>
              <span style={{ font: `600 12px/1 ${font.mono}`, color: color.text }}>
                {String(value)}
              </span>
            </span>
          ))}
      </div>

      {otherArtifacts.length > 0 && (
        <div
          style={{
            borderTop: `1px solid ${color.borderInner}`,
            paddingTop: 10,
            display: 'flex',
            flexDirection: 'column',
            gap: 5,
          }}
        >
          {otherArtifacts.map(([key, uri]) => (
            <div key={key} style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <span
                style={{ font: `500 10.5px/1.5 ${font.mono}`, color: color.textMuted, minWidth: 130 }}
              >
                {key}
              </span>
              <span
                style={{
                  font: `400 11px/1.5 ${font.mono}`,
                  color: color.textStrong,
                  overflowWrap: 'anywhere',
                }}
              >
                {uri}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
