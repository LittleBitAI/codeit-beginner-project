/**
 * EDA 시트.
 *
 * 고른 dataset을 model 없이 뜯어본 결과를 봅니다. 여기 있는 숫자는 어떤 학습
 * 결과와도 무관해서, "학습이 이상한가"와 "데이터가 다른가"를 갈라 말할 수 있습니다.
 */

import { useEffect, useState } from 'react';

import { api, ApiError } from '../api/client';
import type { EdaDistribution, EdaReport, EdaState } from '../api/types';
import { Button, MicroLabel, Sheet } from '../components/primitives';
import { color, type } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';

const RUNNING_INTERVAL_MS = 2000;

function Row({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'baseline', padding: '7px 0' }}>
      <div style={{ ...type.monoSpec, color: color.textMuted, minWidth: 190 }}>{label}</div>
      <div style={{ ...type.monoValue, color: color.textStrong }}>{value}</div>
      {hint && <div style={{ ...type.monoSpec, color: color.textMuted }}>{hint}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 26 }}>
      <MicroLabel style={{ marginBottom: 8 }}>{title}</MicroLabel>
      {children}
    </div>
  );
}

const number = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? '-' : value.toFixed(digits);

const spread = (value: EdaDistribution | null) =>
  value ? `${number(value.median, 4)}  (p10 ${number(value.p10, 4)} ~ p90 ${number(value.p90, 4)})` : '-';

const rgb = (value: number[] | null | undefined) =>
  value ? value.map((part) => Math.round(part)).join(', ') : '-';

function ReportView({ report }: { report: EdaReport }) {
  const size = report.object_size;
  const ratio = size.test_over_train;
  return (
    <div>
      <Section title="모양">
        <Row label="클래스 수" value={String(report.classes.class_count)} />
        <Row
          label="클래스별 학습 이미지"
          value={spread(report.classes.train_images_per_class)}
          hint={`불균형 ${number(report.classes.imbalance_ratio, 1)}배`}
        />
        <Row
          label="학습 / 검증 이미지"
          value={`${report.shape.train?.images ?? 0} / ${report.shape.validation?.images ?? 0}`}
        />
        <Row
          label="같은 class 두 번 나온 이미지"
          value={String(report.shape.train?.images_with_a_repeated_class ?? 0)}
          hint="정답에서는 0이어야 합니다"
        />
      </Section>

      <Section title="조합과 split">
        <Row
          label="조합 수 (학습 / 검증)"
          value={`${report.combinations.train.groups} / ${report.combinations.validation.groups}`}
        />
        <Row
          label="양쪽에 걸친 조합"
          value={String(report.combinations.groups_in_both_splits)}
          hint={
            report.combinations.groups_in_both_splits > 0
              ? `누수: ${report.combinations.leaked_group_sample.join(', ')}`
              : '누수 없음'
          }
        />
        <Row
          label="촬영 조건 종류"
          value={String(Object.keys(report.combinations.capture_conditions).length)}
        />
      </Section>

      <Section title="물체 크기">
        <Row
          label="자 보정 (train)"
          value={number(size.calibration.measured_over_annotation)}
          hint={
            size.calibration.trustworthy
              ? `믿음 구간 ${size.calibration.limits.join(' ~ ')} 안`
              : '구간 밖 — 비교를 내주지 않습니다'
          }
        />
        <Row label="학습 전경 비율" value={spread(size.train_foreground_fraction)} />
        <Row label="test 전경 비율" value={spread(size.test_foreground_fraction)} />
        <Row
          label="test / train"
          value={ratio ? `변 길이 ${number(ratio.length_ratio)}배` : '재지 못했습니다'}
          hint={ratio ? `면적 ${number(ratio.area_ratio)}배` : undefined}
        />
      </Section>

      <Section title="촬영 부스와 조명">
        <Row label="배경색 (train)" value={rgb(report.appearance.train_background_color)} />
        <Row label="배경색 (test)" value={rgb(report.appearance.test_background_color)} />
        <Row
          label="배경색 거리"
          value={number(report.appearance.background_color_distance, 1)}
          hint="0~441 척도"
        />
        <Row
          label="물체색 거리"
          value={number(report.appearance.foreground_color_distance, 1)}
        />
      </Section>
    </div>
  );
}

export function EdaSheet({ onClose }: { onClose: () => void }) {
  const [sample, setSample] = useState('200');
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const status = usePolling(() => api.edaStatus(), RUNNING_INTERVAL_MS);
  const state: EdaState | undefined = status.data?.eda;
  const running = state?.status === 'running';
  const progress = state?.progress;

  // 끝나는 순간 한 번 더 읽어 리포트를 받아 옵니다.
  useEffect(() => {
    if (state?.status === 'succeeded' && !state.report) status.refresh();
  }, [state, status]);

  async function start(overwrite: boolean) {
    setStarting(true);
    setError(null);
    try {
      await api.startEda({ image_sample: Number.parseInt(sample, 10) || 200, overwrite });
      status.refresh();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'EDA를 시작하지 못했습니다.');
    } finally {
      setStarting(false);
    }
  }

  return (
    <Sheet title="EDA" onClose={onClose}>
      <div style={{ ...type.body, color: color.textBody, marginBottom: 20, textWrap: 'pretty' }}>
        지금 고른 dataset을 model 없이 뜯어봅니다. 여기 있는 숫자는 어떤 학습 결과와도
        무관하므로, 대회 점수가 낮을 때 원인이 학습에 있는지 데이터에 있는지 가르는 데
        씁니다. 이미지를 전부 열어야 해서 몇 분 걸립니다.
      </div>

      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginBottom: 18 }}>
        <label style={{ display: 'block' }}>
          <MicroLabel style={{ marginBottom: 6 }}>학습 이미지 표본</MicroLabel>
          <input
            value={sample}
            onChange={(event) => setSample(event.target.value)}
            disabled={running}
            style={{
              ...type.monoValue,
              width: 90,
              padding: '7px 10px',
              background: color.panel,
              color: color.textStrong,
              border: `1px solid ${color.border}`,
            }}
          />
        </label>
        <Button onClick={() => start(false)} disabled={running || starting}>
          {running ? '분석 중…' : 'EDA 실행'}
        </Button>
        {state?.report && (
          <Button kind="ghost" onClick={() => start(true)} disabled={running || starting}>
            다시 분석
          </Button>
        )}
      </div>

      {error && (
        <div style={{ ...type.body, color: color.danger, marginBottom: 14 }}>{error}</div>
      )}
      {state && state.status !== 'idle' && (
        <div style={{ ...type.monoSpec, color: color.textMuted, marginBottom: 18 }}>
          {state.message}
          {running && progress?.available && progress.read?.done
            ? ` · ${progress.read.stage} ${progress.read.done} / ${progress.read.total}`
            : ''}
        </div>
      )}

      {state?.report ? (
        <ReportView report={state.report} />
      ) : (
        <div style={{ ...type.body, color: color.textMuted }}>
          아직 리포트가 없습니다. 위에서 실행하세요.
        </div>
      )}
    </Sheet>
  );
}
