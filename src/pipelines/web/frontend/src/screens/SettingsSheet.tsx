/**
 * 설정 시트. 이 도구가 지금 무엇에 붙어 있는지와, 평가를 언제 돌릴지를 모읍니다.
 *
 * 저장소·팀·GPU는 서버가 환경 변수와 하드웨어에서 읽는 값이라 여기서 고칠 수
 * 없습니다. 고칠 수 있는 것은 **평가 실행** 하나뿐이고, 그것만 저장 버튼을 받습니다.
 */

import { useState, type ReactNode } from 'react';

import { api, ApiError } from '../api/client';
import { EPOCH_METRIC_NAMES } from '../api/types';
import type { AppSettings, GpuStatus, RegistryScope } from '../api/types';
import { AlertRow, Button, Field, MicroLabel, Sheet, controlStyle } from '../components/primitives';
import { color, font, radius, type } from '../design/tokens';
import { megabytes, percent } from '../lib/format';
import { useTeam } from '../team/TeamContext';

type EvaluationMode = 'parallel' | 'serial';

const MODES: { key: EvaluationMode; label: string; note: string }[] = [
  {
    key: 'parallel',
    label: '학습과 함께 (병렬)',
    note: '학습이 도는 중에도 곧바로 평가합니다. 평가가 VRAM을 약 1.8GB 더 쓰므로 아래 막대에 그만큼 자리가 남아 있을 때만 고르세요.',
  },
  {
    key: 'serial',
    label: '학습이 끝난 뒤 (직렬)',
    note: '도는 학습이 하나도 없을 때까지 기다렸다 평가합니다. 8GB 카드에서 안전하고, 밤새 대기열을 돌릴 때 쓰기 좋습니다.',
  },
];

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'baseline',
        justifyContent: 'space-between',
        gap: 16,
        padding: '12px 0',
        borderTop: `1px solid ${color.borderRow}`,
      }}
    >
      <span style={{ ...type.bodySmall, color: color.textMuted }}>{label}</span>
      <span
        style={{
          font: `500 13px/1.5 ${font.mono}`,
          color: color.text,
          textAlign: 'right',
          overflowWrap: 'anywhere',
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function SettingsSheet({
  gpu,
  scope,
  settings,
  onClose,
  onSaved,
}: {
  gpu: GpuStatus | null;
  scope: RegistryScope | undefined;
  settings: AppSettings | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const team = useTeam();
  const [mode, setMode] = useState<EvaluationMode | null>(settings?.evaluation_mode ?? null);
  // 순서가 곧 가중치입니다(3:2:1). 고르지 않은 자리는 빈 문자열입니다.
  const [metrics, setMetrics] = useState<string[]>(settings?.epoch_metrics ?? ['', '', '']);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const picked = metrics.filter((name) => name !== '');
  const metricsChosen = picked.length === 3 && new Set(picked).size === 3;
  const metricsTouched = picked.length > 0;
  const metricsInvalid = metricsTouched && !metricsChosen;

  const device = gpu?.telemetry.devices[0] ?? null;
  const totalMb = device?.memory_total_mb ?? null;
  const usedMb = device?.memory_used_mb ?? null;
  const usedRatio = usedMb !== null && totalMb ? Math.min(1, usedMb / totalMb) : null;
  // 평가는 GPU에서 약 1.8GB를 더 씁니다. 남는 자리를 눈으로 확인하고 고를 수 있게
  // 학습이 쓰는 만큼 옆에 평가 몫을 함께 그립니다.
  const evaluateRatio = totalMb ? Math.min(1 - (usedRatio ?? 0), 1800 / totalMb) : null;
  const tight = totalMb !== null && usedMb !== null && totalMb - usedMb < 1800;

  async function save() {
    if (mode === null || metricsInvalid) return;
    setSaving(true);
    setError(null);
    try {
      // 고르다 만 상태는 보내지 않습니다. 보내지 않은 값은 서버가 그대로 둡니다.
      await api.saveSettings({
        evaluation_mode: mode,
        epoch_metrics: metricsChosen ? picked : null,
      });
      onSaved();
      onClose();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '설정을 저장하지 못했습니다.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <Sheet
      title="설정"
      onClose={onClose}
      footer={
        <>
          <Button
            kind="primary"
            disabled={mode === null || metricsInvalid || saving}
            onClick={() => void save()}
          >
            {saving ? '저장 중…' : '저장'}
          </Button>
          <Button kind="ghost" onClick={onClose}>
            닫기
          </Button>
        </>
      }
    >
      <MicroLabel style={{ marginBottom: 16 }}>평가 실행</MicroLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
        {MODES.map((item) => {
          const on = mode === item.key;
          return (
            <button
              key={item.key}
              type="button"
              aria-pressed={on}
              onClick={() => setMode(item.key)}
              style={{
                textAlign: 'left',
                border: `1px solid ${on ? color.accentLine : color.border}`,
                borderRadius: radius.control,
                padding: '16px 18px',
                background: on ? color.fill : 'transparent',
              }}
            >
              <span style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 8 }}>
                <span
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: '50%',
                    flex: 'none',
                    background: on ? color.accent : 'transparent',
                    border: `1px solid ${on ? color.accent : color.border}`,
                  }}
                />
                <span
                  style={{
                    font: `600 14.5px/1.3 ${font.sans}`,
                    color: on ? color.text : color.textBody,
                  }}
                >
                  {item.label}
                </span>
              </span>
              <span
                style={{
                  display: 'block',
                  ...type.bodySmall,
                  color: color.textMuted,
                  paddingLeft: 19,
                  textWrap: 'pretty',
                }}
              >
                {item.note}
              </span>
            </button>
          );
        })}
      </div>

      {/* 고르기 전에는 자동 평가가 꺼져 있다는 사실이 화면에 드러나야 합니다.
          안 그러면 "왜 평가가 안 돌지"가 됩니다. */}
      <div style={{ ...type.note, color: color.textMuted, marginBottom: 26, textWrap: 'pretty' }}>
        {settings?.evaluation_mode == null
          ? '아직 고르지 않아 자동 평가가 꺼져 있습니다. 지금은 학습 화면에서 직접 눌러야 평가가 돕니다. 하나를 고르고 저장하면 학습이 성공할 때마다 이어서 평가합니다.'
          : '학습이 성공하면 이 방식으로 이어서 평가합니다. 평가가 실패한 학습은 다시 집지 않습니다 — 학습 화면에서 직접 다시 누르세요.'}
      </div>

      <MicroLabel style={{ marginBottom: 16 }}>epoch 훑기 기준</MicroLabel>
      <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
        {[0, 1, 2].map((slot) => (
          <div key={slot} style={{ minWidth: 150, flex: 1 }}>
            <Field label={`${slot + 1}순위`}>
              <select
                value={metrics[slot] ?? ''}
                onChange={(event) => {
                  const next = [...metrics];
                  next[slot] = event.target.value;
                  setMetrics(next);
                }}
                style={controlStyle}
              >
                <option value="">고르지 않음</option>
                {EPOCH_METRIC_NAMES.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        ))}
      </div>
      <div style={{ ...type.note, color: color.textMuted, marginBottom: 26, textWrap: 'pretty' }}>
        {metricsInvalid
          ? '서로 다른 지표 3개를 모두 골라야 저장할 수 있습니다.'
          : settings?.epoch_metrics
            ? '훑기가 후보 epoch을 이 순서로 줄 세웁니다. 1순위에 가장 큰 몫(3:2:1)이 가고, 척도가 다른 지표를 섞어도 되도록 후보들 사이에서 0~1로 펴서 더합니다.'
            : '아직 고르지 않아 epoch 훑기를 시작할 수 없습니다. 무엇이 Kaggle 점수를 예측하는지 모르는 것이 이 기능을 만든 이유라, 기본값을 두지 않았습니다.'}
      </div>

      {error && (
        <div style={{ marginBottom: 22 }}>
          <AlertRow level="error" title="저장하지 못했습니다">
            {error}
          </AlertRow>
        </div>
      )}

      <MicroLabel style={{ margin: '0 0 16px' }}>GPU</MicroLabel>
      {usedRatio === null ? (
        <div style={{ ...type.bodySmall, color: color.textMuted }}>
          {gpu?.telemetry.reason ?? gpu?.torch.reason ?? 'GPU 메모리 사용량을 읽지 못했습니다.'}
        </div>
      ) : (
        <>
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              gap: 16,
              marginBottom: 12,
            }}
          >
            <span style={{ ...type.bodySmall, color: color.textMuted }}>지금 VRAM</span>
            <span style={{ ...type.monoValue, color: color.text, fontVariantNumeric: 'tabular-nums' }}>
              {megabytes(usedMb)} / {megabytes(totalMb)}
            </span>
          </div>
          <div style={{ height: 6, background: color.border, borderRadius: 3, overflow: 'hidden', display: 'flex' }}>
            <div style={{ width: `${usedRatio * 100}%`, height: '100%', background: color.accent }} />
            {evaluateRatio !== null && evaluateRatio > 0 && (
              <div style={{ width: `${evaluateRatio * 100}%`, height: '100%', background: color.accentLine }} />
            )}
          </div>
          <div
            style={{
              display: 'flex',
              gap: 20,
              marginTop: 12,
              font: `400 12px/1.5 ${font.mono}`,
              color: color.textMuted,
              flexWrap: 'wrap',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{ width: 10, height: 3, background: color.accent }} />
              지금 {megabytes(usedMb)}
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <span style={{ width: 10, height: 3, background: color.accentLine }} />
              평가 ~1.8 GB
            </span>
            <span>사용률 {percent(device?.utilization_percent)}</span>
          </div>
          <div style={{ ...type.body, color: color.textBody, marginTop: 18, textWrap: 'pretty' }}>
            {tight
              ? '지금 남은 VRAM이 평가 몫보다 적습니다. 병렬로 두면 둘 다 out of memory로 잃을 수 있으니 직렬을 고르세요.'
              : '평가는 GPU에서 약 2분, CPU에서는 2,942장 기준 약 55분 걸려 시간 제한에 걸립니다. 위 막대에 평가 몫이 들어갈 자리가 남아 있으면 병렬로 둬도 됩니다.'}
          </div>
        </>
      )}

      <MicroLabel style={{ margin: '30px 0 4px' }}>저장소</MicroLabel>
      <Row
        label="실험 목록"
        value={scope ? (scope.shared ? '팀 공유 (S3)' : '이 컴퓨터') : '읽는 중'}
      />
      <Row label="backend" value={scope?.backend ?? '읽는 중'} />
      <div style={{ ...type.note, color: color.textMuted, marginTop: 12, textWrap: 'pretty' }}>
        {!scope
          ? '등록된 실험 목록을 아직 읽고 있습니다. index 전체를 훑기 때문에 기록이 많으면 수십 초가 걸립니다.'
          : scope.shared
            ? '팀원이 등록한 실험도 목록과 캔버스에 함께 나옵니다.'
            : 'PILL_STORAGE_S3_BUCKET을 설정한 뒤 서버를 다시 시작하면 팀원 기록까지 읽습니다.'}
      </div>

      <MicroLabel style={{ margin: '30px 0 4px' }}>팀</MicroLabel>
      <Row label="연결" value={team.config.enabled ? '켜짐' : '꺼짐'} />
      {team.config.enabled && (
        <>
          <Row label="team_id" value={team.config.team_id ?? '-'} />
          <Row label="로그인" value={team.user?.username ?? team.config.actor ?? '로그인하지 않음'} />
          <div style={{ ...type.note, color: color.textMuted, marginTop: 12, textWrap: 'pretty' }}>
            팀원이 지금 돌리는 학습과 그 실시간 로그는 기록 목록의{' '}
            <b style={{ color: color.textBody }}>학습 중</b> 표에서 봅니다.
          </div>
          {team.user && (
            <div style={{ marginTop: 16 }}>
              <Button kind="ghost" onClick={() => void team.logout()}>
                로그아웃
              </Button>
            </div>
          )}
        </>
      )}
    </Sheet>
  );
}
