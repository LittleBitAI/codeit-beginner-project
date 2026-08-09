import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type { DataSource, Defaults, FieldSpec, ValidationResult } from '../api/types';
import {
  AlertRow,
  Button,
  Field,
  Panel,
  ScreenIntro,
  controlStyle,
  invalidControlStyle,
} from '../components/primitives';
import { IconShield } from '../components/Icon';
import { color, font, radius } from '../design/tokens';
import { dataMatchesSource } from '../lib/dataSource';
import {
  EARLY_STOPPING_FIELDS,
  isEarlyStoppingOn,
  messageFor,
  toPayload,
} from '../lib/draftPayload';
import { resolveTrainCapability } from '../lib/trainCapabilities';
import { useDraft } from '../state/DraftContext';

type TabKey = 'basic' | 'hyper' | 'output';

const TABS: { key: TabKey; label: string; fields: string[] }[] = [
  {
    key: 'basic',
    label: '기본 정보',
    fields: [
      'architecture',
      'optimizer',
      'augmentation',
      'precision',
      'run_id',
      'seed',
      'device',
      'pretrained',
    ],
  },
  {
    key: 'hyper',
    label: '하이퍼파라미터',
    fields: [
      'epochs',
      'batch_size',
      'learning_rate',
      'momentum',
      'weight_decay',
      'beta1',
      'beta2',
      'epsilon',
      'num_workers',
      'early_stopping',
      'early_stopping_patience',
      'early_stopping_min_delta',
    ],
  },
  { key: 'output', label: '출력', fields: ['output_dir', 'output_prefix'] },
];

export function NewExperiment({
  defaults,
  source,
}: {
  defaults: Defaults | null;
  source: DataSource | null;
}) {
  const navigate = useNavigate();
  const { draft, setTrainField, setDataField, setDataFields, setSaved } = useDraft();
  const [tab, setTab] = useState<TabKey>('basic');
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const fields = defaults?.fields ?? [];
  const payload = useMemo(() => toPayload(draft, fields), [draft, fields]);

  // 고른 데이터셋과 지금 칸의 값이 다른지 확인합니다.
  // 실제로 화면에는 새 데이터셋이 보이는데 예전 데이터로 학습된 적이 있어서,
  // 다르면 눈에 띄게 알리고 한 번에 맞출 수 있게 합니다.
  const mismatched = !dataMatchesSource(draft.data, source);

  // 입력이 멈추면 서버에 검증을 맡깁니다. 판단 기준은 언제나 서버입니다.
  useEffect(() => {
    if (!defaults) return;
    const timer = window.setTimeout(() => {
      void api
        .validate(payload)
        .then(setResult)
        .catch(() => setResult(null));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [payload, defaults]);

  const errors = result?.errors ?? [];
  const warnings = result?.warnings ?? [];
  // 서버가 지어 준 이름입니다. 규칙은 backend 한 곳에만 있습니다.
  const autoRunId =
    typeof result?.normalized?.train?.run_id === 'string'
      ? result.normalized.train.run_id
      : null;
  const dataFilled = (defaults?.data_fields ?? []).every(
    (spec) => (draft.data[spec.name] ?? '').trim() !== '',
  );
  const canSave = Boolean(result?.valid) && dataFilled && !saving;

  async function save() {
    setSaving(true);
    setSaveError(null);
    try {
      const created = await api.createConfig(payload);
      setSaved(created);
      navigate('/review');
    } catch (caught) {
      setSaveError(caught instanceof ApiError ? caught.message : '설정을 저장하지 못했습니다.');
    } finally {
      setSaving(false);
    }
  }

  if (!defaults) {
    return <Panel>설정 정보를 불러오는 중입니다.</Panel>;
  }

  const capability = resolveTrainCapability(defaults);
  const selectedOptimizer = draft.train.optimizer || capability.optimizer.default;
  const earlyStoppingOn = isEarlyStoppingOn(draft.train, fields);
  const activeTab = TABS.find((item) => item.key === tab) ?? TABS[0]!;
  const tabHasError = (item: (typeof TABS)[number]) =>
    item.fields.some((name) => messageFor(errors, `train.${name}`) !== undefined);

  return (
    <div style={{ maxWidth: 1320 }}>
      <ScreenIntro
        title="학습에 쓸 설정을 만듭니다"
        terms={[
          { term: 'batch size', meaning: '한 번에 처리할 이미지 수. GPU 메모리에 가장 큰 영향을 줍니다' },
          { term: 'learning rate', meaning: '한 번에 얼마나 크게 배울지. 너무 크면 학습이 발산합니다' },
          { term: 'seed', meaning: '무작위성을 고정하는 값. 같으면 같은 결과가 나옵니다' },
        ]}
      >
        값을 비워 두면 backend의 기본값을 씁니다. 저장하기 전에는 학습을 시작할 수 없고, 저장 버튼은 모든
        검증을 통과해야 열립니다.
      </ScreenIntro>

      <div style={{ marginBottom: 14 }}>
        <AlertRow
          level="info"
          title={
            capability.source === 'legacy_fallback'
              ? 'Train capability 호환 기본값을 사용합니다'
              : 'Train capability를 반영했습니다'
          }
        >
          모델 <code style={{ fontFamily: font.mono }}>{capability.model.default}</code> · optimizer{' '}
          <code style={{ fontFamily: font.mono }}>{capability.optimizer.default}</code>
          {capability.source === 'legacy_fallback'
            ? '를 기본으로 씁니다. Train이 capability를 아직 제공하지 않아 현재 실제 구현과 맞춘 선택 목록입니다.'
            : '를 기본 구성으로 사용합니다.'}
        </AlertRow>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-start' }}>
        <div style={{ flex: '3 1 400px', minWidth: 0 }}>
          <div style={{ display: 'flex', gap: 2 }}>
            {TABS.map((item) => {
              const active = item.key === tab;
              return (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setTab(item.key)}
                  style={{
                    padding: '9px 15px',
                    background: active ? color.surface : '#EDF1F6',
                    border: `1px solid ${active ? color.borderControl : 'transparent'}`,
                    borderBottomColor: active ? color.surface : 'transparent',
                    borderRadius: '5px 5px 0 0',
                    font: `${active ? 600 : 500} 12px/1 ${font.sans}`,
                    color: active ? color.text : color.textBody,
                    position: 'relative',
                    top: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  {item.label}
                  {tabHasError(item) && (
                    <span
                      style={{ width: 5, height: 5, borderRadius: '50%', background: color.red }}
                    />
                  )}
                </button>
              );
            })}
          </div>

          <div
            style={{
              background: color.surface,
              border: `1px solid ${color.borderControl}`,
              borderRadius: '0 6px 6px 6px',
              padding: '20px 22px',
              display: 'flex',
              flexDirection: 'column',
              gap: 18,
            }}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
                gap: 12,
              }}
            >
              {activeTab.fields.map((name) => {
                if (selectedOptimizer === 'SGD' && ['beta1', 'beta2', 'epsilon'].includes(name)) {
                  return null;
                }
                if (selectedOptimizer !== 'SGD' && name === 'momentum') return null;
                // 조기 종료를 끄면 관련 숫자 칸도 감춥니다. 보이면 그 값이 학습에
                // 쓰이는 것처럼 읽히고, 서버도 쓰지 않는 값이라며 거부합니다.
                if (!earlyStoppingOn && EARLY_STOPPING_FIELDS.includes(name)) return null;
                const spec = fields.find((item) => item.name === name);
                if (!spec) return null;
                // 이름을 비워 두면 서버가 설정을 읽어 지어 줍니다. 규칙을 여기에
                // 옮겨 적지 않고, 매 입력마다 받는 검증 결과의 이름을 그대로 보여 줍니다.
                if (name === 'run_id') {
                  return (
                    <TrainField
                      key={name}
                      spec={spec}
                      value={draft.train.run_id ?? ''}
                      error={messageFor(errors, 'train.run_id')}
                      devices={defaults.devices}
                      onChange={(value) => setTrainField('run_id', value)}
                      note={
                        (draft.train.run_id ?? '').trim() === '' && autoRunId
                          ? `자동 이름: ${autoRunId}`
                          : undefined
                      }
                    />
                  );
                }
                const optimizerDefault = spec.defaults_by_optimizer?.[selectedOptimizer];
                const shownSpec =
                  optimizerDefault === undefined
                    ? spec
                    : { ...spec, default: optimizerDefault };
                return (
                  <TrainField
                    key={name}
                    spec={shownSpec}
                    value={draft.train[name] ?? ''}
                    error={messageFor(errors, `train.${name}`)}
                    devices={defaults.devices}
                    onChange={(value) => setTrainField(name, value)}
                  />
                );
              })}
            </div>

            {tab === 'basic' && (
              <div
                style={{
                  borderTop: `1px solid ${color.borderInner}`,
                  paddingTop: 14,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 10,
                    flexWrap: 'wrap',
                  }}
                >
                  <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.text }}>
                    data pipeline artifact
                  </span>
                  {source?.complete && (
                    <span
                      style={{
                        font: `400 11.5px/1.4 ${font.mono}`,
                        color: color.textMuted,
                        overflowWrap: 'anywhere',
                      }}
                    >
                      전처리 데이터셋에서 자동으로 채움 · {source.directory}
                    </span>
                  )}
                </div>
                {!source?.complete && (
                  <span style={{ font: `400 12px/1.6 ${font.sans}`, color: color.textBody }}>
                    학습 개요 화면에서 전처리 데이터셋을 고르면 이 네 칸이 자동으로 채워집니다.
                  </span>
                )}
                {mismatched && source && (
                  <AlertRow
                    level="warning"
                    title="고른 데이터셋과 아래 값이 다릅니다"
                    action={
                      <Button
                        onClick={() => setDataFields({ ...source.data }, null)}
                        title="고른 데이터셋의 값으로 맞춥니다"
                      >
                        데이터셋에 맞추기
                      </Button>
                    }
                  >
                    이대로 학습하면 <b>아래 칸에 적힌 데이터</b>로 돌아갑니다. 고른 데이터셋은{' '}
                    <code style={{ fontFamily: font.mono }}>{source.directory}</code> 입니다.
                  </AlertRow>
                )}
                {defaults.data_fields.map((spec) => (
                  <Field
                    key={spec.name}
                    label={spec.label}
                    hint={spec.hint}
                    error={messageFor(errors, `inputs.data.${spec.name}`)}
                  >
                    <input
                      value={draft.data[spec.name] ?? ''}
                      placeholder="artifacts/data/... 또는 s3://bucket/..."
                      onChange={(event) => setDataField(spec.name, event.target.value)}
                      style={
                        messageFor(errors, `inputs.data.${spec.name}`)
                          ? invalidControlStyle
                          : controlStyle
                      }
                    />
                  </Field>
                ))}
                <div style={{ display: 'flex', gap: 9, alignItems: 'flex-start' }}>
                  <span style={{ marginTop: 2 }}>
                    <IconShield color={color.textBody} />
                  </span>
                  <span style={{ font: `400 12.5px/1.55 ${font.sans}`, color: color.textBody }}>
                    이 네 값은 data pipeline이 만든 결과의 위치입니다. 저장소 기준 상대 경로나{' '}
                    <code style={{ fontFamily: font.mono }}>s3://bucket/key</code> 형식만 받습니다.
                    절대 경로와 <code style={{ fontFamily: font.mono }}>..</code>는 거부됩니다.
                  </span>
                </div>
              </div>
            )}

            {tab === 'output' && (
              <span style={{ font: `400 12.5px/1.55 ${font.sans}`, color: color.textBody }}>
                data 입력이 <code style={{ fontFamily: font.mono }}>s3://</code>면 S3 backend로,
                아니면 로컬 디스크로 저장합니다. S3 bucket 이름은 환경 변수에서 읽으며 설정 파일에 남기지
                않습니다.
              </span>
            )}
          </div>
        </div>

        <div style={{ flex: '2 1 320px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Panel title="생성될 설정" bodyStyle={{ padding: 0 }}>
            <pre
              style={{
                background: color.navy,
                color: color.logText,
                margin: 0,
                padding: '13px 15px',
                maxHeight: 330,
                overflow: 'auto',
                font: `400 11.5px/1.65 ${font.mono}`,
              }}
            >
              {JSON.stringify(result?.normalized ?? payload, null, 2)}
            </pre>
          </Panel>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {errors.map((item) => (
              <AlertRow key={`${item.field}-${item.message}`} level="error" title={item.field}>
                {item.message}
              </AlertRow>
            ))}
            {errors.length === 0 && !dataFilled && (
              <AlertRow level="warning" title="data artifact 위치가 비어 있습니다">
                네 값을 모두 채워야 저장할 수 있습니다.
              </AlertRow>
            )}
            {errors.length === 0 &&
              dataFilled &&
              warnings.map((item) => (
                <AlertRow key={`${item.field}-${item.message}`} level="warning" title={item.field}>
                  {item.message}
                </AlertRow>
              ))}
            {errors.length === 0 && dataFilled && warnings.length === 0 && result?.valid && (
              <AlertRow level="success" title="검증 통과">
                설정 형식, 값 범위, 경로, 실행 이름 중복 검사에서 문제가 발견되지 않았습니다.
              </AlertRow>
            )}
            {saveError && (
              <AlertRow level="error" title="저장 실패">
                {saveError}
              </AlertRow>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <Button
              kind="primary"
              disabled={!canSave}
              onClick={() => void save()}
              style={{ padding: '9px 0', borderRadius: radius.control }}
              title={canSave ? undefined : '검증을 통과해야 저장할 수 있습니다.'}
            >
              {saving ? '저장 중…' : '설정 저장 후 검토 →'}
            </Button>
            <span
              style={{
                font: `400 11.5px/1.5 ${font.sans}`,
                color: color.textMuted,
                textAlign: 'center',
              }}
            >
              설정을 저장하지 않으면 학습을 시작할 수 없습니다.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function TrainField({
  spec,
  value,
  error,
  devices,
  onChange,
  note,
}: {
  spec: FieldSpec;
  value: string;
  error?: string;
  devices: Defaults['devices'];
  onChange: (value: string) => void;
  /** 힌트 대신 보여 줄 한 줄. 지금은 자동으로 지어진 실행 이름에만 씁니다. */
  note?: string;
}) {
  const style = error ? invalidControlStyle : controlStyle;
  const placeholder =
    spec.placeholder ?? (spec.default !== undefined && spec.default !== null ? `기본값 ${String(spec.default)}` : '');

  if (spec.name === 'device') {
    // 고를 수 있는 목록은 devices에서 오지만 **출발값은 spec.default**입니다. 여기에
    // cpu를 박아 두면 GPU가 있는 컴퓨터에서 서버가 cuda를 내려줘도 화면이 늘 cpu로
    // 시작하고, 바꾸는 것을 잊은 학습이 몇 분에서 몇 시간짜리가 됩니다.
    const fallback = typeof spec.default === 'string' ? spec.default : 'cpu';
    return (
      <Field label={spec.label} hint={spec.hint} error={error}>
        <select value={value || fallback} onChange={(event) => onChange(event.target.value)} style={style}>
          {devices.map((device) => (
            <option key={device.value} value={device.value} disabled={!device.available}>
              {device.value}
              {device.available ? '' : ` (${device.reason ?? '사용 불가'})`}
            </option>
          ))}
        </select>
      </Field>
    );
  }

  if (spec.type === 'enum') {
    const selected = value || (typeof spec.default === 'string' ? spec.default : '');
    return (
      <Field label={spec.label} hint={spec.hint} error={error}>
        <select value={selected} onChange={(event) => onChange(event.target.value)} style={style}>
          {(spec.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>{choice}</option>
          ))}
        </select>
      </Field>
    );
  }

  if (spec.type === 'boolean') {
    // 서버가 알려 준 기본값으로 시작합니다. 'false'로 못박아 두면 기본값을 바꿔도
    // 화면이 따라가지 않습니다.
    const selected = value || String(spec.default === true);
    return (
      <Field label={spec.label} hint={spec.hint} error={error}>
        <select value={selected} onChange={(event) => onChange(event.target.value)} style={style}>
          <option value="false">사용하지 않음</option>
          <option value="true">사용함</option>
        </select>
      </Field>
    );
  }

  return (
    <Field label={spec.label} hint={note ?? spec.hint} error={error}>
      <input
        value={value}
        placeholder={placeholder}
        inputMode={spec.type === 'integer' || spec.type === 'number' ? 'decimal' : 'text'}
        onChange={(event) => onChange(event.target.value)}
        style={style}
      />
    </Field>
  );
}
