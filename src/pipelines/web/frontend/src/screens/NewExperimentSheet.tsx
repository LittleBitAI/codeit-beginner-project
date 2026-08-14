/**
 * 새 실험을 만드는 시트입니다.
 *
 * 예전에는 "설정 만들기 → 검토 → 시작"이 화면 세 개였습니다. 한 판으로 합친
 * 이유는 그 셋이 한 번의 결정이기 때문입니다. 대신 **저장·검증은 그대로 서버가**
 * 합니다: 시작을 누르면 설정을 먼저 만들고, 서버가 거부하면 시작하지 않습니다.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type { DataSource, Defaults, FieldSpec, ValidationResult } from '../api/types';
import {
  AlertRow,
  Button,
  Chip,
  Field,
  MicroLabel,
  Sheet,
  controlStyle,
  invalidControlStyle,
} from '../components/primitives';
import { color, font, type } from '../design/tokens';
import { dataMatchesSource } from '../lib/dataSource';
import {
  EARLY_STOPPING_FIELDS,
  LR_FIELDS,
  isEarlyStoppingOn,
  lrFieldsFor,
  messageFor,
  selectedArchitecture,
  selectedSchedule,
  toPayload,
} from '../lib/draftPayload';
import { datasetLabel } from '../lib/runSpec';
import { resolveTrainCapability } from '../lib/trainCapabilities';
import { useDraft } from '../state/DraftContext';
import { useTeam } from '../team/TeamContext';

type TabKey = 'basic' | 'hyper' | 'output';

const TABS: { key: TabKey; label: string; fields: string[] }[] = [
  {
    key: 'basic',
    label: '기본',
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
    label: '고급',
    fields: [
      'epochs',
      'batch_size',
      // GPU 메모리가 모자라 batch size를 못 올릴 때 쓰는 값이라 그 옆에 둡니다.
      'gradient_accumulation_steps',
      // MMDetection 모델만 쓰는 값입니다. 다른 모델을 고르면 서버가 거부합니다.
      'input_size',
      'learning_rate',
      'momentum',
      'weight_decay',
      'beta1',
      'beta2',
      'epsilon',
      'num_workers',
      'lr_scheduler',
      'lr_warmup_steps',
      'lr_warmup_start_factor',
      'lr_min_factor',
      'lr_step_size',
      'lr_gamma',
      'early_stopping',
      'early_stopping_patience',
      'early_stopping_min_delta',
    ],
  },
  { key: 'output', label: '출력', fields: ['output_dir', 'output_prefix'] },
];

/** 표에 세울 순서. 화면에서 채운 순서 그대로 읽히게 둡니다. */
const TAB_FIELD_ORDER = TABS.flatMap((tab) => tab.fields);

/**
 * 보낼 설정을 표 한 장으로 펼칩니다.
 *
 * 값은 서버가 정규화한 것을 그대로 씁니다. 화면의 draft를 쓰면 비워 둔 칸이 빈
 * 줄로 나오는데, 실제로 학습에 쓰이는 것은 서버가 채운 기본값입니다.
 *
 * `lr_scheduler`처럼 값이 다시 묶음인 칸은 한 겹 펼쳐 `lr_scheduler.name`으로
 * 적습니다. `[object Object]`는 아무것도 말해 주지 않습니다.
 */
function settingRows(train: Record<string, unknown>): { key: string; value: string }[] {
  const ordered = [
    ...TAB_FIELD_ORDER.filter((name) => name in train),
    ...Object.keys(train).filter((name) => !TAB_FIELD_ORDER.includes(name)),
  ];
  // 이름은 창 제목에 이미 크게 있습니다.
  return ordered
    .filter((name) => name !== 'run_id')
    .flatMap((name) => {
      const value = train[name];
      if (value !== null && typeof value === 'object') {
        return Object.entries(value as Record<string, unknown>).map(([inner, nested]) => ({
          key: `${name}.${inner}`,
          value: String(nested),
        }));
      }
      return [{ key: name, value: String(value) }];
    });
}

/**
 * 시작하기 전에 설정을 한 번 더 펼쳐 보여 줍니다.
 *
 * 시트를 닫고 나면 무엇으로 돌고 있는지 다시 볼 자리가 없어, "내가 뭘 세팅했는지
 * 모르겠다"는 말이 나왔습니다. 예전에는 이 자리에 설정을 문장으로 풀어 쓴 문단이
 * 있었지만 아무도 읽지 않았습니다 — 값을 확인하는 일에는 표가 맞습니다.
 */
function ConfirmStart({
  runId,
  dataset,
  train,
  mode,
  pending,
  onCancel,
  onConfirm,
}: {
  runId: string;
  /** 어느 데이터셋으로 도는지. 설정 하나하나보다 먼저 확인할 값입니다. */
  dataset: string | null;
  train: Record<string, unknown>;
  mode: 'start' | 'queue';
  pending: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  // 창이 포커스를 받아야 ESC가 닿습니다. 열기 전에 눌렀던 단추에 포커스가 남아 있으면
  // 아래 onKeyDown은 한 번도 실행되지 않습니다. 닫을 때는 눌렀던 자리로 돌려줍니다 —
  // 그러지 않으면 키보드로 쓰는 사람이 하던 자리를 잃고 문서 맨 위에서 다시 찾습니다.
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const opener = document.activeElement;
    dialogRef.current?.focus();
    return () => {
      if (opener instanceof HTMLElement) opener.focus();
    };
  }, []);

  // 원시 값 둘로는 실제 갱신 규모가 보이지 않습니다. 지운 설명 문단이 계산해 주던 값이라
  // 표에서 사라지면 그대로 잃습니다.
  const accumulation = Number(train.gradient_accumulation_steps ?? 1);
  const effectiveBatch =
    accumulation > 1 ? Number(train.batch_size) * accumulation : null;

  return (
    <>
      <div
        onClick={onCancel}
        style={{ position: 'fixed', inset: 0, background: 'rgba(8,6,4,.55)', zIndex: 65 }}
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={`${runId} 시작 확인`}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === 'Escape') onCancel();
        }}
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          zIndex: 70,
          width: 'min(520px, calc(100vw - 40px))',
          background: color.sheet,
          border: `1px solid ${color.border}`,
          borderRadius: 4,
          padding: '20px 22px',
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
        }}
      >
        <span style={{ ...type.subTitle, color: color.text }}>이 설정으로 시작할까요?</span>
        <div style={{ ...type.monoValue, color: color.textStrong, overflowWrap: 'anywhere' }}>
          {runId}
        </div>
        {dataset && (
          <div style={{ ...type.monoSpec, color: color.textMuted, overflowWrap: 'anywhere' }}>
            {dataset}
          </div>
        )}
        <div style={{ maxHeight: 320, overflow: 'auto', borderTop: `1px solid ${color.border}` }}>
          {settingRows(train).map((row) => (
            <div
              key={row.key}
              style={{
                display: 'flex',
                gap: 16,
                justifyContent: 'space-between',
                padding: '7px 0',
                borderBottom: `1px solid ${color.borderRow}`,
              }}
            >
              <span style={{ ...type.monoSpec, color: color.textMuted }}>{row.key}</span>
              <span
                style={{
                  ...type.monoSpec,
                  color: color.text,
                  textAlign: 'right',
                  overflowWrap: 'anywhere',
                }}
              >
                {row.value}
              </span>
            </div>
          ))}
        </div>
        {effectiveBatch !== null && (
          <div style={{ ...type.bodySmall, color: color.textBody }}>
            유효 batch {effectiveBatch} — batch {String(train.batch_size)}개를 {accumulation}번
            모아 한 번씩 가중치를 갱신합니다.
          </div>
        )}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <Button kind="ghost" onClick={onCancel} disabled={pending}>
            다시 고치기
          </Button>
          <Button kind="primary" onClick={onConfirm} disabled={pending}>
            {pending ? '보내는 중…' : mode === 'start' ? '시작' : '대기열에 넣습니다'}
          </Button>
        </div>
      </div>
    </>
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
    spec.placeholder ??
    (spec.default !== undefined && spec.default !== null ? `기본값 ${String(spec.default)}` : '');

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
            <option key={choice} value={choice}>
              {choice}
            </option>
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

export function NewExperimentSheet({
  defaults,
  source,
  datasetKey,
  queuedCount,
  busy,
  onClose,
  onStarted,
}: {
  defaults: Defaults | null;
  source: DataSource | null;
  datasetKey: string | null;
  /** 지금 줄 서 있는 학습 수. 이 설정이 몇 번째로 들어가는지 말해 줍니다. */
  queuedCount: number;
  /** 지금 도는 학습이 있는지. 있으면 바로 시작할 수 없습니다. */
  busy: boolean;
  onClose: () => void;
  onStarted: () => void;
}) {
  const navigate = useNavigate();
  const team = useTeam();
  const { draft, setTrainField, setDataField, setDataFields, setSaved } = useDraft();
  const [tab, setTab] = useState<TabKey>('basic');
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [showJson, setShowJson] = useState(false);
  const [pending, setPending] = useState<'start' | 'queue' | null>(null);
  const [error, setError] = useState<string | null>(null);
  // 시작하기 전에 보낼 설정을 한 번 더 펼쳐 보여 줍니다.
  const [confirming, setConfirming] = useState<'start' | 'queue' | null>(null);

  const [validatedKey, setValidatedKey] = useState<string | null>(null);

  const fields = defaults?.fields ?? [];
  const payload = useMemo(() => toPayload(draft, fields), [draft, fields]);
  // 확인 창은 **마지막 검증 결과**를 보여 주고 만들기는 **지금 payload**를 보냅니다.
  // 둘이 어긋난 채로 시작할 수 있으면 확인한 것과 다른 설정으로 학습이 도므로, 검증이
  // 따라잡을 때까지(250ms) 시작을 잠급니다.
  const payloadKey = useMemo(() => JSON.stringify(payload), [payload]);

  // 고른 데이터셋과 지금 칸의 값이 다른지 확인합니다. 실제로 화면에는 새 데이터셋이
  // 보이는데 예전 데이터로 학습된 적이 있어서, 다르면 눈에 띄게 알립니다.
  const mismatched = !dataMatchesSource(draft.data, source);

  // 입력이 멈추면 서버에 검증을 맡깁니다. 판단 기준은 언제나 서버입니다.
  //
  // 이미 떠난 요청은 취소할 수 없으므로 **늦게 온 답을 버립니다.** 그러지 않으면 옛
  // 설정의 답이 새 설정의 답을 덮어써, 지금 화면의 값은 멀쩡한데 시작 단추가 잠긴 채로
  // 남습니다. 그 상태는 다시 무언가를 고치기 전까지 풀리지 않습니다.
  useEffect(() => {
    if (!defaults) return;
    let live = true;
    const timer = window.setTimeout(() => {
      void api
        .validate(payload)
        .then((value) => {
          if (!live) return;
          setResult(value);
          setValidatedKey(payloadKey);
        })
        .catch(() => {
          if (live) setResult(null);
        });
    }, 250);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [payload, payloadKey, defaults]);

  // 확인 창이 떠 있는 동안 값이 바뀌면 창을 닫습니다. 창은 마지막 검증 결과를 보여 주고
  // 만들기는 지금 payload를 보내므로, 열어 둔 채로 두면 확인한 것과 다른 설정이 실려
  // 갑니다. 뒤쪽 칸은 창이 떠 있어도 키보드로 닿습니다.
  useEffect(() => {
    setConfirming(null);
  }, [payloadKey]);

  if (!defaults) {
    return (
      <Sheet title="새 실험" onClose={onClose}>
        <div style={{ ...type.body, color: color.textMuted }}>설정 정보를 불러오고 있습니다.</div>
      </Sheet>
    );
  }

  const errors = result?.errors ?? [];
  const warnings = result?.warnings ?? [];
  // 서버가 지어 준 이름입니다. 규칙은 backend 한 곳에만 있습니다.
  const autoRunId =
    typeof result?.normalized?.train?.run_id === 'string' ? result.normalized.train.run_id : null;
  const dataFilled = defaults.data_fields.every(
    (spec) => (draft.data[spec.name] ?? '').trim() !== '',
  );
  const ready =
    Boolean(result?.valid) && dataFilled && validatedKey === payloadKey && pending === null;

  const capability = resolveTrainCapability(defaults);
  const selectedOptimizer = draft.train.optimizer || capability.optimizer.default;
  // 감추는 규칙과 payload에서 빼는 규칙이 같은 모델 이름을 봐야 합니다. 둘이 어긋나면
  // 화면에 없는 값이 실려 가고, 사용자는 오류가 난 칸을 찾지 못합니다.
  const architecture = selectedArchitecture(draft.train, fields) || capability.model.default;
  const earlyStoppingOn = isEarlyStoppingOn(draft.train, fields);
  // 고른 schedule이 쓰지 않는 칸은 감춥니다. 보이면 그 값이 학습에 쓰이는 것처럼
  // 읽히고, 서버도 쓰지 않는 값이라며 거부합니다. payload의 제외 규칙과 같은 함수를
  // 씁니다 — 둘이 어긋나면 화면에 없는 값 때문에 저장이 막힙니다.
  const shownLrFields = new Set(lrFieldsFor(selectedSchedule(draft.train, fields)));
  const activeTab = TABS.find((item) => item.key === tab) ?? (TABS[0] as (typeof TABS)[number]);
  const tabErrorCount = (item: (typeof TABS)[number]) =>
    item.fields.filter((name) => messageFor(errors, `train.${name}`) !== undefined).length;
  // data 칸은 기본 표에만 그려집니다. 다른 표를 보고 있으면 그 오류는 화면에 없습니다.
  const hiddenErrors = errors.filter((item) =>
    item.field.startsWith('inputs.data.')
      ? tab !== 'basic'
      : !activeTab.fields.includes(item.field.replace(/^train\./, '')),
  );

  /** 설정을 만든 뒤 곧바로 시작하거나 줄을 세웁니다. 만들기가 실패하면 아무것도 하지 않습니다. */
  async function submit(mode: 'start' | 'queue') {
    setPending(mode);
    setError(null);
    try {
      const created = await api.createConfig(payload);
      setSaved(created);
      const token = await team.getAccessToken();
      if (mode === 'start') {
        const job = await api.startJob(created.config_id, token);
        onStarted();
        onClose();
        navigate(`/monitor/${job.job_id}`);
        return;
      }
      const queue = await api.addToQueue(created.config_id, token);
      onStarted();
      onClose();
      // 비어 있었으면 곧바로 시작되므로 그 학습 화면으로, 줄을 섰으면 목록에 남습니다.
      if (queue.started) navigate(`/monitor/${queue.started.job_id}`);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : mode === 'start'
            ? '학습을 시작하지 못했습니다.'
            : '대기열에 넣지 못했습니다.',
      );
      // 확인 창을 닫아야 시트에 붙은 오류가 보입니다.
      setConfirming(null);
    } finally {
      setPending(null);
    }
  }

  return (
    <Sheet
      title="새 실험"
      subtitle={datasetKey ?? '데이터셋을 아직 고르지 않았습니다'}
      onClose={onClose}
      footer={
        <>
          <Button kind="primary" disabled={!ready} onClick={() => setConfirming('queue')}>
            {pending === 'queue' ? '넣는 중…' : '대기열에 추가'}
          </Button>
          <Button
            kind="secondary"
            disabled={!ready || busy}
            title={busy ? '다른 학습이 도는 중이라 바로 시작할 수 없습니다' : undefined}
            onClick={() => setConfirming('start')}
          >
            {pending === 'start' ? '시작하는 중…' : '바로 시작'}
          </Button>
          <Button kind="ghost" onClick={() => setShowJson((value) => !value)}>
            JSON 보기
          </Button>
        </>
      }
    >
      <div style={{ display: 'flex', gap: 8, marginBottom: 26, flexWrap: 'wrap' }}>
        {TABS.map((item) => (
          <Chip
            key={item.key}
            active={item.key === tab}
            count={tabErrorCount(item) > 0 ? `!${tabErrorCount(item)}` : item.fields.length}
            onClick={() => setTab(item.key)}
          >
            {item.label}
          </Chip>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '22px 18px', marginBottom: 30 }}>
        {activeTab.fields.map((name) => {
          if (selectedOptimizer === 'SGD' && ['beta1', 'beta2', 'epsilon'].includes(name)) return null;
          if (selectedOptimizer !== 'SGD' && name === 'momentum') return null;
          // 조기 종료를 끄면 관련 숫자 칸도 감춥니다. 보이면 그 값이 학습에 쓰이는
          // 것처럼 읽히고, 서버도 쓰지 않는 값이라며 거부합니다.
          if (!earlyStoppingOn && EARLY_STOPPING_FIELDS.includes(name)) return null;
          if (LR_FIELDS.includes(name) && !shownLrFields.has(name)) return null;
          const spec = fields.find((item) => item.name === name);
          if (!spec) return null;
          // 이 칸을 쓰지 않는 모델에서는 감춥니다. 어떤 모델이 쓰는지는 서버가
          // 알려 줍니다. 여기에 옮겨 적으면 목록이 어긋나도 아무도 모릅니다.
          if (spec.only_for_architectures && !spec.only_for_architectures.includes(architecture)) {
            return null;
          }
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
                    ? `비워 두면 ${autoRunId}`
                    : undefined
                }
              />
            );
          }
          // 기본값이 고른 optimizer나 모델에 따라 달라지는 칸이 있습니다. 하나만
          // 보여 주면 비워 둔 사람에게 실제와 다른 값을 안내합니다.
          const variableDefault =
            spec.defaults_by_optimizer?.[selectedOptimizer] ??
            spec.defaults_by_architecture?.[architecture];
          return (
            <TrainField
              key={name}
              spec={variableDefault === undefined ? spec : { ...spec, default: variableDefault }}
              value={draft.train[name] ?? ''}
              error={messageFor(errors, `train.${name}`)}
              devices={defaults.devices}
              onChange={(value) => setTrainField(name, value)}
            />
          );
        })}
      </div>

      {tab === 'basic' && (
        <div style={{ paddingTop: 4, marginBottom: 26 }}>
          <MicroLabel style={{ marginBottom: 16 }}>이 학습이 읽을 데이터</MicroLabel>
          {source?.complete && (
            <div style={{ ...type.monoSpec, color: color.textMuted, marginBottom: 12, overflowWrap: 'anywhere' }}>
              전처리 데이터셋에서 자동으로 채움 · {source.directory}
            </div>
          )}
          {!source?.complete && (
            <div style={{ ...type.note, color: color.textMuted, marginBottom: 12 }}>
              왼쪽 <b style={{ color: color.textBody }}>dataset 준비</b>에서 전처리 폴더를 고르면 이
              네 칸이 자동으로 채워집니다.
            </div>
          )}
          {mismatched && source && (
            <div style={{ marginBottom: 12 }}>
              <AlertRow
                level="warning"
                title="고른 데이터셋과 아래 값이 다릅니다"
                action={
                  <Button
                    onClick={() => setDataFields({ ...source.data }, null)}
                    title="고른 데이터셋의 값으로 맞춥니다"
                  >
                    맞추기
                  </Button>
                }
              >
                이대로 학습하면 <b>아래 칸에 적힌 데이터</b>로 돌아갑니다.
              </AlertRow>
            </div>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
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
                    messageFor(errors, `inputs.data.${spec.name}`) ? invalidControlStyle : controlStyle
                  }
                />
              </Field>
            ))}
          </div>
          <div style={{ ...type.note, color: color.textMuted, marginTop: 12 }}>
            저장소 기준 상대 경로나 <code style={{ fontFamily: font.mono }}>s3://bucket/key</code>{' '}
            형식만 받습니다. 절대 경로와 <code style={{ fontFamily: font.mono }}>..</code>는
            거부됩니다.
          </div>
        </div>
      )}

      {tab === 'output' && (
        <div style={{ ...type.note, color: color.textMuted, marginBottom: 26 }}>
          data 입력이 <code style={{ fontFamily: font.mono }}>s3://</code>면 S3 backend로, 아니면
          로컬 디스크로 저장합니다. S3 bucket 이름은 환경 변수에서 읽으며 설정 파일에 남기지
          않습니다.
        </div>
      )}

      {/* 지금 보이는 칸의 오류는 그 칸 아래에 이미 붙어 있습니다. 여기에는 **다른 표에
          가려진** 오류만 모읍니다 — 같은 말을 두 번 적으면 시트가 오류로 뒤덮입니다. */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 26 }}>
        {hiddenErrors.map((item) => (
          <AlertRow key={`${item.field}-${item.message}`} level="error" title={item.field}>
            {item.message}
          </AlertRow>
        ))}
        {errors.length === 0 && !dataFilled && (
          <AlertRow level="warning" title="data artifact 위치가 비어 있습니다">
            네 값을 모두 채워야 시작할 수 있습니다.
          </AlertRow>
        )}
        {errors.length === 0 &&
          dataFilled &&
          warnings.map((item) => (
            <AlertRow key={`${item.field}-${item.message}`} level="warning" title={item.field}>
              {item.message}
            </AlertRow>
          ))}
        {error && (
          <AlertRow level="error" title="시작하지 못했습니다">
            {error}
          </AlertRow>
        )}
      </div>

      <div style={{ paddingTop: 26, borderTop: `1px solid ${color.border}` }}>
        <MicroLabel style={{ marginBottom: 16 }}>시작하면</MicroLabel>
        <div style={{ ...type.monoValue, color: color.textStrong, marginBottom: 14, overflowWrap: 'anywhere' }}>
          {draft.train.run_id?.trim() || autoRunId || '이름은 저장할 때 지어집니다'}
        </div>
        <div style={{ ...type.body, color: color.textBody, maxWidth: '34em', textWrap: 'pretty' }}>
          {busy ? (
            <>
              지금 학습 하나가 돌고 있어{' '}
              <b style={{ fontWeight: 600, color: color.accent }}>대기열 {queuedCount + 1}번</b>으로
              들어갑니다. 앞의 {queuedCount}건이 끝나면 시작합니다.
            </>
          ) : queuedCount > 0 ? (
            <>
              대기열에 {queuedCount}건이 있어 <b style={{ fontWeight: 600, color: color.accent }}>
                {queuedCount + 1}번
              </b>
              으로 들어갑니다.
            </>
          ) : (
            '지금 도는 학습이 없어 대기열에 넣으면 곧바로 시작합니다.'
          )}
        </div>
      </div>

      {showJson && (
        <div style={{ background: color.panel, padding: '18px 20px', marginTop: 22 }}>
          <MicroLabel style={{ marginBottom: 12 }}>생성될 설정</MicroLabel>
          <pre
            style={{
              ...type.code,
              color: color.textBody,
              margin: 0,
              maxHeight: 320,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              overflowWrap: 'break-word',
            }}
          >
            {JSON.stringify(result?.normalized ?? payload, null, 2)}
          </pre>
        </div>
      )}

      {confirming && result?.normalized && (
        <ConfirmStart
          runId={draft.train.run_id?.trim() || autoRunId || '(이름은 저장할 때 지어집니다)'}
          dataset={datasetLabel(result.normalized.inputs?.data)}
          train={result.normalized.train as Record<string, unknown>}
          mode={confirming}
          pending={pending !== null}
          onCancel={() => setConfirming(null)}
          onConfirm={() => void submit(confirming)}
        />
      )}
    </Sheet>
  );
}
