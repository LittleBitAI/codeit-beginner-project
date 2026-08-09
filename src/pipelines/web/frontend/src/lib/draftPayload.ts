import type { ConfigDraftPayload } from '../api/client';
import type { FieldSpec } from '../api/types';
import type { Draft } from '../state/DraftContext';

const INTEGER_PATTERN = /^-?\d+$/;
const NUMBER_PATTERN = /^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$/;

/** 조기 종료를 껐을 때 함께 사라지는 칸입니다. */
export const EARLY_STOPPING_FIELDS = ['early_stopping_patience', 'early_stopping_min_delta'];

/**
 * 조기 종료 스위치의 현재 값입니다. 손대지 않았으면 서버가 알려 준 기본값입니다.
 *
 * 화면의 숨김 규칙과 payload의 제외 규칙이 항상 같아야 해서 한 곳에 둡니다.
 * 숨겨 놓고 값을 실어 보내면 서버가 "쓰지 않는 값"이라며 저장을 막습니다.
 */
export function isEarlyStoppingOn(
  train: Record<string, string>,
  fields: FieldSpec[],
): boolean {
  const spec = fields.find((item) => item.name === 'early_stopping');
  const value = train.early_stopping?.trim() || String(spec?.default === true);
  return value === 'true';
}

/**
 * form의 문자열 값을 backend가 기대하는 타입으로 바꿉니다.
 *
 * 숫자로 보이지 않는 값은 **바꾸지 않고 문자열 그대로 보냅니다**. ``parseInt('3abc')``가
 * 3이 되는 식으로 조용히 통과시키면, train이 거부할 값을 GUI가 받아들이게 됩니다.
 * 서버의 검증이 유일한 판단 기준이어야 합니다.
 */
export function toPayload(draft: Draft, fields: FieldSpec[]): ConfigDraftPayload {
  const train: Record<string, unknown> = {};
  const optimizerSpec = fields.find((spec) => spec.name === 'optimizer');
  const optimizer =
    draft.train.optimizer?.trim() ||
    (typeof optimizerSpec?.default === 'string' ? optimizerSpec.default : 'SGD');
  const irrelevant =
    optimizer === 'SGD'
      ? new Set(['beta1', 'beta2', 'epsilon'])
      : new Set(['momentum']);
  if (!isEarlyStoppingOn(draft.train, fields)) {
    for (const name of EARLY_STOPPING_FIELDS) irrelevant.add(name);
  }

  for (const spec of fields) {
    if (irrelevant.has(spec.name)) continue;
    const raw = draft.train[spec.name];
    if (raw === undefined || raw.trim() === '') {
      // 새 enum 선택은 명시적으로 저장해 legacy config와 구분합니다.
      //
      // device도 예외가 아닙니다. 예전에는 device만 빼고 보냈는데, GPU가 있는
      // 컴퓨터에서 서버가 device=cuda와 precision=amp를 짝으로 내려 주면서
      // 화면에는 cuda가 보이는데 payload에는 없는 상태가 됐습니다. 서버는 값이
      // 없으면 train 기본값 cpu를 쓰므로, 폼을 열자마자 "amp 정밀도는 device가
      // cuda일 때만" 오류가 떴습니다. **화면이 보여 주는 값을 그대로 보냅니다.**
      if (spec.type === 'enum' && typeof spec.default === 'string') {
        train[spec.name] = spec.default;
      }
      // boolean도 마찬가지입니다. 손대지 않았다고 빼 버리면 서버 fallback이
      // 화면이 안내한 기본값을 덮어씁니다.
      if (spec.type === 'boolean' && typeof spec.default === 'boolean') {
        train[spec.name] = spec.default;
      }
      continue;
    }
    const text = raw.trim();

    switch (spec.type) {
      case 'integer':
        train[spec.name] = INTEGER_PATTERN.test(text) ? Number.parseInt(text, 10) : text;
        break;
      case 'number':
        train[spec.name] = NUMBER_PATTERN.test(text) ? Number.parseFloat(text) : text;
        break;
      case 'boolean':
        train[spec.name] = text === 'true';
        break;
      default:
        train[spec.name] = text;
    }
  }

  const data: Record<string, string> = {};
  for (const [name, value] of Object.entries(draft.data)) {
    if (value.trim() !== '') data[name] = value.trim();
  }

  return { train, inputs: { data } };
}

/** 특정 field에 붙은 오류 메시지를 찾습니다. */
export function messageFor(
  messages: { field: string; message: string }[],
  field: string,
): string | undefined {
  return messages.find((item) => item.field === field)?.message;
}
