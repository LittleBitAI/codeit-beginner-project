import type { ConfigDraftPayload } from '../api/client';
import type { FieldSpec } from '../api/types';
import type { Draft } from '../state/DraftContext';

const INTEGER_PATTERN = /^-?\d+$/;
const NUMBER_PATTERN = /^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$/;

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

  for (const spec of fields) {
    if (irrelevant.has(spec.name)) continue;
    const raw = draft.train[spec.name];
    if (raw === undefined || raw.trim() === '') {
      // 새 enum 선택은 명시적으로 저장해 legacy config와 구분합니다.
      if (
        spec.type === 'enum' &&
        spec.name !== 'device' &&
        typeof spec.default === 'string'
      ) {
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
