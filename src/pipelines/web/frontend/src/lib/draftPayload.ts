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

  for (const spec of fields) {
    const raw = draft.train[spec.name];
    if (raw === undefined || raw.trim() === '') {
      // 비워 두면 backend의 기본값을 씁니다.
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
