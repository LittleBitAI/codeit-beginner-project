/**
 * 모델 칸과 backbone 칸을 architecture 이름 하나로 오가는 표 조회입니다.
 *
 * 저장하고 서버에 보내는 값은 **언제나 architecture 이름 하나**입니다. 두 칸은 그
 * 이름을 나눠 보여 줄 뿐이고, 어느 쪽을 바꾸든 다시 이름 하나로 합쳐 올립니다. 값을
 * 둘로 늘리면 서로 어긋날 수 있고, 어긋난 쪽은 멈추지 않고 점수만 나빠집니다.
 */

/** `{ dino: { resnet50: 'dino_r50_4scale', ... } }` 꼴입니다. 서버가 내려 줍니다. */
export type BackboneTable = Record<string, Record<string, string>>;

/** architecture가 속한 갈래 이름입니다. 접히지 않는 모델이면 undefined입니다. */
export function familyOf(table: BackboneTable, architecture: string): string | undefined {
  return Object.keys(table).find((family) =>
    Object.values(table[family] ?? {}).includes(architecture),
  );
}

/** architecture가 쓰는 backbone 이름입니다. */
export function backboneOf(table: BackboneTable, architecture: string): string | undefined {
  const family = familyOf(table, architecture);
  if (family === undefined) return undefined;
  const entries = table[family] ?? {};
  return Object.keys(entries).find((backbone) => entries[backbone] === architecture);
}

/**
 * 모델 칸에 세울 목록입니다. 갈래에 속한 이름들을 갈래 이름 하나로 접습니다.
 *
 * **접는 일은 화면에서만 합니다.** 서버가 내려 주는 `choices`는 계약의 진짜
 * architecture 이름 그대로여야 합니다 — 거기에 `dino` 같은 접힌 이름을 실으면 그
 * 목록을 그대로 보내는 다른 소비자가 서버에게 거절당합니다.
 */
export function displayChoices(table: BackboneTable, choices: string[]): string[] {
  const shown: string[] = [];
  for (const choice of choices) {
    const label = familyOf(table, choice) ?? choice;
    if (!shown.includes(label)) shown.push(label);
  }
  return shown;
}

/** 갈래와 backbone을 architecture 이름으로 합칩니다. 표에 없으면 undefined입니다. */
export function architectureOf(
  table: BackboneTable,
  family: string,
  backbone: string,
): string | undefined {
  return (table[family] ?? {})[backbone];
}

/**
 * 갈래를 골랐을 때 놓을 architecture입니다. 지금 값이 그 갈래면 그대로 두어, 모델 칸을
 * 건드리지 않았는데 backbone이 기본값으로 되돌아가는 일이 없게 합니다.
 */
export function architectureForFamily(
  table: BackboneTable,
  defaults: Record<string, string>,
  family: string,
  current: string,
): string {
  if (table[family] === undefined) return family;
  if (familyOf(table, current) === family) return current;
  const fallback = defaults[family];
  return (
    architectureOf(table, family, fallback ?? '') ?? Object.values(table[family])[0] ?? family
  );
}
