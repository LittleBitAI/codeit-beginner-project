/** train이 요구하는 data artifact 4개. 화면에 보여 줄 순서이기도 합니다. */
export const DATA_KEYS = [
  'train_manifest_uri',
  'validation_manifest_uri',
  'class_map_uri',
  'dataset_summary_uri',
] as const;

export type DataKey = (typeof DATA_KEYS)[number];

/** Train에는 선택이지만 Evaluate의 대회 submission 생성에 필요합니다. */
export const OPTIONAL_DATA_KEYS = ['test_manifest_uri'] as const;

export const ALL_DATA_KEYS = [...DATA_KEYS, ...OPTIONAL_DATA_KEYS] as const;
