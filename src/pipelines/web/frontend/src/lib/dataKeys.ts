/** train이 요구하는 data artifact 4개. 화면에 보여 줄 순서이기도 합니다. */
export const DATA_KEYS = [
  'train_manifest_uri',
  'validation_manifest_uri',
  'class_map_uri',
  'dataset_summary_uri',
] as const;

export type DataKey = (typeof DATA_KEYS)[number];
