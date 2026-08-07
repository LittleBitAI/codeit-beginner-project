import type { RuntimeConfig } from '../api/types';

/**
 * 설정을 한국어 문장으로 풀어 씁니다.
 *
 * 설정 검토 화면에서 딥러닝을 모르는 사람도 무엇이 실행될지 읽을 수 있게 하는 것이
 * 목적입니다. 값은 전부 config에서 오고, 없는 값을 지어내지 않습니다.
 */
export function describeRun(config: RuntimeConfig | null): string {
  if (!config) return '설정이 아직 준비되지 않았습니다.';

  const train = config.train as Record<string, unknown>;
  const data = config.inputs?.data ?? {};
  const device = train.device === 'cuda' ? 'CUDA GPU' : 'CPU';
  const start = train.pretrained === true ? 'COCO 사전학습 가중치' : '무작위 초기 가중치';
  const backend = (config.storage as { backend?: string }).backend === 's3' ? 'S3' : '로컬 디스크';
  const sources = Object.keys(data).length;
  const earlyStopping = train.early_stopping as { patience?: unknown; min_delta?: unknown } | null;
  // 켰을 때만 말합니다. 쓰지 않는 설명을 붙이면 안 쓰는 기능을 쓰는 줄 압니다.
  const stopSentence = earlyStopping
    ? `검증 손실이 ${String(earlyStopping.min_delta)}보다 크게 좋아지지 않는 상태가 ` +
      `${String(earlyStopping.patience)} epoch 이어지면 남은 epoch를 채우지 않고 조기 종료합니다. `
    : '';

  return (
    `data pipeline이 만든 artifact ${sources}개(학습 manifest, 검증 manifest, 클래스 맵, 데이터셋 요약)로 ` +
    `torchvision Faster R-CNN 모델을 ${start}에서 시작해 ${String(train.epochs)} epoch 동안 학습합니다. ` +
    `batch ${String(train.batch_size)}, SGD optimizer(learning rate ${String(train.learning_rate)}, ` +
    `momentum ${String(train.momentum)}, weight decay ${String(train.weight_decay)})를 쓰고, ` +
    `random seed는 ${String(train.seed)}이라 같은 데이터면 같은 결과가 나옵니다. ` +
    `DataLoader worker는 ${String(train.num_workers)}개이며 실행 대상은 ${device}입니다. ` +
    stopSentence +
    `결과 checkpoint와 학습 이력은 ${backend}의 '${String(train.output_dir)}/${String(train.run_id)}'에 저장됩니다.`
  );
}

/** 기본값과 다른 항목만 뽑아 diff로 보여 줍니다. */
export interface ConfigDiffRow {
  key: string;
  before: string;
  after: string;
}

export function diffAgainstDefaults(
  train: Record<string, unknown>,
  defaults: Record<string, unknown>,
): ConfigDiffRow[] {
  const rows: ConfigDiffRow[] = [];
  for (const [key, fallback] of Object.entries(defaults)) {
    if (!(key in train)) continue;
    const current = train[key];
    if (String(current) === String(fallback)) continue;
    rows.push({ key: `train.${key}`, before: String(fallback), after: String(current) });
  }
  return rows;
}
