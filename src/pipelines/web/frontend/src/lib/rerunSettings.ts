import type { ExperimentSummary } from '../api/types';

/**
 * 끝난 실행의 설정을 새 실험 화면이 쓰는 평평한 초안 값으로 되돌립니다.
 *
 * 화면의 칸은 평평하고(`lr_warmup_steps`) train이 받는 config는 중첩이라
 * (`lr_scheduler.warmup_steps`), 그 사이를 되돌리는 곳이 여기 하나입니다.
 * `train_config.py`의 `_LR_FIELDS`가 가는 방향이고 이 파일이 오는 방향입니다.
 *
 * **dataset은 담지 않습니다.** 어떤 데이터로 학습할지는 dataset 준비에서 고르는
 * 것이고, 남의 실행을 눌렀다는 이유로 내 데이터 선택이 바뀌면 안 됩니다.
 */

/** 화면의 평평한 칸 이름 <- `lr_scheduler` 안쪽 key. */
const LR_FIELDS: Record<string, string> = {
  warmup_steps: 'lr_warmup_steps',
  warmup_start_factor: 'lr_warmup_start_factor',
  min_lr_factor: 'lr_min_factor',
  step_size: 'lr_step_size',
  gamma: 'lr_gamma',
};

const EARLY_STOPPING_FIELDS: Record<string, string> = {
  patience: 'early_stopping_patience',
  min_delta: 'early_stopping_min_delta',
};

function put(into: Record<string, string>, name: string, value: unknown): void {
  // null은 "그 실행이 쓰지 않았다"는 뜻이라 칸을 비워 둡니다. 0과 false는 값입니다.
  if (value === null || value === undefined) return;
  into[name] = typeof value === 'boolean' ? String(value) : String(value);
}

function putNested(
  into: Record<string, string>,
  source: Record<string, unknown> | null,
  names: Record<string, string>,
): void {
  if (!source) return;
  for (const [key, field] of Object.entries(names)) put(into, field, source[key]);
}

export function rerunSettings(experiment: ExperimentSummary): Record<string, string> {
  const { model, optimizer, training } = experiment;
  const values: Record<string, string> = {};

  put(values, 'architecture', model.architecture);
  put(values, 'pretrained', model.pretrained);
  put(values, 'optimizer', optimizer.name);
  for (const key of ['learning_rate', 'momentum', 'weight_decay', 'beta1', 'beta2', 'epsilon'] as const) {
    put(values, key, optimizer[key]);
  }
  for (const key of [
    'device',
    'epochs',
    'batch_size',
    'num_workers',
    'gradient_accumulation_steps',
    'input_size',
    'precision',
    'checkpoint_every',
    'seed',
  ] as const) {
    put(values, key, training[key]);
  }

  put(values, 'augmentation', training.augmentation?.preset);
  if (training.lr_scheduler) {
    put(values, 'lr_scheduler', training.lr_scheduler.name);
    putNested(values, training.lr_scheduler, LR_FIELDS);
  }
  if (training.early_stopping) {
    // 화면은 켜고 끄는 칸이 따로 있습니다. 값만 채우면 꺼진 채로 남습니다.
    put(values, 'early_stopping', true);
    putNested(values, training.early_stopping, EARLY_STOPPING_FIELDS);
  }
  return values;
}
