import type { Defaults, TrainCapability } from '../api/types';

const LEGACY_OPTIMIZER = 'SGD';

function groupLooksUsable(
  value: TrainCapability['model'] | undefined,
): value is TrainCapability['model'] {
  return Boolean(
    value &&
      typeof value.default === 'string' &&
      value.default !== '' &&
      Array.isArray(value.choices) &&
      value.choices.length > 0 &&
      value.choices.every((choice) => typeof choice === 'string' && choice !== '') &&
      value.choices.includes(value.default) &&
      typeof value.selection_supported === 'boolean',
  );
}

function looksUsable(value: TrainCapability | undefined): value is TrainCapability {
  return Boolean(
    value &&
      value.schema_version === 1 &&
      (value.source === 'train' || value.source === 'legacy_fallback') &&
      (value.fallback_reason === null ||
        value.fallback_reason === 'train_capability_unavailable' ||
        value.fallback_reason === 'train_capability_invalid') &&
      groupLooksUsable(value.model) &&
      groupLooksUsable(value.optimizer),
  );
}

/** capability field가 없는 구버전 backend 응답도 현재 고정 구성으로 읽습니다. */
export function resolveTrainCapability(defaults: Defaults): TrainCapability {
  if (looksUsable(defaults.train_capability)) return defaults.train_capability;
  return {
    schema_version: 1,
    source: 'legacy_fallback',
    fallback_reason: 'train_capability_unavailable',
    model: {
      default: defaults.architecture,
      choices: [defaults.architecture],
      selection_supported: false,
    },
    optimizer: {
      default: LEGACY_OPTIMIZER,
      choices: [LEGACY_OPTIMIZER],
      selection_supported: false,
    },
  };
}
