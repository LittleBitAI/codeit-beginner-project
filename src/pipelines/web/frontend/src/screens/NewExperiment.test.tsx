import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import type { Defaults } from '../api/types';
import { DraftProvider } from '../state/DraftContext';

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    validate: vi.fn().mockResolvedValue({
      valid: false,
      errors: [],
      warnings: [],
      normalized: null,
    }),
    createConfig: vi.fn(),
  },
}));

const { NewExperiment } = await import('./NewExperiment');

const LEGACY_DEFAULTS: Defaults = {
  architecture: 'fasterrcnn_mobilenet_v3_large_320_fpn',
  architecture_note: '고정 모델',
  fields: [],
  data_fields: [],
  devices: [],
};

describe('NewExperiment · Train capability 호환', () => {
  it('capability field가 없는 응답에도 실제 고정 기본값을 안내한다', () => {
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={LEGACY_DEFAULTS} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    expect(screen.getByText('Train capability 호환 기본값을 사용합니다')).toBeInTheDocument();
    expect(screen.getByText('fasterrcnn_mobilenet_v3_large_320_fpn')).toBeInTheDocument();
    expect(screen.getByText('SGD')).toBeInTheDocument();
  });

  it('증강 preset을 화면에서 고를 수 있다', () => {
    // 데이터가 적을 때 과적합을 줄이는 유일한 수단인데, 화면에 칸이 없으면
    // train의 기본값 none으로만 학습됩니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        { name: 'augmentation', type: 'enum', default: 'none', choices: ['none', 'pill_basic'], label: '증강 preset', hint: '' },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    const field = screen.getByLabelText('증강 preset');
    expect(field).toHaveValue('none');
    fireEvent.change(field, { target: { value: 'pill_basic' } });
    expect(field).toHaveValue('pill_basic');
  });

  it('boolean 칸은 서버가 알려 준 기본값으로 시작한다', () => {
    // 예전에는 무조건 "사용하지 않음"으로 시작해, 서버가 기본값을 바꿔도 화면이
    // 따라가지 않았습니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        { name: 'pretrained', type: 'boolean', default: true, label: 'Pretrained 가중치', hint: '' },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    expect(screen.getByLabelText('Pretrained 가중치')).toHaveValue('true');
  });

  it('조기 종료를 켰을 때만 patience와 min delta를 묻는다', () => {
    // 끈 채로 숫자 칸을 보여 주면 그 값이 학습에 쓰이는 것처럼 읽힙니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        { name: 'early_stopping', type: 'boolean', default: false, label: '조기 종료', hint: '' },
        { name: 'early_stopping_patience', type: 'integer', default: 5, label: 'Patience', hint: '' },
        { name: 'early_stopping_min_delta', type: 'number', default: 0, label: 'Min delta', hint: '' },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByText('하이퍼파라미터'));
    expect(screen.queryByLabelText('Patience')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Min delta')).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('조기 종료'), { target: { value: 'true' } });

    expect(screen.getByLabelText('Patience')).toBeInTheDocument();
    expect(screen.getByLabelText('Min delta')).toBeInTheDocument();
  });

  it('모델과 optimizer를 고르고 profile에 맞는 수치만 보여 준다', () => {
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      train_capability: {
        schema_version: 1,
        source: 'legacy_fallback',
        fallback_reason: 'train_capability_unavailable',
        model: {
          default: 'fasterrcnn_mobilenet_v3_large_320_fpn',
          choices: ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_resnet50_fpn_v2'],
          selection_supported: true,
        },
        optimizer: {
          default: 'AdamW',
          choices: ['AdamW', 'SGD', 'Adam'],
          selection_supported: true,
        },
      },
      fields: [
        { name: 'architecture', type: 'enum', default: 'fasterrcnn_mobilenet_v3_large_320_fpn', choices: ['fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_resnet50_fpn_v2'], label: '모델', hint: '' },
        { name: 'optimizer', type: 'enum', default: 'AdamW', choices: ['AdamW', 'SGD', 'Adam'], label: 'Optimizer', hint: '' },
        { name: 'learning_rate', type: 'number', default: 0.0001, defaults_by_optimizer: { AdamW: 0.0001, SGD: 0.005, Adam: 0.0001 }, label: 'Learning rate', hint: '' },
        { name: 'momentum', type: 'number', default: 0.9, label: 'Momentum', hint: '' },
        { name: 'beta1', type: 'number', default: 0.9, label: 'Beta 1', hint: '' },
        { name: 'beta2', type: 'number', default: 0.999, label: 'Beta 2', hint: '' },
        { name: 'epsilon', type: 'number', default: 1e-8, label: 'Epsilon', hint: '' },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    expect(screen.getByLabelText('모델')).toHaveValue('fasterrcnn_mobilenet_v3_large_320_fpn');
    expect(screen.getByLabelText('Optimizer')).toHaveValue('AdamW');
    fireEvent.click(screen.getByText('하이퍼파라미터'));
    expect(screen.getByLabelText('Beta 1')).toBeInTheDocument();
    expect(screen.queryByLabelText('Momentum')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Learning rate')).toHaveAttribute(
      'placeholder',
      '기본값 0.0001',
    );

    fireEvent.click(screen.getByText('기본 정보'));
    fireEvent.change(screen.getByLabelText('Optimizer'), { target: { value: 'SGD' } });
    fireEvent.click(screen.getByText('하이퍼파라미터'));
    expect(screen.getByLabelText('Momentum')).toBeInTheDocument();
    expect(screen.queryByLabelText('Beta 1')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Learning rate')).toHaveAttribute(
      'placeholder',
      '기본값 0.005',
    );
  });
});
