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

  it('연산 정밀도를 화면에서 고를 수 있다', () => {
    // train이 amp를 받아도 화면에 칸이 없으면 fp32로만 학습됩니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        { name: 'precision', type: 'enum', default: 'fp32', choices: ['fp32', 'amp'], label: '연산 정밀도', hint: '' },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    const field = screen.getByLabelText('연산 정밀도');
    expect(field).toHaveValue('fp32');
    fireEvent.change(field, { target: { value: 'amp' } });
    expect(field).toHaveValue('amp');
  });

  it('고른 schedule이 쓰지 않는 칸은 감춘다', () => {
    // 보이면 그 값이 학습에 쓰이는 것처럼 읽히고, 서버도 쓰지 않는 값이라며
    // 저장을 막습니다. warmup 칸은 어느 schedule에서나 씁니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        {
          name: 'lr_scheduler',
          type: 'enum',
          default: 'none',
          choices: ['none', 'cosine', 'step'],
          label: 'Learning rate schedule',
          hint: '',
        },
        { name: 'lr_warmup_steps', type: 'integer', default: 0, label: 'Warmup steps', hint: '' },
        { name: 'lr_min_factor', type: 'number', default: 0.01, label: '최저 배율', hint: '' },
        { name: 'lr_step_size', type: 'integer', default: 3, label: '줄이는 간격', hint: '' },
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

    expect(screen.getByLabelText('Warmup steps')).toBeInTheDocument();
    expect(screen.queryByLabelText('최저 배율')).toBeNull();
    expect(screen.queryByLabelText('줄이는 간격')).toBeNull();

    fireEvent.change(screen.getByLabelText('Learning rate schedule'), {
      target: { value: 'cosine' },
    });

    expect(screen.getByLabelText('최저 배율')).toBeInTheDocument();
    expect(screen.queryByLabelText('줄이는 간격')).toBeNull();
  });

  it('Device 칸은 서버가 알려 준 기본값으로 시작한다', () => {
    // GPU가 달린 컴퓨터에서는 서버가 cuda를 기본값으로 내려줍니다. 화면이 그것을
    // 무시하고 늘 cpu로 시작하면, 바꾸는 것을 잊은 사람의 학습이 몇 분에서 몇
    // 시간짜리가 됩니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        { name: 'device', type: 'enum', default: 'cuda', choices: ['cpu', 'cuda'], label: 'Device', hint: '' },
      ],
      devices: [
        { value: 'cpu', available: true, reason: null },
        { value: 'cuda', available: true, reason: null },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    expect(screen.getByLabelText('Device')).toHaveValue('cuda');
  });

  it('CUDA가 없으면 Device 칸이 cpu로 시작한다', () => {
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        { name: 'device', type: 'enum', default: 'cpu', choices: ['cpu', 'cuda'], label: 'Device', hint: '' },
      ],
      devices: [
        { value: 'cpu', available: true, reason: null },
        { value: 'cuda', available: false, reason: '이 컴퓨터에서 CUDA를 사용할 수 없습니다.' },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    expect(screen.getByLabelText('Device')).toHaveValue('cpu');
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

describe('NewExperiment · 자동 실행 이름', () => {
  const defaults: Defaults = {
    ...LEGACY_DEFAULTS,
    fields: [
      { name: 'run_id', type: 'string', label: '실행 이름', hint: '결과 폴더 이름입니다.' },
    ],
  };

  async function showWithGeneratedName(runId: string) {
    const { api } = await import('../api/client');
    vi.mocked(api.validate).mockResolvedValue({
      valid: true,
      errors: [],
      warnings: [],
      normalized: {
        project: { name: 'pill' },
        execution: { mode: 'real' },
        storage: {},
        train: { run_id: runId },
        inputs: { data: {} },
      },
    });
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );
  }

  it('이름을 비워 두면 서버가 지어 준 이름을 미리 보여 준다', async () => {
    // 규칙은 backend 한 곳에만 있습니다. 화면은 검증 결과의 이름을 그대로 씁니다.
    await showWithGeneratedName('retina-basic-e15-b4-lr6e3-s42-a7f3');

    expect(
      await screen.findByText('자동 이름: retina-basic-e15-b4-lr6e3-s42-a7f3'),
    ).toBeInTheDocument();
  });

  it('이름을 직접 쓰면 자동 이름 안내를 감춘다', async () => {
    await showWithGeneratedName('retina-basic-e15-b4-lr6e3-s42-a7f3');
    // Field는 label 안에 힌트까지 넣으므로 접근 이름에 힌트가 딸려 옵니다.
    fireEvent.change(await screen.findByLabelText(/실행 이름/), { target: { value: 'my-run' } });

    expect(screen.queryByText(/자동 이름:/)).not.toBeInTheDocument();
    expect(screen.getByText('결과 폴더 이름입니다.')).toBeInTheDocument();
  });

  it('입력 크기 칸은 그 값을 쓰는 모델을 골랐을 때만 보인다', () => {
    // 서버가 칸을 내밀어도 이 화면은 고정 목록만 그립니다. 목록에 넣지 않으면 서버
    // test는 통과하는데 사용자는 값을 조정할 수 없습니다. 반대로 늘 보여 주면 그 값을
    // 쓰지 않는 모델에서도 정할 수 있는 것처럼 읽히는데 서버는 거부합니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        {
          name: 'architecture',
          type: 'enum',
          default: 'fasterrcnn_mobilenet_v3_large_320_fpn',
          choices: ['fasterrcnn_mobilenet_v3_large_320_fpn', 'dino_r50_4scale'],
          label: '모델',
          hint: '',
        },
        {
          name: 'gradient_accumulation_steps',
          type: 'integer',
          default: 1,
          label: 'Gradient accumulation',
          hint: '',
        },
        {
          name: 'input_size',
          type: 'integer',
          default: 640,
          only_for_architectures: ['dino_r50_4scale'],
          label: '입력 크기',
          hint: '',
        },
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

    // 모으는 수는 모든 모델이 씁니다.
    expect(screen.getByLabelText('Gradient accumulation')).toBeInTheDocument();
    expect(screen.queryByLabelText('입력 크기')).toBeNull();

    fireEvent.click(screen.getByText('기본 정보'));
    fireEvent.change(screen.getByLabelText('모델'), {
      target: { value: 'dino_r50_4scale' },
    });
    fireEvent.click(screen.getByText('하이퍼파라미터'));

    expect(screen.getByLabelText('입력 크기')).toBeInTheDocument();

    // 값을 적어 둔 뒤 모델을 되돌려도 칸은 사라집니다. draft에 남은 그 값을 빼는 것은
    // toPayload의 몫이라 lib.test.ts가 함께 지킵니다.
    fireEvent.change(screen.getByLabelText('입력 크기'), { target: { value: '800' } });
    fireEvent.click(screen.getByText('기본 정보'));
    fireEvent.change(screen.getByLabelText('모델'), {
      target: { value: 'fasterrcnn_mobilenet_v3_large_320_fpn' },
    });
    fireEvent.click(screen.getByText('하이퍼파라미터'));

    expect(screen.queryByLabelText('입력 크기')).toBeNull();
  });

  it('고른 모델에 따라 안내하는 기본값이 달라진다', () => {
    // 하나만 보여 주면 MMDetection을 고르고 비워 둔 사람에게 1이라고 안내하면서
    // 실제로는 8로 돕니다. 화면이 거짓말을 하는 셈입니다.
    const defaults: Defaults = {
      ...LEGACY_DEFAULTS,
      fields: [
        {
          name: 'architecture',
          type: 'enum',
          default: 'fasterrcnn_mobilenet_v3_large_320_fpn',
          choices: ['fasterrcnn_mobilenet_v3_large_320_fpn', 'dino_r50_4scale'],
          label: '모델',
          hint: '',
        },
        {
          name: 'gradient_accumulation_steps',
          type: 'integer',
          default: 1,
          defaults_by_architecture: { dino_r50_4scale: 8 },
          label: 'Gradient accumulation',
          hint: '',
        },
      ],
    };
    render(
      <MemoryRouter>
        <DraftProvider>
          <NewExperiment defaults={defaults} source={null} />
        </DraftProvider>
      </MemoryRouter>,
    );

    // draft는 화면 밖에서 이어지므로 앞선 test가 고른 모델이 남아 있을 수 있습니다.
    // 여기서 두 방향을 모두 확인해 그 상태와 무관하게 만듭니다.
    fireEvent.click(screen.getByText('기본 정보'));
    fireEvent.change(screen.getByLabelText('모델'), {
      target: { value: 'dino_r50_4scale' },
    });
    fireEvent.click(screen.getByText('하이퍼파라미터'));
    expect(screen.getByLabelText('Gradient accumulation')).toHaveAttribute(
      'placeholder',
      '기본값 8',
    );

    fireEvent.click(screen.getByText('기본 정보'));
    fireEvent.change(screen.getByLabelText('모델'), {
      target: { value: 'fasterrcnn_mobilenet_v3_large_320_fpn' },
    });
    fireEvent.click(screen.getByText('하이퍼파라미터'));
    expect(screen.getByLabelText('Gradient accumulation')).toHaveAttribute(
      'placeholder',
      '기본값 1',
    );
  });
});
