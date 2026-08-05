import { render, screen } from '@testing-library/react';
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
});
