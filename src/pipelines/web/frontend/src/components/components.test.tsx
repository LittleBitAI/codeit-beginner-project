import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { AlertRow, EstimatedValue, KpiCard, ProgressBar, StatusBadge } from './primitives';
import { LogStream } from './LogStream';
import { LossBreakdown } from './LossBreakdown';
import { LossChart } from './LossChart';
import type { EpochRecord, LogLine } from '../api/types';

describe('AlertRow', () => {
  it('왼쪽 세로 강조 막대를 쓰지 않는다', () => {
    // 디자인 하드 제약: alert에 좌측 accent bar 금지. 색은 아이콘과 제목에만.
    const { container } = render(
      <AlertRow level="error" title="실패했습니다">
        원인 설명
      </AlertRow>,
    );
    const box = container.firstElementChild as HTMLElement;
    const computed = window.getComputedStyle(box);

    // 네 변의 테두리 굵기가 같아야 합니다. 왼쪽만 두꺼우면 accent bar입니다.
    expect(computed.borderLeftWidth).toBe(computed.borderRightWidth);
    expect(computed.borderLeftWidth).toBe(computed.borderTopWidth);
    expect(computed.borderLeftColor).toBe(computed.borderRightColor);
  });

  it('제목과 설명을 함께 보여 준다', () => {
    render(
      <AlertRow level="warning" title="주의">
        무슨 일이 있었는지
      </AlertRow>,
    );

    expect(screen.getByText('주의')).toBeInTheDocument();
    expect(screen.getByText('무슨 일이 있었는지')).toBeInTheDocument();
  });
});

describe('EstimatedValue', () => {
  it('추정값은 ~ 접두와 기울임으로 측정값과 구분한다', () => {
    render(<EstimatedValue>12분</EstimatedValue>);
    const node = screen.getByText('~12분');

    expect(node).toHaveClass('estimated');
  });
});

describe('StatusBadge', () => {
  it('상태 라벨을 그대로 보여 준다', () => {
    render(<StatusBadge status="running" label="실행 중" />);

    expect(screen.getByText('실행 중')).toBeInTheDocument();
  });
});

describe('ProgressBar', () => {
  it('진행률을 모르면 채우지 않고 값도 알리지 않는다', () => {
    render(<ProgressBar ratio={null} />);
    const bar = screen.getByRole('progressbar');

    expect(bar.getAttribute('aria-valuenow')).toBeNull();
    expect(bar.children.length).toBe(0);
  });

  it('진행률을 알면 그만큼 채운다', () => {
    render(<ProgressBar ratio={0.5} />);

    expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('50');
  });
});

describe('KpiCard', () => {
  it('값과 쉬운 설명을 함께 보여 준다', () => {
    render(<KpiCard label="VAL LOSS" value="0.5109" note="낮을수록 좋습니다" />);

    expect(screen.getByText('0.5109')).toBeInTheDocument();
    expect(screen.getByText('낮을수록 좋습니다')).toBeInTheDocument();
  });
});

describe('LogStream', () => {
  const lines: LogLine[] = [
    { seq: 1, stream: 'system', level: 'info', text: '학습 시작', ts: '2026-08-05T00:00:00Z' },
    { seq: 2, stream: 'stderr', level: 'error', text: 'CUDA out of memory', ts: '2026-08-05T00:00:01Z' },
  ];

  it('줄을 순서대로 보여 준다', () => {
    render(<LogStream lines={lines} streaming />);

    expect(screen.getByText('학습 시작')).toBeInTheDocument();
    expect(screen.getByText('CUDA out of memory')).toBeInTheDocument();
    expect(screen.getByText('스트리밍 중')).toBeInTheDocument();
  });

  it('출력이 없으면 그렇다고 말한다', () => {
    render(<LogStream lines={[]} streaming={false} />);

    expect(screen.getByText('아직 출력이 없습니다.')).toBeInTheDocument();
  });
});

describe('LossChart', () => {
  it('epoch이 없으면 빈 상태를 알리고 곡선을 그리지 않는다', () => {
    const { container } = render(<LossChart epochs={[]} totalEpochs={10} currentEpoch={null} />);

    expect(screen.getByText(/그릴 데이터가 없습니다/)).toBeInTheDocument();
    expect(container.querySelector('polyline')).toBeNull();
  });

  it('실제 epoch 값이 있을 때만 곡선을 그린다', () => {
    const epochs: EpochRecord[] = [
      { epoch: 1, train_loss: 1.2, validation_loss: 1.1, epoch_seconds: 2, is_best: true },
      { epoch: 2, train_loss: 0.9, validation_loss: 1.0, epoch_seconds: 2, is_best: true },
    ];
    const { container } = render(<LossChart epochs={epochs} totalEpochs={4} currentEpoch={2} />);

    expect(container.querySelectorAll('polyline').length).toBe(2);
  });
});

describe('LossBreakdown', () => {
  it('한쪽에만 있는 이름은 반대쪽을 지어내지 않고 "-"로 둔다', () => {
    const epochs: EpochRecord[] = [
      {
        epoch: 7,
        train_loss: 1.0,
        validation_loss: 1.1,
        train_loss_components: { classification: 0.6, bbox_regression: 0.4 },
        validation_loss_components: { classification: 0.7 },
        epoch_seconds: 2,
        is_best: true,
      },
    ];

    render(<LossBreakdown epochs={epochs} />);

    expect(screen.getByText('손실 분해 · epoch 7')).toBeInTheDocument();
    expect(screen.getByText('0.4000')).toBeInTheDocument();
    expect(screen.getByText('-')).toBeInTheDocument();
  });
});
