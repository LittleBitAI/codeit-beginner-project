/**
 * 앙상블 화면이 **낭비를 실제로 막는지** 봅니다.
 *
 * 이 화면의 값어치는 고르는 편의가 아니라 경고를 보여 주는 데 있습니다. 경고가 안
 * 보이면 사람이 Kaggle 제출을 한 번 버립니다. 그래서 test도 "화면에 떴는가"를 잽니다.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '../api/client';
import type { EnsembleCandidate, EnsembleDiagnosis } from '../api/types';
import { Ensemble } from './Ensemble';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function candidate(overrides: Partial<EnsembleCandidate> = {}): EnsembleCandidate {
  return {
    run_id: 'dino-a',
    checkpoint_uri: 's3://bucket/dino-a/best_checkpoint.pt',
    test_predictions_uri: 's3://bucket/dino-a/test_predictions.json',
    ready: true,
    dataset_label: 'v5-seed42-8020-group',
    kaggle_score: 0.62437,
    created_at: null,
    ...overrides,
  };
}

function diagnosis(overrides: Partial<EnsembleDiagnosis> = {}): EnsembleDiagnosis {
  return {
    run_ids: ['dino-a', 'dino-b'],
    checks: [],
    diversity: { pairs: [] },
    expected: {},
    blocking: false,
    ...overrides,
  };
}

function stub(candidates: EnsembleCandidate[], result: EnsembleDiagnosis = diagnosis()) {
  vi.spyOn(api, 'ensembleCandidates').mockResolvedValue({ candidates });
  vi.spyOn(api, 'ensembleStatus').mockResolvedValue({ status: 'idle' });
  const diagnose = vi.spyOn(api, 'diagnoseEnsemble').mockResolvedValue(result);
  return diagnose;
}

/** 후보 둘을 고릅니다. 진단은 둘 이상일 때만 돕니다. */
function pickTwo() {
  const boxes = screen.getAllByRole('checkbox') as HTMLElement[];
  fireEvent.click(boxes[0] as HTMLElement);
  fireEvent.click(boxes[1] as HTMLElement);
}

describe('앙상블 화면', () => {
  it('체크포인트만 있는 실행도 고를 수 있고, 추론이 필요하다고 알린다', async () => {
    // 예측이 있는 실행만 고르게 하면 이 저장소에서 47개 중 2개만 남습니다.
    stub([candidate(), candidate({ run_id: 'dino-fresh', ready: false, kaggle_score: 0.61 })]);

    render(<Ensemble />);

    expect(await screen.findByText('dino-fresh')).toBeTruthy();
    expect(screen.getByText('추론 필요')).toBeTruthy();
  });

  it('둘 미만이면 합치기를 막고 진단을 부르지 않는다', async () => {
    const diagnose = stub([candidate(), candidate({ run_id: 'dino-b' })]);

    render(<Ensemble />);
    await screen.findByText('dino-a');

    fireEvent.click(screen.getAllByRole('checkbox')[0] as HTMLElement);

    expect(screen.getByRole('button', { name: /합치기/ })).toHaveProperty('disabled', true);
    expect(diagnose).not.toHaveBeenCalled();
  });

  it('경고를 화면에 그대로 보여 준다', async () => {
    stub(
      [candidate(), candidate({ run_id: 'dino-b' })],
      diagnosis({
        checks: [
          {
            id: 'diversity',
            level: 'warn',
            title: '거의 같은 예측입니다 (일치 97.7%)',
            detail: '합칠 것이 거의 없습니다.',
          },
        ],
      }),
    );

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();

    expect(await screen.findByText('거의 같은 예측입니다 (일치 97.7%)')).toBeTruthy();
    expect(screen.getByText('합칠 것이 거의 없습니다.')).toBeTruthy();
  });

  it('경고가 있어도 합치기를 막지 않는다', async () => {
    // 막아 버리면 예측이 틀렸을 때 반증할 길까지 막힙니다.
    stub(
      [candidate(), candidate({ run_id: 'dino-b' })],
      diagnosis({
        checks: [
          { id: 'dilution', level: 'warn', title: '약한 실행 1개', detail: '평균 쪽으로 끌려갑니다.' },
        ],
      }),
    );

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();
    await screen.findByText('약한 실행 1개');

    expect(screen.getByRole('button', { name: /합치기/ })).toHaveProperty('disabled', false);
  });

  it('시험지가 다르다는 경고가 있을 때만 확인 칸을 묻는다', async () => {
    const consent = '사진이 같은데 위치만 다른 것을 확인했습니다 (fusion_allow_copied_images)';
    stub(
      [candidate(), candidate({ run_id: 'dino-b' })],
      diagnosis({
        checks: [
          { id: 'test_set', level: 'warn', title: '시험지가 다릅니다', detail: 'v5와 v6' },
        ],
      }),
    );

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();

    expect(await screen.findByText(consent)).toBeTruthy();
  });

  it('시험지 경고가 없으면 확인 칸을 묻지 않는다', async () => {
    // 늘 보이면 뜻 없이 켜집니다.
    stub([candidate(), candidate({ run_id: 'dino-b' })], diagnosis({ checks: [] }));

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();
    await waitFor(() => expect(api.diagnoseEnsemble).toHaveBeenCalled());

    expect(screen.queryByText(/fusion_allow_copied_images/)).toBeNull();
  });

  it('고른 조합을 그대로 보내 실행한다', async () => {
    stub([candidate(), candidate({ run_id: 'dino-b' })]);
    const start = vi
      .spyOn(api, 'startEnsemble')
      .mockResolvedValue({ status: 'running', run_id: 'fusion-2' });

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();
    await waitFor(() => expect(api.diagnoseEnsemble).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', { name: /합치기/ }));

    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[0].run_ids).toEqual(['dino-a', 'dino-b']);
  });

  it('고른 것이 바뀌면 지난 동의를 버린다', async () => {
    // A와 B가 같은 사진이라고 확인했다고 해서 C까지 확인한 것이 아닙니다. 남겨 두면
    // 새 진단 전에 실행할 때 C까지 위치 검사를 면제해, 다른 사진을 조용히 합칩니다.
    const consent = '사진이 같은데 위치만 다른 것을 확인했습니다 (fusion_allow_copied_images)';
    stub(
      [candidate(), candidate({ run_id: 'dino-b' }), candidate({ run_id: 'dino-c' })],
      diagnosis({
        run_ids: ['dino-a', 'dino-b'],
        checks: [{ id: 'test_set', level: 'warn', title: '시험지가 다릅니다', detail: 'v5와 v6' }],
      }),
    );

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();
    const box = (await screen.findByText(consent)).querySelector('input') as HTMLInputElement;
    fireEvent.click(box);
    expect(box.checked).toBe(true);

    fireEvent.click(screen.getAllByRole('checkbox')[2] as HTMLElement);

    await waitFor(() => {
      const again = screen.queryByText(consent)?.querySelector('input') as HTMLInputElement | null;
      expect(again === null || again.checked === false).toBe(true);
    });
  });

  it('진단이 지금 고른 것의 답이 아니면 실행을 막는다', async () => {
    // 옛 조합의 진단 위에서 실행을 결정하면, 화면이 보여 준 경고와 실제로 합치는
    // 것이 달라집니다.
    stub(
      [candidate(), candidate({ run_id: 'dino-b' }), candidate({ run_id: 'dino-c' })],
      diagnosis({ run_ids: ['dino-a', 'dino-b'] }),
    );

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /합치기/ })).toHaveProperty('disabled', false),
    );

    fireEvent.click(screen.getAllByRole('checkbox')[2] as HTMLElement);

    expect(screen.getByRole('button', { name: /합치기/ })).toHaveProperty('disabled', true);
    // 세 개짜리 진단이 도착하는 것까지 기다립니다. 안 기다리면 test가 끝난 뒤 상태가
    // 바뀌어 React가 act(...) 경고를 냅니다.
    await waitFor(() => expect(api.diagnoseEnsemble).toHaveBeenCalledTimes(2));
  });

  it('고른 embedding을 재순위로 함께 보낸다', async () => {
    // 화면에서 골랐는데 요청에 빠지면, 사람은 재순위한 제출이라고 믿고 Kaggle에
    // 올립니다. 점수가 왜 그대로인지 알아낼 방법이 없습니다.
    stub([candidate(), candidate({ run_id: 'dino-b' })]);
    vi.spyOn(api, 'embeddingRuns').mockResolvedValue({
      runs: [
        {
          run_id: 'emb-r18',
          job_id: 'job-1',
          status: 'succeeded',
          backbone: 'resnet18',
          epochs: 30,
          checkpoint_uri: 'artifacts/emb/best.pt',
          crop_bank_uri: 'datasets/v5/crop_bank.tar',
          created_at: null,
          ready: true,
        },
      ],
    });
    const start = vi
      .spyOn(api, 'startEnsemble')
      .mockResolvedValue({ status: 'running', run_id: 'fusion-2' });

    render(<Ensemble />);
    await screen.findByText('dino-a');
    pickTwo();
    await screen.findByText('emb-r18');
    // 후보 둘 다음에 오는 것이 embedding 칸입니다.
    fireEvent.click(screen.getAllByRole('checkbox')[2] as HTMLElement);
    await waitFor(() => expect(api.diagnoseEnsemble).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', { name: /합치기|재순위/ }));

    await waitFor(() => expect(start).toHaveBeenCalled());
    expect(start.mock.calls[0]?.[0].embedding_run_ids).toEqual(['emb-r18']);
  });

  it('추론 중인지 합치는 중인지 구분해 보여 준다', async () => {
    // 45개를 고르면 GPU가 몇 시간 돕니다. 어느 단계인지 안 보이면 멈춘 줄 압니다.
    vi.spyOn(api, 'ensembleCandidates').mockResolvedValue({ candidates: [candidate()] });
    vi.spyOn(api, 'diagnoseEnsemble').mockResolvedValue(diagnosis());
    vi.spyOn(api, 'ensembleStatus').mockResolvedValue({
      status: 'running',
      run_id: 'fusion-3',
      stage: 'harvest',
      harvesting: 'dino-fresh',
      harvest_progress: [2, 5],
    });

    render(<Ensemble />);

    expect(await screen.findByText(/test 예측을 만드는 중 dino-fresh \(2\/5\)/)).toBeTruthy();
  });
});
