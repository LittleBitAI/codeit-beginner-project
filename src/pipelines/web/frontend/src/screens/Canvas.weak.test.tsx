/**
 * 약한 class 표와 그 옆 길들.
 *
 * 값이 없는 것을 0으로 그리면 화면이 거짓말을 합니다. 그리고 기록에서 누르면 이제
 * 이 화면으로 오므로, 이 컴퓨터가 돌린 실행의 로그로 가는 길이 여기 있어야 합니다.
 */

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useParams } from 'react-router-dom';

import type { ExperimentSummary } from '../api/types';
import type { RunRecord } from '../lib/records';
import { Canvas } from './Canvas';

afterEach(cleanup);

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    runId: 'run-a',
    family: 'retinanet_resnet50_fpn_v2',
    datasetKey: 'v6',
    spec: 'e15 · b4',
    status: 'succeeded',
    statusLabel: '완료',
    at: '2026-08-05T00:00:00Z',
    jobId: null,
    registered: true,
    evaluated: true,
    submitted: false,
    metrics: {
      kaggle: null,
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      bestValidationLoss: null,
      bestEpoch: null,
      epochs: 15,
      elapsedSeconds: null,
    },
    ...overrides,
  } as RunRecord;
}

/** 재지 못한 AP(null)를 담은 약한 class 하나를 든 실험입니다. */
function experiment(weakCount = 2): ExperimentSummary {
  return {
    experiment_id: 'run-a',
    run_id: 'run-a',
    status: 'succeeded',
    status_label: '등록 완료',
    created_at: '2026-08-05T00:00:00Z',
    started_at: null,
    finished_at: null,
    elapsed_seconds: null,
    dataset: { identity: 'v6', identity_source: 'artifact_set', artifacts_complete: true, label: 'v6' },
    model: { architecture: 'retinanet_resnet50_fpn_v2', pretrained: true, source: 'record' },
    optimizer: {
      name: 'AdamW',
      source: 'record',
      learning_rate: 0.0001,
      momentum: null,
      weight_decay: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    training: {
      device: 'cuda',
      epochs: 15,
      batch_size: 4,
      num_workers: 0,
      gradient_accumulation_steps: 1,
      input_size: null,
      seed: 42,
    },
    per_class_summary: {
      min_truth_count: 5,
      top_n: 10,
      counts: { weak: weakCount, sparse: 0, unmeasured: 0 },
      weak: [
        { category_id: 16548, name: '가바토파정 100mg', ap: 0.12 },
        // 표본은 충분한데 AP를 재지 못한 줄입니다. evaluate가 허용하는 모양입니다.
        { category_id: 16232, name: '리피토정 20mg', ap: null },
      ],
      sparse: [],
    },
    metrics: {
      map: null,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      best_epoch: null,
      best_validation_loss: null,
      final_train_loss: null,
      final_validation_loss: null,
      kaggle_score: null,
    },
  } as unknown as ExperimentSummary;
}

function MonitorStub() {
  const { jobId } = useParams();
  return <div>모니터 화면 {jobId}</div>;
}

function show(records: RunRecord[], runs = ['run-a']) {
  return render(
    <MemoryRouter initialEntries={[`/canvas?${runs.map((id) => `run=${id}`).join('&')}`]}>
      <Routes>
        <Route
          path="/canvas"
          element={
            <Canvas
              datasetKey="v6"
              records={records}
              loading={false}
              onScoreSaved={() => {}}
              onNewExperiment={() => {}}
            />
          }
        />
        {/* 주소만 맞는 것으로는 부족합니다. 어느 job으로 갔는지까지 드러냅니다. */}
        <Route path="/monitor/:jobId" element={<MonitorStub />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('약한 class 표', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({ experiments: [experiment()], missing: [], curves: {} }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it('재지 못한 AP를 0으로 그리지 않는다', async () => {
    show([record()]);

    expect(await screen.findByText('리피토정 20mg')).toBeTruthy();
    // 0.120은 실제로 잰 값이라 나와야 하고, null은 0.000이 되면 안 됩니다.
    expect(screen.getByText('0.120')).toBeTruthy();
    expect(screen.queryByText('0.000')).toBeNull();
    // 그 자리에 무엇이 적히는지까지 봅니다. 안 적히는 것만 보면 빈 칸도 통과합니다.
    expect(screen.getByText('미측정')).toBeTruthy();
  });

  it('이 컴퓨터가 돌린 실행이면 그 job의 로그 화면으로 보낸다', async () => {
    show([record({ jobId: 'job-77' })]);

    fireEvent.click(await screen.findByText('로그 보기'));

    // 어느 job인지까지 봅니다. 주소만 맞고 엉뚱한 job이면 안 됩니다.
    expect(await screen.findByText('모니터 화면 job-77')).toBeTruthy();
  });

  it('목록이 잘려 있으면 없는 class를 약하지 않다고 적지 않는다', async () => {
    // run-b는 상위 1개만 받았고(counts 9 > 목록 1) 그 안에 가바토파정이 없습니다.
    // 실제로 약한데 순위 밖일 수 있으므로 "-"(약하지 않음)라고 말하면 안 됩니다.
    const other = {
      ...experiment(9),
      experiment_id: 'run-b',
      run_id: 'run-b',
      per_class_summary: {
        min_truth_count: 5,
        top_n: 1,
        counts: { weak: 9, sparse: 0, unmeasured: 0 },
        weak: [{ category_id: 99999, name: '다른 알약', ap: 0.05 }],
        sparse: [],
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        // 두 실행을 실제로 요청했는지 봅니다. 부분 문자열로 보면 `run-a-wrong`도
        // 통과하므로 보낸 배열을 그대로 비교합니다.
        const sent = JSON.parse(String(init?.body ?? '{}')) as { run_ids?: string[] };
        expect(sent.run_ids).toEqual(['run-a', 'run-b']);
        return new Response(
          JSON.stringify({ experiments: [experiment(), other], missing: [], curves: {} }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      }),
    );
    show([record(), record({ runId: 'run-b' })], ['run-a', 'run-b']);

    expect(await screen.findByText('가바토파정 100mg')).toBeTruthy();
    // run-b 칸이 "순위 밖"이어야 합니다. "-"로 적으면 약하지 않다고 단정하는 것입니다.
    expect(screen.getAllByText('순위 밖').length).toBeGreaterThan(0);
  });

  it('class 요약이 없는 실행을 약하지 않다고 적지 않는다', async () => {
    // per_class_summary가 없는 실행입니다. 평가 전일 수도 있고, 평가는 했지만 이
    // 요약이 생기기 전에 등록된 기록일 수도 있습니다. 여기서는 후자를 씁니다 —
    // mAP는 있는데 "평가 없음"이라고 적으면 같은 화면이 서로 다른 말을 합니다.
    const notEvaluated = {
      ...experiment(),
      experiment_id: 'run-b',
      run_id: 'run-b',
      metrics: { ...experiment().metrics, map: 0.51 },
    };
    delete (notEvaluated as { per_class_summary?: unknown }).per_class_summary;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(
          JSON.stringify({ experiments: [experiment(), notEvaluated], missing: [], curves: {} }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );
    show([record(), record({ runId: 'run-b' })], ['run-a', 'run-b']);

    expect(await screen.findByText('가바토파정 100mg')).toBeTruthy();
    // 안 돌린 실행이 "약하지 않음"으로 보이면 그 실행이 가장 좋아 보입니다.
    expect(screen.getAllByText('class 요약 없음').length).toBeGreaterThan(0);
    // 평가를 안 했다고 단정하면 안 됩니다. mAP가 있는 옛 기록도 여기 걸립니다.
    expect(screen.queryByText('평가 없음')).toBeNull();
    expect(screen.queryByText('약하지 않음')).toBeNull();
  });

  it('어느 칸이 어느 실행인지 머리글로 밝힌다', async () => {
    show([record()]);

    // 실행 이름은 곡선 범례에도 있습니다. 표 안에서 찾아야 머리글을 지킵니다.
    const table = (await screen.findByText('약한 class')).parentElement as HTMLElement;
    expect(within(table).getAllByText('run-a').length).toBeGreaterThan(0);
  });

  it('잴 만한 class가 없었던 것을 좋은 결과로 적지 않는다', async () => {
    // weak는 AP가 낮은 목록이 아니라 정답이 충분한 목록입니다. 비었다는 것은
    // 잴 만한 class가 없었다는 뜻이지 좋다는 뜻이 아닙니다.
    const clean = {
      ...experiment(),
      per_class_summary: {
        min_truth_count: 5,
        top_n: 10,
        counts: { weak: 0, sparse: 0, unmeasured: 0 },
        weak: [],
        sparse: [],
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ experiments: [clean], missing: [], curves: {} }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    show([record()]);

    expect(await screen.findByText(/class가 없어 약한지 판단할 수 없습니다/)).toBeTruthy();
  });

  it('요약이 아예 없으면 평가부터 다시 하라고 안내한다', async () => {
    const bare = { ...experiment() };
    delete (bare as { per_class_summary?: unknown }).per_class_summary;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ experiments: [bare], missing: [], curves: {} }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    show([record()]);

    // 등록만으로는 생기지 않습니다. 그 말을 빼면 틀린 복구 안내가 됩니다.
    expect(await screen.findByText(/평가를 다시 실행해 등록하면/)).toBeTruthy();
  });

  it('요약이 있는 실행과 없는 실행이 섞이면 둘 다 말한다', async () => {
    // run-a는 요약이 있고 약한 class가 0개(좋은 결과), run-b는 요약이 아예 없습니다.
    // "AP가 낮은 것이 없습니다"만 적으면 run-b가 좋은 결과로 읽힙니다.
    const clean = {
      ...experiment(),
      per_class_summary: {
        min_truth_count: 5,
        top_n: 10,
        counts: { weak: 0, sparse: 0, unmeasured: 0 },
        weak: [],
        sparse: [],
      },
    };
    const bare = { ...experiment(), experiment_id: 'run-b', run_id: 'run-b' };
    delete (bare as { per_class_summary?: unknown }).per_class_summary;
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ experiments: [clean, bare], missing: [], curves: {} }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    show([record(), record({ runId: 'run-b' })], ['run-a', 'run-b']);

    // 두 문장이 **같은 렌더에** 다 있어야 합니다. 하나만 보면 다른 하나가
    // 사라져도 통과합니다.
    expect(await screen.findByText(/class가 없어 약한지 판단할 수 없습니다/)).toBeTruthy();
    expect(screen.getByText(/run-b에는 class별 요약이 없어/)).toBeTruthy();
  });

  it('다른 실행에서 표본이 부족한 class를 약하지 않다고 적지 않는다', async () => {
    // run-b에서 가바토파정은 정답이 적어 sparse로 분류됐습니다. weak에 없다는
    // 이유로 "약하지 않음"이라고 적으면 표본 부족을 성능 양호로 읽습니다.
    const other = {
      ...experiment(),
      experiment_id: 'run-b',
      run_id: 'run-b',
      per_class_summary: {
        min_truth_count: 5,
        top_n: 10,
        counts: { weak: 0, sparse: 1, unmeasured: 0 },
        weak: [],
        sparse: [{ category_id: 16548, name: '가바토파정 100mg', ap: 0.4 }],
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ experiments: [experiment(), other], missing: [], curves: {} }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    show([record(), record({ runId: 'run-b' })], ['run-a', 'run-b']);

    expect(await screen.findByText('가바토파정 100mg')).toBeTruthy();
    // 가바토파정 칸이 "표본 부족"이어야 합니다. 다른 class(리피토정)는 run-b에서
    // 정말로 약하지 않으므로 그 표기가 남는 것이 맞습니다.
    expect(screen.getByText('표본 부족')).toBeTruthy();
  });

  it('정답이 하나도 없던 class를 약하지 않다고 적지 않는다', async () => {
    const other = {
      ...experiment(),
      experiment_id: 'run-b',
      run_id: 'run-b',
      per_class_summary: {
        min_truth_count: 5,
        top_n: 10,
        counts: { weak: 0, sparse: 0, unmeasured: 1 },
        weak: [],
        sparse: [],
        unmeasured: [{ category_id: 16548, name: '가바토파정 100mg', ap: null }],
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ experiments: [experiment(), other], missing: [], curves: {} }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    show([record(), record({ runId: 'run-b' })], ['run-a', 'run-b']);

    expect(await screen.findByText('가바토파정 100mg')).toBeTruthy();
    expect(screen.getByText('정답 없음')).toBeTruthy();
  });

  it('sparse 목록이 잘려 있어도 약하지 않다고 단정하지 않는다', async () => {
    // counts.sparse가 목록 길이보다 큽니다. 그 밖의 class는 표본이 부족했을 수도
    // 있으므로 "약하지 않음"이라고 말할 근거가 없습니다.
    const other = {
      ...experiment(),
      experiment_id: 'run-b',
      run_id: 'run-b',
      per_class_summary: {
        min_truth_count: 5,
        top_n: 1,
        counts: { weak: 0, sparse: 9, unmeasured: 0 },
        weak: [],
        sparse: [{ category_id: 99999, name: '다른 알약', ap: 0.3 }],
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ experiments: [experiment(), other], missing: [], curves: {} }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );
    show([record(), record({ runId: 'run-b' })], ['run-a', 'run-b']);

    expect(await screen.findByText('가바토파정 100mg')).toBeTruthy();
    expect(screen.getAllByText('순위 밖').length).toBeGreaterThan(0);
    expect(screen.queryByText('약하지 않음')).toBeNull();
  });

  it('팀원이 돌린 실행에는 로그 링크를 내지 않는다', async () => {
    show([record({ jobId: null })]);

    expect(await screen.findByText('리피토정 20mg')).toBeTruthy();
    expect(screen.queryByText('로그 보기')).toBeNull();
  });
});
