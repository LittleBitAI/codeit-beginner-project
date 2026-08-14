import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import type { JobRecord, QueueState, TeamRun } from '../api/types';
import type { RunRecord } from '../lib/records';
import { Runs } from './Runs';

afterEach(cleanup);

function record(overrides: Partial<RunRecord> = {}): RunRecord {
  return {
    runId: 'retina-e15-b4-a7f3',
    family: 'retinanet_resnet50_fpn_v2',
    datasetKey: 'v5-118cls',
    spec: 'e15 · b4 · lr 0.006 · seed 42',
    status: 'succeeded',
    statusLabel: '완료',
    at: '2026-08-05T00:00:00Z',
    jobId: null,
    registered: true,
    evaluated: true,
    submitted: false,
    metrics: {
      kaggle: null,
      map: 0.52,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      bestValidationLoss: 0.41,
      bestEpoch: 12,
      epochs: 15,
      elapsedSeconds: 7200,
    },
    ...overrides,
  };
}

/** 결과를 남기지 못하고 끝난 기록. 검증 오차가 없다는 것이 판단 기준입니다. */
function failedRecord(overrides: Partial<RunRecord> = {}): RunRecord {
  return record({
    runId: 'oom',
    status: 'failed',
    statusLabel: '실패',
    registered: false,
    evaluated: false,
    metrics: { ...record().metrics, bestValidationLoss: null },
    ...overrides,
  });
}

function cancelledRecord(overrides: Partial<RunRecord> = {}): RunRecord {
  return failedRecord({
    runId: 'stopped',
    status: 'cancelled',
    statusLabel: '취소됨',
    ...overrides,
  });
}

function liveJob(): JobRecord {
  return {
    job_id: 'job-1',
    config_id: 'cfg-1',
    run_id: 'retina-live',
    status: 'running',
    status_label: '학습 중',
    created_at: '2026-08-05T00:00:00Z',
    started_at: '2026-08-05T00:00:00Z',
    finished_at: null,
    elapsed_seconds: 600,
    exit_code: null,
    message: null,
    artifacts: {},
    summary: {},
    settings: {},
    data_inputs: {},
    progress: {
      available: true,
      reason: null,
      message: null,
      architecture: 'retinanet_resnet50_fpn_v2',
      device: 'cuda',
      total_epochs: 15,
      current_epoch: 3,
      completed_epochs: 3,
      percent: 20,
      eta_seconds: 1800,
      epochs: [
        { epoch: 1, train_loss: 0.9, validation_loss: 0.88, epoch_seconds: 200, is_best: true },
        { epoch: 3, train_loss: 0.6, validation_loss: 0.62, epoch_seconds: 190, is_best: true },
      ],
      best: { epoch: 3, validation_loss: 0.62 },
    },
    log_lines: 120,
    orphan_note: null,
  };
}

function show(props: Partial<Parameters<typeof Runs>[0]> = {}) {
  const queue: QueueState = { entries: [], paused: false };
  return render(
    <MemoryRouter>
      <Runs
        datasetKey="v5-118cls"
        records={[record()]}
        liveJob={null}
        queue={queue}
        scope={{ backend: 'local', shared: false }}
        unnamedCount={0}
        teamRuns={[]}
        teamAvailable={false}
        error={null}
        onNewExperiment={() => {}}
        onRemoveFromQueue={() => {}}
        onResumeQueue={() => {}}
        onCancelJob={() => {}}
        {...props}
      />
    </MemoryRouter>,
  );
}

describe('Runs', () => {
  it('지금 도는 학습이 있으면 그 값을 맨 위에 크게 세운다', () => {
    show({ liveJob: liveJob() });

    expect(screen.getByText('지금 학습 중')).toBeInTheDocument();
    expect(screen.getByText('retina-live')).toBeInTheDocument();
    // best validation loss가 큰 숫자 자리에 옵니다.
    expect(screen.getByText('0.6200')).toBeInTheDocument();
    // 첫 epoch에서 얼마나 내려왔는지. 0.88 - 0.62 입니다.
    expect(screen.getByText('↓ 0.2600')).toBeInTheDocument();
    // 남은 시간은 loss와 같은 자리·같은 크기입니다.
    expect(screen.getByText('남은 시간')).toBeInTheDocument();
    expect(screen.getByText('~30분')).toBeInTheDocument();
  });

  it('남은 시간을 못 재면 0이 아니라 모른다고 적는다', () => {
    const job = liveJob();
    show({ liveJob: { ...job, progress: { ...job.progress, eta_seconds: null } } });

    expect(screen.getByText('알 수 없음')).toBeInTheDocument();
  });

  it('도는 학습이 없으면 라이브 카드를 아예 그리지 않는다', () => {
    show();

    expect(screen.queryByText('지금 학습 중')).not.toBeInTheDocument();
  });

  it('조건에 맞는 기록이 없으면 감췄다고 말한다', () => {
    show({ records: [record({ evaluated: false, registered: true })] });

    fireEvent.click(screen.getByRole('button', { name: /평가 완료/ }));

    expect(screen.getByText(/고른 조건에 맞는 기록이 없습니다/)).toBeInTheDocument();
  });

  it('Kaggle 점수가 없는 기록은 정렬해도 위로 올라오지 않는다', () => {
    show({
      records: [
        record({ runId: 'no-score', metrics: { ...record().metrics, kaggle: null } }),
        record({ runId: 'scored', metrics: { ...record().metrics, kaggle: 0.61 } }),
      ],
    });

    fireEvent.click(screen.getByRole('button', { name: 'Kaggle' }));

    const ids = screen.getAllByText(/^(no-score|scored)$/).map((node) => node.textContent);
    expect(ids[0]).toBe('scored');
  });

  // 35건 중 32건이 결과 없이 끝난 기록이라 볼 것 3건이 가운데 묻혀 있었습니다.
  it('결과 없이 끝난 기록은 접어 두고 몇 건인지 말한다', () => {
    show({ records: [record({ runId: 'good' }), failedRecord(), cancelledRecord()] });

    expect(screen.getByText('good')).toBeInTheDocument();
    expect(screen.queryByText('oom')).toBeNull();
    expect(screen.queryByText('stopped')).toBeNull();
    expect(screen.getByText('2건 (실패 1 · 취소·중단 1)')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /결과 없이 끝남/ }));

    // 접은 것을 전부 되돌려 줘야 합니다. 하나라도 빠지면 그 기록은 어디에서도 못 봅니다.
    expect(screen.getByText('oom')).toBeInTheDocument();
    expect(screen.getByText('stopped')).toBeInTheDocument();
  });

  // 등록되지 않은 이유가 실패라면 배지 두 개가 같은 말을 두 번 합니다. 그렇다고 상태를
  // 통째로 지우면 취소·중단 줄에 아무 표시도 남지 않아, 성공한 기록과 구별되지 않습니다.
  it('끝난 이유는 한 번만, 그러나 반드시 적는다', () => {
    show({
      records: [
        failedRecord(),
        cancelledRecord({ metrics: { ...record().metrics, bestValidationLoss: 0.5 } }),
        record({ runId: 'done', registered: false }),
      ],
    });

    // 결과가 남은 취소와 미등록 성공은 접히지 않습니다.
    expect(screen.getByText('취소됨')).toBeInTheDocument();
    expect(screen.getByText('미등록')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /결과 없이 끝남/ }));

    expect(screen.getByText('실패')).toBeInTheDocument();
    // 실패한 줄에는 미등록이 겹치지 않습니다 — 미등록 배지는 위의 성공 줄 하나뿐입니다.
    expect(screen.getAllByText('미등록')).toHaveLength(1);
  });

  // 접기는 전체 표에서만 합니다. 이미 좁혀 놓은 표에서 또 접으면 "12건이라는데 아무것도
  // 안 보인다"가 됩니다.
  it('미등록·실패 표에서는 접지 않는다', () => {
    show({ records: [record({ runId: 'good' }), failedRecord(), cancelledRecord()] });

    fireEvent.click(screen.getByRole('button', { name: /미등록·실패/ }));

    expect(screen.queryByText(/결과 없이 끝남/)).toBeNull();
    expect(screen.getByText('oom')).toBeInTheDocument();
    expect(screen.getByText('stopped')).toBeInTheDocument();
  });

  it('대기열에서 빼기를 누르면 그 항목만 알려 준다', () => {
    const onRemoveFromQueue = vi.fn();
    show({
      queue: {
        entries: [
          { entry_id: 'e1', config_id: 'c1', run_id: 'queued-1', queued_at: '2026-08-05T01:00:00Z' },
        ],
        paused: false,
      },
      onRemoveFromQueue,
    });

    fireEvent.click(screen.getByRole('button', { name: /학습 대기열/ }));
    fireEvent.click(screen.getByRole('button', { name: '빼기' }));

    expect(onRemoveFromQueue).toHaveBeenCalledWith('e1');
  });

  it('대기열이 멈춰 있으면 다시 돌릴 수 있다고 말한다', () => {
    const onResumeQueue = vi.fn();
    show({
      queue: {
        entries: [
          { entry_id: 'e1', config_id: 'c1', run_id: 'queued-1', queued_at: '2026-08-05T01:00:00Z' },
        ],
        paused: true,
      },
      onResumeQueue,
    });

    fireEvent.click(screen.getByRole('button', { name: /학습 대기열/ }));
    expect(screen.getByText('대기열이 멈춰 있습니다')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '다시 돌리기' }));
    expect(onResumeQueue).toHaveBeenCalled();
  });

  it('대기열 탭은 지금 도는 학습부터 세우고 그 오른쪽에 취소를 둔다', () => {
    const onCancelJob = vi.fn();
    show({ liveJob: liveJob(), onCancelJob });

    fireEvent.click(screen.getByRole('button', { name: /학습 대기열/ }));

    expect(screen.getByText('지금')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '취소' }));
    expect(onCancelJob).toHaveBeenCalledWith('job-1');
  });

  it('추가 버튼은 대기열 맨 끝에 있다', () => {
    const onNewExperiment = vi.fn();
    show({
      liveJob: liveJob(),
      queue: {
        entries: [
          { entry_id: 'e1', config_id: 'c1', run_id: 'q1', queued_at: '2026-08-05T01:00:00Z' },
          { entry_id: 'e2', config_id: 'c2', run_id: 'q2', queued_at: '2026-08-05T02:00:00Z' },
        ],
        paused: false,
      },
      onNewExperiment,
    });

    fireEvent.click(screen.getByRole('button', { name: /학습 대기열/ }));

    const add = screen.getByRole('button', { name: '+ 대기열에 추가' });
    const last = screen.getByText('q2');
    // 마지막 대기열 줄보다 뒤에 있어야 합니다. 줄의 맨 끝이 새로 넣는 것이 들어갈 자리입니다.
    expect(last.compareDocumentPosition(add) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    fireEvent.click(add);
    expect(onNewExperiment).toHaveBeenCalled();
  });

  it('로컬 저장소면 팀원 기록이 왜 없는지 화면이 말한다', () => {
    show();

    expect(screen.getByText('이 컴퓨터')).toBeInTheDocument();
    expect(screen.getByText(/PILL_STORAGE_S3_BUCKET/)).toBeInTheDocument();
  });

  it('dataset을 알 수 없어 감춘 기록이 있으면 몇 건인지 말한다', () => {
    show({ unnamedCount: 17 });

    expect(screen.getByText(/알 수 없는 기록 17건은 왼쪽 목록에 세우지 않았습니다/)).toBeInTheDocument();
  });

  it('학습 중 표는 팀원 것과 내 것을 함께 세우고 같은 학습을 두 번 세지 않는다', () => {
    const shared: TeamRun = {
      teamId: 't',
      cloudRunId: 'c1',
      localJobId: 'job-9',
      runId: 'mate-run',
      actorSub: 'sub-1',
      actorName: '김팀원',
      actorSource: 'cognito',
      status: 'running',
      settings: {},
      dataInputs: {},
      progress: { current_epoch: 7, total_epochs: 15, best: { validation_loss: 0.0612 } },
      summary: {},
      artifacts: {},
      evaluation: {},
      message: null,
      createdAt: '2026-08-05T00:00:00Z',
      startedAt: '2026-08-05T00:00:00Z',
      finishedAt: null,
      heartbeatAt: new Date().toISOString(),
      revision: 1,
    };
    // 내 학습도 팀에 올라가 있습니다. 그대로 두면 같은 학습이 두 줄이 됩니다.
    const mine: TeamRun = { ...shared, cloudRunId: 'c2', runId: 'retina-live', actorName: '나' };

    show({ liveJob: liveJob(), teamRuns: [shared, mine], teamAvailable: true });

    fireEvent.click(screen.getByText('학습 중'));

    expect(screen.getByText('mate-run')).toBeInTheDocument();
    expect(screen.getByText('김팀원')).toBeInTheDocument();
    expect(screen.getAllByText('retina-live')).toHaveLength(1);
    expect(screen.getByText('나 (이 컴퓨터)')).toBeInTheDocument();
  });

  it('heartbeat가 끊긴 팀 학습은 목록에도 개수에도 넣지 않는다', () => {
    const stale: TeamRun = {
      teamId: 't',
      cloudRunId: 'c9',
      localJobId: 'job-9',
      runId: 'stale-run',
      actorSub: 'sub-9',
      actorName: '박팀원',
      actorSource: 'cognito',
      status: 'running',
      settings: {},
      dataInputs: {},
      progress: {},
      summary: {},
      artifacts: {},
      evaluation: {},
      message: null,
      createdAt: '2026-08-05T00:00:00Z',
      startedAt: '2026-08-05T00:00:00Z',
      finishedAt: null,
      // 2분을 훌쩍 넘긴 heartbeat. 도는 척만 하고 있습니다.
      heartbeatAt: '2026-08-05T00:00:00Z',
      revision: 1,
    };

    show({ teamRuns: [stale], teamAvailable: true });

    expect(screen.getByText('학습 중').parentElement).toHaveTextContent('학습 중0');
    fireEvent.click(screen.getByText('학습 중'));
    expect(screen.queryByText('stale-run')).not.toBeInTheDocument();
    expect(screen.getByText('지금 돌고 있는 학습이 없습니다.')).toBeInTheDocument();
  });

  it('팀 연결이 꺼져 있으면 내 것만 보인다고 말한다', () => {
    show({ liveJob: liveJob(), teamAvailable: false });

    fireEvent.click(screen.getByText('학습 중'));

    expect(screen.getByText(/팀 실시간 연결이 꺼져 있어/)).toBeInTheDocument();
  });

  it('registry를 아직 못 읽었으면 이 컴퓨터뿐이라고 단정하지 않는다', () => {
    show({ scope: undefined });

    expect(screen.getByText('읽는 중')).toBeInTheDocument();
    expect(screen.queryByText('이 컴퓨터')).not.toBeInTheDocument();
  });
});
