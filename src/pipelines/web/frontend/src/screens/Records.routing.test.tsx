/**
 * 기록을 눌렀을 때 어디로 가는가.
 *
 * 예전에는 이 컴퓨터가 돌렸는지로 갈라, 같은 실험인데 내 것과 팀원 것이 다른 화면을
 * 열었습니다. 이제는 **그 화면이 보여 줄 수 있는지**로 가릅니다. 캔버스는 등록된
 * 실행만 목록에 올리므로, 등록 안 된 것을 보내면 빈 화면이 됩니다.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';

import type { RunRecord } from '../lib/records';
import { Records } from './Records';

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
      kaggle: 0.6,
      map: 0.5,
      map50: null,
      map75: null,
      precision50: null,
      recall50: null,
      bestValidationLoss: 0.4,
      bestEpoch: 3,
      epochs: 15,
      elapsedSeconds: 60,
    },
    ...overrides,
  } as RunRecord;
}

function show(records: RunRecord[]) {
  return render(
    <MemoryRouter initialEntries={['/records']}>
      <Routes>
        <Route
          path="/records"
          element={
            <Records
              datasets={[{ key: 'v6', sub: '기록 1건', count: 1 }]}
              datasetKey="v6"
              onPickDataset={() => {}}
              records={records}
              scope={{ shared: false, backend: 'local' } as never}
              unnamedCount={0}
              error={null}
              onNewExperiment={() => {}}
            />
          }
        />
        <Route path="/canvas" element={<div>견줄 실행 화면</div>} />
        <Route path="/monitor/:jobId" element={<div>모니터 화면</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('기록에서 실행 열기', () => {
  it('등록된 실행은 이 컴퓨터가 돌렸어도 견줄 실행으로 간다', () => {
    show([record({ registered: true, jobId: 'job-77' })]);

    fireEvent.click(screen.getByText('run-a'));

    expect(screen.getByText('견줄 실행 화면')).toBeTruthy();
  });

  it('팀원이 돌린 등록 실행도 같은 화면으로 간다', () => {
    show([record({ registered: true, jobId: null })]);

    fireEvent.click(screen.getByText('run-a'));

    expect(screen.getByText('견줄 실행 화면')).toBeTruthy();
  });

  it('아직 등록되지 않은 내 실행은 모니터로 간다', () => {
    // 캔버스는 등록된 것만 목록에 올리므로 보내면 빈 화면이 됩니다.
    show([record({ registered: false, jobId: 'job-77', evaluated: false })]);

    fireEvent.click(screen.getByText('run-a'));

    expect(screen.getByText('모니터 화면')).toBeTruthy();
  });
});
