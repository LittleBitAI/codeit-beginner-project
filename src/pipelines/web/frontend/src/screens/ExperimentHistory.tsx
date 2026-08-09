/**
 * 팀원 것까지 포함해 끝까지 마친 실험만 보는 화면.
 *
 * 학습 개요는 이 컴퓨터가 시작한 학습만 보여 줍니다. 여기는 registry index를 읽으므로,
 * S3 bucket을 쓰는 팀이라면 팀원이 등록한 실험도 함께 나옵니다. 그 구분을 화면이
 * 말해 주지 않으면, 로컬 저장소를 쓰는 사람은 "팀원 기록이 왜 없지"에서 멈춥니다.
 */

import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api } from '../api/client';
import { ExperimentTable } from '../components/ExperimentTable';
import { AlertRow, Panel, ScreenIntro } from '../components/primitives';
import { color, font } from '../design/tokens';
import { usePolling } from '../hooks/usePolling';
import { isComplete } from '../lib/completion';

export function ExperimentHistory() {
  const navigate = useNavigate();
  const listing = usePolling(() => api.listExperiments(), 5000);
  const [onlyComplete, setOnlyComplete] = useState(true);

  const experiments = useMemo(() => listing.data?.experiments ?? [], [listing.data]);
  const complete = useMemo(() => experiments.filter(isComplete), [experiments]);
  const shown = onlyComplete ? complete : experiments;
  const hidden = experiments.length - complete.length;
  const scope = listing.data?.scope;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1320 }}>
      <ScreenIntro
        title="팀원 것까지, 끝까지 마친 실험의 결과를 봅니다"
        terms={[
          { term: '평가', meaning: 'checkpoint로 mAP 같은 지표를 낸 단계입니다' },
          { term: '제출', meaning: '대회에 낼 submission.csv를 만들고 등록까지 마친 단계입니다' },
        ]}
      >
        <b>행을 누르면 그 실험의 세팅과 평가 결과 전체가 열립니다.</b> 학습 개요와 달리 이
        목록은 registry에서 읽으므로, 팀이 같은 S3 저장소를 쓰면 팀원이 등록한 실험도 함께
        나옵니다. 여러 실험을 견주려면 실험 비교 화면을 쓰세요.
      </ScreenIntro>

      {listing.error && (
        <AlertRow level="error" title="실험 목록을 불러오지 못했습니다">
          {listing.error}
        </AlertRow>
      )}

      {scope && !scope.shared && (
        <AlertRow level="info" title="지금은 이 컴퓨터에 등록된 실험만 보입니다">
          실험 목록은 저장소 설정을 그대로 따라갑니다. 지금 backend가{' '}
          <code style={{ fontFamily: font.mono }}>{scope.backend}</code>라서 팀이 공유하는
          기록을 읽지 않습니다. 팀원 것까지 보려면 <code style={{ fontFamily: font.mono }}>
            PILL_STORAGE_S3_BUCKET
          </code>
          을 설정한 뒤 서버를 다시 시작하세요.
        </AlertRow>
      )}

      <Panel
        title="실험 내역"
        right={
          <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMuted }}>
            {shown.length}건
          </span>
        }
        bodyStyle={{ padding: 0 }}
      >
        <label
          style={{
            display: 'flex',
            gap: 8,
            alignItems: 'center',
            padding: '10px 13px',
            borderBottom: `1px solid ${color.borderInner}`,
            cursor: 'pointer',
          }}
        >
          <input
            type="checkbox"
            checked={onlyComplete}
            onChange={() => setOnlyComplete((value) => !value)}
          />
          <span style={{ font: `500 12px/1.4 ${font.sans}`, color: color.textStrong }}>
            평가와 제출까지 끝난 실험만 보기
          </span>
          {/* 감춘 것이 있으면 몇 건인지 항상 말해 줍니다. 조용히 빼면 그만큼이 없는 줄 압니다. */}
          {onlyComplete && hidden > 0 && (
            <span style={{ font: `400 11.5px/1.4 ${font.sans}`, color: color.textMuted }}>
              아직 끝나지 않은 {hidden}건을 감췄습니다
            </span>
          )}
        </label>

        {listing.loading && experiments.length === 0 ? (
          <div style={{ padding: 20, font: `400 13px/1.6 ${font.sans}`, color: color.textBody }}>
            실험 기록을 불러오고 있습니다.
          </div>
        ) : (
          <ExperimentTable
            experiments={shown}
            onOpen={(experiment) => navigate(`/history/${encodeURIComponent(experiment.run_id)}`)}
            emptyMessage={
              onlyComplete && experiments.length > 0
                ? '평가와 제출까지 끝난 실험이 아직 없습니다. 위 체크를 풀면 나머지가 보입니다.'
                : '등록된 실험이 아직 없습니다. 학습을 마치고 평가를 돌리면 여기에 쌓입니다.'
            }
          />
        )}
      </Panel>
    </div>
  );
}
