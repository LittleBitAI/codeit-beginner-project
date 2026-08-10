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
import { completionOf } from '../lib/completion';

export function ExperimentHistory() {
  const navigate = useNavigate();
  const listing = usePolling(() => api.listExperiments(), 5000);
  const [onlyEvaluated, setOnlyEvaluated] = useState(true);
  const [onlySubmitted, setOnlySubmitted] = useState(false);
  // 기록된 실제 mAP는 기본으로 잠급니다. 표를 지나가다 누른 저장이 이미 적어 둔
  // 점수를 갈아치우면 그 값이 무엇이었는지 아무도 모릅니다. 고칠 때만 켭니다.
  const [editingScores, setEditingScores] = useState(false);

  const experiments = useMemo(() => listing.data?.experiments ?? [], [listing.data]);
  const shown = useMemo(
    () =>
      experiments.filter((experiment) => {
        const completion = completionOf(experiment);
        return (
          (!onlyEvaluated || completion.evaluated) &&
          (!onlySubmitted || completion.submitted)
        );
      }),
    [experiments, onlyEvaluated, onlySubmitted],
  );
  const hidden = experiments.length - shown.length;
  const scope = listing.data?.scope;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 1320 }}>
      <ScreenIntro
        title="팀원 것까지, 끝까지 마친 실험의 결과를 봅니다"
        terms={[
          { term: '평가', meaning: 'checkpoint로 mAP 같은 지표를 낸 단계입니다' },
          { term: '제출', meaning: 'Kaggle에 올린 뒤 실제 점수까지 기록한 단계입니다' },
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
          <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ font: `400 12px/1 ${font.mono}`, color: color.textMuted }}>
              {shown.length}건
            </span>
            <button
              type="button"
              aria-pressed={editingScores}
              title={
                editingScores
                  ? '다 고쳤으면 다시 눌러 잠그세요'
                  : '이미 기록된 실제 mAP를 고칩니다'
              }
              onClick={() => setEditingScores((value) => !value)}
              style={{
                font: `${editingScores ? 600 : 500} 11.5px/1 ${font.sans}`,
                padding: '6px 10px',
                borderRadius: 4,
                color: editingScores ? '#fff' : color.textBody,
                background: editingScores ? color.amber : color.surface,
                border: `1px solid ${editingScores ? color.amber : color.borderControl}`,
              }}
            >
              실제 mAP 수정
            </button>
          </span>
        }
        bodyStyle={{ padding: 0 }}
      >
        {/* 켠 것을 잊고 표를 만지면 잠근 뜻이 없어지므로, 켜져 있는 동안 계속 말합니다. */}
        {editingScores && (
          <div
            style={{
              display: 'flex',
              gap: 8,
              alignItems: 'center',
              padding: '10px 13px',
              background: color.amberTint,
              borderBottom: `1px solid ${color.borderInner}`,
              font: `400 12px/1.5 ${font.sans}`,
              color: color.textStrong,
            }}
          >
            <b style={{ color: color.amber }}>실제 mAP를 고칠 수 있습니다.</b>
            잘못 적은 칸의 숫자를 바꾼 뒤 그 줄의 <b>수정</b>을 누르세요. 다 고쳤으면 위
            <b> 실제 mAP 수정</b>을 다시 눌러 잠그세요.
          </div>
        )}

        <div
          style={{
            display: 'flex',
            gap: 16,
            alignItems: 'center',
            padding: '10px 13px',
            borderBottom: `1px solid ${color.borderInner}`,
            flexWrap: 'wrap',
          }}
        >
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={onlyEvaluated}
              onChange={() => setOnlyEvaluated((value) => !value)}
            />
            <span style={{ font: `500 12px/1.4 ${font.sans}`, color: color.textStrong }}>
              평가가 끝난 실험만 보기
            </span>
          </label>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={onlySubmitted}
              onChange={() => setOnlySubmitted((value) => !value)}
            />
            <span style={{ font: `500 12px/1.4 ${font.sans}`, color: color.textStrong }}>
              Kaggle 제출까지 끝난 실험만 보기
            </span>
          </label>
          {/* 감춘 것이 있으면 몇 건인지 항상 말해 줍니다. 조용히 빼면 그만큼이 없는 줄 압니다. */}
          {(onlyEvaluated || onlySubmitted) && hidden > 0 && (
            <span style={{ font: `400 11.5px/1.4 ${font.sans}`, color: color.textMuted }}>
              조건에 맞지 않는 {hidden}건을 감췄습니다
            </span>
          )}
        </div>

        {listing.loading && experiments.length === 0 ? (
          <div style={{ padding: 20, font: `400 13px/1.6 ${font.sans}`, color: color.textBody }}>
            실험 기록을 불러오고 있습니다.
          </div>
        ) : (
          <ExperimentTable
            experiments={shown}
            onOpen={(experiment) => navigate(`/history/${encodeURIComponent(experiment.run_id)}`)}
            onKaggleScoreSave={async (runId, score, overwrite) => {
              await api.saveKaggleScore(runId, score, overwrite);
              listing.refresh();
            }}
            kaggleScoreEditable={editingScores}
            emptyMessage={
              (onlyEvaluated || onlySubmitted) && experiments.length > 0
                ? '선택한 완료 조건에 맞는 실험이 없습니다. 위 체크를 풀면 나머지가 보입니다.'
                : '등록된 실험이 아직 없습니다. 학습을 마치고 평가를 돌리면 여기에 쌓입니다.'
            }
          />
        )}
      </Panel>
    </div>
  );
}
