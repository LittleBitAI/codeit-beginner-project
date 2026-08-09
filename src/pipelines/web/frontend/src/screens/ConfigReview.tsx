import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { api, ApiError } from '../api/client';
import type { Defaults, JobListing } from '../api/types';
import { AlertRow, Button, Panel, ScreenIntro, StatusBadge } from '../components/primitives';
import { IconCheck } from '../components/Icon';
import { color, font, radius } from '../design/tokens';
import { describeRun, diffAgainstDefaults } from '../lib/describeRun';
import { useDraft } from '../state/DraftContext';
import { useTeam } from '../team/TeamContext';

export function ConfigReview({
  defaults,
  listing,
  onStarted,
}: {
  defaults: Defaults | null;
  listing: JobListing | null;
  onStarted: () => void;
}) {
  const navigate = useNavigate();
  const { saved } = useDraft();
  const team = useTeam();
  const [starting, setStarting] = useState(false);
  const [queueing, setQueueing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!saved) {
    return (
      <div style={{ maxWidth: 900 }}>
        <ScreenIntro title="검토할 설정이 없습니다">
          새 실험 화면에서 설정을 저장하면 여기에서 마지막으로 확인한 뒤 학습을 시작할 수 있습니다.
        </ScreenIntro>
        <Button kind="primary" onClick={() => navigate('/new')}>
          새 실험 만들기
        </Button>
      </div>
    );
  }

  const train = saved.config.train as Record<string, unknown>;
  const optimizer = String(train.optimizer ?? 'SGD');
  const defaultValues = Object.fromEntries(
    (defaults?.fields ?? [])
      .filter((spec) => spec.default !== undefined && spec.default !== null)
      .map((spec) => [
        spec.name,
        spec.defaults_by_optimizer?.[optimizer] ?? spec.default,
      ]),
  );
  const diff = diffAgainstDefaults(train, defaultValues);
  const busy = Boolean(listing?.active_job_id);

  const checks = [
    '설정 형식과 값 범위가 train pipeline의 규칙과 같습니다.',
    'data artifact 위치가 저장소 안의 상대 경로이거나 s3:// URI입니다.',
    `실행 이름 '${String(train.run_id)}'과 같은 결과가 아직 없습니다.`,
    `출력 경로 '${String(train.output_dir)}'가 저장소를 벗어나지 않습니다.`,
    train.device === 'cuda' ? 'CUDA를 사용할 수 있습니다.' : 'CPU로 실행합니다.',
  ];

  async function addToQueue() {
    setQueueing(true);
    setError(null);
    try {
      const queue = await api.addToQueue(saved!.config_id, await team.getAccessToken());
      onStarted();
      // 비어 있었으면 곧바로 시작되므로 그 학습 화면으로, 줄을 섰으면 개요로 갑니다.
      navigate(queue.started ? `/monitor/${queue.started.job_id}` : '/');
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '대기열에 넣지 못했습니다.');
    } finally {
      setQueueing(false);
    }
  }

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const job = await api.startJob(saved!.config_id, await team.getAccessToken());
      onStarted();
      navigate(`/monitor/${job.job_id}`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '학습을 시작하지 못했습니다.');
    } finally {
      setStarting(false);
    }
  }

  return (
    <div style={{ maxWidth: 1320, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <ScreenIntro title="GPU 시간을 쓰기 전 마지막 확인입니다">
        아래 문장은 저장된 설정을 그대로 풀어 쓴 것입니다. 시작하면 이 설정으로 학습 process가 실행되고,
        끝날 때까지 다른 학습은 바로 시작되지 않습니다. 지금 도는 학습이 있으면 대기열에 넣어 두세요.
      </ScreenIntro>

      <Panel>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ font: `700 17px/1.2 ${font.mono}`, color: color.text }}>
            {saved.run_id}
          </span>
          <StatusBadge status="queued" label="저장됨 · 미실행" />
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            borderTop: `1px solid ${color.borderInner}`,
            marginTop: 14,
          }}
        >
          {[
            ['MODEL', String(train.architecture ?? defaults?.architecture ?? 'Faster R-CNN')],
            ['OPTIMIZER', optimizer],
            ['DEVICE', String(train.device)],
            ['EPOCHS', String(train.epochs)],
            ['BATCH', String(train.batch_size)],
            ['LR', String(train.learning_rate)],
            ['SEED', String(train.seed)],
            ...(optimizer === 'SGD'
              ? [['MOMENTUM', String(train.momentum)]]
              : [
                  ['BETA 1', String(train.beta1)],
                  ['BETA 2', String(train.beta2)],
                  ['EPSILON', String(train.epsilon)],
                ]),
          ].map(([label, value], index, all) => (
            <div
              key={label}
              style={{
                padding: '11px 13px',
                borderRight: index === all.length - 1 ? undefined : `1px solid ${color.borderInner}`,
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
                minWidth: 0,
              }}
            >
              <span style={{ font: `500 11px/1.3 ${font.mono}`, color: color.textMuted }}>
                {label}
              </span>
              <span
                style={{
                  font: `600 13px/1.2 ${font.mono}`,
                  color: color.text,
                  overflowWrap: 'anywhere',
                }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
        <p
          style={{
            font: `400 13.5px/1.75 ${font.sans}`,
            color: color.textStrong,
            borderTop: `1px solid ${color.borderInner}`,
            paddingTop: 13,
            margin: '13px 0 0',
          }}
        >
          {describeRun(saved.config)}
        </p>

        {/* 어떤 데이터로 도는지 시작 전에 눈으로 확인할 수 있어야 합니다. */}
        <div
          style={{
            borderTop: `1px solid ${color.borderInner}`,
            paddingTop: 12,
            marginTop: 13,
            display: 'flex',
            flexDirection: 'column',
            gap: 5,
          }}
        >
          <span style={{ font: `600 12.5px/1 ${font.sans}`, color: color.textStrong }}>
            이 학습이 읽을 데이터
          </span>
          {Object.entries(saved.config.inputs?.data ?? {}).map(([key, uri]) => (
            <div key={key} style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <span
                style={{
                  font: `500 11.5px/1.5 ${font.mono}`,
                  color: color.textMuted,
                  minWidth: 170,
                }}
              >
                {key}
              </span>
              <span
                style={{
                  font: `400 12px/1.5 ${font.mono}`,
                  color: color.textStrong,
                  overflowWrap: 'anywhere',
                }}
              >
                {String(uri)}
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-start' }}>
        <Panel title="기본값과 다른 항목" style={{ flex: '2 1 340px' }} bodyStyle={{ padding: 0 }}>
          {diff.length === 0 ? (
            <div style={{ padding: '20px 16px', font: `400 12.5px/1.6 ${font.sans}`, color: color.textBody }}>
              모든 값이 기본값과 같습니다.
            </div>
          ) : (
            <div
              style={{
                background: color.navy,
                padding: '14px 0',
                font: `400 12px/1.75 ${font.mono}`,
                maxHeight: 420,
                overflow: 'auto',
              }}
            >
              <div style={{ padding: '1px 15px', color: '#7F91A8' }}>@@ train @@</div>
              {diff.map((row) => (
                <div key={row.key}>
                  {/* diff에서만 배경 색조를 씁니다. Primer diff 관례입니다. */}
                  <div
                    style={{
                      padding: '1px 15px',
                      color: '#F0A0A0',
                      background: 'rgba(198,40,40,.16)',
                      whiteSpace: 'pre',
                    }}
                  >
                    - {row.key}: {row.before}
                  </div>
                  <div
                    style={{
                      padding: '1px 15px',
                      color: '#8FE0A8',
                      background: 'rgba(31,138,59,.16)',
                      whiteSpace: 'pre',
                    }}
                  >
                    + {row.key}: {row.after}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <div style={{ flex: '1 1 290px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Panel title="실행 전 검사">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {checks.map((text) => (
                <div key={text} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                  <span style={{ marginTop: 2 }}>
                    <IconCheck size={12} color={color.green} />
                  </span>
                  <span style={{ font: `400 12.5px/1.5 ${font.sans}`, color: color.textStrong }}>
                    {text}
                  </span>
                </div>
              ))}
            </div>
          </Panel>

          {saved.warnings.map((item) => (
            <AlertRow key={`${item.field}-${item.message}`} level="warning" title={item.field}>
              {item.message}
            </AlertRow>
          ))}

          {busy && (
            <AlertRow level="warning" title="다른 학습이 실행 중입니다">
              한 번에 하나만 실행할 수 있어 지금 바로 시작하지는 못합니다. 대기열에 넣으면 앞
              학습이 끝나는 대로 이 설정이 이어서 시작됩니다.
            </AlertRow>
          )}

          {error && (
            <AlertRow level="error" title="시작하지 못했습니다">
              {error}
            </AlertRow>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <Button
              kind="teal"
              disabled={busy || starting}
              onClick={() => void start()}
              style={{ padding: '11px 0', font: `600 13px/1 ${font.sans}`, borderRadius: radius.control }}
            >
              {starting ? '시작하는 중…' : '학습 시작'}
            </Button>
            {/* 여러 설정을 줄 세워 두고 자러 갈 때 씁니다. 비어 있으면 곧바로 시작합니다.
                다른 학습이 도는 중에도 눌러야 합니다. 줄을 세우는 기능을 정작 줄 세울
                상황에서 막으면 남는 쓸모가 없습니다. */}
            <Button
              disabled={starting || queueing}
              onClick={() => void addToQueue()}
              style={{ padding: '9px 0' }}
            >
              {queueing ? '넣는 중…' : '대기열에 추가'}
            </Button>
            <Button onClick={() => navigate('/new')} style={{ padding: '9px 0' }}>
              ← 설정으로 돌아가기
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
