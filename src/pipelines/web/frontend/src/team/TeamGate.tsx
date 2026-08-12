import type { ReactNode } from 'react';

import { Button, Panel } from '../components/primitives';
import { color, font } from '../design/tokens';
import { useTeam } from './TeamContext';

export function TeamGate({ children }: { children: ReactNode }) {
  const team = useTeam();
  if (!team.ready) {
    return <div style={{ padding: 32, fontFamily: font.sans }}>팀 설정을 확인하고 있습니다…</div>;
  }
  // actor가 있으면 서버가 "여기는 로그인할 수 없는 환경"이라고 알려 준 것입니다.
  // Colab 주소는 세션마다 바뀌어 Cognito redirect에 등록할 수 없으므로, 막으면
  // 화면을 아예 못 씁니다. 학습은 그 이름으로 팀에 기록됩니다.
  if (!team.config.enabled || team.user || team.config.actor) return <>{children}</>;
  return (
    <main
      style={{
        minHeight: '100vh',
        display: 'grid',
        placeItems: 'center',
        background: color.page,
        padding: 24,
      }}
    >
      <Panel title="팀 학습 관제판 로그인" style={{ width: 'min(440px, 100%)' }}>
        <p style={{ font: `400 13.5px/1.7 ${font.sans}`, color: color.textBody, marginTop: 0 }}>
          관리자에게 초대받은 팀원만 학습 설정, 실시간 로그와 결과를 볼 수 있습니다.
        </p>
        {team.error && <p style={{ color: color.danger }}>{team.error}</p>}
        <Button kind="primary" onClick={() => void team.login()}>
          Cognito로 로그인
        </Button>
      </Panel>
    </main>
  );
}
