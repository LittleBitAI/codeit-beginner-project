/**
 * 재순위에 쓸 crop embedding을 **고르기만** 합니다.
 *
 * 학습은 왼쪽 메뉴의 `embedding 학습`에 있습니다. 여기 함께 있었을 때는 한 화면이
 * "이미 있는 것을 고르는 자리"와 "GPU를 몇 시간 쓰는 자리"를 겸했고, 무엇을 누르는
 * 중인지 화면만 보고는 알 수 없었습니다.
 *
 * 고르지 않으면 지금까지와 똑같은 제출이 나갑니다. 재순위는 상자를 바꾸지 않고
 * 남은 행의 점수만 고쳐 class 안 순위를 바꿉니다.
 */

import { useEffect, useState } from 'react';

import { api } from '../api/client';
import type { EmbeddingRun } from '../api/types';
import { color, type } from '../design/tokens';

interface Props {
  /** 재순위에 쓰기로 고른 embedding 이름들. */
  selected: string[];
  onToggle: (runId: string) => void;
  disabled?: boolean;
  /** 하나도 고르지 않았을 때 그것이 왜 문제인지. 화면마다 다릅니다. */
  hint?: string;
}

export function EmbeddingPanel({ selected, onToggle, disabled = false, hint }: Props) {
  const [runs, setRuns] = useState<EmbeddingRun[]>([]);

  // 화면을 닫은 뒤 응답이 도착하면 사라진 화면의 상태를 건드립니다.
  useEffect(() => {
    let alive = true;
    api
      .embeddingRuns()
      .then((result) => {
        if (alive) setRuns(result.runs);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <p style={{ ...type.note, color: color.textMid, margin: 0 }}>
        {hint ??
          '고르면 합친 뒤 그 embedding으로 점수만 다시 매깁니다. 상자와 행 수는 그대로입니다.'}
      </p>

      {runs.length === 0 ? (
        <p style={{ ...type.body, color: color.textMuted, margin: 0 }}>
          학습해 둔 embedding이 없습니다. 왼쪽 메뉴{' '}
          <b style={{ color: color.textMid }}>embedding 학습</b>에서 하나 학습하세요.
        </p>
      ) : (
        <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid' }}>
          {runs.map((item) => (
            <li key={item.run_id} style={{ borderTop: `1px solid ${color.borderRow}` }}>
              <label
                style={{
                  display: 'flex',
                  gap: 10,
                  alignItems: 'baseline',
                  cursor: item.ready ? 'pointer' : 'default',
                  padding: '9px 2px',
                }}
              >
                <input
                  type="checkbox"
                  checked={selected.includes(item.run_id)}
                  onChange={() => onToggle(item.run_id)}
                  disabled={disabled || !item.ready}
                />
                <span style={{ ...type.body, color: color.text, flex: 1, minWidth: 0 }}>
                  {item.run_id}
                </span>
                <span style={{ ...type.note, color: color.textMuted }}>{item.backbone}</span>
                {item.ready ? null : (
                  <span style={{ ...type.note, color: color.warn }}>{item.status}</span>
                )}
              </label>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default EmbeddingPanel;
