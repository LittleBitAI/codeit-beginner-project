import { color, font } from '../design/tokens';
import type { LogLine } from '../api/types';

const LEVEL_COLOR: Record<LogLine['level'], string> = {
  warn: color.logWarn,
  error: color.logError,
  info: color.logText,
};

function lineColor(line: LogLine): string {
  if (line.level === 'info' && line.text.includes('최고 기록')) return color.logGood;
  return LEVEL_COLOR[line.level] ?? color.logText;
}

export function LogStream({
  lines,
  streaming,
  height = 352,
}: {
  lines: LogLine[];
  streaming: boolean;
  height?: number;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div
        style={{
          background: color.navy,
          padding: '11px 13px',
          height,
          overflow: 'auto',
          // 최신 줄이 아래에 붙고 자동으로 따라 내려가도록 뒤집어 쌓습니다.
          display: 'flex',
          flexDirection: 'column-reverse',
        }}
      >
        <div>
          {lines.length === 0 ? (
            <span style={{ ...typeLine, color: color.textFaint }}>아직 출력이 없습니다.</span>
          ) : (
            lines.map((line) => (
              <div key={line.seq} style={{ ...typeLine, color: lineColor(line) }}>
                {line.text}
              </div>
            ))
          )}
        </div>
      </div>
      <div
        style={{
          padding: '6px 13px',
          borderTop: `1px solid ${color.borderInner}`,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          background: color.surfaceAlt,
        }}
      >
        <span
          className={streaming ? 'pulse-dot' : undefined}
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: streaming ? color.teal : color.textFaint,
            flex: 'none',
          }}
        />
        <span style={{ font: `400 11.5px/1.4 ${font.sans}`, color: color.textMuted }}>
          {streaming ? '스트리밍 중' : `연결되지 않음 · ${lines.length}줄`}
        </span>
      </div>
    </div>
  );
}

const typeLine = {
  font: `400 11.5px/1.6 ${font.mono}`,
  whiteSpace: 'pre-wrap' as const,
  overflowWrap: 'anywhere' as const,
};
