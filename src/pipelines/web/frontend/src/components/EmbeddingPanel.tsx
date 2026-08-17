/**
 * 재순위에 쓸 crop embedding을 고르고, 없으면 여기서 학습을 겁니다.
 *
 * 융합 화면 안에 두는 이유는 이 학습이 **재순위 말고 쓸 데가 없기** 때문입니다.
 * 따로 떼면 무엇에 쓰는 model인지 화면만 보고는 알 수 없습니다.
 *
 * 고르지 않으면 지금까지와 똑같은 제출이 나갑니다. 재순위는 상자를 바꾸지 않고
 * 남은 행의 점수만 고쳐 class 안 순위를 바꿉니다.
 */

import { useCallback, useEffect, useState } from 'react';

import { api } from '../api/client';
import type { EmbeddingDefaults, EmbeddingRun, ProcessedDataset } from '../api/types';
import { AlertRow, Button, MicroLabel, controlStyle } from './primitives';
import { color, type } from '../design/tokens';

/** 은행과 class map은 전처리 폴더 안에서 이 이름으로 놓입니다(data pipeline). */
const CROP_BANK_FILE = 'crop_bank.tar';
const CLASS_MAP_FILE = 'class_map.json';

interface Props {
  /** 재순위에 쓰기로 고른 embedding 이름들. */
  selected: string[];
  onToggle: (runId: string) => void;
  disabled?: boolean;
}

export function EmbeddingPanel({ selected, onToggle, disabled = false }: Props) {
  const [runs, setRuns] = useState<EmbeddingRun[]>([]);
  const [defaults, setDefaults] = useState<EmbeddingDefaults | null>(null);
  const [datasets, setDatasets] = useState<ProcessedDataset[]>([]);
  const [open, setOpen] = useState(false);
  const [directory, setDirectory] = useState('');
  const [backbone, setBackbone] = useState('');
  const [name, setName] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api
      .embeddingRuns()
      .then((result) => setRuns(result.runs))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .embeddingRuns()
      .then((result) => {
        if (alive) setRuns(result.runs);
      })
      .catch(() => undefined);
    api
      .embeddingDefaults()
      .then((result) => {
        if (!alive) return;
        setDefaults(result);
        setBackbone(result.defaults.backbone);
      })
      .catch(() => undefined);
    api
      .listDatasets()
      .then((result) => {
        if (alive) setDatasets(result.datasets.filter((item) => item.has_crop_bank));
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const start = useCallback(() => {
    setError(null);
    setNotice(null);
    api
      .startEmbedding({
        crop_bank_uri: `${directory}${CROP_BANK_FILE}`,
        class_map_uri: `${directory}${CLASS_MAP_FILE}`,
        backbone: backbone || undefined,
        run_id: name.trim() || undefined,
      })
      .then((result) => {
        setNotice(`${result.run_id} 학습을 걸었습니다. 진행은 모니터 화면에서 봅니다.`);
        setName('');
        refresh();
      })
      .catch((cause) =>
        setError(cause instanceof Error ? cause.message : '학습을 걸지 못했습니다.'),
      );
  }, [backbone, directory, name, refresh]);

  return (
    <section style={{ display: 'grid', gap: 8 }}>
      <MicroLabel>재순위 embedding ({selected.length}개 선택)</MicroLabel>
      <p style={{ ...type.note, color: color.textMid, margin: 0 }}>
        고르면 합친 뒤 그 embedding으로 <strong>점수만</strong> 다시 매깁니다. 상자와 행 수는
        그대로입니다. 고르지 않으면 지금까지와 똑같은 제출이 나갑니다.
      </p>

      {runs.length === 0 ? (
        <p style={{ ...type.body, color: color.textMid, margin: 0 }}>
          학습해 둔 embedding이 없습니다. 아래에서 하나 학습하세요.
        </p>
      ) : (
        <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 6 }}>
          {runs.map((item) => (
            <li key={item.run_id}>
              <label style={{ display: 'flex', gap: 10, alignItems: 'baseline', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={selected.includes(item.run_id)}
                  onChange={() => onToggle(item.run_id)}
                  disabled={disabled || !item.ready}
                />
                <span style={{ ...type.body, color: color.text, flex: 1 }}>{item.run_id}</span>
                <span style={{ ...type.note, color: color.textMid }}>{item.backbone}</span>
                {item.ready ? null : (
                  <span style={{ ...type.note, color: color.warn }}>{item.status}</span>
                )}
              </label>
            </li>
          ))}
        </ul>
      )}

      <div>
        <Button onClick={() => setOpen((value) => !value)}>
          {open ? '학습 칸 닫기' : '새 embedding 학습'}
        </Button>
      </div>

      {open ? (
        <div style={{ display: 'grid', gap: 8 }}>
          {datasets.length === 0 ? (
            <p style={{ ...type.note, color: color.warn, margin: 0 }}>
              crop 은행이 있는 전처리 폴더가 없습니다. 데이터 준비에서 은행을 함께 만드세요.
            </p>
          ) : null}
          <select
            style={controlStyle}
            value={directory}
            onChange={(event) => setDirectory(event.target.value)}
            aria-label="crop 은행"
          >
            <option value="">참조 crop을 고르세요</option>
            {datasets.map((item) => (
              <option key={item.name} value={item.directory}>
                {item.name}
              </option>
            ))}
          </select>
          <select
            style={controlStyle}
            value={backbone}
            onChange={(event) => setBackbone(event.target.value)}
            aria-label="backbone"
          >
            {(defaults?.backbones ?? []).map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <input
            style={controlStyle}
            placeholder="이름 (비우면 자동)"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <div>
            <Button onClick={start} disabled={!directory}>
              학습 걸기
            </Button>
            <span style={{ ...type.note, color: color.textMid, marginLeft: 10 }}>
              학습 대기열을 함께 씁니다. 도는 학습이 있으면 그 뒤에 섭니다.
            </span>
          </div>
        </div>
      ) : null}

      {notice ? (
        <AlertRow level="success" title="embedding">
          {notice}
        </AlertRow>
      ) : null}
      {error ? (
        <AlertRow level="error" title="embedding">
          {error}
        </AlertRow>
      ) : null}
    </section>
  );
}

export default EmbeddingPanel;
