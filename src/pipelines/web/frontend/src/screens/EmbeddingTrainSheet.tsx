/**
 * crop embedding 하나를 학습으로 겁니다.
 *
 * 앙상블 화면 안에 있던 것을 여기로 옮겼습니다. **고르는 자리와 GPU를 쓰는 자리는
 * 다른 일입니다.** 한 화면에 있으면 "합치기"를 하러 들어온 사람 앞에 몇 시간짜리
 * 학습 버튼이 함께 놓입니다.
 *
 * detector 폼과 칸을 섞지 않습니다. 받는 이름이 절반도 겹치지 않아, 한 폼으로 묶으면
 * detector 설정에 backbone 선택이 뜹니다.
 *
 * 학습은 detector와 **같은 대기열**을 지납니다 — GPU로 가는 문이 둘이 되면 밤새
 * 돌리는 목록이 서로를 밀어냅니다. 진행과 취소는 모니터 화면에서 봅니다.
 */

import { useCallback, useEffect, useState } from 'react';

import { api } from '../api/client';
import type { EmbeddingDefaults, ProcessedDataset } from '../api/types';
import { AlertRow, Button, Field, Sheet, controlStyle } from '../components/primitives';
import { color, type } from '../design/tokens';
import { useTeam } from '../team/TeamContext';

/** 은행과 class map은 전처리 폴더 안에서 이 이름으로 놓입니다(data pipeline). */
const CROP_BANK_FILE = 'crop_bank.tar';
const CLASS_MAP_FILE = 'class_map.json';

export function EmbeddingTrainSheet({ onClose }: { onClose: () => void }) {
  const team = useTeam();
  const [defaults, setDefaults] = useState<EmbeddingDefaults | null>(null);
  const [datasets, setDatasets] = useState<ProcessedDataset[]>([]);
  const [directory, setDirectory] = useState('');
  const [backbone, setBackbone] = useState('');
  const [name, setName] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .embeddingDefaults()
      .then((result) => {
        if (!alive) return;
        setDefaults(result);
        setBackbone(result.defaults.backbone);
      })
      .catch(() => undefined);
    // 은행은 두 자리에 있습니다. 준비가 만든 것은 전처리 폴더 안에, 손으로 자른
    // 것은 `crop-bank/` 밑에. 한쪽만 보면 나머지로는 학습을 걸 방법이 없습니다.
    Promise.all([api.listDatasets(), api.listCropBanks()])
      // 은행이 없는 폴더로 걸면 대기열에 들어간 뒤 자기 차례에 실패합니다.
      .then(([processed, banks]) => {
        if (!alive) return;
        setDatasets(
          [...processed.datasets, ...banks.datasets].filter((item) => item.has_crop_bank),
        );
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setNotice(null);
    try {
      // 학습 화면과 같은 token을 보냅니다. 이 학습도 같은 대기열을 지나므로,
      // token 없이 넣으면 꺼내 시작할 때 팀 기록을 못 만들어 거절당합니다.
      const token = await team.getAccessToken();
      const result = await api.startEmbedding(
        {
          crop_bank_uri: `${directory}${CROP_BANK_FILE}`,
          class_map_uri: `${directory}${CLASS_MAP_FILE}`,
          backbone: backbone || undefined,
          run_id: name.trim() || undefined,
        },
        token,
      );
      setNotice(`${result.run_id} 학습을 걸었습니다. 진행은 모니터 화면에서 봅니다.`);
      setName('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '학습을 걸지 못했습니다.');
    }
  }, [backbone, directory, name, team]);

  return (
    <Sheet
      title="embedding 학습"
      subtitle="앙상블 화면이 재순위에 쓸 crop embedding"
      onClose={onClose}
      footer={
        <>
          <Button kind="primary" onClick={start} disabled={!directory}>
            학습 걸기
          </Button>
          <span style={{ ...type.note, color: color.textMuted }}>
            학습 대기열을 함께 씁니다. 도는 학습이 있으면 그 뒤에 섭니다.
          </span>
        </>
      }
    >
      <div style={{ display: 'grid', gap: 20 }}>
        <p style={{ ...type.body, color: color.textBody, margin: 0, textWrap: 'pretty' }}>
          여기서 학습한 embedding은 <b style={{ color: color.text }}>앙상블 화면</b>에서 고릅니다.
          제출에 남은 상자의 점수만 다시 매기는 데 쓰고, 상자 자체는 바꾸지 않습니다.
        </p>

        {datasets.length === 0 ? (
          <AlertRow level="warning" title="참조 crop이 없습니다">
            은행이 있는 전처리 폴더도, 손으로 올린 은행도 없습니다. dataset 준비에서 은행을
            함께 만드세요.
          </AlertRow>
        ) : null}

        <Field label="참조 crop (crop 은행)" hint="이 폴더의 crop_bank.tar와 class_map.json을 씁니다.">
          <select
            style={controlStyle}
            value={directory}
            onChange={(event) => setDirectory(event.target.value)}
            aria-label="crop 은행"
          >
            <option value="">참조 crop을 고르세요</option>
            {/* 두 자리를 이어 붙이므로 이름은 겹칠 수 있습니다. 폴더가 진짜 열쇠입니다. */}
            {datasets.map((item) => (
              <option key={item.directory} value={item.directory}>
                {item.name}
              </option>
            ))}
          </select>
        </Field>

        <Field label="backbone">
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
        </Field>

        <Field label="이름" hint="비우면 자동으로 짓습니다.">
          <input
            style={controlStyle}
            placeholder="이름 (비우면 자동)"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>

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
      </div>
    </Sheet>
  );
}

export default EmbeddingTrainSheet;
