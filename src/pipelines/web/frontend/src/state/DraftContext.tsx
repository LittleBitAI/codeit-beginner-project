import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

import type { CreatedConfig } from '../api/types';

/** 새 실험 화면이 만드는 설정 초안. 값은 전부 문자열로 들고 다니다가 보낼 때 바꿉니다. */
export interface Draft {
  train: Record<string, string>;
  data: Record<string, string>;
  /** artifact 값을 어느 데이터셋에서 채웠는지. 데이터셋이 바뀐 것을 알아채는 데 씁니다. */
  sourceKey?: string | null;
}

export const EMPTY_DRAFT: Draft = { train: {}, data: {}, sourceKey: null };

const STORAGE_KEY = 'pill-training-draft';

interface DraftContextValue {
  draft: Draft;
  setTrainField: (name: string, value: string) => void;
  setDataField: (name: string, value: string) => void;
  /** 전처리 데이터셋을 고르면 4칸을 한꺼번에 채웁니다. */
  setDataFields: (values: Record<string, string>, sourceKey?: string | null) => void;
  resetDraft: () => void;
  saved: CreatedConfig | null;
  setSaved: (value: CreatedConfig | null) => void;
}

const DraftContext = createContext<DraftContextValue | null>(null);

function readStored(): Draft {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY_DRAFT;
    const parsed = JSON.parse(raw) as Draft;
    return { train: parsed.train ?? {}, data: parsed.data ?? {} };
  } catch {
    return EMPTY_DRAFT;
  }
}

function persist(draft: Draft): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
  } catch {
    // sessionStorage를 못 써도 화면은 계속 동작해야 합니다.
  }
}

export function DraftProvider({ children }: { children: ReactNode }) {
  const [draft, setDraft] = useState<Draft>(readStored);
  const [saved, setSaved] = useState<CreatedConfig | null>(null);

  const update = useCallback((next: Draft) => {
    setDraft(next);
    persist(next);
    // 값이 바뀌면 저장해 둔 설정은 더 이상 최신이 아닙니다.
    setSaved(null);
  }, []);

  const setTrainField = useCallback(
    (name: string, value: string) => {
      update({ ...draft, train: { ...draft.train, [name]: value } });
    },
    [draft, update],
  );

  const setDataField = useCallback(
    (name: string, value: string) => {
      update({ ...draft, data: { ...draft.data, [name]: value } });
    },
    [draft, update],
  );

  const setDataFields = useCallback(
    (values: Record<string, string>, sourceKey?: string | null) => {
      update({
        ...draft,
        data: { ...draft.data, ...values },
        sourceKey: sourceKey === undefined ? draft.sourceKey : sourceKey,
      });
    },
    [draft, update],
  );

  const resetDraft = useCallback(() => update(EMPTY_DRAFT), [update]);

  const value = useMemo(
    () => ({ draft, setTrainField, setDataField, setDataFields, resetDraft, saved, setSaved }),
    [draft, setTrainField, setDataField, setDataFields, resetDraft, saved],
  );

  return <DraftContext.Provider value={value}>{children}</DraftContext.Provider>;
}

export function useDraft(): DraftContextValue {
  const value = useContext(DraftContext);
  if (!value) throw new Error('DraftProvider 안에서만 쓸 수 있습니다.');
  return value;
}
