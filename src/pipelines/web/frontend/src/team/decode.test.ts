import { describe, expect, it } from 'vitest';

import { decodeJson, decodeLines } from './decode';

const SETTINGS = { architecture: 'retinanet_resnet50_fpn_v2', optimizer: 'AdamW', epochs: 30 };
const LINES = [{ seq: 1, stream: 'system', level: 'info', text: '학습 시작', ts: 'now' }];

describe('decodeJson', () => {
  it('두 번 감싼 AWSJSON에서도 값을 꺼낸다', () => {
    // 배포된 resolver가 AWSJSON field에 문자열을 돌려주면 AppSync가 그 문자열을 한 번
    // 더 감싼다. 한 번만 풀면 객체가 아니라 문자열이 나와서 화면이 값을 통째로 잃었다.
    expect(decodeJson(JSON.stringify(JSON.stringify(SETTINGS)))).toEqual(SETTINGS);
  });

  it('한 번만 감싼 값도 그대로 읽는다', () => {
    // resolver를 고쳐 배포하면 이 모양으로 온다. 두 모양이 한동안 섞인다.
    expect(decodeJson(JSON.stringify(SETTINGS))).toEqual(SETTINGS);
  });

  it('이미 객체면 그대로 쓴다', () => {
    expect(decodeJson(SETTINGS)).toEqual(SETTINGS);
  });

  it('값이 없거나 JSON이 아니면 빈 객체를 준다', () => {
    // 빈 객체를 주어야 화면이 "값이 아직 없다"로 그린다. 예외로 화면을 죽이지 않는다.
    expect(decodeJson(null)).toEqual({});
    expect(decodeJson('학습 중')).toEqual({});
    expect(decodeJson(JSON.stringify(42))).toEqual({});
  });
});

describe('decodeLines', () => {
  it('두 번 감싼 로그 batch에서도 줄을 꺼낸다', () => {
    expect(decodeLines(JSON.stringify(JSON.stringify(LINES)))).toEqual(LINES);
  });

  it('배열이 아니면 빈 목록을 준다', () => {
    expect(decodeLines(JSON.stringify({ seq: 1 }))).toEqual([]);
    expect(decodeLines(undefined)).toEqual([]);
  });
});
