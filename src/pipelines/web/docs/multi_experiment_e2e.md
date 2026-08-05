# Web multi-experiment E2E 시나리오

이 시나리오는 같은 dataset으로 설정을 달리한 학습 두 건이 Web에 저장되고, 서버를
다시 띄운 뒤에도 실험 비교 화면까지 도달하는지 확인합니다.

## 검증하는 흐름

1. 공용 JSON fixture에서 같은 data artifact 4개와 서로 다른 학습 설정 두 개를 읽습니다.
2. Web API로 config를 저장하고 학습 job을 차례로 시작합니다.
3. 실제 GPU 학습 대신 정상적인 pipeline stdout을 반환하는 작은 fake process를 씁니다.
4. 메모리의 `JobManager`를 새 instance로 바꿔 디스크 record 재로딩을 재현합니다.
5. `/api/train/experiments`가 두 실험, 같은 dataset identity, 서로 다른 설정과 loss를
   반환하는지 확인합니다.
6. Frontend 전체 `App`을 `/compare`에서 열고 최근 두 실험을 선택해 같은 dataset 안내와
   비교표가 표시되는지 확인합니다.

Fixture는
`src/pipelines/web/frontend/src/test/fixtures/multiExperiment.json` 하나를 backend와
frontend가 함께 사용합니다. 실제 dataset, checkpoint, log는 만들지 않습니다.

## 실행

저장소 root에서 backend 시나리오를 실행합니다.

```text
python -m pytest src/pipelines/web/tests/test_web_multi_experiment_e2e.py -q
```

Frontend 시나리오는 frontend directory에서 실행합니다.

```text
npm test -- --run src/test/multiExperiment.e2e.test.tsx
```

둘 중 하나가 실패하면 출력에 표시된 첫 API 단계 또는 화면 assertion부터 확인합니다.
테스트는 임시 저장소 fixture를 사용하므로 실패하더라도 실제 Web job 기록을 수정하거나
별도의 cleanup을 요구하지 않습니다.
