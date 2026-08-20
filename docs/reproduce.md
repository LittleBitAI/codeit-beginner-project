# 처음 보는 사람이 재현하기

이 저장소를 clone한 사람이 팀 AWS 자격 증명 없이 **최고 점수 제출을 재현하고**, 화면에서 데이터 준비부터 제출까지 전 과정을 직접 해 보는 절차입니다.

## 1. 무엇을 재현하나

| 제출 | Kaggle | 무엇인가 |
| --- | --- | --- |
| **fusion-top3-ensemble** | **0.63594** | 아래 셋을 WBF로 합치고(IoU 0.55) crop 임베딩 셋(resnet18/34/50)의 margin 평균으로 점수를 다시 매긴 것 |
| dino-basic-e12-b1-lr1e4-s42-4675 | 0.62437 | 단일 model 최고. 융합·재순위의 기준이 된 실행 |
| dino-basic-e12-b1-lr2e4-s42-b711 | 0.61613 | 융합 입력 |
| dino-basic-e12-b1-lr1e4-s42-bfeb | 0.61098 | 융합 입력 |

재순위는 상자와 class를 **바꾸지 않습니다.** 검출 3,368개(842장 × 최대 4개)는 그대로 두고 점수만 `score' = score × (1 + margin) / 2`로 다시 매깁니다. `margin`은 잘라 낸 알약이 자기 class의 참조 crop과 얼마나 닮았는지에서 다른 class와 닮은 정도를 뺀 값입니다.

## 2. 준비물

- **NVIDIA GPU가 있는 Windows 또는 Linux.** MMDetection detector는 `device="cuda"`가 아니면 시작을 거부하고, `mmcv` 설치 파일은 Windows·Linux × Python 3.11·3.12용만 있습니다. macOS에서는 학습도 재현도 되지 않습니다.
- Python 3.11 (conda 권장), Node.js 22, 디스크 여유 10 GB 이상.

## 3. 설치

```
git clone https://github.com/LittleBitAI/codeit-beginner-project.git
cd codeit-beginner-project
conda create -n pill-detection python=3.11 -y
conda activate pill-detection
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -c "import mmcv._ext; print('mmcv ops OK')"
```

마지막 줄이 통과해야 detector를 만들 수 있습니다. 자세한 확인은 `onboarding/docs/onboarding.md`에 있습니다.

## 4. 번들 받기

데이터와 재현 재료는 Git에 담지 않습니다. 저장소 root에서 받아 풉니다. **자격 증명은 필요 없습니다.**

```
curl -L -o bundle.tar.gz <번들 주소>
tar xzf bundle.tar.gz
```

`<번들 주소>`는 번들을 공개 위치에 올린 뒤 이 자리에 적습니다.

받은 뒤 확인합니다(2.74 GB, 파일 3,294개).

```
cd datasets/pill_detection/reproduce && sha256sum -c SHA256SUMS && cd -
```

| 자리 | 무엇 |
| --- | --- |
| `datasets/pill_detection/raw/v90/` | 데모 학습용 표본 원본. train 511장 + 대회 test 842장 전부 |
| `datasets/pill_detection/reproduce/` | 최고 제출 재현 재료. 융합 입력 3개, 임베딩 3개, crop 은행, manifest, **원본 제출 CSV** |

번들의 `test_manifest.json`과 융합 입력 3개는 **위치 문자열만** 팀 S3 URI에서 번들 안 상대 경로로 바뀌어 있습니다. 자격 증명 없이 열 수 있게 하려는 것이고, id·크기·category·예측 값은 원본 그대로입니다. 그래서 이 두 종류는 S3 원본과 바이트 단위로 같지 않습니다.

`v90`은 발표 데모용 표본이라는 뜻의 판 번호입니다. 학습에 실제로 쓴 판(v5, 이미지 10,553장)이 아닙니다.

## 5. 최고 제출 재현

```
python -m src.main_pipeline --only evaluate --config configs/reproduce.best.json
```

RTX 3080에서 **5~6분**이 걸렸고(두 번 재서 290초와 366초), 결과는 `artifacts/reproduce/best/`에 놓입니다. 융합 3개와 임베딩 3개가 모두 실렸는지는 `test_predictions.json`의 `fused_from`과 `rerank`에서 확인합니다.

**여기서 만들어진 CSV를 Kaggle에 올리지 마세요.** 기록과 같지 않습니다 — 실제로 채점받아 보니 **0.00096 낮습니다.** 이 실행은 *방법이 재현된다*는 것을 보이는 용도이고, 제출 파일을 대체하지 않습니다.

제출할 파일은 번들에 든 원본입니다.

```
datasets/pill_detection/reproduce/reference-submission.csv
```

GPU가 없으면 `configs/reproduce.best.json`의 `"device": "cuda"`를 `"cpu"`로 바꿉니다. 훨씬 오래 걸립니다.

## 6. 화면으로 전 과정 해 보기

```
cd src/pipelines/web/frontend && npm ci && npm run build && cd ../../../..
python -m src.pipelines.web.server
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. 로그인은 필요 없습니다.

1. **데이터 준비** — 원본 경로에 `datasets/pill_detection/raw/v90/`을 넣고 8:2, seed 42로 실행합니다. 끝나면 `v90-seed42-8020-group`이 목록에 뜹니다(train 288 / validation 219 / 118종).
2. **EDA** — 준비한 판을 그대로 읽어 class 분포와 상자 크기를 잽니다.
3. **새 실험** — 위쪽 `점수를 받은 설정 채우기 · 최고 점수 detector`를 누르면 0.62437을 받은 설정이 그대로 채워집니다. **발표 시연이라면 `epochs`만 1이나 2로 낮추세요.** 12 epoch은 표본이 아닌 진짜 판에서 하루 가까이 걸립니다.
4. **평가와 제출** — 학습이 끝나면 그 실행에서 평가를 돌립니다. test manifest가 있으면 제출 CSV가 함께 만들어지고, 화면의 `내려받기`로 바로 받습니다.
5. **앙상블** — 끝난 실행 둘 이상을 골라 합칩니다. 합치기 전에 진단이 얼마나 닮았는지 알려 줍니다. 약한 실행을 넣으면 점수가 내려갑니다(7개 0.62087 < 단독 0.62437 < 상위 3개 0.62645).
**임베딩 학습은 이 문서 기준으로 화면에서 시작할 수 없습니다.** 학습에는 참조 crop을 모은 crop 은행이 필요한데, 준비 화면에 그것을 만드는 칸이 아직 없습니다. 은행은 판마다 따로 만들어져 나중에 덧붙일 수도 없습니다. 그 칸이 들어온 뒤에 시연하세요. 5절의 재현은 **번들에 든 은행과 임베딩**을 쓰므로 이것과 무관하게 그대로 됩니다.

## 7. 알아 둘 것

- **번들의 임베딩과 crop 은행을 다시 만들지 마세요.** 참조 crop이 바뀌면 margin이 전부 바뀌어 다른 점수가 됩니다. 6번의 임베딩 학습은 기능을 보이는 것이고, 5번 재현과 섞지 않습니다.
- 데모 표본은 조합 193개(class마다 2개씩)를 골라 118종을 모두 담았습니다. 나누는 단위가 조합 통째라 조합 하나뿐인 class는 검증에 못 갑니다.
- 표본이 작아 검증 비율이 8:2보다 큽니다(288 / 219). data pipeline은 누수 없이 갈 수 있는 가장 가까운 비율을 고릅니다.
- 로컬 검증 점수는 Kaggle 점수를 예측하지 못합니다. 독립 실험 셋이 모두 무관하거나 반대로 움직였습니다.
