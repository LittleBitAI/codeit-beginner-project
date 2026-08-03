# Handoff: PillDetect — 알약 객체 탐지 실험 플랫폼

## Overview

An internal engineering tool for a pill (alyak) object-detection project. Each image contains 1–4 pills; the model predicts a class and bounding box for every pill. The tool exists so a small ML team can **configure, run, compare, diagnose and ship** detection models reproducibly, without hand-editing YAML or digging through `runs/` folders.

It is a single application shell containing two logical modules:

- **Training GUI** — writes the future. Assembles and validates a run configuration, launches it, streams progress. Primary artifact is a *validated config*, not a model.
- **Evaluation GUI** — reads the past. Interprets finished runs, compares them, finds failure patterns, tunes inference thresholds, produces competition submissions. Primary artifact is *evidence for the next experiment*.

16 screens total. UI chrome and prose are Korean; ML terminology stays in English (`mAP`, `IoU`, `NMS`, `epoch`, `batch`, `AdamW`…). A global **"쉬운 설명" (plain-language) toggle** layers beginner explanations over every metric without removing any data.

---

## About the Design Files

The files in this bundle are **design references created in HTML**. They are prototypes that demonstrate intended look, layout, density, copy and interaction behavior — **they are not production code to copy directly.**

`mockup/PillDetect-Platform.html` is a single self-contained file: open it in any browser, no server, no build, no network. Everything (fonts, runtime, data) is inlined. Click through the left rail to reach all 16 screens.

Your task is to **recreate these designs in the target codebase's existing environment** — React, Vue, Svelte, whatever is already there — using its established component library, state management, routing and styling conventions. If no frontend environment exists yet, pick the most appropriate framework for the project and implement there.

`source/` contains the authoring format for reference. It uses a bespoke template runtime (`support.js`) that is **not** part of the deliverable — read it for logic and copy, do not port it.

---

## Fidelity

**High-fidelity (hifi).** Final colors, typography, spacing, density, copy and interaction behavior. Recreate pixel-for-pixel where the target codebase's design system permits; where it conflicts, prefer the codebase's existing tokens and keep the *structure* and *density* from these mocks.

Three things are deliberately **not** final:

1. **Pill imagery** — all validation/test photos are grey placeholder frames. Bounding-box overlays, labels, confidence values and IoU numbers on top of them **are** designed and should be recreated exactly. Swap the placeholder for a real `<img>`.
2. **Mock data** — 14 experiments, 12 pill classes, 4,000 images, 6 Kaggle submissions. Plausible and internally consistent, but invented. See *Mock Data Contract*.
3. **Kaggle submission format** — assumed `image_id,PredictionString`. Confirm against the real competition spec.

---

## Model Family Matrix

Drive the 새 실험 form from a table like this. Switching family applies the whole row.

| | YOLO11 | RT-DETR | Faster R-CNN | DINO |
|---|---|---|---|---|
| stack | ultralytics | ultralytics | torchvision | mmdetection |
| sizes | `n s m l x` | `l x` | `mobilenet r50 r101` | `r50-4scale r50-5scale swin-l` |
| checkpoints | `yolo11{size}.pt` + own runs | `rtdetr-{size}.pt` + own | `fasterrcnn_{size}_fpn_coco` + own | `dino_{size}_coco` + own |
| loss terms | box 7.5 · cls 0.5 · dfl 1.5 | giou 2.0 · cls 1.0 · l1 5.0 | rpn_cls · rpn_box · roi_cls · roi_box (1.0) | cls 1.0 · bbox 5.0 · giou 2.0 |
| augmentations | all 13 | 12 (no cutmix) | 7 (no mosaic/mixup/cutmix/hsv/occl) | 6 |
| NMS | yes | **NMS-free** | yes | **NMS-free** |
| default lr / batch / opt / warmup | 1e-3 / 16 / AdamW / 3 | 2e-4 / 8 / AdamW / 2 | 5e-3 / 4 / SGD / 1 | 1e-4 / 4 / AdamW / 1 |

Each size also carries a memory coefficient and a seconds-per-image coefficient used by the resource estimator. NMS-free families emit `nms: null` in the YAML with an inline comment, and the generated Korean prose says so.

---

## Cross-Screen State Flow

The prototype's own first review found that screens looked wired but weren't. Three flows must actually connect:

**1. Selected experiment (`selExp`).** Clicking any row in 실험 목록 or 평가 개요 sets it. 실험 상세, 지표 상세, 실패 갤러리, 제출 관리, 모델 리포트 and the top context badge all read from it. Route to the monitor instead when the run is `running` — both tables must share one `openRun(id, status)` handler so the same run never behaves differently on two screens.

**2. Monitored run (`monitorExp`).** The live monitor derives *everything* from the monitored run's record: epoch total (progress bar width, ETA, chart x-scale, tick cap, lr schedule period), model name, device, owner. Nothing may be hardcoded to 120 epochs — exp-013 is 80, exp-005 is 50, exp-008 is 60, and a run started from the form can be anything. Reset the epoch counter and curve when the monitored run changes.

**3. Comparison set (`cmpSel`).** The first table column is a checkbox that toggles membership and must `stopPropagation` so it doesn't navigate. The action button reflects the count and disables below 2.

Starting a run from 설정 검토 registers a new record, points `monitorExp` at it, seeds the log with the drafted config, and toasts.

---

## Design Tokens

### Color — semantics are fixed, do not reassign

| Token | Hex | Use |
|---|---|---|
| `navy` | `#0B2545` | Left rail, code blocks, log viewer background |
| `navy-ink` | `#172030` | Tooltip background |
| `primary` | `#1A56A8` | Primary buttons, active chips/tabs, selected state, links |
| `primary-hover` | `#164A94` | Primary button hover |
| `primary-tint` | `#EAF1FB` | Selected chip fill, context badge fill |
| `primary-border` | `#C9DCF3` | Selected chip border |
| `teal` | `#0D8B84` | Running state, live indicator, "start training" action |
| `teal-dark` | `#0A6E68` | Running badge text, teal button hover |
| `teal-tint` | `#E2F5F3` | Running badge fill, dataset badge fill |
| `green` | `#1F8A3B` | Success icon, improved metric (+) |
| `green-dark` | `#166B2D` | Completed badge text, best-in-column value |
| `green-tint` | `#EAF6EC` | Completed badge fill |
| `amber` | `#B5760A` | Warning icon/title, false positives, estimated values |
| `amber-tint` | `#FBF4E8` | Warning badge fill |
| `red` | `#C1332D` | Error icon/title, failed run, false negatives, regressed metric (−) |
| `red-tint` | `#FBE4E4` | Failed badge fill |
| `text` | `#111C2E` | Primary text, metric values |
| `text-strong` | `#31405A` | Labels, table cell text |
| `text-body` | `#5C6470` | Body prose, hints |
| `text-muted` | `#8A929E` | Micro-labels, units, secondary meta |
| `text-faint` | `#9AA2AD` | Chart axes, disabled, placeholders |
| `border` | `#E4E7EB` | Panel borders — **1px, everywhere** |
| `border-inner` | `#EEF0F3` | Section dividers inside a panel |
| `border-row` | `#F2F4F7` | Table row separators |
| `border-control` | `#D8DCE2` | Input, select, secondary button borders |
| `border-chart` | `#DFE3E8` | Chart axis lines |
| `surface` | `#FFFFFF` | All panels |
| `surface-page` | `#F4F6F9` | Page background |
| `surface-alt` | `#FAFBFC` | Table header rows, subsection strips, hover |
| `surface-sunken` | `#F6F7F9` | Grid lines, empty bar tracks |
| `surface-image` | `#E9ECF0` | Image placeholder fill |

Detection-visualization colors (bounding boxes, TP/FP/FN bars) are separate and slightly desaturated so they read against photos:
`tp #2E7D5B` · `fp #C98A3A` · `fn #C1332D` · `gt-outline #5C6470` (1.5px dashed).

Multi-run comparison series, in order: `#1A56A8`, `#0D8B84`, `#B5760A`, `#7A4FBF`, `#C1332D`.

### Typography — two families, no exceptions

- **Pretendard Variable** — all Korean UI, prose, labels, buttons.
  `https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css`
- **IBM Plex Mono** (400/500/600) — **every number, identifier, hyperparameter name, file path, log line, and column header.** Decimal alignment is a precondition for comparing runs; never render a metric in a proportional font.
  Google Fonts.

| Role | Size / weight / line-height |
|---|---|
| Page title (top bar) | 14.5px / 650 / 1.3 — Pretendard |
| Panel header | 12px / 600 / 1 — Pretendard |
| Screen intro title | 13px / 600 / 1.4 — Pretendard |
| Body prose | 12.5px / 400 / 1.7 — Pretendard |
| Plain-language note | 11px / 400 / 1.5 — Pretendard |
| Field label | 11.5px / 600 / 1 — Pretendard |
| Field hint | 10.5px / 400 / 1.45 — Pretendard |
| Table cell | 11px / 400 / 1.3 — Mono |
| Table column header | 10px / 600 / 1.3, `letter-spacing: .04em` — Mono |
| KPI micro-label | 10px / 500 / 1.3, `letter-spacing: .04–.05em` — Mono |
| KPI value (large) | 22px / 600 / 1 — Mono |
| KPI value (compact) | 15–19px / 600 / 1 — Mono |
| Status badge | 10px / 600 / 1.3 — Mono, uppercase |
| Log line | 10.5px / 400 / 1.6 — Mono |
| Code / YAML | 10.5px / 400 / 1.65–1.75 — Mono, `white-space: pre` |
| Chart axis label | 9px — Mono (SVG only) |

**Minimum size for any meaning-carrying text is 9.5px.** 9px is reserved for SVG chart axis labels. Do not go below.

### Spacing, shape, elevation

- 4px grid. Panel padding 13–20px. Form row gap 12px. Card grid gap 10–12px. Section gap 12–14px.
- Table row height ~34px (`padding: 8px 12px`). Density target: 8/10 — dense but never cramped.
- Border radius: **badge/chip 3–4px · control 4px · panel 5–6px.** Nothing rounder. No pills, no 12px+ radii.
- Elevation: only floating elements (tooltip, dropdown, modal) get `0 4px 14px rgba(11,37,69,.22)`. **Panels are separated by 1px borders, never by shadow.**
- Left rail 216px fixed, full height, `position: sticky`. Content max-width 1560–1720px depending on screen.
- Top bar: 3px teal accent strip, then a 9px-padded sticky context bar.

---

## Hard Visual Constraints

These were explicit client requirements. Violating them fails review.

**Never:**
- colored vertical bars on the left edge of alerts or cards
- large red / yellow / blue / green tinted panels
- cards nested inside cards
- border radius above ~6px
- gradients, glows, glassmorphism, neon
- purple "AI" branding, sparkle icons, decorative AI symbols
- large empty hero areas
- charts that don't inform a decision

**Always:**
- neutral white / light-grey surfaces, thin consistent borders
- section-based layout, compact tables, aligned form grids
- restrained status badges, icons paired with text
- clear primary vs. secondary action
- dense but readable hierarchy

### Alert and error design — three levels, no exceptions

1. **Field-level** — inline, directly below the offending input.
2. **Page-level blocking** — a compact *horizontal* row: SVG icon (13px, `margin-top: 2px`, `flex: none`) → bold colored title → neutral `#5C6470` explanation → recovery action button on the right. White background, 1px `#E4E7EB` border. **Color appears only on the icon and title.**
3. **Transient** — toast.

Every error message must state **what happened, why, and what to do next.** Example from 학습 개요:

> ⛌ **exp-012 학습이 CUDA out of memory로 중단됨**
> epoch 7/120에서 3.42 GiB 추가 할당에 실패했습니다. imgsz 1280 × batch 8 × YOLO11-x는 24GB GPU에 들어가지 않습니다. 아래 중 하나로 재시도하면 같은 config가 그대로 복제됩니다.
> `[batch 8→4로 재시도]` `[imgsz 1280→960으로 재시도]` `[Colab A100으로 이관]` `[전체 로그 보기]`

Icons are inline SVG, 1.25–1.4 stroke width, `currentColor`-ish semantic hex. **No emoji, no text glyphs (`✓`, `!`, `✕`) as icons.**

---

## Application Shell

### Left rail (216px, `#0B2545`)

Logo block → nav groups → user footer (`margin-top: auto`).

Groups, in order: **개요** · `TRAINING` (학습 개요 / 새 실험 / 설정 검토 / 라이브 모니터) · `EXPERIMENTS` (실험 목록 / 실험 상세) · `EVALUATION` (평가 개요 / 실험 비교 / 지표 상세 / 임계값 튜닝) · `PREDICTIONS` (예측 검사기 / 실패 갤러리) · `SUBMISSIONS` (제출 관리 / 모델 리포트) · **설정**

- Group label: 10px/600 mono, `letter-spacing: .09em`, `#5C7290`.
- Item: 12.5px Pretendard, `#A9BAD0` idle → `#fff` on hover with `rgba(255,255,255,.05)`.
- **Active item: `rgba(255,255,255,.09)` background only — no left accent bar.**
- 라이브 모니터 shows a 6px teal dot with `pulse 1.6s infinite` while a run is active.
- Footer: avatar + name, then a teal dot + `wandb · pill-det 연결됨`.

### Top context bar (sticky)

3px `#0D8B84` strip, then a `#FBFCFE` bar with 1px bottom border containing:

**Left** — page title, 1px×15px divider, then always-visible context badges: `dataset v4.2` (teal tint) · `12 classes · 4,000 imgs` (neutral) · `exp-014 · yolo11m-mosaic-v3 · best.pt` (blue tint, truncates).
These three badges are the structural implementation of the rule *"every chart must be traceable to an experiment and a dataset version."* They never disappear.

**Right** — 쉬운 설명 toggle · `⌘K` hint · `+ 새 실험` primary button.

---

## The Plain-Language Layer

A defining feature. Goal: a reader who knows nothing about deep learning can understand every screen, **without any information being removed for the expert.**

**Toggle** in the top bar, default **on**. On: blue tint fill, check-circle icon, "쉬운 설명 켜짐". Off: white, empty circle, "쉬운 설명 꺼짐". Off restores the original dense engineering view. Persist per user.

Three tiers:

**Tier 1 — screen intro strip.** A white panel above every screen's content. Info-circle SVG (`#4C7FBF`) + a plain title + 1–3 sentences of jargon-free Korean explaining what the screen is for. Below, separated by a 1px divider on `#FAFBFC`, 2–4 contextual term definitions laid out in a wrapping flex row (`flex: 1 1 300px` each): term name (mono if ASCII, Pretendard 11.5px/600 if Korean) followed by a one-line meaning.

Example — 임계값 튜닝:
> **확신 기준 조절**
> 모델은 '이건 87% 확률로 약이다'처럼 확신도를 함께 냅니다. 몇 % 이상을 정답으로 받아들일지 정하는 화면입니다. 학습을 다시 하지 않고 결과만 달라집니다.
> `confidence` 모델의 확신 정도. 기준을 높이면 확실한 것만 남고, 낮추면 애매한 것까지 잡습니다
> `NMS` 같은 약을 두 번 세 번 잡았을 때 하나만 남기는 정리 작업
> `F1` Precision과 Recall을 합친 균형 점수

**Tier 2 — metric card plain notes.** Every KPI card on 개요 / 라이브 모니터 / 평가 개요 / 임계값 튜닝 gets an extra line below its value: 11px Pretendard `#5C6470`, separated by a 1px `#F2F4F7` top border with 6px padding. **The live value is interpolated into the sentence**, so it stays informative:

> PRECISION **0.8904** ±0.0000 vs 기본
> `'약이다'라고 한 2,066개 중 1,840개가 진짜였습니다. 헛것을 보지 않는 정도.`

**Tier 3 — hover tooltips.** Jargon terms carry `data-tip`, rendered with a 1px dotted `#A8B0BA` bottom border and `cursor: help`. On hover a dark tooltip shows **bold term — plain definition**.

Implementation note: the tooltip is a **single element appended to `document.body`, positioned `fixed`** via a delegated `mouseover` listener. This is required — panels use `overflow: hidden` and tables use `overflow-x: auto`, so a CSS `::after` tooltip would be clipped. It flips above the anchor when it would overflow the viewport bottom, clamps to the right edge, and hides on scroll, click and route change.

Applied to: all KPI card micro-labels, and table column headers `mAP@0.5 ↓`, `mAP@.5:.95`, `IMGSZ`, `EP`, `BATCH`, `LR`, `OPT`, `P`, `R`, `F1`, `ms/img`, `Kaggle LB`, `MODEL`, `DS`, `CLASS`, `AP`, `GT`.

A glossary map of ~50 terms lives in the source — reuse it verbatim.

---

## Screens

### 01 · 개요 — Overview
**Purpose:** where is the team right now.
**Layout:** intro strip → 5-up KPI grid (`auto-fit, minmax(178px, 1fr)`) → two-column row (`minmax(400px,1fr)` auto-fit): mAP-over-time line chart | stacked (확인이 필요한 항목 / 다음에 할 일).
- KPI cards: 실행 중 · BEST mAP@0.5 · BEST mAP@0.5:0.95 · 총 실험 · Kaggle 제출. Micro-label → 22px mono value → delta line → plain note.
- Chart: SVG line of best mAP@0.5 per experiment, x-axis banded by dataset version (`v3.8 / v4.0 / v4.1 / v4.2`, alternating `#F7F9FC` bands). Latest point is a filled teal dot with its value labeled. Banding is the point — it shows which gains came from data changes vs. model changes.
- 확인이 필요한 항목: level-2 error rows (see alert spec), each with a recovery button.
- 다음에 할 일: numbered `01/02/03` rows, whole row clickable, deep-links to the relevant screen.

### 02 · 학습 개요 — Training Overview
**Purpose:** what is running, what broke.
**Layout:** 4-up resource KPIs → run table → failure detail block.
- Resource cards: local GPU (RTX 4090, 78%, 18.2/24 GB with a progress bar), Colab GPU (A100, session expiry `~1:42`), queue depth, latest best val.
- Run table: 실험명 · MODEL · DS · EPOCH · 상태 · BEST mAP@0.5 · 경과 · 시작 · 소유자. Filter chips above: 전체 / 실행 중 / 대기 / 실패 / 내 실험 (4px radius, blue fill when active).
- Status badges: `RUNNING` teal · `COMPLETED` green · `QUEUED` grey · `CUDA OOM` / `NaN LOSS` red.
- Rows navigate by status: running → 라이브 모니터, done → 실험 상세.
- Empty state names the cause and offers one action: "이 필터에 맞는 런이 0건입니다" + `[필터 초기화]`.
- Failure block: error icon + title + explanation + a dark log excerpt (last 5 lines, `[E]` in `#F08A8A`) + four one-click recovery actions.

### 03 · 새 실험 — New Experiment *(priority screen)*
**Purpose:** assemble the next run.
**Layout: split.** Left `flex: 3 1 400px` tabbed form, right `flex: 2 1 320px` live preview. Wraps to stacked below ~740px.

Five tabs: **기본 정보 · 모델 · 하이퍼파라미터 · Augmentation · 고급.** A tab showing a problem gets a 5px amber dot.

- **기본 정보** — 실험명, 설명·가설 (textarea, hint: *"무엇을 왜 바꿨는가를 남기세요. 실험 상세와 모델 리포트에 그대로 인용됩니다"*), tags, dataset version / split / seed in a 3-col grid, then a shield-icon note: **test split cannot be selected**, split-hash collision blocks execution.
- **모델** — family picker as 4 selectable cards (YOLO11 / RT-DETR / Faster R-CNN / DINO), a size segmented control with a live parameter-count hint, pretrained checkpoint select (COCO or a prior run's `best.pt`, which records lineage), then nc (disabled, derived from dataset), freeze, resume.

  **Changing family rebuilds the whole schema — this is load-bearing, not cosmetic.** Each family is a different framework, so switching must swap: the size list, the checkpoint list, the loss terms, the available augmentations, the default hyperparameters, the memory/time estimation coefficients, and NMS availability. Without this the form emits impossible configs (`DINO-m` + `yolo11m.pt`). See *Model Family Matrix*. A toast names what changed; the experiment name and description auto-derive until the user edits them.
- **하이퍼파라미터** — 9 numeric fields in a 4-col grid (epochs, batch, imgsz, lr0, weight decay, warm-up, patience, workers, grad accumulation — the last showing effective batch), then optimizer / scheduler / device selects, then AMP and DDP toggles.
- **Augmentation** — a warning row (*augmentation은 강할수록 좋지 않습니다*, with the live intensity total), then 13 sliders in 2 columns: fliplr, degrees, scale, translate, hsv_s, brightness, contrast, blur_p, mosaic, mixup, cutmix, random_crop, occlusion_p. Each row: label + mono key + right-aligned value, slider, one-line domain hint (e.g. mosaic: *"4장 합성 — 알약 개수 분포를 왜곡함"*).
- **고급** — loss weights (**family-specific**, headed by a `{family} · {stack}` badge) + class-balancing toggle; validation thresholds **greyed with the badge "학습 중 val에만 적용 · 최종 추론 임계값과 다름"**; checkpoint/logging periods. For NMS-free families the NMS IoU input is disabled and reads `해당 없음`.

**Right rail, all live:**
1. **생성될 설정** — full YAML on `#0B2545`, `max-height: 330px`, regenerates on every keystroke. YAML/JSON/복사 controls.
2. **baseline 대비 변경점** — changed keys only, `old` struck through in red → `new` in green. Footer: *"N개 항목이 baseline과 다릅니다 … 변화의 원인을 좁히려면 한 번에 1~2개만 바꾸세요."*
3. **자원 추정** — GPU memory (bar turns amber >18 GB, red >22 GB) and wall time. Estimates render *italic, grey, `~`-prefixed, dotted underline* — visually separated from measured values.
4. **Live validation** — error/warning rows. Rules: memory overflow (blocking), memory tight, augmentation total >1.35, seed+split reuse, patience ≥ epochs/2. Clean state shows a green check row.
5. **Actions** — `[프리셋으로 저장]` `[설정 저장 후 검토 →]` with the note *"설정을 저장하지 않으면 실행할 수 없습니다."*

### 04 · 설정 검토 — Config Review
**Purpose:** last gate before spending hours of GPU time.
- Header: name + `DRAFT · 미저장` badge + description + a 6-cell summary strip (MODEL / DATASET / SCHEDULE / IMGSZ / DEVICE / SEED) + **a full-sentence Korean paragraph describing the run in prose**, generated from the config. This is what makes the screen readable to a non-specialist.
- Left: unified-diff of `config.yaml` vs baseline. `-` red on `rgba(198,40,40,.16)`, `+` green on `rgba(31,138,59,.16)`, `@@ section @@` in grey. (Diff row tints are the one place tinted rows are correct — it's the Primer diff convention.)
- Right: 실행 전 검사 (schema / dataset integrity / **leakage: test∩train = 0건** / checkpoint reachable, each with an SVG check, plus any warnings), 추정치 panel, then `[프리셋 저장]` `[등록만]` and the teal `[설정 저장 후 학습 시작]`.

### 05 · 라이브 모니터 — Live Monitor *(priority screen)*
**Purpose:** is this run on track, or should I kill it.
- Header: pulsing teal dot + run name + `exp-014`, meta line (device · wandb id · owner · start), progress bar with `epoch 64 / 120` and an italic `남은 시간 ~1시간 2분 · 추정`, then `[일시정지]` `[중지]` `[설정 보기]`.
- 8 live KPI cards (`auto-fit, minmax(126px,1fr)`): train loss, val loss, precision, recall, mAP@0.5 (teal), mAP@0.5:0.95, F1, best ckpt. Each shows a per-epoch delta with ▲/▼ and green/red.
- Left column: **Loss** chart (train `#1A56A8`, val `#B5760A`) and **Validation mAP** chart (mAP@0.5 solid teal, mAP@0.5:0.95 dashed grey), both with a dashed teal "now" line at the current epoch. Below, 4 validation prediction previews with box overlays and captions.
- Right column: system panel (GPU util, memory, lr, throughput, epoch time, best epoch) and a **log stream** — `height: 352px`, `flex-direction: column-reverse`, `[W]` amber, `[E]` red, `new best` green, plus a pulsing "스트리밍 중" indicator.
- **Ticks every 2.6s**: appends an epoch point to both charts and two log lines, recomputes every KPI and delta.

### 06 · 실험 목록 — Experiments
Wide sortable table, `min-width: 1480px` inside a horizontal scroller. Columns: status dot · 실험명 (+ id/owner/date beneath) · MODEL · DS · IMGSZ · EP · BATCH · LR · OPT · P · R · **mAP@0.5 ↓** · mAP@.5:.95 · ms/img · 시간 · Kaggle LB. Best mAP cell gets a green tint chip. Search input + the same filter chips + `[선택 항목 비교]` `[CSV 내보내기]`. Legend line below explains the status dots.

### 07 · 실험 상세 — Experiment Detail
The crossover point between the two modules.
- Header: name + COMPLETED badge + id, hypothesis text, meta (owner · timing · wandb · git sha), actions `[실험 복제]` `[다른 실험과 비교]` `[평가 GUI에서 열기 →]`.
- 7-cell metric strip: mAP@0.5 · mAP@.5:.95 · P · R · BEST EPOCH · ms/img · Kaggle LB.
- Left: two small training charts; full `config.yaml`, **read-only**, labeled *"불변 · 수정하려면 복제하세요"*.
- Right: artifacts list (`best.pt` / `last.pt` / `results.csv` / `predictions_val.json` / `confusion_matrix.png` with sizes), 계보 (parent / weights / children), and a notes block with the researcher's written conclusion.

### 08 · 평가 개요 — Eval Overview
6 summary cards (최고 실험 / 최고 mAP@0.5 / 최고 mAP@.5:.95 / 최고 RECALL / 최저 val loss / Kaggle 후보; the first two use a subtly distinct `#FBFCFD` + `#D3DAE2` surface). Then a warning row: **Kaggle LB is not the selection criterion** — DINO leads the board at 0.9131 but is 8× slower and weaker on local validation. Then the full evaluation table sorted by mAP@0.5.

### 09 · 실험 비교 — Comparison *(priority screen)*
- Chip selector, 2–5 runs.
- **무엇이 바뀌었나** matrix: sticky first column (`180px`), one column per run headed by a name cell with a 3px colored top-left border in the series color. Parameter rows where values differ get `#FAFBFC` background and amber bold values; identical rows stay white. Metric rows highlight the best value green (min for ms/img, max for everything else).
- Overlaid validation-loss chart, one line per run in series color.
- **클래스별 AP 차이**: 12 diverging bars around a center line, green right / red left, delta labeled.
- **결론**: a generated paragraph naming the delta, how many parameters differ, that gains are concentrated in small/reflective classes, and the accuracy-vs-speed tradeoff.
- Under 2 selections: empty state, not a broken chart.

### 10 · 지표 상세 — Metric Detail
- Context strip: which run, which checkpoint, val size, GT box count, active thresholds, `[임계값 변경 →]`.
- **PR curve** — current run solid, baseline dashed, current threshold marked with a labeled dot. Caption interprets the knee: *"conf를 0.35까지 올리면 precision +0.019 / recall −0.041 — 알약 4개 중 1개를 놓치는 비용이 오탐 하나보다 크므로 현재 위치가 타당합니다."*
- **클래스별 AP** — 12 rows: code + Korean name, horizontal bar colored by band (≥.93 green / ≥.86 blue / ≥.79 amber / below red), then AP / P / R / GT right-aligned. Caption: the bottom 3 classes cost 0.031 of overall mAP.
- **Confusion matrix** — 13 columns (12 classes + BG). Diagonal blue by intensity, off-diagonal red by intensity, `<0.005` flat `#F6F7F9`. Caption names the two worst cells and distinguishes *class* errors from *detection* errors.
- **분포** — IoU histogram (median 0.871 → localization is fine) and confidence histogram (FPs cluster 0.25–0.45 → threshold headroom is here).
- **조건별 성능** — grouped table: pill count (1/2/3/4), object size (small/medium/large), capture condition (natural / fluorescent / low-light / reflective packaging). Section headers are `#FAFBFC` rows. mAP colored red <0.87, amber <0.92. Caption connects 4-pill images and small objects to the same root cause and proposes imgsz 960→1280.

### 11 · 임계값 튜닝 — Thresholds *(priority screen)*
- Info row: **these are inference thresholds — NMS is re-applied to stored raw predictions; no retraining occurs**; the confirmed values feed submission generation.
- Left panel (`flex: 1 1 296px`): confidence slider (0.05–0.9), NMS IoU slider (0.3–0.95), max-det segmented `2 3 4 5 10`, TTA toggle, ensemble select. Every control has a consequence hint.
- Right (`flex: 3 1 340px`): 6 recomputing cards (P / R / F1 / 검출 수 / FP / FN) with deltas vs. default and plain notes; a **sweep chart** (precision, recall, F1 across conf 0.05→0.9) with a red line at the current value and a dashed grey line at the F1 optimum; a TP/FP/FN composition bar; and a 확정 panel whose advice text changes with the regime (too low → FP cost, too high → FN is unrecoverable, near-optimal → ship it).

**Calibration requirement.** At the defaults **conf 0.25 / NMS 0.70 / max-det 4 / TTA off**, the screen must reproduce exp-011's published numbers exactly: **P 0.8904 · R 0.8702 · F1 0.8802**, all three deltas `±0.0000`, FP 226, FN 274, GT 2,114. Compute metrics from unrounded counts and round only for display, otherwise integer boxes introduce a 0.0002 discrepancy against every other screen. Wire to `POST /api/runs/:id/evaluate`, 200ms debounce, cache results.

### 12 · 예측 검사기 — Prediction Inspector *(priority screen)*
Three columns: filter rail (`flex: 1 1 184px`, max 280px) | image (`flex: 3 1 356px`) | analysis (`flex: 2 1 278px`).
- **Filters**, grouped 결과 / 객체 조건 / 알약 개수, each with a live count. Active row: `#F0F3F7` + 600 weight.
- **Canvas**: 4:3 frame. GT = 1.5px dashed `#5C6470` inset slightly outside the prediction. Predictions = 2px solid in TP/FP/FN color with a filled label chip above (`K-101 0.94`). `GT / 예측 / 겹침` segmented toggle. Placeholder caption is pinned bottom-left so it never collides with overlays. Prev/next, zoom controls, legend.
- **Analysis**: verdict badge, GT/검출/FP/FN counts (FP amber, FN red when non-zero), a per-object match table (`GT → PRED`, IoU, conf, colored dot; IoU <0.7 amber), metadata (pill count, lighting, background, blur score, smallest object with size class, overlap, resolution), and a written diagnosis for that specific image.
- Thumbnail strip with `OK / FN / FP / CLS` tags; current is blue-bordered.
- 4 fully authored example images — a small-object miss, a duplicate detection, a class confusion, and a clean case. Their diagnoses are the model for what generated text should say.

### 13 · 실패 갤러리 — Failure Gallery *(priority screen)*
- Category selector: 10 tiles (전체 274 · 놓친 알약 93 · 잘못된 클래스 49 · 중복 검출 58 · 위치 오차 33 · 저신뢰 검출 26 · 소형 객체 실패 41 · 겹침 실패 37 · 반사 실패 24 · 배경 혼동 16). Each shows a count and a proportion bar; active gets `#F0F3F7` + `inset 0 -2px 0 #31405A`.
- `[검토 세트로 내보내기]` and **`[이 그룹으로 다음 실험 제안]`** — the latter opens 새 실험 pre-filled with the augmentation/HP changes implied by the failure group. **This link is what makes the two modules one product.**
- Card grid (`auto-fill, minmax(238px,1fr)`): image with overlays, a category chip top-left, filename bottom-left on a translucent plate, then GT / PRED rows, conf + IoU, a one-line cause, and `[열기]` `[검토 세트 +]`.
- Summary line: top 3 categories account for 73% of failures.

### 14 · 제출 관리 — Submissions
- 제출 구성: experiment + checkpoint selects with local val / dataset version / git sha; the confirmed inference thresholds (read-only, with a link back to tuning); a live CSV preview on navy; and a warning about **empty predictions** — images where nothing cleared the threshold, which most metrics score as wrong.
- 제출 이력: 6 rows with 임계값, 로컬 val, Public LB, and the gap. Caption: the gap is consistently −0.005 to −0.010, which is *healthy* — a sudden widening means val overfitting.
- Right: daily quota (5 remaining, 5 segments, KST 09:00 reset), a submission-note textarea, `[제출 파일 생성]` + `[다운로드]` `[제출됨으로 표시]`, and a reminder that Public LB scores only part of the test set.

### 15 · 모델 리포트 — Model Report
A 960px document, 8 numbered sections: 모델 구성 · 주요 지표 (key/value) · best and worst classes · core failure modes · threshold configuration · recommended next experiment · Kaggle history · final rationale. `[Markdown 복사]` `[PDF]`. The prose is the deliverable — it must answer "why this model" months later.

### 16 · 설정 — Settings
Four grouped panels (경로 / 실험 추적 / 연산 자원 / 기본값·정책). Each row is a 3-column grid: label + hint | value (mono) | status badge. Notable policy rows: **설정 저장 없이 실행 = 차단**, **데이터 누수 검사 = 활성**, **Kaggle 일일 한도 = 5회**.

---

## Interactions & Behavior

| Interaction | Behavior |
|---|---|
| Rail navigation | Client-side route swap. Hides any open tooltip. |
| 쉬운 설명 toggle | Shows/hides intro strips and all plain notes. Persist per user. |
| Form field edit | Regenerates YAML, diff, resource estimates and validation **on every keystroke** — no debounce; it is the feedback mechanism. |
| Model family change | Rebuilds sizes, checkpoints, loss terms, augmentations, defaults, estimator coefficients and NMS availability. Toast names the change. |
| Row click (any run table) | Shared `openRun(id, status)`: running → live monitor, else → detail. |
| Comparison checkbox | Toggles `cmpSel`, `stopPropagation` so the row does not navigate. |
| Threshold sliders | Recompute P/R/F1/FP/FN/detections + sweep marker + advice copy. Debounce 200ms against the server; render optimistically. |
| Filter chips | Instant client-side filter; count label updates; empty state names the cause. |
| Compare chips | Toggle membership, max 5; below 2 shows the empty state. |
| Inspector view mode | GT-only / prediction-only / overlay. |
| Inspector prev/next + thumbnails | Change the active image. |
| Gallery category | Filters cards and rewrites the summary line. |
| Live monitor tick | 2.6s: append chart points + 2 log lines, recompute KPIs. Real implementation: `WS /ws/runs/:id`. |
| Pause / resume | Stops and restarts the stream. |
| Tooltip | `mouseover` (delegated, capture) → fixed-position element on `body`; flip/clamp to viewport; hide on scroll, click, route change. |
| Hover | Table rows `#FAFBFC`; secondary buttons darken border to `#9AA2AD`; primary buttons `#164A94`. |
| Responsive | Every multi-column screen is `display: flex; flex-wrap: wrap` with `flex-grow` allowed, so side rails never strand. Verified down to ~655px content width. |

**Animation:** only `pulse 1.6s infinite` on live indicators and a 0.1s tooltip fade. Nothing else moves.

---

## State Management

```
state
  route                                  active screen
  plain                                  plain-language layer on/off (persist)
  selExp                                 experiment the evaluation side is about
  monitorExp                             run the live monitor is watching
  cmpSel[]                               compared run ids (max 5)
  runFilter                              'all' | 'running' | 'queued' | 'failed' | 'mine'
  tab                                    new-experiment tab
  nameTouched / descTouched              stop auto-deriving once the user types
  f { … ~45 keys … }                     the draft config (identity / data / model /
                                         train / augment / loss{} / val / runtime)
  ep, running, loss[], logs[]            live monitor
  conf, nmsIou, maxDet, tta              inference thresholds
  imgIdx, viewMode, insFilter            inspector
  galCat                                 gallery category
  toast                                  transient feedback
```

Derived, never stored: generated YAML · baseline diff · validation warnings · resource estimates · all threshold metrics · sweep series · advice copy.

---

## Backend Contract (as designed)

```
POST /api/experiments                 config JSON → run_id   (register only, does not start)
POST /api/experiments/:id/validate    schema + resource + leakage → warnings[]
POST /api/experiments/:id/start       enqueue. target=local|colab
POST /api/experiments/:id/stop
GET  /api/experiments?status=&owner=
GET  /api/experiments/:id             config + final metrics + artifacts
GET  /api/experiments/:id/config.yaml
GET  /api/runs/:id/predictions        ?filter=fp,fn,wrong_class&conf=0.25
POST /api/runs/:id/evaluate           re-apply thresholds → P/R/F1 (cached)
POST /api/submissions                 CSV generation job
GET  /api/gpu/status                  poll 5s
WS   /ws/runs/:id                     epoch_end · batch_metrics · log_line · error
WS   /ws/queue                        queue reorder · session-expiry warning
```

Three invariants the UI depends on:

1. **create → validate → start are separate calls.** "Cannot run without a saved config" is enforced at the API layer; the start button only enables after `validate` returns 200.
2. **W&B is the source of truth.** The training script logs to W&B directly; the backend relays `wandb_run_url` and summary metrics. Deep-link for detailed charts — do not reimplement them.
3. **Threshold changes re-run NMS on stored raw predictions.** Never retraining. The UI says so explicitly.

Colab sessions that expire leave the run `interrupted` with a **재개** action from the last checkpoint; resumed runs carry a lineage badge.

---

## Mock Data Contract

Replace with real data; keep the *shape* and the *realism*.

- **14 experiments** — 9 completed, 2 running, 2 queued, 1 failed (CUDA OOM), 1 failed (NaN loss). Families: YOLO11 n/s/m/x, RT-DETR-l, Faster R-CNN R50, DINO R50.
- **mAP@0.5 range 0.7431–0.9214**, mAP@0.5:0.95 0.5820–0.7106. Realistic spread, no suspicious clustering.
- **12 pill classes** `K-101`…`K-224`, AP 0.741–0.961. White/round tablets top; **transparent capsules, small scored tablets and reflective coated tablets bottom** — so failures naturally concentrate in specific classes.
- **Validation** 800 images, 2,114 GT boxes. **Test** 3,000 images.
- **Failure distribution** — missed 34% · duplicate 21% · wrong class 18% · localization 12% · reflection 9% · background 6%.
- **6 Kaggle submissions**, LB 0.7288–0.9131, **consistently 0.005–0.010 below local validation.**
- **Metadata per image** — lighting (natural / fluorescent / low-light), background (solid / palm / packaging), blur score, pill count, smallest-object size class.

---

## Assets

No image or icon files. All icons are inline SVG defined in the markup (info, warning, error, success, shield, image placeholder, check-circle). Fonts load from CDN — self-host for production. Pill photography is not included; every image area is a grey placeholder awaiting real assets.

---

## Files

| Path | What it is |
|---|---|
| `mockup/PillDetect-Platform.html` | **Start here.** Self-contained interactive prototype, all 16 screens. Open directly in a browser. |
| `source/PillDetect Platform.dc.html` | Authoring source. Markup + a logic class holding all mock data, the glossary map, metric generators and the tooltip engine. Read for exact copy, colors and formulas. |
| `source/support.js` | Template runtime for the source format. **Not part of the deliverable — do not port.** |
| `source/Design Plan.dc.html` | The approved plan: IA, module boundaries, workflow loop, component inventory, design-system rationale, and the three rejected/accepted form-layout options. |

### Reading the source

- `INTRO` — per-screen plain-language titles, bodies and term definitions.
- `FAMILIES` — the model-family matrix (sizes, checkpoints, loss terms, augmentations, defaults, coefficients, NMS-free flag).
- `GLOSSARY` — ~50 term → plain-Korean definitions used by the tooltips.
- `EXPS` — the 14 experiment records.
- `CLASSES` — the 12 pill classes with AP/P/R/GT.
- `ROUTES` — screen ids, Korean names, page titles, nav grouping.
- `openRun()` / `setFamily()` / `setSize()` — the cross-screen and schema-swap handlers.
- `th()` — the calibrated threshold metric generator (see the calibration note under screen 11).
- `warnings()` / `eta()` — validation rules and resource estimation, both family-aware.
- `initTips()` — the tooltip engine, portable as-is.

Screen markup is grouped under `<sc-if value="{{ isX }}">` blocks in source order matching the rail.

---

## Open Questions

1. **Pill classes and dataset scale** — 12 classes / 4,000 images is assumed. Real class list and counts needed.
2. **Kaggle submission format** — `image_id,PredictionString` with `{class} {conf} {xmin} {ymin} {xmax} {ymax}` is assumed. Confirm.
3. **Sample imagery** — needed to replace placeholders in the inspector, gallery and monitor previews.
4. **Backend** — FastAPI + WebSocket + Redis queue is assumed; adjust the contract to whatever exists.
