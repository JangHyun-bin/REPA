# REPA 재평가 연구: FFHQ-256 기반 Diagnostic Study

**작성일**: 2026-04-15
**상태**: 설계 완료 — 구현 준비 단계
**목표 학회**: 워크샵 제출 (NeurIPS / CVPR 생성 모델 워크샵)
**기간**: 착수일 기준 1개월
**코드베이스**: 새 저장소, `sihyun-yu/REPA` fork

---

## 0. Executive Summary

본 연구는 고정 계수(fixed-coefficient) REPA (Representation Alignment; Yu et al., ICLR 2025)가 이미지 생성에서 실제로 논문이 주장하는 이익을 제공하는지 의심한다. 동기는 주관적 관찰이다: REPA로 학습한 샘플이 동등한 compute의 DiT/DDPM 대비 더 흐릿하고(blurry) 다양성이 떨어지는 것으로 체감됐다. 이를 엄밀하게 검증하기 위해 FFHQ-256에서 5가지 계수 스케줄 — off, fixed, cosine decay, cosine warmup, HASTE 스타일 hard cutoff — 에 대한 factorial study를 수행한다. 평가는 자동 metric (FID, IS, Precision/Recall, FD-DINOv2, sharpness 분포 거리), pairwise human preference, interpolation 기반 diversity probe를 모두 포함한다.

중심 claim은 diagnostic하다: **고정 계수 REPA는 수렴 속도와 샘플 sharpness/diversity 사이에 지금까지 특성화되지 않은 trade-off를 가지며, 본 연구는 이 trade-off를 여러 스케줄에 걸쳐 특성화하고 이를 감지하기 위한 diagnostic protocol을 제시한다.**

본 연구는 의도적으로 **최소한의(minimal), 검증 가능한(verifiable)** 코드베이스로 진행된다. 사전 예비 실험에 포함되어 있던 REPA 외 부가 기법들(SPRINT sparse path, contrastive flow matching, cls-token dual-stream 학습, path-drop regularization, INVAE)은 모두 제거하고, 원본 REPA 레퍼런스 구현을 그대로 사용한다. 이렇게 해야 관찰되는 모든 효과가 REPA 계수 동역학에 귀속될 수 있으며, 부가 기법들의 confounding으로 오염되지 않는다.

---

## 1. 연구 질문 및 동기

### 1.1 연구를 촉발한 개인적 관찰
U-Net DDPM, DiT, SiT+REPA 아키텍처를 비교하던 비공식 실험에서, 동등한 compute 조건임에도 SiT+REPA 샘플이 더 단순한 DDPM/DiT baseline의 샘플보다 주관적으로 더 흐릿하게 체감됐다. 이 관찰은 REPA 논문이 만든 기대 — ImageNet에서 non-REPA baseline 대비 더 빠른 수렴과 더 낮은 FID — 와 상충된다.

세 가지 설명이 가능하다:
1. **주관적 편향**: 실제 차이는 없는데 관찰자가 편향된 것.
2. **숨겨진 trade-off**: REPA가 FID는 개선하지만, 원 논문이 측정하지 않은 다른 품질 축 (sharpness, diversity) 을 저하시키는 것.
3. **Confounder의 효과**: 사전 실험에 섞여 있던 SpeedrunDiT의 부가 기법들 / INVAE 등 때문이며, 순수 REPA baseline에서는 이 현상이 사라지는 것.

본 연구의 목적은 측정된 증거로 이 세 설명을 구분하는 것이다.

### 1.2 공식 연구 질문
> **RQ**: 최소한의 검증 가능한 코드베이스 하에서 고정 계수 REPA는 FFHQ-256에서 수렴 속도와 샘플 sharpness/diversity 사이의 통계적으로 유의미한 trade-off를 보이는가? 만약 그렇다면, 계수 스케줄링(decay, warmup, hard cutoff)으로 이 trade-off를 완화할 수 있는가?

---

## 2. 중심 Claim (Pre-registered)

### 2.1 Claim B — Diagnostic
> **고정 계수 REPA는 FFHQ-256에서 수렴 속도와 샘플 sharpness/diversity 사이에 지금까지 특성화되지 않은 trade-off를 보인다. 본 연구는 네 가지 계수 스케줄 (off, fixed, decay, warmup, hard cutoff) 에 걸쳐 이 trade-off를 특성화하고, 분포 수준 sharpness metric과 paired human preference를 사용한 diagnostic protocol을 제시한다.**

### 2.2 Pre-registration 조건
Main experiment 시작 전에 `docs/prereg.md`를 새 저장소에 commit해야 하며, 아래 판정 규칙이 포함되어야 한다. Commit 후 이 조건은 **frozen** 된다: 사후 수정 금지.

**Positive 결과 (diagnostic claim 성립)**:
- Primary metric (FID-vs-step trajectory AUC, §7.1 참조) 또는 sharpness/diversity metric 중 최소 하나에서 REPA-Fixed가 REPA-Off 대비 통계적으로 유의미하게 (paired t-test, Bonferroni 보정 α=0.05/6) 저하되고, **동시에**
- 스케줄 변형 (decay, warmup, hard cutoff) 중 최소 하나가 fixed REPA의 수렴 속도 이점 (FID-vs-step 곡선) 을 유지하면서 저하된 축을 부분적으로 회복.

**Null 결과**:
- 최종 checkpoint에서 어떤 branch 간에도 sharpness/diversity에서 통계적 유의미한 차이가 없음. 이 경우 full methodology와 함께 null report로 발표.

**Mixed 결과**:
- 자동 metric은 차이를 보이지만 human preference와 일치하지 않거나, 그 반대. Dissociation finding으로 보고.

세 결과 모두 publishable. 연구는 어떤 결과가 나오더라도 유용한 기여를 만들도록 설계되었다.

### 2.3 통계적 power의 사전 명시 (Pre-registration의 일부)

본 연구는 branch당 n=3 seed로 paired t-test를 사용한다. Bonferroni 보정 (α=0.05/6 ≈ 0.0083) 하에서 검출 가능한 최소 effect size는 매우 큼 — 양측 paired t-test, n=3, power=0.80 기준 Cohen's d ≈ 4.5. 즉 본 연구의 사전 등록된 hypothesis test는 *작거나 중간 크기*의 effect를 감지할 수 없으며, **pre-registered 통계 검정의 통과 여부는 effect의 *존재* 가 아닌 *극단성* 만 측정**한다. 이것은 사전 등록된 알려진 한계이다.

이 underpowered 상태는 다음으로 보완한다:

1. **모든 결과를 4가지 형태로 동시 보고**:
   - **Descriptive statistics**: branch별 metric의 mean ± std (n=3 seeds).
   - **Directional consistency**: 몇 개의 seed가 같은 부등식 방향을 가리키는가 (e.g., 3/3, 2/3).
   - **Effect size**: Cohen's d (paired) 점추정과 1000-iter bootstrap 95% CI.
   - **Pre-registered significance test**: Bonferroni 보정 paired t-test의 p-value.
2. **Single test 결론 금지**: 어떤 단일 metric의 significance test 결과만으로 claim을 지지/반박하지 않는다. 4가지 정보가 일관될 때만 strong claim, 불일치 시 weak claim 또는 추가 분석 필요로 보고.
3. **"Underpowered null" vs "true null" 분리**: pre-registered test가 통과되지 않더라도 effect size가 |d| > 0.8 (large) 이면 그것은 underpowered null (실제 effect가 있을 가능성) 로 명시적 보고, "차이 없음" 결론을 짓지 않는다. |d| < 0.2 (small) 인 경우만 진짜 null로 해석.
4. **이 power 한계 자체가 pre-registration의 일부**: 나중에 더 큰 seed로 follow-up을 한다면 "exploratory replication" 으로 별도 발표.

---

## 3. Baseline 및 Purity Principle

### 3.1 Purity Principle
원본 REPA 논문 (Yu et al., *Representation Alignment for Generation*, ICLR 2025) 에 존재하지 않는 모든 구성요소는 baseline에서 **제외**한다. 실험 시스템은 정확히 다음만으로 구성된다:
- SiT backbone (SiT-B/2; §5 참조)
- 원본 REPA projection loss (하나의 중간 DiT layer를 frozen DINOv2 ViT-B feature에 negative cosine similarity로 정렬하는 MLP projector)
- Standard v-prediction flow matching objective
- Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, 8× 공간 압축, 4 latent channels)

### 3.2 명시적으로 제외할 구성요소
사전 예비 실험에 포함되어 있던 다음 구성요소들은 **제거**되며, 새 코드베이스에 등장하면 안 된다:
- **SPRINT sparse path** (token dropping + sparse forward + mask padding + dense-sparse fusion)
- **Contrastive Flow Matching (CFM) auxiliary loss** (출처 불명, REPA 논문에 없음)
- **Class-token dual-stream 학습** (CLS token에 대한 별도 noise/target)
- **Path-drop regularization** (학습 중 stochastic path drop)
- **adaLN-Gaussian 초기화** (원본 SiT/REPA의 zero 초기화로 복귀)
- **INVAE** (REPA 논문의 실제 VAE 선택인 SD-VAE 사용)

### 3.3 코드베이스 설정
구현은 원본 REPA 저장소에서 시작한다:
```bash
git clone https://github.com/sihyun-yu/REPA.git repa-reeval
cd repa-reeval
```

이 코드베이스에 대한 최소 수정사항:
1. FFHQ-256 (및 사전 계산된 SD-VAE latents) 을 읽도록 dataset loader 교체.
2. Class-conditional label 처리 제거 (FFHQ는 unconditional); 고정 null class token으로 대체.
3. 학습 루프에 계수 scheduler hook 추가, off / fixed / cosine decay / cosine warmup / hard cutoff 모드 지원.
4. 본 연구의 metric suite를 재현하는 evaluation harness script 추가.
5. Pre-registration markdown 문서 (`docs/prereg.md`) 추가.

새 저장소의 모든 commit은 태그되어야 한다. Pre-registration commit은 `prereg-v1` 이며 모든 main training commit보다 선행해야 한다. Git tag로 이를 enforce.

---

## 4. 데이터셋: FFHQ-256

### 4.1 선택 이유
- **Compute 실현성**: 256×256 해상도의 70K 이미지는 1개월 예산 내에 단일 consumer GPU에서 학습 가능. ImageNet-1K는 불가능.
- **연구 질문에의 도메인 적합성**: 얼굴은 sharpness와 identity 수준 diversity에 대해 인간 지각이 극도로 민감한 도메인. "Blur"는 ImageNet의 넓은 카테고리 혼합보다 얼굴에서 더 진단적이다.
- **Unconditional benchmark**: FFHQ는 unconditional 생성의 표준 benchmark로, class conditioning으로 인한 confounder 제거.
- **표준 FID reference**: FFHQ-256의 FID reference statistics는 표준화되어 있고 선행 연구에서 널리 보고됨.

### 4.2 원본 REPA로부터의 도메인 이탈(Domain deviation)
원본 REPA 논문은 ImageNet-1K (class-conditional) 에서 평가한다. FFHQ-256 (unconditional, 단일 도메인) 사용은 의도된 이탈이다. 논문의 methods 섹션에 명시적으로 기술:

> "We evaluate on FFHQ-256 rather than ImageNet-1K due to compute constraints. FFHQ, being a high-resolution face dataset, provides a domain where perceptual sharpness and identity-level diversity are particularly salient — making it well-suited for our research question about REPA's quality trade-offs."

이 포지셔닝은 이탈을 연구 설계의 **특징(feature)** 으로 프레이밍하며, 단순한 한계로 취급하지 않는다.

### 4.3 데이터 준비 (Streaming-first workflow)

로컬 디스크 제약 (< 90 GB 가용) 으로 인해 raw FFHQ-256 이미지를 영구 저장하지 않는다. 다음 두 옵션 중 하나 사용:

**옵션 A (Recommended): HuggingFace Hub streaming**
1. HuggingFace Hub의 안정적인 FFHQ-256 mirror에서 `datasets.load_dataset(<id>, split=..., streaming=True)` 로 한 번만 iterate. (사용 mirror id, revision hash는 `data/MANIFEST.json` 에 기록.)
2. 각 이미지에 대해 즉시 (a) 256×256 verify/resize, (b) `stabilityai/sd-vae-ft-mse` 로 SD-VAE encode, (c) latent을 bf16으로 `data/ffhq256_latents/{split}/{index}.pt` 에 저장.
3. Raw 이미지는 RAM에서 즉시 폐기 — 디스크에 절대 영구 저장하지 않음.
4. **순(net) 디스크 사용**: SD-VAE latents 약 600 MB. Raw 이미지 0 GB.
5. `data/MANIFEST.json` 에 (a) 사용한 dataset id + revision hash, (b) 받은 sample 수, (c) 각 latent 파일의 SHA-256, (d) train/eval split 인덱스 list 를 모두 기록 — *content-addressed* manifest로 재현성 확보.

**옵션 B: 임시 다운로드 → 즉시 폐기 (Fallback)**
1. 공식 FFHQ-256 릴리즈를 임시 디렉토리 (e.g., `/tmp/ffhq256_raw`) 에 다운로드 (~28 GB temporary).
2. 위와 동일하게 SD-VAE latent 사전 계산.
3. 사전 계산 완료 *즉시* `/tmp/ffhq256_raw` 디렉토리 삭제. 디스크 영구 저장 0.
4. 옵션 A의 streaming이 불안정하거나 HF mirror를 신뢰할 수 없을 때 사용.

**Train/eval split 고정**:
- 65,000 train, 5,000 eval. Latent file index 기준 (raw filename이 아님 — streaming workflow와 호환).
- 인덱스 list를 `data/ffhq256_train.txt` / `data/ffhq256_eval.txt` 에 기록.
- 두 list 파일과 모든 latent 파일의 SHA-256를 `data/MANIFEST.json` 에 등록.

**Reference statistics (FID, FD-DINOv2)**:
- Eval split의 latent을 SD-VAE decode하여 5K 이미지 (~1 GB) 를 *메모리에만* 일시 생성 → `clean-fid` / DINOv2 feature 계산 → reference 통계 파일만 저장.
- 저장: `data/fid_ref_ffhq256.pt`, `data/fd_dinov2_ref_ffhq256.pt`. 합계 < 200 MB.
- Decode된 reference 이미지는 디스크에 저장하지 않음.

모든 reference statistics 파일은 저장소에 직접 commit한다. 목표는 미래의 reviewer가 정확한 숫자를 재현할 수 있는 것.

### 4.4 저장 공간 예산

가용 디스크 < 90 GB. 모든 artifact를 다음과 같이 budgeting하여 영구 사용량 ~20 GB로 제약:

| Artifact | 최적화 전 (원본 디자인) | 최적화 후 |
|---|---|---|
| Raw FFHQ-256 PNG | ~28 GB | **0 GB** (streaming/temp-delete) |
| SD-VAE latents (70K, bf16) | ~0.6 GB | ~0.6 GB |
| FID/FD-DINOv2 reference statistics | ~0.2 GB | ~0.2 GB |
| Code + git history | ~0.1 GB | ~0.1 GB |
| Checkpoints (15 runs) | ~150 GB (full state × 5 ckpts × fp32) | **~16 GB** (EMA-only bf16, eval ckpt만) |
| Eval samples (60 evaluations × 10K) | ~30 GB (PNG 저장) | **~0.6 GB** (stream-compute → 폐기, 시각화용 16-64장만 보관) |
| Human eval images (600장 PNG) | ~0.15 GB | ~0.15 GB |
| Logs, `results.csv`, plots | ~2 GB | ~2 GB |
| **합계 (영구)** | **~211 GB** | **~20 GB** |
| Buffer (50%) | | ~10 GB |
| **할당** | | **~30 GB** |

**Checkpoint 압축 전략**:
1. **Eval checkpoint만 저장** (4점: 20K/50K/80K/100K). 매 20K마다는 저장 안 함.
2. **EMA-only, bf16** 으로 저장. SiT-B/2 EMA bf16 = 130M × 2 byte = ~260 MB / checkpoint. 4점 × 15 runs = ~16 GB.
3. **Resume용 full state는 저장 안 함**. Run당 wallclock ~3.5h 이므로 fail 시 처음부터 재시작 수용. 단 *현재 step* 의 rolling latest checkpoint 1개만 학습 중 임시 유지 (~2 GB 임시), 다음 eval checkpoint 도달 시 갱신.

**Eval sample 처리**:
- `eval.py` 가 10K 샘플을 메모리에서 batch-by-batch로 생성 → metric 계산 → 폐기.
- 시각화용 첫 16-64 샘플만 PNG로 저장 (`exps/{branch}_{seed}/eval_{step}/preview/`).
- Eval 한 번당 임시 디스크 사용 < 1 GB.

**Worst-case headroom**: 영구 ~20 GB + 임시 (current run의 latest ckpt 2 GB + current eval의 in-flight 1 GB) = 25 GB. 사용자 가용 90 GB의 ~28%. 충분.

---

## 5. 모델 및 학습 설정

### 5.1 모델
- **SiT-B/2**: patch_size=2, depth=12, hidden_size=768, num_heads=12. 130M 파라미터.
- 입력: SD-VAE latent 32×32×4. patch=2 하에서 16×16 = **256 tokens**, 각 토큰은 4×2×2=16 차원을 768로 projection.
- 출력: 동일한 shape.
- Projection head: DiT hidden 768 → DINOv2 embedding dim 768을 매핑하는 단일 layer MLP. 하나의 중간 block에서 projection (REPA 논문 default: SiT-B의 경우 block index 8, 구현 시 reference 코드에서 재확인).

### 5.2 Teacher encoder
- DINOv2 ViT-B (`facebook/dinov2-base`), frozen, `eval()` 모드.
- 입력 전처리: bilinear로 256→224 resize, ImageNet normalization.
- Feature 추출: 최종 layer의 patch tokens (CLS 아님), shape (B, 256, 768). 이것이 cosine-similarity 정렬의 `zs` target.

### 5.3 Optimizer
- AdamW, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0.
- Learning-rate schedule 없음 (constant lr, REPA 논문 일치).
- Gradient clip: max-norm=1.0.
- EMA: 모델 가중치 0.9999 decay; sample은 EMA copy에서 추출.

### 5.4 Loss 구성
학습 objective는 정확히:
```
L(t) = L_denoise(t) + λ(t) · L_proj(t)
```
여기서
- `L_denoise`는 SD-VAE latent 상의 standard v-prediction flow matching MSE loss,
- `L_proj`는 projection된 DiT hidden state와 frozen DINOv2 feature 사이의 평균 negative cosine similarity,
- `λ(t)`는 §6에서 branch별로 정의되는 계수 스케줄.

다른 loss 항 없음. cls-token dual loss 없음. Contrastive flow matching 없음. 어떤 보조 항도 없음.

### 5.5 학습 하이퍼파라미터 (모든 branch에 공통 고정)
| Hyperparameter | 값 |
|---|---|
| Batch size | 32 |
| Precision | bf16 (accelerate mixed precision) |
| Total steps | 100,000 |
| Warmup steps (lr) | 0 |
| Eval checkpoints | 20K, 50K, 80K, 100K |
| Save checkpoints | Eval checkpoints만 (20K/50K/80K/100K), **EMA-only bf16** (§4.4 참조). Resume 시 처음부터 재시작. |
| Seeds | {42, 1337, 2024} — branch당 3 seed |
| Data augmentation | horizontal flip만 |

---

## 6. Research Branches

각 branch는 계수 스케줄 `λ(t)` 로 정의되며 `t ∈ [0, T]`, `T = 100,000`. 최대 계수는 원본 REPA 논문 SiT-B/2 variant의 default에 맞춰 **0.5**로 설정.

### B0 — REPA Off (control)
```
λ(t) = 0  for all t
```
**목적**: 절대 control. 어떤 alignment 압력도 없이 SiT-B/2가 FFHQ-256에서 달성하는 수준을 측정. REPA의 기여분을 측정하는 ground truth.

### B1 — REPA Fixed (standard REPA)
```
λ(t) = 0.5  for all t
```
**목적**: REPA 논문의 default configuration 재현. 이 branch 없이는 "REPA의 효과"에 대한 어떤 claim도 불가능.

### B2 — Cosine Decay (1→0)
```
λ(t) = 0.5 · 0.5 · (1 + cos(π · t / T))
```
**목적**: Full REPA로 시작해 zero까지 decay. 가설: REPA의 가속 효과는 학습 초반 현상이며, 후반에 제거하면 본 연구에서 조사 중인 후반 품질 저하를 방지한다. 끝점: λ(0)=0.5, λ(T/2)=0.25, λ(T)=0.

### B3 — Cosine Warmup (0→1)
```
λ(t) = 0.5 · 0.5 · (1 - cos(π · t / T))
```
**목적**: B2의 역. Free 학습으로 시작해 REPA 압력을 후반에 추가. REPA의 trade-off가 정말 후반 붕괴라면 이 branch가 가장 나쁜 sharpness/diversity를 보여야 한다. B2 가설의 falsification check 역할. 끝점: λ(0)=0, λ(T/2)=0.25, λ(T)=0.5.

### B4 — HASTE 스타일 Hard Cutoff
```
λ(t) = 0.5  if t < T/2
       0    if t ≥ T/2
```
**목적**: HASTE 2025 ("REPA Works Until It Doesn't") 논문의 처방과 일치. Prior-art 방어에 결정적: HASTE의 hard cutoff가 이미 본 연구의 smooth 스케줄이 달성하는 것을 달성한다면 우리 기여가 약해짐. 이 branch가 "smooth이 hard보다 낫다" 혹은 "smooth은 hard와 동등하다" 의 입장을 잡게 해줌 — 두 statement 모두 publishable.

### Branch 요약
| Branch | 스케줄 | 최대 계수 | 총 run 수 |
|---|---|---|---|
| B0 | off | 0 | 3 (seeds) |
| B1 | fixed | 0.5 | 3 |
| B2 | cosine decay | 0.5 → 0 | 3 |
| B3 | cosine warmup | 0 → 0.5 | 3 |
| B4 | hard cutoff | 0.5 → 0 at t=T/2 | 3 |
| **합계** | | | **15 runs** |

---

## 7. Evaluation Protocol

### 7.1 자동 metric (매 eval checkpoint에서 계산)
15 runs 각각에 대해 4개의 eval checkpoint (20K, 50K, 80K, 100K) 에서, **10,000 샘플** 생성 (50-step DDPM sampler, FFHQ unconditional이므로 CFG 비활성) 후 아래 metric 계산:

| Metric | 도구 | 목적 |
|---|---|---|
| **FID-vs-step AUC** (20K → 100K) | 4 eval checkpoint의 FID에 대한 trapezoid integration | **Primary metric** — 단일 endpoint가 아닌 수렴 trajectory 전체를 포착. §2.2, §8 참조. |
| FID @ 100K | `clean-fid` | Secondary; REPA 논문과의 cross-paper 비교 가능성 유지 |
| Inception Score | `torch-fidelity` | Sanity check only (FFHQ는 단일 도메인이라 IS의 정보량 거의 없음) |
| Precision / Recall | Kynkäänniemi et al. 2019 구현 | Diversity (recall) 와 fidelity (precision) 분리 |
| FD-DINOv2 | 자체 (DINOv2 feature + Frechet distance) | DINO feature 공간에서의 semantic fidelity |
| Sharpness 분포 거리 | 자체 (§7.2 참조) | "Blurry" 관찰의 객관화 |

출력 형식: 모든 metric을 포함한 `{branch}_{seed}_{step}.json`. `python scripts/aggregate.py` 헬퍼로 `results.csv` 에 집계.

### 7.2 Sharpness metric (자체 구현)
본 연구를 촉발한 "blurry" 관찰은 객관적으로 측정 가능해야 한다. 단일 이미지별 점수보다 분포 수준 metric을 사용:

1. 각 이미지 `x` 에 대해 세 개의 고전적 sharpness 통계 계산:
   - **Laplacian variance**: grayscale 변환 후 `var(Laplacian(x))`.
   - **High-frequency energy ratio**: FFT magnitude 중 radius-cutoff (Nyquist의 50%) 바깥 부분과 총 FFT magnitude의 비율.
   - **Sobel gradient mean**: Sobel magnitude map의 평균.

2. 생성 샘플 (N=10,000) 과 reference 이미지 (FFHQ eval split에서 N=5,000) 에 대해 각 통계량을 계산, 총 여섯 개의 1-D 분포 생성.

3. 각 통계량에 대해 생성 분포와 reference 분포 사이의 Wasserstein-1 거리 계산.

4. 세 개의 Wasserstein 거리를 **별도로** 보고. 해석: 작을수록 해당 sharpness 통계량의 생성 분포가 실제 이미지 분포에 더 가깝다.

이 metric은 "더 sharp vs 덜 sharp"를 순위 매기지 않는다 — "실제 분포에 가까움 vs 멀어짐"을 순위 매긴다. Reference 이미지가 ground-truth sharpness 분포를 정의한다고 가정.

### 7.3 Human evaluation (최종 checkpoint에서만)
**목적**: 본 연구를 촉발한 "blurry"라는 주관적 관찰을 객관적으로 검증 또는 반증.

**프로토콜**: Two-alternative forced choice (2AFC) pairwise 비교.

- **Pair 조합**: B0 또는 B1을 포함하는 5 branches의 모든 2-조합 (primary 비교), 그리고 스케줄 비교를 위한 B2–B1 와 B4–B2:
  - B0 vs B1 (REPA가 도움이 되는가 해가 되는가?)
  - B0 vs B2 (decay 스케줄이 도움이 되는가?)
  - B0 vs B3 (warmup 스케줄이 도움이 되는가?)
  - B1 vs B2 (decay가 fixed보다 개선하는가?)
  - B1 vs B4 (hard cutoff가 fixed보다 개선하는가?)
  - B2 vs B4 (smooth vs hard?)

  **총 6 pair 조합**.

- **Pair 조합당 이미지 쌍 수**: 50개, 매칭된 noise seed로 생성 (즉, 동일한 초기 noise를 두 branch의 EMA checkpoint에 통과시킴). 이로써 샘플의 무작위 변동 통제.

- **이미지 쌍당 rater 수**: 5. 초기 pilot에는 self + 지인 2-3명; 결과가 promising하면 독립적 rater를 위해 Prolific (USD 50-100 예산) 으로 확장.

- **이미지 쌍당 질문 수**: 2 — "Which image looks sharper?" 와 "Which image looks more like a real photograph?"

- **총 응답 수**: 6 pairs × 50 image pairs × 5 raters × 2 questions = **3,000 responses**.

- **Platform**: Minimal 로컬 Flask 또는 static HTML + JavaScript 앱. 좌우 랜덤화 및 CSV 응답 저장. 별도 툴 구매 불필요.

**분석**: 각 branch에 대해 본인이 등장하는 모든 비교에 걸쳐 Bradley-Terry preference score를 계산. Bootstrap resampling (1000 iterations) 으로 95% 신뢰 구간. 각 자동 sharpness metric이 human preference를 예측하는지 검증하기 위해 Bradley-Terry score와 자동 metric 간 Pearson 상관관계 계산.

### 7.4 Interpolation study (diversity probe, 최종 checkpoint에서만)
**목적**: REPA가 latent 공간 붕괴를 유발한다면, random latent 사이의 interpolation은 부드러운 traversal이 아니라 "mode snapping" (급작스런 점프) 으로 나타날 것. 이는 질적이자 양적으로 측정 가능한 signature.

**프로토콜**:
1. 각 branch에 대해, 최종 EMA checkpoint 사용:
   - Random 초기 noise latent 쌍 `(z_a, z_b)` 10개 생성.
   - 각 쌍에 대해 9개 값 `t ∈ {0, 0.125, 0.25, ..., 1.0}` 에서 spherical linear interpolation (slerp).
   - 각 쌍의 9-step 생성 전체를 decode.
2. 각 decode 이미지에 대해 DINOv2 ViT-B feature (patch-token mean) 계산.
3. 각 쌍에 대해 연속적인 interpolation 단계들 사이의 pairwise feature distance 시퀀스 (길이 8 시퀀스) 계산.
4. **Jaggedness score**: `var(distances) / mean(distances)`. 높은 jaggedness = mode snapping. 낮은 jaggedness = 부드러운 traversal.
5. Branch별로 10 쌍에 걸친 평균 jaggedness ± std 보고.

**시각화**: 논문용으로 branch당 interpolation grid 하나 (5 × 9 이미지) 를 결과 섹션에 배치.

---

## 8. Pre-registration 요구사항

Main training run 이전에 새 저장소에는 `prereg-v1` 태그로 commit된 `docs/prereg.md` 가 있어야 한다. 이 파일 내용:

1. **Hypothesis 진술문** (§2.1 Claim B의 복사본).
2. **Pre-registered 결과 조건** (§2.2의 복사본).
3. **Primary metric**: **FID-vs-step trajectory AUC** (4 eval checkpoint 20K/50K/80K/100K 에 대한 trapezoid integration), paired t-test (Bonferroni α=0.05/6) 로 비교. **Secondary**: 100K step의 FID 단일점 (legacy comparator로 REPA 논문과 cross-paper 비교성 유지). Primary로 trajectory를 사용하는 이유: 본 연구의 hypothesis는 *수렴 가속 vs 후반 품질* trade-off이며, 단일 endpoint metric은 trajectory 정보를 수렴 도달 여부와 분리하지 못함.
4. **Secondary metrics**: 100K FID 단일점, Precision, Recall, FD-DINOv2, 세 개의 sharpness Wasserstein 거리, human eval의 Bradley-Terry score, jaggedness score.
5. **중단 조건**: Pilot run (§9.1.2) 에서 20K step까지 3개 이상의 branch에 걸쳐 어느 metric에서도 식별 가능한 signal이 없으면 일시 중단하고 연구 질문 재평가. Compute 예산이 GPU 연속 사용 5일을 넘으면 일시 중단하고 우선순위 재정의.
6. **Pre-registered branch 순위 기대값** (데이터 보기 전 실험자의 사전 추측):
   - Sharpness: B0 > B2 ≈ B4 > B1 > B3 (실험자 가설; confirmation bias 감지용)
   - FID 수렴 속도 (목표 FID까지의 step 수): B1 ≈ B4 > B2 > B3 > B0
   - 사후 confirmation-bias check 가능하게 기록. 실제 결과가 pre-registered 기대와 거의 완벽히 일치하면 더 *의심스러운* 것이지 덜 의심스러운 것이 아님.
7. **분석자 identity**: 실험자 이름과 pre-registration 날짜.

**엄격한 규칙**: `git tag prereg-v1` 이후 `prereg.md` 의 내용은 수정 불가. Pre-registration에 포함되지 않은 추가 분석은 논문에서 "exploratory" 로 명시.

---

## 9. Timeline — 1개월

총 기간: 28일 (4주). GPU 예산: 단일 consumer GPU (RTX 4080 Super 16GB 또는 동급).

### Week 1 — 설정 및 Pilot (Days 1–7)
**Day 1–2: 코드베이스 및 데이터**
- `sihyun-yu/REPA` 를 새 저장소로 fork.
- 환경 설정 (conda/venv, pytorch, accelerate, clean-fid, torch-fidelity, SD-VAE용 diffusers, dinov2).
- FFHQ-256 다운로드, split 고정, SHA-256 manifest 계산.
- SD-VAE latents 사전 계산 (4080S에서 1–2 시간).
- FID와 FD-DINOv2 reference statistics 사전 계산.

**Day 3: 최소한의 코드 수정**
- FFHQ-256 latents용 dataset loader 교체.
- Class-conditional 처리 제거; null token 사용.
- 학습 루프에 계수 scheduler hook 추가 (함수 `schedule_off`, `schedule_fixed`, `schedule_cosine_decay`, `schedule_cosine_warmup`, `schedule_hard_cutoff`).
- `schedule_off` 가 `λ=0` 강제 REPA와 동일한 behavior를 생성하는지 검증.
- `schedule_fixed` 가 동일 config 하에서 원본 REPA repo의 fixed-coefficient run과 1000-step loss trajectory 가 일치 (부동소수점 노이즈 제외) 하는지 검증.

**Day 4: Evaluation harness**
- Checkpoint 경로를 받아 모든 metric의 JSON을 생성하는 `eval.py` 작성.
- 무작위 초기화된 모델에서 `eval.py` 가 end-to-end로 실행되는지 확인 (FID는 터무니없이 높을 것 — 그건 괜찮음; pipeline 테스트지 모델 테스트가 아님).
- 모든 result JSON을 읽어 `results.csv` 를 생성하는 `scripts/aggregate.py` 작성.

**Day 5: Pre-registration**
- `docs/prereg.md` 를 §2 와 §8 에 맞춰 작성.
- Commit 및 `git tag prereg-v1`.
- 이 순간부터 hypothesis는 frozen.

**Day 6–7: Pilot runs**
- Pilot 실행: 5 branches × 1 seed × 20,000 steps. 약 5 × 40분 = **3.5 GPU-hours**.
- Pilot checkpoint 평가. 각 branch의 스케줄이 측정 가능하게 다른 loss trajectory 를 생성하는지 확인. Evaluation harness 가 합리적인 숫자를 생성하는지 확인.
- Pilot run이 crash하거나 pathological behavior를 보이면 main run 진행 전에 debug.

### Week 2 — Main Training Runs (Days 8–14)
**Day 8–11: Main runs**
- 15 runs 순차 실행: 5 branches × 3 seeds × 100,000 steps.
- 각 run: 4080S에서 약 3.5 GPU-hours (이전 SiT-B/1 벤치마크 기반 외삽).
- 총: 15 × 3.5 = **52.5 GPU-hours** ≈ **2.2일** 연속 GPU 사용.
- 여유: 재시작, 디버그, 예기치 않은 crash를 위한 1.8일 buffer.
- §4.4 storage 정책에 따라 **eval checkpoint만** (20K/50K/80K/100K) **EMA-only bf16** 으로 저장. 매 20K 보관 안 함. Run 중 fail 시 처음부터 재시작.

**Day 12–14: 자동 평가**
- 15 runs × 4 eval checkpoints = 60 evaluation을 `eval.py` 로 실행.
- 각 evaluation: 약 20-30분 (10K sample 생성 + metric 계산).
- 총: 60 × 25분 = **25 GPU-hours** ≈ 1일 연속.
- `results.csv` 생성.
- 예비 plot (branch별 FID vs step) 생성 및 검토. 명확한 signal 또는 명확한 null 탐색.

### Week 3 — Human Eval, Interpolation, Analysis (Days 15–21)
**Day 15–16: Human eval platform**
- 2AFC pairwise 비교를 위한 minimal Flask 또는 static HTML 앱 구축.
- Synthetic 쌍으로 테스트.
- 6 × 50 = 300 이미지 쌍 생성 (각 쌍은 비교 대상 두 branch의 최종 EMA checkpoint 사이의 매칭된 noise seed 사용).

**Day 17–18: Human eval 응답 수집**
- 첫 패스 (pilot) 를 위한 self + 지인 2-3명.
- Pilot 결과가 명확한 차별화를 보이면 Prolific 등으로 독립 rater 5명 모집.
- 모든 응답을 CSV로 저장.

**Day 19: Interpolation study**
- 각 branch의 최종 EMA checkpoint에 대해 10 × 9 = 90 slerp 이미지 생성.
- DINOv2 feature 및 jaggedness score 계산.
- 논문용 interpolation grid 생성.

**Day 20–21: 통계 분석**
- 모든 결과 집계.
- Pre-registered 검정 실행: Bonferroni 보정 paired t-test.
- Human eval로부터 Bradley-Terry score 계산.
- 자동 sharpness metric과 human preference 사이의 상관관계 계산.
- 모든 최종 plot 생성 (FID vs step, sharpness bar chart, jaggedness bar chart, CI 포함 Bradley-Terry score, 상관관계 scatter).
- Pre-registered 판정 규칙 (§2.2) 적용. 결과를 positive / null / mixed 로 기록.

### Week 4 — Writing 및 제출 (Days 22–28)
**Day 22–24: 초안**
- 논문 초안 작성: introduction, related work, method, experiments, results, discussion.
- 목표: 4–8 page 워크샵 포맷.
- Pre-registration을 appendix로 포함.

**Day 25–26: Figure 및 Table**
- 모든 figure를 publication 품질로 finalize.
- Main results table 생성.
- 모든 원본 숫자를 appendix에 포함.

**Day 27–28: 수정 및 제출**
- Self-review 패스.
- 가능하면 외부 reader 한 명에게 전달.
- 목표 워크샵에 제출.

### Critical path 요약
| Phase | Days | GPU-hours |
|---|---|---|
| Setup + pilot | 7 | ~5 |
| Main runs + eval | 7 | ~78 |
| Human eval + interp + analysis | 7 | ~3 |
| Writing | 7 | 0 |
| **합계** | **28** | **~86** |

Compute가 binding constraint. 28일 동안 consumer GPU에서 86 GPU-hours는 GPU가 약 13% 가동되는 것 — 재시작과 디버그를 위한 충분한 여유.

---

## 10. 리스크 및 완화

| 리스크 | 확률 | 영향 | 완화 |
|---|---|---|---|
| HASTE가 이미 동등한 finding을 발표함 | 높음 | 기여 약화 | Week 1에 HASTE 꼼꼼히 읽기; B4 branch가 우리를 "smooth vs hard" 비교로 포지셔닝. Null 결과도 publishable. |
| **n=3가 모든 hypothesis test를 underpower 시킴** | **높음** | Pre-registered test가 실제 effect를 놓침 | §2.3에 명시된 4-way 보고 (descriptive / directional consistency / effect size / pre-registered test) 로 결론. Underpowered null과 진짜 null을 명시적 분리. Effect size + directional consistency는 hypothesis test와 독립적으로 보고. |
| **저장 공간 부족 (가용 < 90 GB)** | 중간 | 학습/eval 중단 | §4.4의 storage budget 강제: streaming 데이터셋, EMA-only bf16 checkpoint, eval sample stream-compute. 영구 ~20 GB로 제약. 학습 중 디스크 사용량 모니터링 (자동 alert). |
| Null 결과 (어떤 branch도 유의하게 다르지 않음) | 중간 | Claim 실패 | Claim B는 null로도 publishable하게 설계됨. 프레이밍을 "trade-off 존재" 에서 "순수 REPA에는 trade-off 없음, 주관적 인상은 이전 confounder로 설명됨" 으로 전환. **단** §2.3의 effect-size 기반 underpowered-null vs true-null 구분을 적용하여 결론. |
| 자동 sharpness metric이 human preference와 상관 없음 | 중간 | Dissociation finding | Sub-finding으로 프레이밍: "기존 자동 metric은 인간이 지각하는 sharpness 차이를 포착하지 못함; 미래 REPA 연구에는 human eval을 권장." |
| GPU compute 예산 초과 | 낮음 | Timeline slip | Seed를 3개에서 2개로 줄임 (통계적 power 감소하지만 예산 유지). Branch는 줄이지 말 것. |
| FFHQ-SD-VAE 불일치 (SD-VAE가 FFHQ를 잘 재구성 못함) | 낮음 | Baseline 손상 | Week 1 Day 1에 확인: held-out FFHQ batch를 encode/decode 후 PSNR + LPIPS 계산. **PSNR ≥ 28 dB AND LPIPS ≤ 0.10** 이어야 함. PSNR 임계 28 dB는 SD-VAE-ft-mse 의 알려진 동작 범위 (face/natural image 27-30 dB) 와 일치 (2026-04-15: 두 mirror에서 ~29.6 dB로 수렴 측정 후 spec frozen 전 조정). LPIPS는 perceptual 지표로 함께 검증. |
| Pre-registration이 사후 수정됨 | 중간 | Diagnostic claim 무효화 | Git tag `prereg-v1` 및 명시적 "이 commit 이후 frozen" 텍스트로 enforce. 모든 수정은 논문에서 별도 exploratory 분석으로 처리. |
| λ=0 에 늦게 도달하는 스케줄 (B2) 의 학습 불안정성 | 낮음 | 특정 branch 실패 | 학습 중 loss 곡선 모니터링. 불안정성 발생 시 해당 branch 중단 및 보고. |

---

## 11. Deliverables

28일 종료 시점에 새 저장소에 존재하는 것:

### 코드
- Fork되고 최소 수정된 REPA 학습 코드
- `scripts/precompute_latents.py`, `scripts/precompute_fid_ref.py`, `scripts/precompute_dino_ref.py`
- `scripts/train.py` (REPA 원본에서 수정)
- `scripts/eval.py`
- `scripts/aggregate.py`
- `scripts/human_eval_server.py` (로컬 Flask / static HTML)
- `scripts/interpolation_study.py`
- `scripts/plots.py`

### 데이터 아티팩트
- `data/MANIFEST.json` — FFHQ-256 split 해시
- `data/fid_ref_ffhq256.pt`
- `data/fd_dinov2_ref_ffhq256.pt`

### 결과
- `exps/{branch}_{seed}/checkpoints/ema_{step}.pt` — EMA-only bf16, eval checkpoint만 (20K/50K/80K/100K). 15 runs × 4 ckpt ≈ 16 GB. (§4.4 참조)
- `exps/{branch}_{seed}/loss_log.csv` — step별 loss trajectory (text only, 작음)
- `exps/{branch}_{seed}/eval_{step}/preview/` — 시각화용 첫 16-64개 sample PNG만
- `results.csv` — 모든 metric (FID-vs-step AUC primary 포함), 모든 eval checkpoint, 모든 run
- `human_eval/responses.csv` — 모든 human eval 응답
- `interpolation/{branch}/` — interpolation grid 및 jaggedness score

### 문서
- `docs/prereg.md` — pre-registration (frozen, `prereg-v1` 태그)
- `docs/BASELINE.md` — 원본 REPA로부터의 deviation을 포함한 정확한 baseline 구성
- `docs/EVAL_PROTOCOL.md` — 전체 evaluation 방법론
- `README.md` — 재현 instructions

### 논문
- `paper/draft.tex` 또는 `paper/draft.md`
- `paper/figures/` — 모든 figure를 publication 품질로
- `paper/tables/` — 모든 table을 CSV + LaTeX로

---

## 12. 부록: 주요 명령어 및 설정 스니펫

### 12.1 저장소 설정
```bash
git clone https://github.com/sihyun-yu/REPA.git repa-reeval
cd repa-reeval
git checkout -b reeval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install clean-fid torch-fidelity
```

### 12.2 FFHQ 전처리
```bash
# FFHQ-256이 data/ffhq256_raw/ 에 다운로드되었다고 가정
python scripts/fix_splits.py --src data/ffhq256_raw --out data/
python scripts/precompute_latents.py --split train --vae stabilityai/sd-vae-ft-mse
python scripts/precompute_latents.py --split eval  --vae stabilityai/sd-vae-ft-mse
python scripts/precompute_fid_ref.py   --split eval
python scripts/precompute_dino_ref.py  --split eval
```

### 12.3 Pre-registration
```bash
# docs/prereg.md 작성 후:
git add docs/prereg.md
git commit -m "prereg: pre-registered hypotheses and outcome conditions"
git tag prereg-v1
git push --tags
```

### 12.4 Training 명령어 (branch × seed당 하나)
```bash
# B0
python scripts/train.py --branch off     --seed 42   --total_steps 100000 --out exps/b0_s42
# B1
python scripts/train.py --branch fixed   --seed 42   --total_steps 100000 --out exps/b1_s42
# B2
python scripts/train.py --branch decay   --seed 42   --total_steps 100000 --out exps/b2_s42
# B3
python scripts/train.py --branch warmup  --seed 42   --total_steps 100000 --out exps/b3_s42
# B4
python scripts/train.py --branch cutoff  --seed 42   --total_steps 100000 --out exps/b4_s42
# 각각 --seed 1337 과 --seed 2024 로도 반복
```

### 12.5 Evaluation
```bash
# 각 run × eval checkpoint 별로:
python scripts/eval.py \
    --ckpt  exps/b0_s42/checkpoint-20000.pt \
    --out   results/b0_s42_20000.json \
    --n_samples 10000

# 집계:
python scripts/aggregate.py --in results/ --out results.csv
```

### 12.6 계수 스케줄 (reference 구현)
```python
import math

def schedule_off(t, T):
    return 0.0

def schedule_fixed(t, T, max_coeff=0.5):
    return max_coeff

def schedule_cosine_decay(t, T, max_coeff=0.5):
    # t=0 에서 0.5, t=T 에서 0
    return max_coeff * 0.5 * (1 + math.cos(math.pi * t / T))

def schedule_cosine_warmup(t, T, max_coeff=0.5):
    # t=0 에서 0, t=T 에서 0.5
    return max_coeff * 0.5 * (1 - math.cos(math.pi * t / T))

def schedule_hard_cutoff(t, T, max_coeff=0.5, cutoff_frac=0.5):
    return max_coeff if t < T * cutoff_frac else 0.0
```

### 12.7 Sharpness metric (reference 구현)
```python
import numpy as np
import cv2
from scipy.stats import wasserstein_distance

def laplacian_variance(img_gray_uint8):
    return cv2.Laplacian(img_gray_uint8, cv2.CV_64F).var()

def high_freq_ratio(img_gray_uint8, cutoff_frac=0.5):
    f = np.fft.fft2(img_gray_uint8.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r_cut = int(cutoff_frac * min(cy, cx))
    mask = np.ones_like(mag, dtype=bool)
    y, x = np.ogrid[:h, :w]
    mask[(y - cy)**2 + (x - cx)**2 < r_cut**2] = False
    return mag[mask].sum() / (mag.sum() + 1e-8)

def sobel_mean(img_gray_uint8):
    gx = cv2.Sobel(img_gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2).mean()

def sharpness_wasserstein(gen_imgs_uint8, ref_imgs_uint8):
    stats_fns = [laplacian_variance, high_freq_ratio, sobel_mean]
    out = {}
    for name, fn in zip(["lapvar", "hffreq", "sobel"], stats_fns):
        gen = np.array([fn(to_gray(x)) for x in gen_imgs_uint8])
        ref = np.array([fn(to_gray(x)) for x in ref_imgs_uint8])
        out[name] = wasserstein_distance(gen, ref)
    return out
```

---

## 13. Scope 제외 항목

다음 항목은 본 1개월 연구에서 의도적으로 제외된다. Future work로 등장할 수 있음:

- **iREPA (Conv2d projection)**: 별도의 아키텍처 질문. 계수 스케줄 trade-off 조사의 일부가 아님.
- **REPA-E / INVAE**: 별도의 VAE 측 질문. Baseline을 순수하게 유지하기 위해 의도적으로 제외.
- **Teacher encoder ablation (DINO, MAE, CLIP)**: Scope 밖. DINOv2가 REPA 논문에 따라 teacher로 고정됨.
- **Model scale study (SiT-L/XL)**: Compute 이유로 scope 밖. SiT-B/2 만.
- **ImageNet 비교**: Compute 이유로 scope 밖. FFHQ 만, 명시적 domain-deviation 정당화와 함께.
- **Text conditioning / CFG**: Scope 밖. Unconditional 만.
- **Video DiT 확장**: 별도의 연구 방향, future work.

---

## 14. 성공 기준 (Claim 결과와 독립)

본 1개월 연구는 다음 조건을 충족하면 성공으로 간주한다:

1. 15개의 main run 모두가 4개의 eval checkpoint 모두와 함께 완료됨.
2. 모든 자동 metric이 계산되고 저장됨.
3. Human eval이 ≥1,000 응답 수집 (최소 self + pilot rater).
4. Interpolation study가 모든 5 branches에 대해 완료됨.
5. Pre-registration 판정이 §2.2에 따라 수정 없이 적용됨.
6. Positive/null/mixed 결과와 무관하게 Day 28에 논문 초안 존재.

여섯 조건 모두 충족은 과학적 finding과 무관하게 연구 과정이 엄밀했음을 의미한다. Publishable 이든 아니든, 이 사이클을 완주한 경험 자체가 부차 목표였던 연구 방법론 훈련이다.

---

*설계 문서 끝. 이 문서는 self-contained하며, 본 연구의 유일한 specification으로 새 저장소에 가지고 갈 수 있도록 의도되었다.*
