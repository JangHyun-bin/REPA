# REPA Preliminary Baseline Comparison — Spec

**작성일**: 2026-04-15
**상태**: 설계 완료, 구현 대기
**목적**: Full 5-branch study 착수 전 가설 premise de-risk
**Parent spec**: `spec-ko.md` (full 5-branch study)
**기간**: 2-3 일 (약 36 GPU-hours)
**코드베이스**: `sihyun-yu/REPA` fork, branch `prelim-baseline` (git worktree at `/home/famoz/projects/dl/REPA-prelim/`)

---

## 0. Executive Summary

본 문서는 full 5-branch REPA 재평가 연구 (`spec-ko.md`) 착수 전 실행할 **minimal preliminary comparison** 을 설계합니다. 목적은 단 하나: full study 52 GPU-hours + 1 개월 일정을 투입하기 *전에* 연구 가설의 두 가지 premise 가 우리 setup 에서 실제로 관측되는지 확인해서 연구 자체를 de-risk.

**두 가지 premise**:
1. **Q1 — 재현 가능성**: 우리 코드베이스가 REPA 논문의 핵심 positive 결과 (수렴 속도 개선) 를 FFHQ-256 setup 에서 재현하는가?
2. **Q2 — Trade-off 존재**: 그와 *동시에* REPA 가 실제로 sharpness/diversity 를 저하시키는 signature 가 측정 가능한가?

둘 다 관측되어야 full study 가 과학적으로 의미 있습니다. 어느 하나라도 관측 안 되면 full study 를 *먼저* 재고해야 합니다.

본 preliminary 는 2-branch (B0 pure SiT, B1 SiT+REPA-fixed) × 1 seed × 400K steps 로, full study 와 독립된 git branch `prelim-baseline` 에서 격리되어 실행됩니다.

---

## 1. Research Questions

### 1.1 Q1 — 재현 가능성
> **우리 코드베이스 + FFHQ-256 + SD-VAE latent setup 에서 SiT+REPA-fixed (B1) 가 pure SiT (B0) 대비 더 빠른 FID 수렴을 보이는가?**

이는 REPA 논문 (Yu et al., ICLR 2025) 의 **main positive claim** 이며, 우리의 hook 주입·dataset loader 교체·null class token 변경 등의 코드 수정이 REPA 의 작동을 깨뜨리지 않았음을 확인하는 *sanity check* 역할을 겸합니다.

### 1.2 Q2 — Trade-off 존재
> **동일 run 에서 SiT+REPA-fixed (B1) 의 sample 이 pure SiT (B0) 대비 측정 가능한 sharpness/diversity 저하를 보이는가?**

이는 full study 의 핵심 diagnostic claim. 본 preliminary 는 이 signature 가 *아예 존재하는지* 만 확인 — 정량적 trade-off 특성화는 full study 에서.

### 1.3 왜 두 질문을 *동시에* 묻는가

Q1 ✓ Q2 ✗ 이면 REPA 는 작동하지만 우리가 찾던 trade-off 는 *없음* → full study 의 diagnostic claim 이 근거를 잃음.
Q1 ✗ Q2 — 이면 우리 코드베이스 설정에 문제 → debug 필요.
Q1 ✓ Q2 ✓ 면 가설 premise 성립 → full study 진행.

한 실험으로 두 질문에 동시 답함으로써 compute 효율 극대화.

---

## 2. Branches + Training Setup

### 2.1 Branch 정의 (단 2 개)

| Branch | λ(t) | 설명 |
|---|---|---|
| **B0 — pure SiT** | 0 (모든 t) | REPA projection loss 완전 제거. 순수 v-prediction flow matching on SD-VAE latents. |
| **B1 — SiT+REPA-fixed** | 0.5 (모든 t) | REPA 논문 default. 중간 DiT layer 를 frozen DINOv2 ViT-B patch token 에 negative cosine similarity 로 정렬. |

Full study 의 B2 (cosine decay), B3 (cosine warmup), B4 (hard cutoff) 는 preliminary 에 **포함되지 않음**. 이 두 극단만으로 premise 검증에 충분.

### 2.2 공통 하이퍼파라미터

본 preliminary 는 `spec-ko.md` §5 의 하이퍼파라미터와 **정확히 일치**하는 것이 원칙. 예외는 (a) total steps, (b) seed 수, (c) eval cadence.

| Hyperparameter | 값 | 출처 |
|---|---|---|
| Backbone | SiT-B/2 (130M, patch=2, depth=12, hidden=768, heads=12) | spec §5.1 |
| Input | SD-VAE latent 4×32×32, 256 tokens | spec §5.1 |
| Teacher (B1 only) | DINOv2 ViT-B frozen, patch tokens (CLS 제외) | spec §5.2 |
| Projection layer (B1 only) | Single MLP block 8 hidden → 768 dim | spec §5.1 |
| Batch size | 32 | spec §5.5 |
| Precision | bf16 mixed (accelerate) | spec §5.5 |
| Optimizer | AdamW lr=1e-4, betas=(0.9, 0.999), wd=0 | spec §5.3 |
| Gradient clip | max-norm 1.0 | spec §5.3 |
| LR schedule | None (constant) | spec §5.3 |
| EMA decay | 0.9999 | spec §5.3 |
| Augmentation | horizontal flip only | spec §5.5 |
| Loss objective | v-prediction flow matching + λ·proj (B1 only) | spec §5.4 |
| **Total steps** | **400,000** | preliminary-only |
| **Seeds** | **42 only** (n=1 per branch) | preliminary-only |

> **400K 근거**: REPA 논문의 training regime (ImageNet 400K × batch 256) 과 effective FLOPs 가 유사. 또한 "수렴 후" 의 late-training behavior 에서 trade-off 가 나타날 가능성이 높아 충분한 training time 확보가 중요.

> **n=1 근거**: Preliminary 는 statistical claim 을 만들지 않음. "signal 이 *아예* 보이는지" 만 확인. Full study 에서 n=3 + 통계 검정.

### 2.3 Dataset

- **Source**: `merkol/ffhq-256`, HF revision `f23c0e21f6fe222b4429272a86be00e4583b0298`
- **Verified**: reeval branch sanity check 통과 (PSNR 29.58 dB, LPIPS 0.030, threshold {PSNR ≥ 28, LPIPS ≤ 0.10})
- **Total**: 70,000 samples, 256×256 PNG
- **Split**: 65K train + 5K eval (reeval 의 split 재사용 또는 동일 seed 20260415 로 재생성)
- **Preprocessing**: SD-VAE-ft-mse latent 사전 계산 (bf16, 4×32×32 per image)
- **License**: FFHQ 원본 상속 (CC BY-NC-SA 4.0)

Dataset manifest 와 SD-VAE sanity 결과는 reeval branch 에서 그대로 복사 (§7.2 참조).

---

## 3. Evaluation Protocol

### 3.1 Checkpoints (8 eval 포인트)

50K step 간격으로 training trajectory 전체를 촘촘히 관측:

**{50K, 100K, 150K, 200K, 250K, 300K, 350K, 400K}**

Full study 는 4 포인트 (20K, 50K, 80K, 100K) 이지만, preliminary 는 trajectory 전체 관측이 핵심이므로 더 촘촘히. 특히 200K 이후의 수렴 후 행동이 Q2 trade-off 탐지에 결정적.

### 3.2 각 checkpoint 에서 측정

| Metric | 구현 | 목적 |
|---|---|---|
| **FID** | `clean-fid` vs precomputed eval reference stats | Q1 수렴 속도 비교 |
| **Sharpness Wasserstein (3 stats)** | 자체: Laplacian variance, HF energy ratio (50% Nyquist cutoff), Sobel mean | Q2 trade-off signature, spec §7.2 와 동일 |
| **Preview samples** | 16-32 장 PNG 저장 (매칭된 noise seed) | Q2 qualitative visual inspection |
| **Precision / Recall** | Kynkäänniemi et al. 2019, VGG-16 feature k-NN | Q2 diversity 축 보조 신호 |

### 3.3 Sampling 설정

- **Samples per evaluation**: **5,000** (full study 의 10K 보다 적지만 preliminary 에서 충분)
- **Sampler**: 50-step DDPM (`samplers.py` 의 upstream REPA 구현 사용)
- **CFG**: disabled (unconditional)
- **Noise seed**: 동일 (B0 과 B1 에서 같은 초기 noise → within-pair variance 제거)
- **EMA**: ON (0.9999 decay)

### 3.4 생략 항목 (preliminary 에 불필요)

- **Inception Score**: FFHQ 단일 도메인에서 정보량 거의 0
- **FD-DINOv2**: full study 에서만 (preliminary 의 Q1/Q2 결정에 불필요)
- **Human eval (2AFC)**: full study 에서만
- **Interpolation jaggedness**: full study 에서만

---

## 4. Success Criteria + Decision Rules

### 4.1 결과 시나리오 × 행동 매트릭스

네 가지 결과를 명시적으로 구분하고 각각의 다음 행동 정의:

| 결과 | Q1 (재현) | Q2 (trade-off) | 해석 | 다음 행동 |
|---|---|---|---|---|
| **Positive** | ✓ | ✓ | 가설 premise 둘 다 성립. Full study 가치 있음. | **Full study 진행** (reeval branch 복귀, Day 3 이후 tutorial 계속) |
| **Codebase OK, 가설 반박** | ✓ | ✗ | REPA 는 작동하지만 우리가 찾던 trade-off 가 이 regime 에선 없음. 주관적 관찰은 confounder 탓일 가능성. | **Full study 재검토**. (a) null-finding paper 로 direction 변경, (b) 다른 regime (더 큰 model, 다른 dataset) 탐색, (c) 연구 중단. |
| **Codebase 문제** | ✗ | (무관) | B1 FID 가 B0 FID 보다 높거나 같음 → 우리 REPA hook 에 bug, 또는 FFHQ 에서 REPA 효과 약함. | **Debug 필수**. (a) equivalence test (`schedule_off ≡ REPA-disabled`) 재실행, (b) projection loss 의 gradient flow 확인, (c) REPA 원본 hyperparameter 대조. Full study 착수 금지. |
| **양쪽 모두 null** | ✗ | ✗ | 학습이 미수렴이거나 우리 FFHQ-SD-VAE setup 이 SiT-B/2 에 부적합. | **Hyperparameter 재검토**. batch, lr, step budget 조정. Full study 중단 또는 장기 연기. |

### 4.2 정량 기준

Decision rule 이 ambiguous 하지 않도록 수치 threshold 명시:

**Q1 pass 조건**:
- **200K step 이후** 2개 이상 checkpoint 에서 B1 FID 가 B0 FID 대비 **≥5% 낮음**
- 즉 `(FID_B0 - FID_B1) / FID_B0 ≥ 0.05`
- 200K 이후만 보는 이유: 초기 100K 는 warmup/초기 수렴 phase, trade-off 와 재현 signal 은 수렴 후에 안정화

**Q2 pass 조건**:
- **200K step 이후** 2개 이상 checkpoint 에서
- 3 개 sharpness Wasserstein statistic (lapvar / hffreq / sobel) 중 **최소 2 개**가
- B0 sharpness W1 가 B1 sharpness W1 대비 **≥15% 낮음** (즉 B0 가 real dist 에 더 가까움)
- 즉 `(W1_B1 - W1_B0) / W1_B1 ≥ 0.15` for 2+ statistics

두 조건은 독립적으로 평가되어 4.1 의 시나리오를 결정.

### 4.3 수동 시각 검사 (보조)

자동 metric 에 의존하지 않고, 200K / 400K 의 preview 32 장을 사람 눈으로 비교:
- B1 이 B0 보다 흐릿해 보이는가?
- B1 의 얼굴이 더 "generic" / 덜 다양해 보이는가?

자동 metric 과 시각 검사가 **일치**하면 strong signal, **불일치**하면 metric 해석 재검토 필요.

### 4.4 결정 문서화

Preliminary 종료 시 `results_prelim.md` 파일에 다음 기록:
1. 각 checkpoint 의 FID, sharpness W1 (3 stat), precision, recall 수치표
2. Q1, Q2 pass/fail 판정
3. 적용된 시나리오 (4.1 표 중 어느 row)
4. 다음 행동 결정
5. 결정 일자 + 분석자

이 문서가 full study 착수 (혹은 중단) 의 공식 기록.

---

## 5. Compute Budget + Timeline

| Phase | 예상 wallclock | 비고 |
|---|---|---|
| Worktree 생성 + 환경 setup | 30 min | `git worktree add`, dl conda env 재사용 |
| 유용 스크립트 cherry-pick | 30 min | reeval 에서 disk_monitor, sanity, MANIFEST 복사 |
| 최소 코드 수정 | 2-4 h | dataset loader, `--repa_on/off` flag, checkpoint policy |
| Latent precompute (70K) | ~30 min | reeval 의 precompute_latents 재사용 |
| FID reference stats | ~10 min | reeval 의 precompute_fid_ref 재사용 |
| **B0 학습 (400K steps)** | **~14 h** | batch=32, 4080, bf16 (가정, wallclock pilot 전) |
| **B1 학습 (400K steps)** | **~14 h** | B0 완료 후 순차 실행 |
| Eval (16 checkpoints × ~15 min) | ~4 h | 학습과 병행 가능하면 wallclock 절감 |
| 분석 + 결정 | 2 h | plot 생성, 4.1 매트릭스 적용 |
| **합계** | **~36 h** | ≈ **1.5-2 일 연속 GPU** |

Buffer (debug, 재시작) 포함 **2-3 일** 예상. Full study budget 52h 대비 약 60%.

### 5.1 Wallclock 가정의 불확실성

현 시점 (Day 2 끝) 까지 **실제 wallclock 측정은 없음**. 14h/run 은 spec §9 의 외삽에 기반한 추정. Preliminary 의 첫 번째 run (B0) 의 처음 5K step 에서 실제 속도 측정 → 400K 외삽 → 만약 예상보다 현저히 느리면 (**> 20h/run**) 중단 후 step budget 조정 고려.

---

## 6. File Structure + Data Sharing Strategy

### 6.1 REPA-prelim worktree 내부 구조

```
/home/famoz/projects/dl/REPA-prelim/
├── .git                            # 파일 (pointer → isREPAgood/REPA/.git/worktrees/REPA-prelim)
├── train.py                        # upstream, 수정됨 (dataset + repa toggle)
├── models/                         # upstream SiT, 변경 없음
├── loss.py                         # upstream REPA loss, 변경 없음
├── samplers.py                     # upstream, 변경 없음
├── utils.py                        # upstream
├── dataset.py                      # 수정됨: FFHQ latent loader
├── requirements-prelim.txt         # NEW: dl env 에 추가할 의존성 (실상 reeval 것과 동일)
├── scripts/
│   ├── disk_monitor.py             # cherry-pick from reeval
│   ├── sd_vae_sanity.py            # cherry-pick from reeval
│   ├── precompute_latents.py       # NEW (or cherry-pick 후 완성)
│   ├── precompute_fid_ref.py       # NEW
│   ├── eval_prelim.py              # NEW: FID + sharpness + P/R + preview
│   └── plot_prelim.py              # NEW: FID vs step, sharpness bars
├── data/
│   ├── MANIFEST.json               # copy from reeval
│   ├── sd_vae_recon_sanity.json    # copy from reeval
│   ├── ffhq256_train.txt           # NEW: 65K indices (seed 20260415)
│   ├── ffhq256_eval.txt            # NEW: 5K indices
│   ├── ffhq256_latents/train/      # NEW: bf16 latents (gitignored)
│   └── fid_ref_ffhq256.npz         # NEW
├── exps/
│   ├── b0_s42/                     # pure SiT
│   │   ├── checkpoints/ema_*.pt    # 8 ckpts (50K 간격, EMA only bf16)
│   │   ├── loss_log.csv
│   │   └── eval_*/
│   │       ├── metrics.json
│   │       └── preview/*.png
│   └── b1_s42/                     # SiT+REPA (동일 구조)
├── results_prelim.csv              # 집계된 모든 metric
├── results_prelim.md               # §4.4 결정 문서
└── figures/                        # FID-vs-step plot, sharpness bars, preview grid
```

### 6.2 Data 공유 전략

`data/ffhq256_latents/train/` (70K × bf16, ~600 MB) 는 prelim 과 full study 에서 **완전히 동일**해야 baseline 일치가 보장됩니다. 전략:

**옵션 A (권장): prelim 에서 먼저 precompute, reeval 이 나중에 symlink 재사용**

1. Prelim worktree 에서 `data/ffhq256_latents/train/` 생성
2. Latent precompute 완료 후, reeval worktree 에서:
   ```bash
   cd /home/famoz/projects/dl/isREPAgood/REPA
   mkdir -p data
   ln -s /home/famoz/projects/dl/REPA-prelim/data/ffhq256_latents data/ffhq256_latents
   ```
3. 디스크 중복 0, 두 worktree 에서 동일 latent 파일 참조
4. Preliminary 결과와 무관하게 latents 는 live (reeval 이 필요 시 사용)

**Manifest + reference stats 공유**: `data/MANIFEST.json`, `data/fid_ref_ffhq256.npz`, `data/ffhq256_train.txt`, `data/ffhq256_eval.txt` 는 **작으므로 복사** (branch 별로 독립 관리). Symlink 면 branch 간 변경이 꼬일 수 있음.

---

## 7. Workflow — Worktree 생성부터 결정까지

### 7.1 Worktree 생성 (현재 위치: `isREPAgood/REPA`, branch = reeval)

```bash
cd /home/famoz/projects/dl/isREPAgood/REPA
git worktree add /home/famoz/projects/dl/REPA-prelim 67f7145
cd /home/famoz/projects/dl/REPA-prelim
git checkout -b prelim-baseline
```

### 7.2 Reeval 에서 유용 파일 cherry-pick (같은 .git 공유이므로 show 로 바로 복사)

```bash
# disk monitor + sanity script + requirements
git show reeval:scripts/disk_monitor.py > scripts/disk_monitor.py
git show reeval:scripts/sd_vae_sanity.py > scripts/sd_vae_sanity.py
git show reeval:requirements-reeval.txt > requirements-prelim.txt
chmod +x scripts/disk_monitor.py scripts/sd_vae_sanity.py

# dataset manifest + sanity result (이미 PASSED)
mkdir -p data
git show reeval:data/MANIFEST.json > data/MANIFEST.json
git show reeval:data/sd_vae_recon_sanity.json > data/sd_vae_recon_sanity.json
git show reeval:data/disk_baseline.txt > data/disk_baseline.txt

# gitignore entries
cat >> .gitignore << 'EOF'

# === prelim workflow ===
data/ffhq256_latents/
exps/
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
EOF

git add scripts/disk_monitor.py scripts/sd_vae_sanity.py requirements-prelim.txt \
        data/MANIFEST.json data/sd_vae_recon_sanity.json data/disk_baseline.txt .gitignore
git commit -m "prelim: import disk monitor, sanity script, dataset manifest from reeval"
```

### 7.3 최소 코드 변경 (4 가지)

1. **Dataset loader 교체**: `dataset.py` 를 FFHQ latent loader 로 교체 (or 새 모듈 추가 + train.py 가 선택)
2. **Class conditioning 제거**: `train.py` 에서 class label 을 항상 0 (null) 으로 전달
3. **REPA toggle**: `--repa_on` / `--repa_off` CLI flag 추가 (5 schedule 아님, 단순 on/off)
4. **Checkpoint policy**: 50K 간격, EMA only, bf16 저장

각 변경마다 별도 commit. 변경은 `prelim-baseline` branch 에서만, reeval 에 영향 없음.

### 7.4 Precompute (latent + FID ref)

```bash
# 이미 sanity PASSED 므로 sanity 재실행 불필요
python scripts/precompute_latents.py --hf_dataset merkol/ffhq-256 \
    --hf_split train --out data/ffhq256_latents/train
python scripts/precompute_fid_ref.py
```

Train/eval split 고정:
```bash
python -c "
import random, hashlib, json
random.seed(20260415)
indices = [f'{i:06d}' for i in range(70000)]
random.shuffle(indices)
open('data/ffhq256_train.txt','w').write('\n'.join(sorted(indices[:65000])) + '\n')
open('data/ffhq256_eval.txt','w').write('\n'.join(sorted(indices[65000:])) + '\n')
"
```

### 7.5 학습 실행 (tmux 권장, 연속 ~28h)

```bash
tmux new -s prelim
python scripts/disk_monitor.py --min_gb 15 --interval 300 --exit_on_alert &

# B0 (pure SiT)
python train.py --repa_off --seed 42 --total_steps 400000 --out exps/b0_s42

# B0 끝나고 B1 (SiT+REPA-fixed)
python train.py --repa_on --seed 42 --total_steps 400000 --out exps/b1_s42
```

### 7.6 Eval + 분석

```bash
# 각 run × 8 checkpoint 에 대해
for B in b0 b1; do
    for STEP in 050000 100000 150000 200000 250000 300000 350000 400000; do
        python scripts/eval_prelim.py \
            --ckpt exps/${B}_s42/checkpoints/ema_${STEP}.pt \
            --out exps/${B}_s42/eval_${STEP}/metrics.json
    done
done

python scripts/aggregate_prelim.py   # → results_prelim.csv
python scripts/plot_prelim.py        # → figures/*.png
```

### 7.7 결정 + 문서화

`results_prelim.md` 작성 (§4.4 에 정의된 5 항목) → commit → push.

결정에 따라:
- **Positive** → reeval worktree 로 복귀 (`cd /home/famoz/projects/dl/isREPAgood/REPA`), Day 3 이후 `tutorial-ko.md` 계속 진행
- **다른 결과** → 4.1 매트릭스 의 해당 행동 수행

---

## 8. Scope — 의도적으로 *제외* 한 것 (YAGNI)

Preliminary 는 다음 항목을 **포함하지 않음**. Full study 가 진행되면 그 때 포함:

- Pre-registration (`prereg-v1` git tag) — preliminary 는 사전 등록 없이 진행, full study 착수 시 등록
- Power analysis (§2.3 4-way reporting) — n=1 이라 의미 없음
- Multi-seed, paired t-test, Bonferroni 보정 — n=1
- 5-branch factorial (decay, warmup, cutoff) — 2-branch 만
- Human evaluation 2AFC + Bradley-Terry
- Interpolation study + jaggedness
- FD-DINOv2 metric
- Confirmation-bias ranking check
- Underpowered null vs true null 구분

Preliminary 는 설계상 *간단*해야 하며, 이 모든 complexity 는 *preliminary 결과가 긍정적인 경우* full study 에서만 추가됩니다.

---

## 9. Full Study 와의 관계

### 9.1 Code reuse

Preliminary 에서 작성한 것 중 full study 에서 재사용되는 항목:
- Dataset loader (`FFHQ256LatentDataset`) — 약간의 확장으로 full study 에 사용 가능
- Checkpoint save helper (EMA bf16, step 간격 지정)
- Sharpness Wasserstein metric 구현
- FID / PR / eval pipeline skeleton
- `precompute_latents.py`, `precompute_fid_ref.py` (이미 reeval 에서 사용 예정)

### 9.2 Branch 관계

```
67f7145 Update README.md ← main (upstream mirror)
    │                      ← prelim-baseline (preliminary 커밋 쌓일 곳)
    │
    └─► be3edb2 scaffold
            │
            └─► e63bc33 env+ops
                     │
                     └─► 62627d1 sanity script
                              │
                              └─► 1ec794f mirror choice ← reeval (full study 진행 대기)
```

두 branch 는 **독립 진화**. Preliminary 가 positive 면 reeval branch 로 복귀, decision 은 full study 설계에 반영되지만 preliminary 의 commit 들은 reeval 에 merge 하지 않음 (commit 스타일과 의도가 다름 — preliminary 는 "minimal sanity", reeval 은 "full infrastructure").

### 9.3 Data reuse

§6.2 의 symlink 전략으로 `data/ffhq256_latents/` 는 두 branch 가 공유. Full study 진행 시 재 precompute 불필요.

### 9.4 Preliminary 종료 후 worktree 처리

- **Positive → full study 진행**: worktree 유지 (나중에 필요 시 참조). `git worktree list` 에 계속 표시됨.
- **Negative → full study 중단**: worktree 유지 (결과 보존), 선택적으로 나중에 `git worktree remove` 로 정리.
- **Debug 필요**: worktree 에서 debug 작업 계속.

---

## 10. Deliverables

Preliminary 종료 시 `REPA-prelim` worktree 에 존재해야 할 것:

### 코드
- 수정된 `train.py` (dataset + repa toggle)
- 새 `dataset.py` (FFHQ latent loader)
- `scripts/eval_prelim.py`, `scripts/aggregate_prelim.py`, `scripts/plot_prelim.py`

### 데이터 (git tracked, 작음)
- `data/MANIFEST.json`, `data/ffhq256_train.txt`, `data/ffhq256_eval.txt`
- `data/fid_ref_ffhq256.npz`, `data/sd_vae_recon_sanity.json`

### 결과 (git tracked)
- `results_prelim.csv` — 16 rows (2 runs × 8 checkpoints) × 모든 metric
- `results_prelim.md` — §4.4 결정 문서
- `figures/` — FID-vs-step, sharpness bars, preview grid

### 학습 산출물 (gitignored)
- `exps/b0_s42/`, `exps/b1_s42/` — checkpoints + logs + previews

---

## 11. 성공 기준 (결정과 독립)

본 preliminary 는 다음 조건을 충족하면 성공으로 간주 (결과의 positive/negative 와 무관):

1. 두 run (B0, B1) 모두 400K step 완주
2. 모든 8 checkpoint eval 완료
3. `results_prelim.csv` 가 16 row × 모든 metric 채워짐
4. `results_prelim.md` 에 §4.4 규칙이 *수정 없이* 적용됨
5. 다음 행동이 명시적으로 결정되고 문서화됨
6. Full study 로 가든 중단하든, 결정 근거가 재현 가능하도록 기록됨

이 여섯 조건 모두 충족 시, preliminary 는 "과학적 의사결정 과정" 으로서 성공. 결정이 어떤 방향이든 full study 의 다음 phase 에 정당화를 제공.

---

*Spec 끝. 본 문서는 `spec-ko.md` (full study) 의 prerequisite 로 작성되었으며, preliminary 실행 중 *수정되지 않음*. 실행 중 발견된 이슈는 별도 commit 으로 `results_prelim.md` 의 "known issues" 섹션에 append.*
