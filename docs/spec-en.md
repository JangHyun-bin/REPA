# Re-evaluating REPA in Image Generation: A Diagnostic Study on FFHQ-256

**Date**: 2026-04-15
**Status**: Design — approved, ready for implementation
**Target venue**: Workshop submission (NeurIPS / CVPR generative workshop)
**Timeline**: 1 month from start
**Codebase**: Fresh repo, forked from `sihyun-yu/REPA`

---

## 0. Executive Summary

This study questions whether fixed-coefficient REPA (Representation Alignment; Yu et al., ICLR 2025) actually delivers its advertised benefits in image generation, motivated by the subjective observation that REPA-trained samples appear blurrier and less diverse than samples from comparable DiT/DDPM baselines. We run a controlled factorial study across five coefficient schedules — off, fixed, cosine decay, cosine warmup, and HASTE-style hard cutoff — on FFHQ-256, and evaluate with a full suite of automatic metrics (FID, IS, Precision/Recall, FD-DINOv2, sharpness distribution distance), pairwise human preference, and an interpolation-based diversity probe.

The central claim is diagnostic: **fixed-coefficient REPA exhibits a previously uncharacterized trade-off between convergence speed and sample sharpness/diversity, and we characterize this trade-off across schedules while providing a diagnostic protocol for detecting it.**

The study is deliberately conducted with a minimal, verifiable codebase. We discard all non-REPA extras present in prior preliminary experiments (SPRINT sparse path, contrastive flow matching, cls-token dual-stream, path-drop regularization, INVAE) and build on the original REPA reference implementation unchanged. This ensures that any observed effect can be attributed to REPA coefficient dynamics and not to confounding training tricks.

---

## 1. Research Question & Motivation

### 1.1 Personal observation motivating the study
In informal experiments comparing U-Net DDPM, DiT, and SiT+REPA architectures, SiT+REPA samples appeared subjectively blurrier than samples from the simpler DDPM/DiT baselines at comparable compute. This contradicts the expectation created by the REPA paper, which reports both faster convergence and improved FID over non-REPA baselines on ImageNet.

Three explanations are possible:
1. The observation is a subjective bias (no real difference exists).
2. The observation reflects a hidden trade-off: REPA improves FID while degrading a different quality axis (sharpness, diversity) that the original paper did not measure.
3. The observation is due to confounders in prior experiments (SpeedrunDiT extras, INVAE, etc.) and vanishes under a clean REPA baseline.

The purpose of this study is to discriminate between these three explanations with measured evidence.

### 1.2 Formal research question
> **RQ**: Does fixed-coefficient REPA, under a minimal and verifiable codebase, exhibit a statistically detectable trade-off between convergence speed and sample sharpness/diversity on FFHQ-256? And if so, can coefficient scheduling (decay, warmup, hard cutoff) mitigate this trade-off?

---

## 2. Central Claim (Pre-registered)

### 2.1 Claim B — Diagnostic
> **Fixed-coefficient REPA exhibits a previously uncharacterized trade-off between convergence speed and sample sharpness/diversity on FFHQ-256. We characterize this trade-off across four coefficient schedules (off, fixed, decay, warmup, hard cutoff) and provide a diagnostic protocol for detecting it using distribution-level sharpness metrics and paired human preference.**

### 2.2 Pre-registration conditions
Before running main experiments, `docs/prereg.md` must be committed to the new repo with the following judgment rules. After commit, these conditions are **frozen**: no post-hoc modification.

**Positive result (diagnostic claim holds)**:
- On the **primary metric (FID-vs-step trajectory AUC, see §7.1)** OR on at least one sharpness/diversity metric, REPA-Fixed shows statistically significant (paired t-test, Bonferroni-corrected α=0.05/6) degradation compared to REPA-Off, **AND**
- At least one scheduled variant (decay, warmup, hard cutoff) partially recovers the degraded axis while preserving fixed REPA's convergence-speed advantage (FID-vs-step curve).

**Null result**:
- No statistically significant difference on sharpness/diversity metrics between any branches at final checkpoint. The study is published as a null report with full methodology.

**Mixed result**:
- Automatic metrics show difference but human preference does not align, or vice versa. Reported as a dissociation finding.

All three outcomes are publishable. The study is designed so that any result produces a useful contribution.

### 2.3 Pre-specified statistical power (part of pre-registration)

The study uses n=3 seeds per branch with a paired t-test. Under Bonferroni correction (α=0.05/6 ≈ 0.0083), the minimum detectable effect size is very large — for a two-sided paired t-test with n=3 and power=0.80, this corresponds to Cohen's d ≈ 4.5. **The pre-registered hypothesis test therefore measures the *extremity* of an effect, not its *existence***. This is a known and pre-registered limitation.

The underpowered configuration is compensated by:

1. **Reporting every result in four parallel forms**:
   - **Descriptive statistics**: per-branch mean ± std across n=3 seeds.
   - **Directional consistency**: how many seeds point in the same inequality direction (e.g., 3/3, 2/3).
   - **Effect size**: paired Cohen's d point estimate with 1000-iter bootstrap 95% CI.
   - **Pre-registered significance test**: Bonferroni-corrected paired t-test p-value.
2. **No single-test conclusions**: no claim is supported or refuted by a single metric's significance test alone. Strong claims require all four forms to agree; mixed evidence is reported as a weak claim or as needing further analysis.
3. **"Underpowered null" vs "true null"**: if the pre-registered test does not reach significance but the effect size satisfies |d| > 0.8 (large), the result is explicitly labeled "underpowered null" (effect may genuinely exist) and not interpreted as "no difference". Only |d| < 0.2 (small) is interpreted as a true null.
4. **The power limitation itself is part of the pre-registration**: any later replication with more seeds will be reported separately as "exploratory replication", not retroactively merged into the main study.

---

## 3. Baseline & Purity Principle

### 3.1 Purity principle
Any component not present in the original REPA paper (Yu et al., *Representation Alignment for Generation*, ICLR 2025) is **excluded** from the baseline. The experimental system consists of exactly and only:
- SiT backbone (SiT-B/2, see §5)
- Original REPA projection loss (MLP projector aligning one intermediate DiT layer with DINOv2 ViT-B frozen features via negative cosine similarity)
- Standard v-prediction flow matching objective
- Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`, 8× spatial compression, 4 latent channels)

### 3.2 Explicitly excluded components
The following components from prior preliminary experiments are **removed** and must not appear in the new codebase:
- **SPRINT sparse path** (token dropping + sparse forward + mask padding + dense-sparse fusion)
- **Contrastive Flow Matching (CFM) auxiliary loss** (source unknown, not in REPA paper)
- **Class-token dual-stream training** (separate noise/target for CLS token)
- **Path-drop regularization** (stochastic path drop in training)
- **adaLN-Gaussian initialization** (revert to zero initialization as in original SiT/REPA)
- **INVAE** (use SD-VAE instead — the REPA paper's actual VAE choice)

### 3.3 Codebase setup
The implementation starts from the original REPA repository:
```bash
git clone https://github.com/sihyun-yu/REPA.git repa-reeval
cd repa-reeval
```

Minimal modifications to this codebase:
1. Replace the dataset loader to read FFHQ-256 (and its pre-computed SD-VAE latents).
2. Remove class-conditional label handling (FFHQ is unconditional); replace with a fixed null class token.
3. Add a coefficient scheduler hook to the training loop, supporting off / fixed / cosine decay / cosine warmup / hard cutoff modes.
4. Add an evaluation harness script that reproduces our metric suite.
5. Add a pre-registration markdown document (`docs/prereg.md`).

Every commit to the new repo must be tagged. The pre-registration commit is `prereg-v1` and must precede all main training commits. This is enforceable via git tags.

---

## 4. Dataset: FFHQ-256

### 4.1 Choice rationale
- **Compute feasibility**: 70K images at 256×256 is trainable on a single consumer GPU within the 1-month budget. ImageNet-1K is not.
- **Domain appropriateness for the research question**: Faces are a domain where human perception is extremely sensitive to sharpness and identity-level diversity. "Blur" is more diagnostic on faces than on ImageNet's wide-category mixture.
- **Unconditional benchmark**: FFHQ is the standard unconditional generation benchmark, removing confounders from class conditioning.
- **Standard FID reference**: FFHQ-256 FID reference statistics are standardized and widely reported in prior work.

### 4.2 Domain deviation from original REPA
The original REPA paper evaluates on ImageNet-1K (class-conditional). Using FFHQ-256 (unconditional, single domain) is a deliberate deviation. This must be stated explicitly in the paper's methods section:

> "We evaluate on FFHQ-256 rather than ImageNet-1K due to compute constraints. FFHQ, being a high-resolution face dataset, provides a domain where perceptual sharpness and identity-level diversity are particularly salient — making it well-suited for our research question about REPA's quality trade-offs."

This positioning frames the deviation as a feature of the study design rather than a limitation.

### 4.3 Data preparation (streaming-first workflow)

Local disk is constrained (< 90 GB free); we therefore **never permanently store raw FFHQ-256 images**. One of the two options below is used:

**Option A (Recommended): HuggingFace Hub streaming**
1. Iterate once through a stable HuggingFace Hub mirror of FFHQ-256 via `datasets.load_dataset(<id>, split=..., streaming=True)`. The exact mirror id and revision hash are recorded in `data/MANIFEST.json`.
2. For each image, immediately (a) verify/resize to 256×256, (b) encode with `stabilityai/sd-vae-ft-mse`, (c) save the latent in bf16 to `data/ffhq256_latents/{split}/{index}.pt`.
3. Raw image bytes are discarded from RAM immediately and never written to disk.
4. **Net disk use**: ~600 MB of SD-VAE latents. Zero raw images.
5. `data/MANIFEST.json` records (a) the dataset id + revision hash, (b) the number of samples received, (c) per-latent SHA-256, and (d) train/eval split index lists. The manifest is *content-addressed* so the run is reproducible from any working mirror with matching latent hashes.

**Option B: Temporary download → immediate delete (fallback)**
1. Download the official FFHQ-256 release into a temporary directory (e.g., `/tmp/ffhq256_raw`), ~28 GB temporary.
2. Pre-compute SD-VAE latents as in Option A.
3. **Immediately** delete `/tmp/ffhq256_raw` after pre-computation. No permanent raw image storage.
4. Use only when streaming is unstable or when no trustworthy HF mirror exists.

**Train/eval split**:
- 65,000 train, 5,000 eval. Indexed by **latent file index**, not raw filename — compatible with the streaming workflow.
- Index lists in `data/ffhq256_train.txt` / `data/ffhq256_eval.txt`.
- SHA-256 hashes of both list files and all latent files recorded in `data/MANIFEST.json`.

**Reference statistics (FID, FD-DINOv2)**:
- SD-VAE-decode the eval split latents in memory (~1 GB transient) → compute `clean-fid` and DINOv2 features → save only the reference statistics files.
- Outputs: `data/fid_ref_ffhq256.pt`, `data/fd_dinov2_ref_ffhq256.pt`. Combined < 200 MB.
- Decoded reference images are never persisted.

All reference statistics files are committed directly to the repo. The goal is that any future reviewer can reproduce the exact numbers.

### 4.4 Storage budget

Available disk < 90 GB. All artifacts are budgeted to keep persistent usage at ~20 GB:

| Artifact | Pre-optimization (original design) | Post-optimization |
|---|---|---|
| Raw FFHQ-256 PNG | ~28 GB | **0 GB** (streaming / temp-delete) |
| SD-VAE latents (70K, bf16) | ~0.6 GB | ~0.6 GB |
| FID/FD-DINOv2 reference statistics | ~0.2 GB | ~0.2 GB |
| Code + git history | ~0.1 GB | ~0.1 GB |
| Checkpoints (15 runs) | ~150 GB (full state × 5 ckpts × fp32) | **~16 GB** (EMA-only bf16, eval ckpts only) |
| Eval samples (60 evaluations × 10K) | ~30 GB (PNG persisted) | **~0.6 GB** (stream-compute → discard, only 16-64 visualization samples kept) |
| Human eval images (600 PNGs) | ~0.15 GB | ~0.15 GB |
| Logs, `results.csv`, plots | ~2 GB | ~2 GB |
| **Total (persistent)** | **~211 GB** | **~20 GB** |
| Buffer (50%) | | ~10 GB |
| **Allocated** | | **~30 GB** |

**Checkpoint compression strategy**:
1. **Save only at eval checkpoints** (4 points: 20K/50K/80K/100K). No saves every 20K beyond these.
2. **EMA-only, bf16**. SiT-B/2 EMA in bf16 = 130M × 2 bytes ≈ 260 MB / checkpoint. 4 × 15 runs ≈ 16 GB.
3. **No resume-state saved**. Per-run wallclock is ~3.5 h, so a failure is recovered by restarting from scratch — acceptable. Only a single rolling latest checkpoint (~2 GB) is held transiently during training and replaced at the next eval checkpoint.

**Eval sample handling**:
- `eval.py` generates 10K samples batch-by-batch in memory → metric computation → discarded.
- Only the first 16-64 samples per evaluation are saved as PNG (`exps/{branch}_{seed}/eval_{step}/preview/`) for visualization.
- Transient disk per evaluation < 1 GB.

**Worst-case headroom**: persistent ~20 GB + transient (current run latest ckpt 2 GB + current eval in-flight 1 GB) ≈ 25 GB. ~28% of the user's available 90 GB. Comfortable.

---

## 5. Model & Training Configuration

### 5.1 Model
- **SiT-B/2**: patch_size=2, depth=12, hidden_size=768, num_heads=12. 130M parameters.
- Input: SD-VAE latent 32×32×4. With patch=2, this produces 16×16 = **256 tokens**, each of dimension 4×2×2=16 projected to 768.
- Output: same shape.
- Projection head: single-layer MLP mapping DiT hidden 768 → DINOv2 embedding dimension 768. Projected from one intermediate block (following the REPA paper default: block index 8 for SiT-B, to be confirmed from the reference codebase at implementation time).

### 5.2 Teacher encoder
- DINOv2 ViT-B (`facebook/dinov2-base`), frozen, `eval()` mode.
- Input preprocessing: resize 256→224 via bilinear, ImageNet normalization.
- Feature extraction: patch tokens (not CLS) from the final layer, shape (B, 256, 768). These are the `zs` targets for cosine-similarity alignment.

### 5.3 Optimizer
- AdamW, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0.
- No learning-rate schedule (constant lr throughout, matching REPA paper).
- Gradient clip: max-norm=1.0.
- EMA: 0.9999 decay on model weights; samples are drawn from the EMA copy.

### 5.4 Loss composition
The training objective is exactly:
```
L(t) = L_denoise(t) + λ(t) · L_proj(t)
```
where
- `L_denoise` is the standard v-prediction flow matching MSE loss on SD-VAE latents,
- `L_proj` is the mean negative cosine similarity between the projected DiT hidden state and the frozen DINOv2 feature,
- `λ(t)` is the coefficient schedule, defined per branch in §6.

No other loss terms. No cls-token dual loss. No contrastive flow matching. No auxiliary terms.

### 5.5 Training hyperparameters (fixed across all branches)
| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Precision | bf16 (mixed precision via accelerate) |
| Total steps | 100,000 |
| Warmup steps (lr) | 0 |
| Eval checkpoints | 20K, 50K, 80K, 100K |
| Save checkpoints | Eval checkpoints only (20K/50K/80K/100K), **EMA-only bf16** (see §4.4). On failure, restart from scratch. |
| Seeds | {42, 1337, 2024} — three seeds per branch |
| Data augmentation | horizontal flip only |

---

## 6. Research Branches

Each branch is defined by its coefficient schedule `λ(t)` where `t ∈ [0, T]` and `T = 100,000`. Maximum coefficient is set to **0.5**, matching the original REPA paper default for SiT-B/2 variants.

### B0 — REPA Off (control)
```
λ(t) = 0  for all t
```
**Purpose**: Absolute control. Measures what SiT-B/2 achieves on FFHQ-256 without any alignment pressure. This is the ground truth against which REPA's contribution is measured.

### B1 — REPA Fixed (standard REPA)
```
λ(t) = 0.5  for all t
```
**Purpose**: Reproduces the REPA paper's default configuration. Without this branch, no claim about "REPA's effect" is possible.

### B2 — Cosine Decay (1→0)
```
λ(t) = 0.5 · 0.5 · (1 + cos(π · t / T))
```
**Purpose**: Start with full REPA, decay to zero. Hypothesis: REPA's acceleration benefit is an early-training phenomenon; removing it late prevents the late-training quality degradation we are investigating. Endpoints: λ(0)=0.5, λ(T/2)=0.25, λ(T)=0.

### B3 — Cosine Warmup (0→1)
```
λ(t) = 0.5 · 0.5 · (1 - cos(π · t / T))
```
**Purpose**: The inverse of B2. Start free, add REPA pressure late. If REPA's trade-off is really late-training collapse, this branch should show the worst sharpness/diversity. Serves as a falsification check for B2's hypothesis. Endpoints: λ(0)=0, λ(T/2)=0.25, λ(T)=0.5.

### B4 — HASTE-style Hard Cutoff
```
λ(t) = 0.5  if t < T/2
       0    if t ≥ T/2
```
**Purpose**: Matches the HASTE 2025 paper's prescription ("REPA Works Until It Doesn't"). Critical for prior-art defense: if HASTE's hard cutoff already achieves what our smooth schedules achieve, our contribution is weaker. This branch lets us position our contribution as "smooth is preferable to hard" or "smooth is equivalent to hard" — either statement is publishable.

### Branch summary
| Branch | Schedule | Max coeff | Total runs |
|---|---|---|---|
| B0 | off | 0 | 3 (seeds) |
| B1 | fixed | 0.5 | 3 |
| B2 | cosine decay | 0.5 → 0 | 3 |
| B3 | cosine warmup | 0 → 0.5 | 3 |
| B4 | hard cutoff | 0.5 → 0 at t=T/2 | 3 |
| **Total** | | | **15 runs** |

---

## 7. Evaluation Protocol

### 7.1 Automatic metrics (computed at every eval checkpoint)
At each of the 4 eval checkpoints (20K, 50K, 80K, 100K) for each of the 15 runs, generate **10,000 samples** (50-step DDPM sampler, CFG disabled since FFHQ is unconditional) and compute:

| Metric | Tool | Purpose |
|---|---|---|
| **FID-vs-step AUC** (20K → 100K) | Trapezoid integration of FID across 4 eval checkpoints | **Primary metric** — captures the full convergence trajectory rather than a single endpoint. See §2.2, §8. |
| FID @ 100K | `clean-fid` | Secondary; preserves a single-point comparator with the REPA paper |
| Inception Score | `torch-fidelity` | Sanity check only (FFHQ is single-domain; IS carries little information here) |
| Precision / Recall | Kynkäänniemi et al. 2019 implementation | Diversity (recall) and fidelity (precision) separation |
| FD-DINOv2 | Custom (DINOv2 features + Frechet distance) | Semantic fidelity in DINO feature space |
| Sharpness distribution distance | Custom (see §7.2) | Objectifies the "blurry" observation |

Output format: `{branch}_{seed}_{step}.json` containing all metrics. Aggregated into `results.csv` via a `python scripts/aggregate.py` helper.

### 7.2 Sharpness metric (custom)
The "blurry" observation that motivated this study must be objectively measurable. We use a distribution-level metric rather than a per-image score:

1. For each image `x`, compute three classical sharpness statistics:
   - **Laplacian variance**: `var(Laplacian(x))` after grayscale conversion.
   - **High-frequency energy ratio**: the ratio of FFT magnitude outside a radius-cutoff (50% of the Nyquist frequency) to total FFT magnitude.
   - **Sobel gradient mean**: mean of the Sobel magnitude map.

2. For generated samples (N=10,000) and reference images (N=5,000 from FFHQ eval split), compute each statistic, producing six 1-D distributions.

3. For each statistic, compute the Wasserstein-1 distance between the generated and reference distributions.

4. Report the three Wasserstein distances separately. The interpretation: smaller means the generated distribution of that sharpness statistic is closer to the real image distribution.

This metric does not rank "sharper vs less sharp" — it ranks "closer to real distribution vs further from real distribution." Reference images are assumed to define the ground-truth sharpness distribution.

### 7.3 Human evaluation (at final checkpoint only)
**Purpose**: Objectively validate or falsify the subjective "blurry" observation that motivated the study.

**Protocol**: Two-alternative forced choice (2AFC) pairwise comparison.

- **Pair combinations**: All 2-combinations of the 5 branches that include either B0 or B1 (the primary comparisons), plus B2–B1 and B4–B2 for schedule comparisons:
  - B0 vs B1 (does REPA help or hurt?)
  - B0 vs B2 (does decay schedule help?)
  - B0 vs B3 (does warmup schedule help?)
  - B1 vs B2 (does decay improve over fixed?)
  - B1 vs B4 (does hard cutoff improve over fixed?)
  - B2 vs B4 (smooth vs hard?)

  **Total: 6 pair combinations**.

- **Image pairs per combination**: 50, generated from matched noise seeds (i.e., same initial noise passed through both branches' EMA checkpoints). This controls for random variation in the sample.

- **Raters per image pair**: 5. Initial pilot with 2–3 self/acquaintance raters; if results are promising, extend to Prolific (USD 50-100 budget) for independent raters.

- **Questions per image pair**: 2 — "Which image looks sharper?" and "Which image looks more like a real photograph?"

- **Total responses**: 6 pairs × 50 image pairs × 5 raters × 2 questions = **3,000 responses**.

- **Platform**: Minimal local Flask or static HTML + JavaScript app that randomizes left/right presentation and saves responses to CSV. No tool purchase required.

**Analysis**: For each branch, compute a Bradley-Terry preference score across all comparisons in which it appears, with 95% confidence intervals via bootstrap resampling (1000 iterations). Compute Pearson correlation between the Bradley-Terry score and each automatic sharpness metric to test whether any automatic metric predicts human preference.

### 7.4 Interpolation study (diversity probe, final checkpoint only)
**Purpose**: If REPA causes latent-space collapse, interpolations between random latents will show "mode snapping" (sudden jumps) rather than smooth traversal. This is a qualitatively and quantitatively measurable signature.

**Protocol**:
1. For each branch, using the final EMA checkpoint:
   - Draw 10 pairs of random initial noise latents `(z_a, z_b)`.
   - For each pair, interpolate via spherical linear interpolation (slerp) at 9 values of `t ∈ {0, 0.125, 0.25, ..., 1.0}`.
   - Decode the full 9-step generation for each pair.
2. Compute the DINOv2 ViT-B feature for each decoded image (patch-token mean).
3. For each pair, compute the sequence of pairwise feature distances between consecutive interpolation steps (length-8 sequence).
4. **Jaggedness score**: `var(distances) / mean(distances)`. High jaggedness = mode snapping. Low jaggedness = smooth traversal.
5. Report mean jaggedness ± std across the 10 pairs, per branch.

**Visualization**: For the paper, produce one interpolation grid per branch (5 × 9 images) placed in the results section.

---

## 8. Pre-registration Requirements

Before any main training run, the new repo must contain `docs/prereg.md` committed at tag `prereg-v1`. This file contains:

1. **Hypothesis statement** (copy of §2.1 Claim B).
2. **Pre-registered outcome conditions** (copy of §2.2).
3. **Primary metric**: **FID-vs-step trajectory AUC** (trapezoid integration over the 4 eval checkpoints at 20K/50K/80K/100K), compared via paired t-test (Bonferroni α=0.05/6). **Secondary**: FID at step 100K (legacy single-point comparator preserving cross-paper comparability with REPA). Rationale for trajectory AUC as primary: the study's hypothesis is a *convergence speed vs. late-training quality* trade-off, and a single-endpoint metric cannot separate trajectory information from convergence-attainment.
4. **Secondary metrics**: FID at step 100K (single point), Precision, Recall, FD-DINOv2, three sharpness Wasserstein distances, Bradley-Terry score from human eval, jaggedness score.
5. **Stop conditions**: If pilot runs (§9.1.2) show no discernible signal by step 20K across any metric for 3 or more branches, pause and re-evaluate the research question. If compute budget exceeds 5 days of continuous GPU use, pause and prioritize.
6. **Pre-registered branch ranking expectation** (experimenter's prior guess before seeing data):
   - Sharpness: B0 > B2 ≈ B4 > B1 > B3 (this is the experimenter's hypothesis; used to detect confirmation bias)
   - FID convergence speed (steps to reach target FID): B1 ≈ B4 > B2 > B3 > B0
   These are recorded to enable a later confirmation-bias check. If the actual results match the pre-registered expectation almost exactly, that is *more* suspicious, not less.
7. **Analyst identity**: experimenter's name and the date of pre-registration.

**Strict rule**: After `git tag prereg-v1`, the content of `prereg.md` is not modified. Any additional analyses not covered in the pre-registration are labeled "exploratory" in the paper.

---

## 9. Timeline — 1 Month

Total duration: 28 days (4 weeks). GPU budget: single consumer GPU (RTX 4080 Super 16GB or equivalent).

### Week 1 — Setup & Pilot (Days 1–7)
**Day 1–2: Codebase and data**
- Fork `sihyun-yu/REPA` into the new repo.
- Set up the environment (conda/venv, pytorch, accelerate, clean-fid, torch-fidelity, diffusers for SD-VAE, dinov2).
- Download FFHQ-256, fix splits, compute SHA-256 manifest.
- Pre-compute SD-VAE latents (1–2 hours on 4080S).
- Pre-compute FID and FD-DINOv2 reference statistics.

**Day 3: Minimal code modifications**
- Replace dataset loader for FFHQ-256 latents.
- Remove class-conditional handling; use null token.
- Add coefficient scheduler hook to training loop (functions `schedule_off`, `schedule_fixed`, `schedule_cosine_decay`, `schedule_cosine_warmup`, `schedule_hard_cutoff`).
- Verify that `schedule_off` produces behavior identical to REPA with `λ=0` forced.
- Verify that `schedule_fixed` reproduces a 1000-step loss trajectory that matches the original REPA repo's fixed-coefficient run with the same config (modulo floating-point noise).

**Day 4: Evaluation harness**
- Write `eval.py` that takes a checkpoint path and produces a JSON of all metrics.
- Test `eval.py` on a randomly initialized model to confirm it runs end-to-end (the FID will be absurdly high — that is fine; we are testing the pipeline, not the model).
- Write `scripts/aggregate.py` that reads all result JSONs and produces `results.csv`.

**Day 5: Pre-registration**
- Write `docs/prereg.md`, matching §2 and §8.
- Commit and tag `git tag prereg-v1`.
- From this moment on, the hypothesis is frozen.

**Day 6–7: Pilot runs**
- Run pilot: 5 branches × 1 seed × 20,000 steps. This consumes approximately 5 × 40 minutes = 3.5 GPU-hours.
- Evaluate pilot checkpoints. Confirm each branch's schedule produces measurably different loss trajectories. Confirm evaluation harness produces sensible numbers.
- If any pilot run crashes or shows pathological behavior, debug before proceeding to main runs.

### Week 2 — Main Training Runs (Days 8–14)
**Day 8–11: Main runs**
- Run 15 runs sequentially: 5 branches × 3 seeds × 100,000 steps.
- Each run: approximately 3.5 GPU-hours on 4080S (extrapolated from prior SiT-B/1 benchmarks).
- Total: 15 × 3.5 = **52.5 GPU-hours** ≈ **2.2 days** continuous GPU use.
- Buffer: 1.8 days for restarts, debugging, or unexpected crashes.
- Per the §4.4 storage policy, save **only at eval checkpoints** (20K/50K/80K/100K) as **EMA-only bf16**. No saves every 20K beyond these. On in-run failure, restart from scratch.

**Day 12–14: Automatic evaluation**
- Run `eval.py` on all 15 runs × 4 eval checkpoints = 60 evaluations.
- Each evaluation: approximately 20–30 minutes (10K sample generation + metric computation).
- Total: 60 × 25 minutes = **25 GPU-hours** ≈ 1 day continuous.
- Produce `results.csv`.
- Produce preliminary plots (FID vs step per branch) and inspect. Look for obvious signal or obvious null.

### Week 3 — Human Evaluation, Interpolation, Analysis (Days 15–21)
**Day 15–16: Human eval platform**
- Build minimal Flask or static HTML app for 2AFC pairwise comparison.
- Test with synthetic pairs.
- Generate the 6 × 50 = 300 image pairs (each pair uses matched noise seeds between the two compared branches' final EMA checkpoints).

**Day 17–18: Collect human eval responses**
- Self + 2–3 acquaintance raters for the first pass (pilot).
- If pilot results show clear differentiation, recruit 5 independent raters via Prolific or similar.
- Save all responses as CSV.

**Day 19: Interpolation study**
- For each branch's final EMA checkpoint, generate the 10 × 9 = 90 slerp images.
- Compute DINOv2 features and jaggedness scores.
- Produce interpolation grids for the paper.

**Day 20–21: Statistical analysis**
- Aggregate all results.
- Run pre-registered tests: paired t-tests with Bonferroni correction.
- Compute Bradley-Terry scores from human eval.
- Compute correlations between automatic sharpness metrics and human preference.
- Produce all final plots (FID vs step, sharpness bar chart, jaggedness bar chart, Bradley-Terry scores with CI, correlation scatter).
- Apply the pre-registered judgment rule (§2.2). Record the result as positive / null / mixed.

### Week 4 — Writing and Submission (Days 22–28)
**Day 22–24: Draft**
- Write paper draft: introduction, related work, method, experiments, results, discussion.
- Target: 4–8 page workshop format.
- Include the pre-registration as an appendix.

**Day 25–26: Figures and tables**
- Finalize all figures at publication quality.
- Produce main results table.
- Produce an appendix with all raw numbers.

**Day 27–28: Revision and submission**
- Self-review pass.
- Send to one external reader if available.
- Submit to target workshop.

### Critical path summary
| Phase | Days | GPU-hours |
|---|---|---|
| Setup + pilot | 7 | ~5 |
| Main runs + eval | 7 | ~78 |
| Human eval + interp + analysis | 7 | ~3 |
| Writing | 7 | 0 |
| **Total** | **28** | **~86** |

Compute is the binding constraint. 86 GPU-hours on a consumer GPU over 28 days means the GPU must be running about 13% of the time — comfortable margin for restarts and debugging.

---

## 10. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| HASTE already published equivalent finding | High | Weakens contribution | Read HASTE carefully in Week 1; B4 branch lets us position our work as "smooth vs hard" comparison. Null result still publishable. |
| **n=3 underpowers all hypothesis tests** | **High** | Pre-registered tests miss real effects | Adopt §2.3's 4-way reporting (descriptive / directional consistency / effect size / pre-registered test). Always separate "underpowered null" from "true null". Effect size and directional consistency are reported independently of the hypothesis test. |
| **Disk space exhaustion (< 90 GB available)** | Medium | Training/eval halts | Enforce the §4.4 storage budget: streaming dataset, EMA-only bf16 checkpoints, stream-compute eval samples. Persistent usage capped at ~20 GB. Disk monitoring with auto-alert during training. |
| Null result (no branch differs significantly) | Medium | Claim fails | Claim B is designed to be publishable as null. Pivot framing from "trade-off exists" to "trade-off does not exist in clean REPA, contradicting subjective impression — prior confounders explain the observation." **However**, apply the §2.3 effect-size-based distinction between underpowered null and true null before concluding. |
| Automatic sharpness metric does not correlate with human preference | Medium | Dissociation finding | Frame as a sub-finding: "existing automatic metrics fail to capture human-perceptible sharpness differences; we recommend human eval for future REPA studies." |
| GPU compute exceeds budget | Low | Timeline slip | Cut seeds from 3 to 2 (loses statistical power but stays in budget). Do not cut branches. |
| FFHQ-SD-VAE mismatch (SD-VAE reconstructs FFHQ poorly) | Low | Baseline corrupted | Verify in Week 1 Day 1: encode/decode a held-out FFHQ batch, compute PSNR + LPIPS. **PSNR ≥ 28 dB AND LPIPS ≤ 0.10** required. The 28 dB PSNR threshold matches the known operating range of SD-VAE-ft-mse on face/natural images (27–30 dB) (2026-04-15: adjusted pre-freeze after empirical measurement on two mirrors converged at ~29.6 dB). LPIPS is verified jointly as the perceptual indicator. |
| Pre-registration gets modified post-hoc | Medium | Invalidates diagnostic claim | Enforce via git tag `prereg-v1` and explicit "frozen after this commit" text. Treat any modification as a separate exploratory analysis in the paper. |
| Training instability with a schedule that reaches λ=0 late (B2) | Low | Specific branch fails | Monitor loss curves during training. If instability occurs, halt that branch and report. |

---

## 11. Deliverables

At the end of 28 days, the new repo contains:

### Code
- Forked and minimally-modified REPA training code
- `scripts/precompute_latents.py`, `scripts/precompute_fid_ref.py`, `scripts/precompute_dino_ref.py`
- `scripts/train.py` (modified from REPA original)
- `scripts/eval.py`
- `scripts/aggregate.py`
- `scripts/human_eval_server.py` (local Flask / static HTML)
- `scripts/interpolation_study.py`
- `scripts/plots.py`

### Data artifacts
- `data/MANIFEST.json` — FFHQ-256 split hashes
- `data/fid_ref_ffhq256.pt`
- `data/fd_dinov2_ref_ffhq256.pt`

### Results
- `exps/{branch}_{seed}/checkpoints/ema_{step}.pt` — EMA-only bf16, eval checkpoints only (20K/50K/80K/100K). 15 runs × 4 ckpts ≈ 16 GB. (See §4.4.)
- `exps/{branch}_{seed}/loss_log.csv` — per-step loss trajectory (text only, small).
- `exps/{branch}_{seed}/eval_{step}/preview/` — only the first 16–64 visualization samples per evaluation, as PNG.
- `results.csv` — all metrics (including the primary FID-vs-step AUC), all eval checkpoints, all runs.
- `human_eval/responses.csv` — all human eval responses.
- `interpolation/{branch}/` — interpolation grids and jaggedness scores.

### Documentation
- `docs/prereg.md` — pre-registration (frozen, tagged `prereg-v1`)
- `docs/BASELINE.md` — exact baseline configuration, including any deviations from original REPA
- `docs/EVAL_PROTOCOL.md` — full evaluation methodology
- `README.md` — reproduction instructions

### Paper
- `paper/draft.tex` or `paper/draft.md`
- `paper/figures/` — all figures at publication quality
- `paper/tables/` — all tables as CSV + LaTeX

---

## 12. Appendix: Key Commands and Config Snippets

### 12.1 Repo setup
```bash
git clone https://github.com/sihyun-yu/REPA.git repa-reeval
cd repa-reeval
git checkout -b reeval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install clean-fid torch-fidelity
```

### 12.2 FFHQ preprocessing
```bash
# Assumes FFHQ-256 downloaded to data/ffhq256_raw/
python scripts/fix_splits.py --src data/ffhq256_raw --out data/
python scripts/precompute_latents.py --split train --vae stabilityai/sd-vae-ft-mse
python scripts/precompute_latents.py --split eval  --vae stabilityai/sd-vae-ft-mse
python scripts/precompute_fid_ref.py   --split eval
python scripts/precompute_dino_ref.py  --split eval
```

### 12.3 Pre-registration
```bash
# Write docs/prereg.md, then:
git add docs/prereg.md
git commit -m "prereg: pre-registered hypotheses and outcome conditions"
git tag prereg-v1
git push --tags
```

### 12.4 Training commands (one per branch × seed)
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
# Repeat each with --seed 1337 and --seed 2024
```

### 12.5 Evaluation
```bash
# For each run × eval checkpoint:
python scripts/eval.py \
    --ckpt  exps/b0_s42/checkpoint-20000.pt \
    --out   results/b0_s42_20000.json \
    --n_samples 10000

# Aggregate:
python scripts/aggregate.py --in results/ --out results.csv
```

### 12.6 Coefficient schedules (reference implementation)
```python
import math

def schedule_off(t, T):
    return 0.0

def schedule_fixed(t, T, max_coeff=0.5):
    return max_coeff

def schedule_cosine_decay(t, T, max_coeff=0.5):
    # 0.5 at t=0, 0 at t=T
    return max_coeff * 0.5 * (1 + math.cos(math.pi * t / T))

def schedule_cosine_warmup(t, T, max_coeff=0.5):
    # 0 at t=0, 0.5 at t=T
    return max_coeff * 0.5 * (1 - math.cos(math.pi * t / T))

def schedule_hard_cutoff(t, T, max_coeff=0.5, cutoff_frac=0.5):
    return max_coeff if t < T * cutoff_frac else 0.0
```

### 12.7 Sharpness metric (reference implementation)
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

## 13. Out of Scope

The following are deliberately excluded from this 1-month study. They may appear as future work:

- **iREPA (Conv2d projection)**: A separate architectural question. Not part of the coefficient-schedule trade-off investigation.
- **REPA-E / INVAE**: A separate VAE-side question. Deliberately excluded to keep the baseline pure.
- **Teacher encoder ablations (DINO, MAE, CLIP)**: Out of scope. DINOv2 is fixed as the teacher per the REPA paper.
- **Model scale study (SiT-L/XL)**: Out of scope for compute reasons. SiT-B/2 only.
- **ImageNet comparison**: Out of scope for compute reasons. FFHQ only, with explicit domain-deviation justification.
- **Text conditioning / CFG**: Out of scope. Unconditional only.
- **Video DiT extension**: Separate research direction, future work.

---

## 14. Success Criteria (for the 1-month study, independent of claim outcome)

This study is considered successful if:

1. All 15 main runs complete with all 4 eval checkpoints.
2. All automatic metrics are computed and stored.
3. Human eval collects ≥1,000 responses (at least self + pilot raters).
4. Interpolation study is completed for all 5 branches.
5. Pre-registration judgment is applied according to §2.2 without modification.
6. A paper draft exists at Day 28, regardless of positive/null/mixed outcome.

Meeting all six criteria means the research process was rigorous regardless of the scientific finding. Publishable or not, the experience of completing this cycle is itself the training in research methodology that was a secondary goal.

---

*End of design document. This document is self-contained and intended to be carried to a fresh repository as the sole specification for the study.*
