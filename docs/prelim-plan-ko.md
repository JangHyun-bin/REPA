# REPA Preliminary Baseline Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full 5-branch study 착수 전 2-branch (B0 pure SiT, B1 SiT+REPA-fixed) × 1 seed × 400K steps 을 FFHQ-256 에서 비교하여 (Q1) REPA 논문의 수렴 가속 효과 재현 + (Q2) sharpness/diversity trade-off signature 탐지.

**Architecture:** 기존 `sihyun-yu/REPA` upstream 의 `train.py` 를 *거의 그대로* 사용하고, (1) FFHQ-256 raw JPEG + SD-VAE latent 을 REPA 의 `CustomDataset` 포맷으로 pre-생성하고, (2) `train.py` 의 checkpoint save 블록을 EMA-only bf16 으로만 *한 번* patch, (3) `--proj-coeff 0` vs `0.5` 로 B0/B1 을 구분 실행. 모든 작업은 git worktree (`/home/famoz/projects/dl/REPA-prelim`, branch `prelim-baseline`, base `67f7145`) 에서 격리 수행.

**Tech Stack:** PyTorch 2.6+cu124, accelerate (bf16 mixed precision), diffusers (SD-VAE-ft-mse), HuggingFace datasets (FFHQ streaming), clean-fid, scipy (Wasserstein), matplotlib. **dl conda env** 재사용 (이미 reeval branch 에서 구축됨).

**Spec 참조:** 본 plan 의 모든 `§X` 참조는 `docs/prelim-spec-ko.md` 의 섹션 (11 섹션, 477 줄). Full study spec (`docs/spec-ko.md`) 은 별도 문서.

**환경 가정:** RTX 4080 16GB, Linux, 가용 디스크 97 GB, Python 3.11.14 (`dl` conda env), bash, tmux.

---

## File Structure (prelim worktree 내부)

```
/home/famoz/projects/dl/REPA-prelim/
├── .git                                 # pointer → isREPAgood/REPA/.git/worktrees/REPA-prelim
├── (sihyun-yu/REPA 67f7145 원본 파일들)
├── train.py                             # 1곳 patch (checkpoint save → EMA bf16)
├── dataset.py                           # unchanged
├── loss.py                              # unchanged
├── models/sit.py                        # unchanged
├── samplers.py                          # unchanged (eval 에서 import)
├── utils.py                             # unchanged
├── requirements-prelim.txt              # NEW (reeval 것 복사)
├── scripts/
│   ├── disk_monitor.py                  # cherry-pick from reeval
│   ├── download_ffhq.py                 # NEW
│   ├── precompute_latents_repa.py       # NEW
│   ├── precompute_fid_ref.py            # NEW
│   ├── eval_prelim.py                   # NEW
│   ├── aggregate_prelim.py              # NEW
│   └── plot_prelim.py                   # NEW
├── data/
│   ├── MANIFEST.json                    # cherry-pick from reeval (mirror: merkol/ffhq-256)
│   ├── sd_vae_recon_sanity.json         # cherry-pick from reeval (이미 PASS)
│   ├── ffhq256_train.txt                # NEW: 65000 indices (seed 20260415)
│   ├── ffhq256_eval.txt                 # NEW: 5000 indices
│   ├── fid_ref_ffhq256.npz              # NEW: clean-fid Inception ref stats
│   ├── disk_baseline.txt                # cherry-pick from reeval
│   └── ffhq256-prelim/                  # NEW: REPA-format dataset root
│       ├── images/
│       │   ├── 000000.jpg               # 70K × ~50 KB JPEG Q95 ≈ 3.5 GB
│       │   ├── 000001.jpg
│       │   └── ... 069999.jpg
│       └── vae-sd/
│           ├── 000000.npy               # (1, 8, 32, 32) SD-VAE moments, fp32
│           ├── 000001.npy
│           ├── ... 069999.npy           # ≈ 2.2 GB
│           └── dataset.json             # REPA labels manifest (모두 label=0)
├── exps/
│   ├── b0_s42/                          # pure SiT (proj-coeff=0)
│   │   ├── args.json
│   │   ├── logs/log.txt
│   │   └── checkpoints/
│   │       ├── 0050000.pt               # EMA-only bf16, ~260 MB
│   │       ├── 0100000.pt
│   │       ├── ... 0400000.pt           # 8 ckpts × ~260 MB ≈ 2 GB
│   └── b1_s42/                          # SiT+REPA (proj-coeff=0.5)
│       └── (동일 구조)
├── results/
│   ├── b0_s42_050000.json               # eval result per checkpoint
│   ├── ... b1_s42_400000.json           # 16 JSON files
│   └── prelim_summary.txt               # flat summary from aggregate script
├── results_prelim.csv                   # aggregated 16 rows
├── results_prelim.md                    # §4.4 decision document
├── figures/
│   ├── fid_vs_step.png                  # B0 vs B1 trajectory
│   ├── sharpness_bars.png               # 3 stats × 2 branches × final checkpoint
│   ├── sharpness_vs_step.png            # trajectory of each sharpness stat
│   └── preview_grid.png                 # B0 vs B1 sample comparison
└── .gitignore                           # extend upstream: data/ffhq256-prelim/, exps/, results/main/
```

**Disk 예산** (영구 사용, preliminary 종료 시점):
- Raw images: ~3.5 GB
- Latents: ~2.2 GB
- Checkpoints: ~4 GB (2 runs × 8 × 260 MB)
- Eval samples (stream-compute, preview 만 보관): <1 GB
- Logs + results + figures: <1 GB
- **총 ~11 GB** (가용 97 GB 내 여유 충분)

---

## Phase 0 — Worktree & Environment Setup

**목표**: Git worktree 생성, 유틸 스크립트 cherry-pick, 환경 검증. **예상 시간**: 30 분.

### Task 0.1: Git worktree 생성

**Files:**
- Create: `/home/famoz/projects/dl/REPA-prelim/` (new working tree)

- [ ] **Step 1**: 현 작업 위치가 reeval worktree 인지 확인.

```bash
cd /home/famoz/projects/dl/isREPAgood/REPA
pwd
git branch --show-current
git status
```

Expected: current branch = `reeval`, working tree clean.

- [ ] **Step 2**: Worktree 추가 (base = 67f7145, detached HEAD).

```bash
git worktree add /home/famoz/projects/dl/REPA-prelim 67f7145
```

Expected output:
```
Preparing worktree (detached HEAD 67f7145)
HEAD is now at 67f7145 Update README.md
```

- [ ] **Step 3**: 새 worktree 로 이동 + branch 생성.

```bash
cd /home/famoz/projects/dl/REPA-prelim
git checkout -b prelim-baseline
```

- [ ] **Step 4**: 검증.

```bash
git worktree list
git branch --show-current
ls
```

Expected `git worktree list`:
```
/home/famoz/projects/dl/isREPAgood/REPA         1ec794f [reeval]
/home/famoz/projects/dl/REPA-prelim             67f7145 [prelim-baseline]
```

Expected `git branch --show-current`: `prelim-baseline`

Expected `ls`: REPA upstream files (`train.py`, `dataset.py`, `models/`, etc.) but NO `scripts/`, NO reeval-specific files (because we're at commit 67f7145).

### Task 0.2: Cherry-pick utility scripts from reeval

**Files:**
- Create: `scripts/disk_monitor.py`
- Create: `scripts/sd_vae_sanity.py`
- Create: `requirements-prelim.txt`
- Create: `data/MANIFEST.json`
- Create: `data/sd_vae_recon_sanity.json`
- Create: `data/disk_baseline.txt`

- [ ] **Step 1**: 스크립트 디렉토리 scaffold.

```bash
mkdir -p scripts data exps results figures
```

- [ ] **Step 2**: Reeval branch 에서 유용 파일 cherry-pick via `git show`.

```bash
git show reeval:scripts/disk_monitor.py > scripts/disk_monitor.py
git show reeval:scripts/sd_vae_sanity.py > scripts/sd_vae_sanity.py
git show reeval:requirements-reeval.txt > requirements-prelim.txt
git show reeval:data/MANIFEST.json > data/MANIFEST.json
git show reeval:data/sd_vae_recon_sanity.json > data/sd_vae_recon_sanity.json
git show reeval:data/disk_baseline.txt > data/disk_baseline.txt
```

- [ ] **Step 3**: 검증.

```bash
ls scripts/ data/
head -3 scripts/disk_monitor.py
head -10 data/MANIFEST.json
```

Expected: 2 script 파일 + 3 data 파일 모두 보임. MANIFEST.json 에 `merkol/ffhq-256` mirror 정보 포함.

- [ ] **Step 4**: .gitignore 확장.

```bash
cat >> .gitignore << 'EOF'

# === prelim workflow ===
data/ffhq256-prelim/
data/ffhq256_latents/
exps/
results/main/
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.tmp
EOF
```

- [ ] **Step 5**: 첫 commit.

```bash
git add scripts/disk_monitor.py scripts/sd_vae_sanity.py \
        requirements-prelim.txt data/MANIFEST.json \
        data/sd_vae_recon_sanity.json data/disk_baseline.txt .gitignore
git commit -m "prelim: import utility scripts + dataset manifest from reeval"
```

Expected: 1 commit on prelim-baseline, 7 files changed.

### Task 0.3: 환경 검증

- [ ] **Step 1**: Conda env 활성화 + GPU 체크.

```bash
source /home/famoz/miniconda3/bin/activate dl
python -c "
import torch, diffusers, datasets, lpips, cleanfid, transformers
print('✓ imports ok')
print('cuda:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0))
print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
"
```

Expected:
```
✓ imports ok
cuda: True / NVIDIA GeForce RTX 4080
vram GB: 16.7
```

- [ ] **Step 2**: Disk baseline 기록.

```bash
python scripts/disk_monitor.py --once --log data/disk_log.csv
cat data/disk_log.csv
df -h .
```

Expected: `data/disk_log.csv` 에 1 row 추가, free > 90 GB.

- [ ] **Step 3**: Commit.

```bash
git add data/disk_log.csv
git commit -m "prelim: env check + disk baseline"
```

---

## Phase 1 — FFHQ-256 Raw Images (JPEG)

**목표**: 70K FFHQ-256 원본을 HF mirror (`merkol/ffhq-256`) 에서 streaming download 하여 JPEG Q95 로 저장. **예상 시간**: 40-60 분 (~3.5 GB 다운로드).

**왜 JPEG Q95?** REPA 의 `CustomDataset` 은 DINO target 을 raw image 에서 계산하므로 원본 픽셀이 필요. PNG (70K × ~400 KB ≈ 28 GB) 보다 JPEG Q95 (70K × ~50 KB ≈ 3.5 GB) 가 디스크 8배 절약. Q95 는 DINO feature 에 거의 영향 없음 (high-quality JPEG).

### Task 1.1: Download script

**Files:**
- Create: `scripts/download_ffhq.py`

- [ ] **Step 1**: Script 작성.

`scripts/download_ffhq.py`:
```python
"""Stream FFHQ-256 from merkol/ffhq-256 and save as JPEG Q95 under
data/ffhq256-prelim/images/000000.jpg ... 069999.jpg

Raw PIL images are decoded in RAM, re-encoded as JPEG, and written.
The original PNG bytes are never persisted to disk.
"""
import argparse, json, time
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    t0 = time.time()
    for i, sample in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        img_key = "image" if "image" in sample else None
        if img_key is None:
            raise RuntimeError(f"sample keys: {list(sample.keys())} — no 'image' field")
        img = sample[img_key].convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.LANCZOS)
        out_path = out_dir / f"{i:06d}.jpg"
        img.save(out_path, format="JPEG", quality=args.quality, optimize=True)
        if (i + 1) % 5000 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            print(f"[{i+1}] rate={rate:.1f} img/s elapsed={dt/60:.1f} min")
    print(f"✓ saved {i+1} images to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dataset", default="merkol/ffhq-256")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="data/ffhq256-prelim/images")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    main(p.parse_args())
```

- [ ] **Step 2**: Dry-run with `--limit 10`.

```bash
source /home/famoz/miniconda3/bin/activate dl
python scripts/download_ffhq.py --limit 10
ls data/ffhq256-prelim/images/ | head
du -sh data/ffhq256-prelim/images/
```

Expected: 10 files `000000.jpg` ~ `000009.jpg`, total ~500 KB.

- [ ] **Step 3**: Commit script.

```bash
git add scripts/download_ffhq.py
git commit -m "feat: FFHQ-256 raw image downloader (JPEG Q95, streaming)"
```

### Task 1.2: Run full download

- [ ] **Step 1**: Clean dry-run files to start fresh indexing.

```bash
rm -rf data/ffhq256-prelim/images
```

- [ ] **Step 2**: Disk pre-flight.

```bash
python scripts/disk_monitor.py --once
df -h .
```

Expected: ≥ 10 GB free.

- [ ] **Step 3**: Full download (uses tmux or nohup for durability).

```bash
python scripts/download_ffhq.py 2>&1 | tee data/download_log.txt
```

Expected: ~15-30 분, rate ~40-80 img/s, 마지막 line: `✓ saved 70000 images to data/ffhq256-prelim/images`.

⚠ **트러블슈팅**:
- **Connection error mid-stream**: script 는 resume 미지원. 삭제 후 재실행.
- **Rate limit from HF Hub**: `huggingface-cli login` 으로 token 등록 재시도.
- **Disk fills up**: 즉시 중단, cleanup 후 원인 점검.

### Task 1.3: Sanity check images

- [ ] **Step 1**: File count + size.

```bash
ls data/ffhq256-prelim/images/ | wc -l
du -sh data/ffhq256-prelim/images/
```

Expected: `70000` files, `~3.5 GB`.

- [ ] **Step 2**: Random sample visual check.

```bash
python -c "
from PIL import Image
import random
for idx in random.sample(range(70000), 5):
    img = Image.open(f'data/ffhq256-prelim/images/{idx:06d}.jpg')
    print(f'{idx:06d}.jpg: size={img.size} mode={img.mode}')
    assert img.size == (256, 256)
    assert img.mode == 'RGB'
print('✓ all checked')
"
```

Expected: 5 파일 모두 256×256 RGB, final `✓ all checked`.

- [ ] **Step 3**: Commit log only (images are gitignored).

```bash
git add data/download_log.txt
git commit -m "data: download 70K FFHQ-256 JPEG Q95 images (~3.5 GB, gitignored)"
```

---

## Phase 2 — SD-VAE Latent Precompute (REPA format)

**목표**: 70K JPEG 를 SD-VAE encode 하여 REPA 의 `CustomDataset` 이 읽을 수 있는 `(1, 8, 32, 32)` moments 포맷으로 저장. **예상 시간**: 20-40 분.

### Task 2.1: Write precompute_latents_repa.py

**Files:**
- Create: `scripts/precompute_latents_repa.py`

- [ ] **Step 1**: Script 작성 (REPA 포맷에 정확히 맞춤).

`scripts/precompute_latents_repa.py`:
```python
"""Encode FFHQ-256 JPEG images to SD-VAE latent moments in REPA's CustomDataset format.

Reads from  data/ffhq256-prelim/images/{idx:06d}.jpg
Writes to   data/ffhq256-prelim/vae-sd/{idx:06d}.npy
Each .npy is shape (1, 8, 32, 32) fp32 = concatenated (mean, std) moments.

Also writes data/ffhq256-prelim/vae-sd/dataset.json with REPA label mapping
(all labels = 0 for unconditional FFHQ).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

def main(args):
    device = "cuda"
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    # REPA's normalization: uint8 [0, 255] -> fp32 [-1, 1]
    # preprocessing/encoders.py line 78: x = x.to(torch.float32) / 127.5 - 1
    # We match that exactly.

    img_files = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        img_files = img_files[:args.limit]
    n_total = len(img_files)
    print(f"encoding {n_total} images → {out_dir}")

    labels = []
    t0 = time.time()
    BS = args.batch_size
    batch_imgs = []
    batch_names = []

    def flush():
        nonlocal batch_imgs, batch_names
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)       # (B, 3, 256, 256) fp32 in [-1, 1]
        with torch.no_grad():
            d = vae.encode(x)["latent_dist"]
            # Concatenate mean and std along channel dim → (B, 8, 32, 32)
            moments = torch.cat([d.mean, d.std], dim=1).cpu().numpy().astype(np.float32)
        for i, name in enumerate(batch_names):
            # Store as (1, 8, 32, 32) to match REPA's preprocessing/dataset_tools.py:407
            arr = moments[i:i+1]
            np.save(out_dir / f"{name}.npy", arr)
            labels.append([f"{name}.npy", 0])
        batch_imgs.clear()
        batch_names.clear()

    for idx, img_path in enumerate(img_files):
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32)      # (H, W, 3) uint8→fp32
        arr = arr.transpose(2, 0, 1)                  # (3, H, W)
        tensor = torch.from_numpy(arr) / 127.5 - 1.0  # [-1, 1]
        batch_imgs.append(tensor)
        batch_names.append(img_path.stem)
        if len(batch_imgs) == BS:
            flush()
        if (idx + 1) % 5000 == 0:
            dt = time.time() - t0
            print(f"[{idx+1}/{n_total}] rate={((idx+1)/dt):.1f} img/s")
    flush()

    dataset_json_path = out_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps({"labels": labels}))
    print(f"✓ saved {len(labels)} latents + dataset.json ({dataset_json_path})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", default="data/ffhq256-prelim/images")
    p.add_argument("--out", default="data/ffhq256-prelim/vae-sd")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
```

- [ ] **Step 2**: Commit.

```bash
git add scripts/precompute_latents_repa.py
git commit -m "feat: SD-VAE encoder in REPA CustomDataset format ((1,8,32,32) moments)"
```

### Task 2.2: Dry run (10 images)

- [ ] **Step 1**: Dry run.

```bash
python scripts/precompute_latents_repa.py --limit 10
```

Expected: `✓ saved 10 latents + dataset.json (...)`.

- [ ] **Step 2**: Verify output shape matches REPA expectation.

```bash
python -c "
import numpy as np, json
arr = np.load('data/ffhq256-prelim/vae-sd/000000.npy')
print('shape:', arr.shape, 'dtype:', arr.dtype)
assert arr.shape == (1, 8, 32, 32), f'unexpected shape {arr.shape}'
assert arr.dtype == np.float32, f'unexpected dtype {arr.dtype}'
meta = json.load(open('data/ffhq256-prelim/vae-sd/dataset.json'))
print('labels count:', len(meta['labels']))
print('first label entry:', meta['labels'][0])
assert meta['labels'][0] == ['000000.npy', 0]
print('✓ format ok')
"
```

Expected:
```
shape: (1, 8, 32, 32) dtype: float32
labels count: 10
first label entry: ['000000.npy', 0]
✓ format ok
```

- [ ] **Step 3**: Cleanup dry-run files.

```bash
rm -rf data/ffhq256-prelim/vae-sd
```

### Task 2.3: Full precompute (70K)

- [ ] **Step 1**: Full run.

```bash
python scripts/precompute_latents_repa.py 2>&1 | tee data/latent_log.txt
```

Expected: ~20-40 분, 마지막 line `✓ saved 70000 latents + dataset.json`.

- [ ] **Step 2**: Verify.

```bash
ls data/ffhq256-prelim/vae-sd/ | wc -l
du -sh data/ffhq256-prelim/vae-sd/
python -c "
import json
meta = json.load(open('data/ffhq256-prelim/vae-sd/dataset.json'))
print('labels:', len(meta['labels']))
assert len(meta['labels']) == 70000
print('first:', meta['labels'][0], 'last:', meta['labels'][-1])
print('✓')
"
```

Expected: `70001` files (70000 .npy + 1 .json), ~2.2 GB, labels count 70000.

- [ ] **Step 3**: REPA CustomDataset smoke test.

```bash
python -c "
import sys; sys.path.insert(0, '.')
from dataset import CustomDataset
ds = CustomDataset('data/ffhq256-prelim')
print('len:', len(ds))
raw, features, label = ds[0]
print('raw shape:', raw.shape, 'dtype:', raw.dtype)
print('features shape:', features.shape, 'dtype:', features.dtype)
print('label:', label.item())
assert len(ds) == 70000
assert raw.shape == (3, 256, 256)  # C, H, W
assert features.shape == (1, 8, 32, 32)
assert label.item() == 0
print('✓ REPA CustomDataset compatible')
"
```

Expected:
```
len: 70000
raw shape: torch.Size([3, 256, 256]) dtype: torch.uint8
features shape: torch.Size([1, 8, 32, 32]) dtype: torch.float32
label: 0
✓ REPA CustomDataset compatible
```

- [ ] **Step 4**: Commit log.

```bash
git add data/latent_log.txt
git commit -m "data: precompute 70K SD-VAE moments in REPA format (~2.2 GB, gitignored)"
```

---

## Phase 3 — Train/Eval Split + FID Reference

**목표**: Reeval 과 동일한 seed 20260415 로 65K/5K split 고정 + clean-fid Inception reference stats 계산. **예상 시간**: 15 분.

### Task 3.1: Split files

**Files:**
- Create: `data/ffhq256_train.txt`
- Create: `data/ffhq256_eval.txt`

- [ ] **Step 1**: Split 생성 (reeval 의 것과 동일한 seed).

```bash
python -c "
import random
random.seed(20260415)
indices = [f'{i:06d}' for i in range(70000)]
random.shuffle(indices)
train = sorted(indices[:65000])
eval_ = sorted(indices[65000:])
open('data/ffhq256_train.txt','w').write('\n'.join(train) + '\n')
open('data/ffhq256_eval.txt','w').write('\n'.join(eval_) + '\n')
print('train:', len(train), 'eval:', len(eval_))
print('train first 3:', train[:3])
print('eval first 3:', eval_[:3])
"
```

Expected:
```
train: 65000 eval: 5000
train first 3: ['000000', '000003', '000004']   (or similar)
eval first 3: ['000001', '000002', '000013']    (or similar)
```

- [ ] **Step 2**: Verify split is disjoint + union covers all.

```bash
python -c "
tr = set(open('data/ffhq256_train.txt').read().splitlines())
ev = set(open('data/ffhq256_eval.txt').read().splitlines())
assert len(tr) == 65000 and len(ev) == 5000
assert tr.isdisjoint(ev)
assert tr | ev == {f'{i:06d}' for i in range(70000)}
print('✓ split ok')
"
```

- [ ] **Step 3**: Commit.

```bash
git add data/ffhq256_train.txt data/ffhq256_eval.txt
git commit -m "data: fix 65K/5K train/eval split (seed 20260415)"
```

### Task 3.2: precompute_fid_ref.py

**Files:**
- Create: `scripts/precompute_fid_ref.py`

- [ ] **Step 1**: Script 작성.

`scripts/precompute_fid_ref.py`:
```python
"""Compute clean-fid Inception statistics for FFHQ-256 eval split.

Loads the 5K eval latents (REPA format), decodes via SD-VAE (using mean only,
not sample_posterior), computes Inception features, writes mu/sigma.
Output: data/fid_ref_ffhq256.npz with keys 'mu' (2048,) and 'sigma' (2048,2048).
"""
import argparse
import numpy as np
import torch
from diffusers import AutoencoderKL
from cleanfid.features import build_feature_extractor

LATENTS_SCALE = 0.18215  # REPA's default, matches SD-VAE-ft-mse scaling factor

def main(args):
    device = "cuda"
    eval_idx = open(args.eval_list).read().splitlines()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    extractor = build_feature_extractor("clean", device)

    feats = []
    batch = []
    BS = args.batch_size
    for i, idx in enumerate(eval_idx):
        # Load (1, 8, 32, 32) moments, take mean channels only [0:4]
        m = np.load(f"{args.latents_dir}/{idx}.npy")   # shape (1, 8, 32, 32)
        mean = torch.from_numpy(m)[0, :4].float()      # (4, 32, 32)
        batch.append(mean)
        if len(batch) == BS or i == len(eval_idx) - 1:
            zs = torch.stack(batch).to(device)         # (BS, 4, 32, 32)
            with torch.no_grad():
                imgs = vae.decode(zs / LATENTS_SCALE).sample           # (BS, 3, 256, 256)
                imgs_uint8 = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
                f = extractor(imgs_uint8)
            feats.append(f.cpu().numpy())
            batch = []
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(eval_idx)}]")

    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    np.savez(args.out, mu=mu, sigma=sigma, n=len(eval_idx))
    print(f"✓ FID reference computed from {len(eval_idx)} samples → {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_list", default="data/ffhq256_eval.txt")
    p.add_argument("--latents_dir", default="data/ffhq256-prelim/vae-sd")
    p.add_argument("--out", default="data/fid_ref_ffhq256.npz")
    p.add_argument("--batch_size", type=int, default=16)
    main(p.parse_args())
```

- [ ] **Step 2**: 실행.

```bash
python scripts/precompute_fid_ref.py
```

Expected: ~5-10 분, 마지막 line `✓ FID reference computed from 5000 samples → data/fid_ref_ffhq256.npz`.

- [ ] **Step 3**: Sanity check output.

```bash
python -c "
import numpy as np
ref = np.load('data/fid_ref_ffhq256.npz')
print('mu shape:', ref['mu'].shape, 'sigma shape:', ref['sigma'].shape, 'n:', int(ref['n']))
assert ref['mu'].shape == (2048,)
assert ref['sigma'].shape == (2048, 2048)
assert int(ref['n']) == 5000
print('✓')
"
```

- [ ] **Step 4**: Commit.

```bash
git add scripts/precompute_fid_ref.py data/fid_ref_ffhq256.npz
git commit -m "data: clean-fid Inception ref stats from eval split (5K)"
```

---

## Phase 4 — train.py Modification (EMA-only bf16 checkpoint)

**목표**: REPA 의 `train.py` 에서 checkpoint save 블록 **1군데만** patch. 다른 수정 없음. **예상 시간**: 30 분.

**왜 patch 가 필요**? REPA 의 기본 save 는 full state (`model`, `ema`, `opt`, `args`, `steps`) 로 checkpoint 당 ~2 GB. 16 checkpoints × 2 runs × 2 GB = 64 GB → spec §4.4 budget 위반. EMA-only bf16 (~260 MB × 16 = ~4 GB) 로 줄임.

### Task 4.1: Patch train.py checkpoint save

**Files:**
- Modify: `train.py` (lines 328-339 approximately)

- [ ] **Step 1**: Patch 위치 확인.

```bash
grep -n "checkpointing_steps\|torch.save" train.py
```

Expected output includes lines around 328-339 (the checkpoint save block).

- [ ] **Step 2**: 직접 수정. `train.py` 의 checkpoint save 블록을 찾아서 다음과 같이 변경 (정확한 indentation 유지):

**BEFORE** (약 lines 328-339):
```python
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
```

**AFTER**:
```python
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # PRELIMINARY MODIFICATION: EMA-only bf16 to fit disk budget.
                    # Trades resumability for disk space (per prelim-spec §6).
                    ema_bf16 = {
                        k: v.detach().to(torch.bfloat16).cpu()
                        for k, v in ema.state_dict().items()
                    }
                    checkpoint = {"ema": ema_bf16, "steps": global_step}
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved EMA-only bf16 checkpoint to {checkpoint_path}")
```

- [ ] **Step 3**: Diff 확인.

```bash
git diff train.py
```

Expected: 위의 블록만 변경되었고 다른 부분은 건드리지 않음. ~12 줄 변경.

- [ ] **Step 4**: 구문 검증.

```bash
python -c "import ast; ast.parse(open('train.py').read()); print('✓ train.py syntax ok')"
```

Expected: `✓ train.py syntax ok`

- [ ] **Step 5**: Commit.

```bash
git add train.py
git commit -m "patch: train.py checkpoint save → EMA-only bf16 (prelim only)"
```

### Task 4.2: Smoke test — 10-step dry run

**목표**: Patch 가 crash 없이 작동 + 다른 코드 경로 망가지지 않았는지 확인. 10 step 만 돌려봄.

- [ ] **Step 1**: Smoke run (conda env 활성화).

```bash
source /home/famoz/miniconda3/bin/activate dl
accelerate launch train.py \
    --exp-name smoke_test \
    --model SiT-B/2 \
    --data-dir data/ffhq256-prelim \
    --batch-size 32 \
    --mixed-precision bf16 \
    --max-train-steps 10 \
    --checkpointing-steps 10 \
    --sampling-steps 9999999 \
    --proj-coeff 0 \
    --num-classes 1 \
    --cfg-prob 0 \
    --encoder-depth 8 \
    --enc-type dinov2-vit-b \
    --path-type linear \
    --prediction v \
    --weighting uniform \
    --seed 42 \
    --output-dir exps \
    --report-to tensorboard \
    --allow-tf32 \
    2>&1 | tail -30
```

⚠ **주의**:
- `--sampling-steps 9999999` → smoke test 에서 wandb sample 생성 건너뜀 (없어도 학습은 작동)
- `--report-to tensorboard` → wandb 의존성 회피 (smoke test 에서는 단순화)
- `--num-classes 1` + `--cfg-prob 0` → unconditional mode
- `--proj-coeff 0` → B0 (REPA off) smoke

Expected: 10 step 후 `Saved EMA-only bf16 checkpoint to exps/smoke_test/checkpoints/0000010.pt`. NaN 없음, 크래시 없음.

⚠ **트러블슈팅**:
- **`ModuleNotFoundError: No module named 'wandb'`**: tensorboard 대신 wandb 가 default 일 수 있음. `--report-to tensorboard` 확실히 전달 or `pip install wandb` 한 번만.
- **`KeyError: num_classes 1`** or model init 실패: num_classes=1 로 embedding table size 문제 가능. 그 경우 num_classes=1000 유지 (전부 label=0 이면 동일 효과). CLI 만 바꾸면 됨.
- **OOM**: batch_size 를 32 → 16 으로.

- [ ] **Step 2**: Checkpoint 파일 size 확인.

```bash
ls -la exps/smoke_test/checkpoints/
```

Expected: `0000010.pt` 가 ~260 MB (EMA bf16 만).

- [ ] **Step 3**: Checkpoint 로드 테스트.

```bash
python -c "
import torch
ckpt = torch.load('exps/smoke_test/checkpoints/0000010.pt', map_location='cpu')
print('keys:', list(ckpt.keys()))
assert set(ckpt.keys()) == {'ema', 'steps'}
print('steps:', ckpt['steps'])
print('ema first param dtype:', next(iter(ckpt['ema'].values())).dtype)
assert next(iter(ckpt['ema'].values())).dtype == torch.bfloat16
print('✓ checkpoint format ok')
"
```

Expected:
```
keys: ['ema', 'steps']
steps: 10
ema first param dtype: torch.bfloat16
✓ checkpoint format ok
```

- [ ] **Step 4**: Smoke run cleanup + commit smoke log.

```bash
rm -rf exps/smoke_test
git status   # should show nothing (exps/ is gitignored)
```

Expected: clean working tree.

---

## Phase 5 — Wallclock Pilot (5K steps)

**목표**: Spec §5.1 의 요구 — 실제 wallclock 측정해 400K 외삽 검증. B0 first 5K 만 시간 측정. **예상 시간**: ~15-25 분.

### Task 5.1: 5K-step pilot run

- [ ] **Step 1**: Disk pre-flight.

```bash
python scripts/disk_monitor.py --once
df -h .
```

- [ ] **Step 2**: Pilot run (B0, seed 42, 5000 steps).

```bash
source /home/famoz/miniconda3/bin/activate dl
SECONDS=0
accelerate launch train.py \
    --exp-name wallclock_pilot_b0 \
    --model SiT-B/2 \
    --data-dir data/ffhq256-prelim \
    --batch-size 32 \
    --mixed-precision bf16 \
    --max-train-steps 5000 \
    --checkpointing-steps 5000 \
    --sampling-steps 9999999 \
    --proj-coeff 0 \
    --num-classes 1 \
    --cfg-prob 0 \
    --encoder-depth 8 \
    --enc-type dinov2-vit-b \
    --path-type linear \
    --prediction v \
    --weighting uniform \
    --seed 42 \
    --output-dir exps \
    --report-to tensorboard \
    --allow-tf32 \
    2>&1 | tee exps/wallclock_pilot_b0.log
echo "ELAPSED: $SECONDS seconds"
```

Expected: 5000 steps 완료, 마지막에 elapsed time 출력.

- [ ] **Step 3**: 외삽 + 결정.

```bash
python -c "
import re, json
log = open('exps/wallclock_pilot_b0.log').read()
m = re.search(r'ELAPSED:\s*(\d+)', log)
if m:
    sec = int(m.group(1))
else:
    # fallback: parse from tqdm progress bar final line
    print('ELAPSED line not found; inspect log manually')
    import sys; sys.exit(1)
hours_per_5k = sec / 3600
hours_per_100k = hours_per_5k * 20
hours_per_400k = hours_per_5k * 80
hours_per_2_runs = hours_per_400k * 2
out = {
    'wallclock_5k_sec': sec,
    'hours_per_5k': hours_per_5k,
    'hours_per_100k_extrap': hours_per_100k,
    'hours_per_400k_extrap': hours_per_400k,
    'hours_2_runs_extrap': hours_per_2_runs,
    'spec_estimate_hours_per_run': 14,
    'spec_estimate_hours_total': 28,
    'within_budget': hours_per_2_runs < 40,
}
print(json.dumps(out, indent=2))
open('data/wallclock_benchmark.json','w').write(json.dumps(out, indent=2))
"
```

🚨 **결정 매트릭스** (prelim-spec §5.1 에 기반):

| `hours_per_400k` 값 | 의미 | 행동 |
|---|---|---|
| ≤ 12h | spec 외삽보다 빠름 | 그대로 main run 진행 |
| 12-16h | spec 정확, 28-32 GPU-hours 총 | 진행 |
| 16-20h | 외삽 underestimate, 32-40h 총 | 진행 (buffer 줄어듦) |
| > 20h | 40h+ 총, timeline 위협 | **STOP**. step budget 을 300K 또는 200K 로 감축 후 재검토. |

- [ ] **Step 4**: Cleanup + commit.

```bash
rm -rf exps/wallclock_pilot_b0
git add data/wallclock_benchmark.json exps/wallclock_pilot_b0.log
git commit -m "data: wallclock pilot B0 5K steps (extrapolate to 400K × 2 runs)"
```

---

## Phase 6 — B0 Training (pure SiT, 400K)

**목표**: `--proj-coeff 0` 로 pure SiT baseline 학습. 400K steps, seed 42. **예상 시간**: ~14 h.

### Task 6.1: Launch B0 in tmux

- [ ] **Step 1**: tmux 세션 시작.

```bash
tmux new -s prelim_b0
```

- [ ] **Step 2**: 세션 안에서 disk monitor 백그라운드 + 학습 명령.

```bash
source /home/famoz/miniconda3/bin/activate dl
python scripts/disk_monitor.py --min_gb 15 --interval 300 --log data/disk_log_b0.csv --exit_on_alert &
DISK_PID=$!

accelerate launch train.py \
    --exp-name b0_s42 \
    --model SiT-B/2 \
    --data-dir data/ffhq256-prelim \
    --batch-size 32 \
    --mixed-precision bf16 \
    --max-train-steps 400000 \
    --checkpointing-steps 50000 \
    --sampling-steps 9999999 \
    --proj-coeff 0 \
    --num-classes 1 \
    --cfg-prob 0 \
    --encoder-depth 8 \
    --enc-type dinov2-vit-b \
    --path-type linear \
    --prediction v \
    --weighting uniform \
    --seed 42 \
    --output-dir exps \
    --report-to tensorboard \
    --allow-tf32 \
    2>&1 | tee exps/b0_s42_train.log

kill $DISK_PID 2>/dev/null || true
```

- [ ] **Step 3**: Detach tmux (`Ctrl+B` then `D`).

- [ ] **Step 4**: 일일 monitoring (별도 터미널에서).

```bash
tmux attach -t prelim_b0   # attach 해서 progress bar 확인
# Ctrl+B, D 로 detach
tail -5 exps/b0_s42_train.log
ls exps/b0_s42/checkpoints/
```

🔍 **Expected progress**:
- 50K step 마다 `Saved EMA-only bf16 checkpoint to exps/b0_s42/checkpoints/00XX0000.pt`
- 총 8 checkpoints 생성 (50K, 100K, ..., 400K)
- 총 wallclock ~14h (pilot 기반)

⚠ **트러블슈팅**:
- **NaN loss**: 학습 발산. Log 확인, 필요 시 재시작. (Pure SiT 는 발산 가능성 낮음.)
- **CUDA OOM**: 다른 프로세스가 GPU 쓰는지 확인 (`nvidia-smi`).
- **tmux 세션 loss (system reboot)**: run 재시작. Full checkpoint save 안 하므로 처음부터 다시. 시간 손실 ~14h.
- **Disk alert**: `data/disk_log_b0.csv` 에서 원인 확인. exps/ 크기 점검.

- [ ] **Step 5**: Run 완료 후 검증.

```bash
ls -la exps/b0_s42/checkpoints/
du -sh exps/b0_s42/
```

Expected: 8 개 `.pt` 파일 (0050000~0400000), 총 ~2 GB.

- [ ] **Step 6**: Commit log (checkpoints 는 gitignored).

```bash
git add exps/b0_s42_train.log data/disk_log_b0.csv
git commit -m "data: B0 (pure SiT, proj-coeff=0) 400K steps complete"
```

---

## Phase 7 — B1 Training (SiT+REPA-fixed, 400K)

**목표**: `--proj-coeff 0.5` 로 REPA-fixed 학습. 400K steps, seed 42. **예상 시간**: ~14 h.

### Task 7.1: Launch B1 in tmux

- [ ] **Step 1**: tmux 새 세션 (B0 끝난 후에만).

```bash
tmux new -s prelim_b1
```

- [ ] **Step 2**: 학습 명령 (B0 과 동일한 인자, `--exp-name` 과 `--proj-coeff` 만 변경).

```bash
source /home/famoz/miniconda3/bin/activate dl
python scripts/disk_monitor.py --min_gb 15 --interval 300 --log data/disk_log_b1.csv --exit_on_alert &
DISK_PID=$!

accelerate launch train.py \
    --exp-name b1_s42 \
    --model SiT-B/2 \
    --data-dir data/ffhq256-prelim \
    --batch-size 32 \
    --mixed-precision bf16 \
    --max-train-steps 400000 \
    --checkpointing-steps 50000 \
    --sampling-steps 9999999 \
    --proj-coeff 0.5 \
    --num-classes 1 \
    --cfg-prob 0 \
    --encoder-depth 8 \
    --enc-type dinov2-vit-b \
    --path-type linear \
    --prediction v \
    --weighting uniform \
    --seed 42 \
    --output-dir exps \
    --report-to tensorboard \
    --allow-tf32 \
    2>&1 | tee exps/b1_s42_train.log

kill $DISK_PID 2>/dev/null || true
```

- [ ] **Step 3**: Detach + 일일 monitoring (Task 6.1 Step 3-4 와 동일 방식).

- [ ] **Step 4**: Run 완료 후 검증.

```bash
ls -la exps/b1_s42/checkpoints/
du -sh exps/b1_s42/
du -sh exps/
```

Expected: B1 에도 8 개 checkpoint, 총 exps/ 크기 ~4 GB (두 run 합친).

- [ ] **Step 5**: Commit log.

```bash
git add exps/b1_s42_train.log data/disk_log_b1.csv
git commit -m "data: B1 (SiT+REPA-fixed, proj-coeff=0.5) 400K steps complete"
```

---

## Phase 8 — Evaluation (16 checkpoints × all metrics)

**목표**: 각 (branch, step) 조합에서 5K samples 생성 → FID + 3-stat sharpness W1 + P/R + preview. **예상 시간**: ~4 h (16 eval × ~15 min).

### Task 8.1: Write eval_prelim.py

**Files:**
- Create: `scripts/eval_prelim.py`

- [ ] **Step 1**: Script 작성.

`scripts/eval_prelim.py`:
```python
"""Evaluate a single (branch, seed, step) checkpoint — preliminary version.

Loads EMA bf16 checkpoint, generates 5K samples via euler_sampler, computes:
- FID (clean-fid vs precomputed reference)
- 3-stat sharpness Wasserstein (lapvar, hffreq, sobel)
- Precision/Recall (Kynkäänniemi, VGG-16 features)
- Saves 16 preview samples

Usage:
    python scripts/eval_prelim.py --ckpt exps/b0_s42/checkpoints/0200000.pt \\
        --out results/b0_s42_0200000.json \\
        --preview_dir exps/b0_s42/eval_0200000/preview
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy.stats import wasserstein_distance
from diffusers import AutoencoderKL
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance

sys.path.insert(0, ".")
from models.sit import SiT_models
from samplers import euler_sampler

LATENTS_SCALE = 0.18215

# --------- Metrics: sharpness Wasserstein (3 stats) ----------
def _to_gray(img_chw_uint8):
    arr = img_chw_uint8.numpy() if hasattr(img_chw_uint8, "numpy") else img_chw_uint8
    return cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

def _lapvar(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def _hffreq(g, cutoff_frac=0.5):
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float64)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = int(cutoff_frac * min(cy, cx))
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 >= r * r
    return float(mag[mask].sum() / (mag.sum() + 1e-8))

def _sobel(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())

def sharpness_wasserstein(gen_uint8, ref_uint8):
    fns = {"lapvar": _lapvar, "hffreq": _hffreq, "sobel": _sobel}
    out = {}
    for name, fn in fns.items():
        gen_vals = np.array([fn(_to_gray(im)) for im in gen_uint8])
        ref_vals = np.array([fn(_to_gray(im)) for im in ref_uint8])
        out[name] = float(wasserstein_distance(gen_vals, ref_vals))
    return out

# --------- Precision/Recall (Kynkäänniemi 2019, VGG-16) ----------
def _vgg_features(imgs_uint8, device, batch_size=32):
    import torchvision.models as M
    import torchvision.transforms as T
    vgg = M.vgg16(weights=M.VGG16_Weights.DEFAULT).features.to(device).eval()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    feats = []
    for i in range(0, len(imgs_uint8), batch_size):
        b = imgs_uint8[i:i+batch_size].to(device).float() / 255.0
        b = norm(b)
        with torch.no_grad():
            f = vgg(b).mean(dim=[2, 3])
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)

def _knn_dist(x, k):
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    return d.topk(k, largest=False).values[:, -1]

def precision_recall(gen_uint8, ref_uint8, device, k=3):
    g = _vgg_features(gen_uint8, device)
    r = _vgg_features(ref_uint8, device)
    r_radii = _knn_dist(r, k)
    g_radii = _knn_dist(g, k)
    d_g_to_r = torch.cdist(g, r)
    prec = (d_g_to_r <= r_radii.unsqueeze(0)).any(dim=1).float().mean().item()
    d_r_to_g = torch.cdist(r, g)
    rec = (d_r_to_g <= g_radii.unsqueeze(0)).any(dim=1).float().mean().item()
    return {"precision": float(prec), "recall": float(rec)}

# --------- Model + sample generation ----------
def load_ema_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ema_state = ckpt["ema"]
    # Build SiT-B/2 with same config as training
    model = SiT_models["SiT-B/2"](
        input_size=32,
        num_classes=1,
        use_cfg=False,
        z_dims=[768],      # DINOv2 ViT-B dim
        encoder_depth=8,
        fused_attn=True,
        qk_norm=False,
    ).to(device).eval()
    model.load_state_dict({k: v.to(torch.float32) for k, v in ema_state.items()})
    return model

def generate_samples(model, n, batch_size, device, seed):
    imgs = []
    n_left = n
    idx = 0
    while n_left > 0:
        bs = min(batch_size, n_left)
        g = torch.Generator(device=device).manual_seed(seed + idx)
        xT = torch.randn(bs, 4, 32, 32, device=device, generator=g)
        y = torch.zeros(bs, dtype=torch.long, device=device)
        with torch.no_grad():
            samples = euler_sampler(
                model, xT, y,
                num_steps=50, heun=False,
                cfg_scale=1.0,           # no CFG
                guidance_low=0.0, guidance_high=1.0,
                path_type="linear",
            ).to(torch.float32)
        imgs.append(samples.cpu())
        n_left -= bs
        idx += bs
    return torch.cat(imgs, dim=0)     # (n, 4, 32, 32) latents

def decode_latents(latents, vae, device, batch_size=32):
    out = []
    for i in range(0, len(latents), batch_size):
        b = latents[i:i+batch_size].to(device)
        with torch.no_grad():
            img = vae.decode(b / LATENTS_SCALE).sample       # (B, 3, 256, 256)
        img_uint8 = ((img.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()
        out.append(img_uint8)
    return torch.cat(out, dim=0)

def load_ref_images(eval_list, latents_dir, vae, device, n=5000):
    """Decode eval-split latents (mean only) to uint8 images (in memory)."""
    idxs = open(eval_list).read().splitlines()[:n]
    batch, out = [], []
    BS = 32
    for i, idx in enumerate(idxs):
        m = np.load(f"{latents_dir}/{idx}.npy")                # (1, 8, 32, 32)
        mean = torch.from_numpy(m)[0, :4].float()              # (4, 32, 32)
        batch.append(mean)
        if len(batch) == BS or i == len(idxs) - 1:
            zs = torch.stack(batch).to(device)
            with torch.no_grad():
                x = vae.decode(zs / LATENTS_SCALE).sample
            out.append(((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu())
            batch = []
    return torch.cat(out, dim=0)

def compute_fid(samples_uint8, ref_mu, ref_sigma, device, batch_size=64):
    ext = build_feature_extractor("clean", device)
    feats = []
    for i in range(0, len(samples_uint8), batch_size):
        b = samples_uint8[i:i+batch_size].to(device)
        with torch.no_grad():
            feats.append(ext(b).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return float(frechet_distance(mu, sigma, ref_mu, ref_sigma))

def main(args):
    t0 = time.time()
    device = "cuda"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

    print(f"Loading model from {args.ckpt}")
    model = load_ema_model(args.ckpt, device)

    print(f"Generating {args.n_samples} samples ...")
    sample_latents = generate_samples(model, args.n_samples, args.batch_size, device, args.sample_seed)
    print("Decoding samples via SD-VAE ...")
    sample_uint8 = decode_latents(sample_latents, vae, device)

    # Save preview (first 16)
    Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
    from torchvision.utils import save_image
    for i in range(min(16, len(sample_uint8))):
        save_image(sample_uint8[i].float() / 255.0,
                   f"{args.preview_dir}/sample_{i:02d}.png")

    print("Loading reference images ...")
    ref_uint8 = load_ref_images("data/ffhq256_eval.txt", "data/ffhq256-prelim/vae-sd", vae, device)

    print("Computing FID ...")
    ref = np.load("data/fid_ref_ffhq256.npz")
    fid = compute_fid(sample_uint8, ref["mu"], ref["sigma"], device)

    print("Computing sharpness Wasserstein ...")
    sharp = sharpness_wasserstein(sample_uint8, ref_uint8)

    print("Computing Precision/Recall ...")
    pr = precision_recall(sample_uint8, ref_uint8, device, k=3)

    out = {
        "ckpt": args.ckpt,
        "n_samples": args.n_samples,
        "fid": fid,
        "sharpness": sharp,
        "precision_recall": pr,
        "wallclock_sec": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("---")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preview_dir", required=True)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--sample_seed", type=int, default=20260415)
    main(p.parse_args())
```

- [ ] **Step 2**: Syntax check.

```bash
python -c "import ast; ast.parse(open('scripts/eval_prelim.py').read()); print('ok')"
```

- [ ] **Step 3**: Commit.

```bash
git add scripts/eval_prelim.py
git commit -m "feat: eval_prelim.py (FID + sharpness W1 + P/R + preview)"
```

### Task 8.2: Smoke test eval on one checkpoint

- [ ] **Step 1**: Run eval on B0 first checkpoint.

```bash
source /home/famoz/miniconda3/bin/activate dl
python scripts/eval_prelim.py \
    --ckpt exps/b0_s42/checkpoints/0050000.pt \
    --out results/b0_s42_0050000.json \
    --preview_dir exps/b0_s42/eval_0050000/preview \
    --n_samples 1000 2>&1 | tail -25
```

⚠ Smoke 용으로 `--n_samples 1000` 으로 줄임. Full eval 은 `--n_samples 5000`.

Expected: 완료 시 `results/b0_s42_0050000.json` 에 FID, sharpness, P/R 출력. Pipeline 전체 동작 확인.

⚠ **트러블슈팅**:
- **model init mismatch**: `SiT_models["SiT-B/2"]` 의 생성 인자가 training 시와 달라 ema state load 실패 가능. `load_ema_model` 의 args 를 train.py 에서 실제 호출한 값으로 정확히 맞춤. 만약 mismatch 면 훈련 시 사용된 파라미터 (`num_classes`, `use_cfg`, `z_dims`, `encoder_depth`, `fused_attn`, `qk_norm`) 를 train.py 의 `args.json` (`exps/b0_s42/args.json`) 에서 읽어 사용.
- **CUDA OOM during sampling**: `--batch_size 16` 으로.
- **euler_sampler signature mismatch**: `samplers.py` 가 예상과 다르면 signature 조정.

- [ ] **Step 2**: Preview 확인.

```bash
ls exps/b0_s42/eval_0050000/preview/
# 이미지 뷰어로 몇 개 열어서 얼굴 인식 가능한지 확인
```

Expected: 16 장 PNG, 50K step 에서는 대략 얼굴 형상이 보이기 시작 (흐릿할 수 있음).

- [ ] **Step 3**: JSON 검증.

```bash
cat results/b0_s42_0050000.json
```

Expected: FID (큰 값, 100-300), sharpness dict 3 entry, precision/recall 값, wallclock_sec.

- [ ] **Step 4**: 50K 에서 sanity 통과하면, smoke 결과 삭제 후 Full eval 로 진행.

```bash
rm -f results/b0_s42_0050000.json
rm -rf exps/b0_s42/eval_0050000
```

### Task 8.3: Full eval on all 16 checkpoints

**Files:**
- Create: `scripts/run_eval_prelim.sh`

- [ ] **Step 1**: Sweep script.

`scripts/run_eval_prelim.sh`:
```bash
#!/bin/bash
set -euo pipefail
source /home/famoz/miniconda3/bin/activate dl

mkdir -p results
START=$(date +%s)
for B in b0 b1; do
    for STEP in 0050000 0100000 0150000 0200000 0250000 0300000 0350000 0400000; do
        CKPT="exps/${B}_s42/checkpoints/${STEP}.pt"
        OUT="results/${B}_s42_${STEP}.json"
        if [ -f "$OUT" ]; then
            echo "[skip] $OUT"
            continue
        fi
        if [ ! -f "$CKPT" ]; then
            echo "[!!] missing $CKPT"
            continue
        fi
        echo "[$(date +%H:%M:%S)] $B s42 step ${STEP}"
        python scripts/eval_prelim.py \
            --ckpt "$CKPT" \
            --out "$OUT" \
            --preview_dir "exps/${B}_s42/eval_${STEP}/preview" \
            --n_samples 5000
    done
done
END=$(date +%s)
HOURS=$(echo "scale=2; ($END - $START) / 3600" | bc)
echo "✓ 16 evals complete in ${HOURS} hours"
```

```bash
chmod +x scripts/run_eval_prelim.sh
```

- [ ] **Step 2**: Run (tmux 추천).

```bash
tmux new -s prelim_eval
bash scripts/run_eval_prelim.sh 2>&1 | tee results/eval_all.log
# Ctrl+B, D
```

Expected: 약 16 × 15 min ≈ 4 GPU-hours.

- [ ] **Step 3**: 완료 후 검증.

```bash
ls results/ | grep -c json    # expect 16
```

- [ ] **Step 4**: Commit.

```bash
git add scripts/run_eval_prelim.sh results/eval_all.log
git commit -m "ops: eval sweep (16 checkpoints × full metrics)"
```

### Task 8.4: Aggregate to results_prelim.csv

**Files:**
- Create: `scripts/aggregate_prelim.py`
- Create: `results_prelim.csv`

- [ ] **Step 1**: Script.

`scripts/aggregate_prelim.py`:
```python
"""Aggregate prelim eval JSONs into results_prelim.csv."""
import csv, json, re
from pathlib import Path

rows = []
for f in sorted(Path("results").glob("*.json")):
    m = re.match(r"^(b[01])_s(\d+)_(\d+)\.json$", f.name)
    if not m:
        continue
    branch, seed, step = m.group(1), int(m.group(2)), int(m.group(3))
    d = json.loads(f.read_text())
    rows.append({
        "branch": branch, "seed": seed, "step": step,
        "fid": d["fid"],
        "precision": d["precision_recall"]["precision"],
        "recall": d["precision_recall"]["recall"],
        "sharp_lapvar": d["sharpness"]["lapvar"],
        "sharp_hffreq": d["sharpness"]["hffreq"],
        "sharp_sobel":  d["sharpness"]["sobel"],
        "wallclock_sec": d["wallclock_sec"],
    })

if not rows:
    print("no eval JSONs found"); exit(1)

fields = list(rows[0].keys())
with open("results_prelim.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"✓ wrote {len(rows)} rows to results_prelim.csv")
print()
print("Summary:")
import pandas as pd
df = pd.read_csv("results_prelim.csv")
for branch in ["b0", "b1"]:
    g = df[df.branch == branch].sort_values("step")
    print(f"\n--- {branch} (pure SiT)" if branch == "b0" else f"\n--- {branch} (SiT+REPA-fixed)")
    print(g[["step", "fid", "sharp_lapvar", "sharp_hffreq", "sharp_sobel", "recall"]].to_string(index=False))
```

- [ ] **Step 2**: 실행.

```bash
python scripts/aggregate_prelim.py
cat results_prelim.csv
```

Expected: 16 rows (2 branches × 8 steps), summary table 출력.

- [ ] **Step 3**: Commit.

```bash
git add scripts/aggregate_prelim.py results_prelim.csv
git commit -m "data: aggregate 16 prelim evals to results_prelim.csv"
```

---

## Phase 9 — Analysis + Decision

**목표**: 결과를 plot + §4.1 decision matrix 적용 → `results_prelim.md` 작성. **예상 시간**: 2 h.

### Task 9.1: Plot FID vs step (B0 vs B1)

**Files:**
- Create: `scripts/plot_prelim.py`
- Create: `figures/fid_vs_step.png`

- [ ] **Step 1**: Script.

`scripts/plot_prelim.py`:
```python
"""Generate preliminary analysis figures."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BRANCH_COLORS = {"b0": "#999999", "b1": "#e41a1c"}
BRANCH_LABELS = {"b0": "B0 pure SiT", "b1": "B1 SiT+REPA-fixed"}

def plot_fid_vs_step(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for branch in ["b0", "b1"]:
        g = df[df.branch == branch].sort_values("step")
        ax.plot(g["step"], g["fid"], marker="o", lw=2,
                color=BRANCH_COLORS[branch], label=BRANCH_LABELS[branch])
    ax.set_xlabel("Training step")
    ax.set_ylabel("FID (clean-fid, 5K samples vs 5K eval ref)")
    ax.set_title("FID vs step — preliminary B0 vs B1 (n=1 seed, 400K steps)")
    ax.axvline(200000, ls="--", color="gray", alpha=0.5, label="Q1/Q2 judgment window")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def plot_sharpness_vs_step(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    stats = ["sharp_lapvar", "sharp_hffreq", "sharp_sobel"]
    titles = ["Laplacian variance W₁", "HF energy ratio W₁", "Sobel mean W₁"]
    for ax, stat, title in zip(axes, stats, titles):
        for branch in ["b0", "b1"]:
            g = df[df.branch == branch].sort_values("step")
            ax.plot(g["step"], g[stat], marker="o", lw=2,
                    color=BRANCH_COLORS[branch], label=BRANCH_LABELS[branch])
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Wasserstein distance to real")
        ax.axvline(200000, ls="--", color="gray", alpha=0.5)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Sharpness Wasserstein distance vs step — lower = closer to real distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def plot_sharpness_bars_final(df, out):
    final = df[df.step == 400000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    stats = ["sharp_lapvar", "sharp_hffreq", "sharp_sobel"]
    titles = ["Laplacian variance W₁", "HF energy ratio W₁", "Sobel mean W₁"]
    for ax, stat, title in zip(axes, stats, titles):
        branches = ["b0", "b1"]
        vals = [float(final[final.branch == b][stat].iloc[0]) for b in branches]
        colors = [BRANCH_COLORS[b] for b in branches]
        ax.bar([BRANCH_LABELS[b] for b in branches], vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel("W₁ distance to real")
    fig.suptitle("Sharpness W₁ at step 400K (lower = closer to real)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def plot_recall_vs_step(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for branch in ["b0", "b1"]:
        g = df[df.branch == branch].sort_values("step")
        ax.plot(g["step"], g["recall"], marker="o", lw=2,
                color=BRANCH_COLORS[branch], label=BRANCH_LABELS[branch])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Improved Recall (Kynkäänniemi 2019)")
    ax.set_title("Recall vs step — higher = more diverse")
    ax.axvline(200000, ls="--", color="gray", alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def main(args):
    df = pd.read_csv(args.results)
    Path("figures").mkdir(exist_ok=True)
    plot_fid_vs_step(df, "figures/fid_vs_step.png")
    plot_sharpness_vs_step(df, "figures/sharpness_vs_step.png")
    plot_sharpness_bars_final(df, "figures/sharpness_bars.png")
    plot_recall_vs_step(df, "figures/recall_vs_step.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results_prelim.csv")
    main(p.parse_args())
```

- [ ] **Step 2**: 실행.

```bash
mkdir -p figures
python scripts/plot_prelim.py
ls figures/
```

Expected: 4 개 PNG (`fid_vs_step`, `sharpness_vs_step`, `sharpness_bars`, `recall_vs_step`).

- [ ] **Step 3**: 시각 검사. 각 figure 를 열어서 확인:
- FID 곡선: 두 branch 모두 감소해야 정상. B1 이 B0 보다 항상 아래면 Q1 확실. 만약 B1 이 B0 위면 Q1 실패.
- Sharpness 곡선: B0 이 B1 보다 낮으면 (즉 real 분포에 가까우면) Q2 signature. 
- Recall 곡선: B0 이 B1 보다 높으면 B0 이 더 diverse.

- [ ] **Step 4**: Commit.

```bash
git add scripts/plot_prelim.py figures/
git commit -m "feat: preliminary figures (fid, sharpness, recall trajectories + final bars)"
```

### Task 9.2: Apply §4.2 decision rule

**Files:**
- Create: `scripts/decide_prelim.py`

- [ ] **Step 1**: Script that mechanically applies §4.2 thresholds.

`scripts/decide_prelim.py`:
```python
"""Mechanically apply prelim-spec §4.2 decision rules to results_prelim.csv.
Outputs boolean q1_pass, q2_pass + scenario label.
"""
import json
import pandas as pd

Q1_THRESH_REL = 0.05   # B1 FID ≥ 5% lower than B0 FID
Q2_THRESH_REL = 0.15   # B0 sharpness W1 ≥ 15% lower than B1 sharpness W1
CHECKPOINTS_AFTER = 200000  # only 200K+ counted

def main():
    df = pd.read_csv("results_prelim.csv")
    late = df[df.step >= CHECKPOINTS_AFTER]
    steps = sorted(late.step.unique())

    q1_passes = []
    for s in steps:
        fid_b0 = float(late[(late.branch == "b0") & (late.step == s)]["fid"].iloc[0])
        fid_b1 = float(late[(late.branch == "b1") & (late.step == s)]["fid"].iloc[0])
        rel = (fid_b0 - fid_b1) / fid_b0
        q1_passes.append({"step": s, "fid_b0": fid_b0, "fid_b1": fid_b1,
                          "rel_improvement": rel, "pass": rel >= Q1_THRESH_REL})
    q1_pass_count = sum(1 for x in q1_passes if x["pass"])
    q1_pass = q1_pass_count >= 2

    q2_details = {}
    q2_pass_count = 0
    for stat in ["sharp_lapvar", "sharp_hffreq", "sharp_sobel"]:
        stat_passes = []
        for s in steps:
            w1_b0 = float(late[(late.branch == "b0") & (late.step == s)][stat].iloc[0])
            w1_b1 = float(late[(late.branch == "b1") & (late.step == s)][stat].iloc[0])
            if w1_b1 == 0:
                continue
            rel = (w1_b1 - w1_b0) / w1_b1
            stat_passes.append({"step": s, "w1_b0": w1_b0, "w1_b1": w1_b1,
                                "rel_gap": rel, "pass": rel >= Q2_THRESH_REL})
        q2_details[stat] = stat_passes
        if sum(1 for x in stat_passes if x["pass"]) >= 2:
            q2_pass_count += 1
    q2_pass = q2_pass_count >= 2

    if q1_pass and q2_pass:
        scenario = "POSITIVE: proceed full study"
    elif q1_pass and not q2_pass:
        scenario = "CODEBASE_OK_HYPOTHESIS_WEAK: re-evaluate full study scope"
    elif not q1_pass:
        scenario = "CODEBASE_PROBLEM: debug before full study"
    else:
        scenario = "UNKNOWN"

    out = {
        "q1_pass": bool(q1_pass), "q2_pass": bool(q2_pass),
        "scenario": scenario,
        "q1_details": q1_passes,
        "q2_details_per_stat": q2_details,
        "q2_pass_stats_count": q2_pass_count,
        "thresholds": {"q1_rel": Q1_THRESH_REL, "q2_rel": Q2_THRESH_REL,
                       "checkpoints_after": CHECKPOINTS_AFTER,
                       "q1_min_pass_checkpoints": 2,
                       "q2_min_pass_stats": 2,
                       "q2_min_pass_checkpoints_per_stat": 2},
    }
    print(json.dumps(out, indent=2))
    open("results/prelim_decision.json", "w").write(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: 실행.

```bash
python scripts/decide_prelim.py
```

Expected: JSON 출력에 `q1_pass`, `q2_pass`, `scenario` 포함.

- [ ] **Step 3**: Commit.

```bash
git add scripts/decide_prelim.py results/prelim_decision.json
git commit -m "analysis: apply prelim-spec §4.2 decision rules"
```

### Task 9.3: Write results_prelim.md

**Files:**
- Create: `results_prelim.md`

- [ ] **Step 1**: `results_prelim.md` 작성. `results/prelim_decision.json` + `results_prelim.csv` 의 내용을 보고 다음 template 을 채움:

`results_prelim.md`:
```markdown
# Preliminary Baseline Comparison — Results

**Date**: 2026-04-X (fill in)
**Branch**: `prelim-baseline` (worktree `REPA-prelim`)
**Spec**: `docs/prelim-spec-ko.md` (unchanged since approval)
**Analyst**: <your name>

## 1. Setup confirmation

- Dataset: merkol/ffhq-256, rev f23c0e21, 70000 samples, 256×256, JPEG Q95
- Model: SiT-B/2, batch=32, bf16, 400K steps
- Branches: B0 (proj-coeff=0) and B1 (proj-coeff=0.5), seed=42 each
- Wallclock (B0): XXh, (B1): XXh
- Checkpoints: 8 per run (50K, 100K, …, 400K)
- Eval per ckpt: 5K samples, FID / sharpness W₁ (3 stats) / P/R

## 2. Q1 — REPA paper reproduction

Pre-registered threshold: B1 FID ≥ 5% lower than B0 FID at 2+ checkpoints (step ≥ 200K).

| Step | B0 FID | B1 FID | (B0-B1)/B0 | Pass? |
|---|---|---|---|---|
| 200K | ... | ... | ...% | Y/N |
| 250K | ... | ... | ...% | Y/N |
| 300K | ... | ... | ...% | Y/N |
| 350K | ... | ... | ...% | Y/N |
| 400K | ... | ... | ...% | Y/N |

**Q1 verdict**: PASS / FAIL

## 3. Q2 — Trade-off signature

Pre-registered threshold: B0 sharpness W₁ ≥ 15% lower than B1 at 2+ checkpoints, on at least 2 of 3 sharpness statistics, at step ≥ 200K.

### 3.1 Laplacian variance W₁
(same table as Q1)

### 3.2 HF energy ratio W₁
(same table)

### 3.3 Sobel mean W₁
(same table)

**Q2 verdict**: PASS / FAIL (X of 3 stats pass)

### 3.4 Recall (secondary)
| Step | B0 Recall | B1 Recall |
|---|---|---|
| 200K | ... | ... |
| ... | ... | ... |

## 4. Visual inspection

Preview samples from step 400K — 8 × 8 grid, B0 top half, B1 bottom half:

![preview](figures/preview_grid.png) (to create from exps/b{0,1}_s42/eval_0400000/preview/)

Qualitative comments:
- Sharpness: …
- Diversity: …
- Artifacts: …

## 5. Applied decision (per prelim-spec §4.1)

Q1 × Q2 combination: (Q1=PASS/FAIL, Q2=PASS/FAIL)

Scenario:
- [ ] **Positive** → Proceed full study
- [ ] **Codebase OK, hypothesis weak** → Re-evaluate full study scope
- [ ] **Codebase problem** → Debug before full study
- [ ] **Both null** → Re-evaluate hyperparameters

**Action**: (chosen action with 1-2 sentence justification)

## 6. Next steps

(concrete follow-up actions)

## 7. Known issues / caveats

(anything that might affect interpretation — e.g., B0/B1 data overlap at 5K eval samples, wallclock mismatch with pilot, etc.)
```

- [ ] **Step 2**: 모든 placeholder 를 실제 값으로 채움. `results_prelim.csv`, `results/prelim_decision.json` 에서 수치 복사.

- [ ] **Step 3**: Preview grid 생성 (수동 옵션, 혹은 아래 한 줄 스크립트).

```bash
python -c "
from PIL import Image
import os
grid_w, grid_h = 8, 8   # 8x8 grid
tile = 128              # each tile 128x128 after downsize
out = Image.new('RGB', (grid_w*tile, grid_h*tile))
b0_files = sorted(os.listdir('exps/b0_s42/eval_0400000/preview'))[:32]
b1_files = sorted(os.listdir('exps/b1_s42/eval_0400000/preview'))[:32]
for i, f in enumerate(b0_files):
    img = Image.open(f'exps/b0_s42/eval_0400000/preview/{f}').resize((tile, tile))
    row, col = i // grid_w, i % grid_w
    out.paste(img, (col*tile, row*tile))
for i, f in enumerate(b1_files):
    img = Image.open(f'exps/b1_s42/eval_0400000/preview/{f}').resize((tile, tile))
    row, col = (i // grid_w) + 4, i % grid_w
    out.paste(img, (col*tile, row*tile))
out.save('figures/preview_grid.png')
print('✓ figures/preview_grid.png')
"
```

- [ ] **Step 4**: Commit.

```bash
git add results_prelim.md figures/preview_grid.png
git commit -m "analysis: write results_prelim.md decision document + preview grid"
```

### Task 9.4: Push prelim-baseline to origin

- [ ] **Step 1**: Push.

```bash
git push -u origin prelim-baseline
```

Expected: 새 branch `prelim-baseline` 이 origin 에 생성됨.

- [ ] **Step 2**: 최종 branch 목록 확인.

```bash
git branch -a
```

Expected: `main`, `reeval`, `planning-docs`, `prelim-baseline` 4 개 local + 같은 4 개 remote tracking.

- [ ] **Step 3**: 4 개 branch 의 간단한 status 보기.

```bash
git log --oneline --all --graph --decorate -20
```

---

## Self-Review

### Spec coverage (vs prelim-spec-ko.md)

- §1 Q1 + Q2 정의 → Task 9.2 decision rule에 구현
- §2.1 B0/B1 정의 → Phase 6 / 7 (proj-coeff 0 vs 0.5)
- §2.2 하이퍼파라미터 → Phase 6/7 의 accelerate launch 인자 (batch=32, bf16, SiT-B/2, AdamW 기본값, grad_clip=1.0)
- §2.3 Dataset → Phase 1-3 (merkol, JPEG, SD-VAE moments, split)
- §3.1 8 eval checkpoints (50K 간격) → Phase 6/7 의 `--checkpointing-steps 50000` + Phase 8 sweep
- §3.2 FID + sharpness W₁ (3 stats) + P/R + preview → Task 8.1 eval_prelim.py 구현
- §3.3 5000 samples, 50-step DDPM, no CFG → eval_prelim.py 구현
- §3.4 Skip IS / FD-DINOv2 / human eval / interpolation → plan 에 아예 없음 ✓
- §4.1 결과 시나리오 × 행동 매트릭스 → Task 9.2 decide_prelim.py 출력 scenario
- §4.2 정량 기준 (5%, 15%, 200K 이후, 3 중 2) → Task 9.2 decide_prelim.py 의 상수
- §4.3 수동 시각 검사 → Task 9.3 의 preview grid + 시각 섹션
- §4.4 결정 문서화 → Task 9.3 `results_prelim.md`
- §5 Compute budget → Phase 5 wallclock pilot 으로 검증
- §6 File structure → 본 plan의 File Structure 섹션
- §7 Workflow → Phase 0-9 전체
- §8 YAGNI → plan 에 pre-registration / power analysis / multi-seed / 5-branch / human eval / interp 모두 없음 ✓
- §9 Full study 와의 관계 → plan 마지막 decision 에서 자동 반영 ( scenario 에 따라)
- §10 Deliverables → 본 plan 의 Phase 9 Task 9.3 commit 내용과 일치
- §11 성공 기준 → Task 9.3 체크리스트 작동

### Placeholder scan

- `train.py` patch line number 는 "약 lines 328-339" 로 비슷한 표현. 이건 spec 위반 아님 — 실제 patch 는 `grep -n "checkpointing_steps"` 로 찾으라고 step 에서 명시.
- `results_prelim.md` 의 `...` 은 템플릿이며 사용자가 채우는 부분. 이건 의도된 템플릿.
- Eval script 의 SiT 생성자 args (`num_classes=1, use_cfg=False, z_dims=[768], encoder_depth=8, ...`) — smoke test step 에서 args.json 에서 읽는 fallback 이 있음.
- 그 외 placeholder 없음.

### Type consistency

- Latent shape: `(1, 8, 32, 32)` — Task 2.1, 2.2, 3.2, 8.1 모두 일관
- Latent scaling: `0.18215` — Task 3.2, 8.1 모두 일관
- Sample uint8 shape: `(N, 3, 256, 256)` — Task 8.1 내 일관
- Branch 이름: `b0`, `b1` — Task 6, 7, 8, 9 모두 일관
- Metric keys: `fid`, `sharpness.{lapvar,hffreq,sobel}`, `precision_recall.{precision,recall}` — Task 8.1, 8.4, 9.1, 9.2 일관
- `--proj-coeff 0` (B0) vs `--proj-coeff 0.5` (B1) — Phase 6 vs 7 정확히 구분
- Eval checkpoint 목록: {0050000, 0100000, ..., 0400000} — 8 개, Task 8.3 sweep script 와 §3.1 일치

### Scope check

- Day 0-5 의 compute work 만 포함. Paper writing / 논문 섹션은 의도적 배제.
- Single implementation plan 가능 (1 worktree, 1 branch)

---

## Execution Handoff

Plan complete and saved to `docs/prelim-plan-ko.md`. 두 가지 실행 옵션:

**1. Subagent-Driven (recommended)** — Task 별로 fresh subagent 를 dispatch, 각 task 완료 시 review, 빠른 iteration

**2. Inline Execution** — 이 session 에서 `superpowers:executing-plans` skill 로 batch 실행, phase 별 checkpoint 마다 review

어떤 방식으로 진행하시겠어요?
