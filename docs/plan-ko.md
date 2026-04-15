# REPA 재평가 연구 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FFHQ-256 위에서 5개 REPA 계수 schedule을 factorial 비교하여, 고정 계수 REPA가 수렴 속도와 sample sharpness/diversity 사이에 trade-off를 가지는지 사전 등록된 diagnostic study로 검증.

**Architecture:** `sihyun-yu/REPA` fork에 대해 (1) FFHQ-256 latent dataset loader 교체, (2) 5개 schedule을 지원하는 coefficient scheduler hook 추가, (3) EMA-only bf16 checkpoint 정책 적용, (4) §4.4 streaming-first storage 정책 적용. Pre-registration commit 후 15 runs (5 branches × 3 seeds × 100K steps) 순차 실행, 자동 metric + 2AFC human eval + interpolation jaggedness 평가, §2.3 4-way reporting으로 결론.

**Tech Stack:** PyTorch, accelerate (mixed precision), diffusers (SD-VAE), HuggingFace `datasets` (streaming), clean-fid, torch-fidelity, transformers (DINOv2), scipy (Wasserstein, t-test), Flask 또는 static HTML+JS (human eval), matplotlib, pandas.

**Spec 참조:** 이 plan의 모든 §X 참조는 `docs/spec-ko.md` 의 섹션. 본 plan은 **Day 1-21 (Phase 0-10)** 의 implementation work만 다룬다. Day 22-28 (paper writing) 은 별도.

**환경 가정:** 단일 RTX 4080 Super 16GB, 가용 디스크 < 90 GB (§4.4 storage budget로 ~30 GB 영구 사용 보장), Linux/bash, Python 3.10+.

---

## File Structure

새 저장소 (`repa-reeval/`) 의 최종 file structure. 기존 REPA fork에 추가/수정되는 파일:

### 신규 (이 plan으로 만드는 파일)

**Schedules / branches:**
- `train/branches.py` — 5개 coefficient schedule 함수 + branch metadata

**Dataset:**
- `train/data/ffhq256_latents.py` — bf16 latent dataset loader
- `data/MANIFEST.json` — content-addressed manifest
- `data/ffhq256_train.txt`, `data/ffhq256_eval.txt` — split index lists
- `data/fid_ref_ffhq256.npz` — clean-fid reference statistics
- `data/fd_dinov2_ref_ffhq256.pt` — DINOv2 feature mean/cov
- `data/sd_vae_recon_sanity.json` — Day 1 PSNR/LPIPS check

**Evaluation modules:**
- `train/eval/sample.py` — sample generation (EMA, batched, in-memory)
- `train/eval/sharpness.py` — 3-statistic sharpness Wasserstein
- `train/eval/fd_dinov2.py` — DINOv2 feature distance
- `train/eval/precision_recall.py` — Kynkäänniemi P/R
- `train/eval/auc.py` — FID-vs-step trapezoid AUC
- `train/eval/jaggedness.py` — interpolation jaggedness
- `train/eval/stats.py` — §2.3 4-way reporting

**Scripts:**
- `scripts/sd_vae_sanity.py` — Day 1 reconstruction sanity check
- `scripts/precompute_latents.py` — streaming SD-VAE encode
- `scripts/precompute_fid_ref.py` — FID reference 생성
- `scripts/precompute_dino_ref.py` — DINOv2 reference 생성
- `scripts/wallclock_pilot.py` — Day 2 wallclock benchmark
- `scripts/train.py` — REPA fork에 대한 wrapper (또는 fork train.py 직접 수정)
- `scripts/eval.py` — 단일 (run, checkpoint) evaluation harness
- `scripts/aggregate.py` — 모든 eval JSON → results.csv
- `scripts/run_pilot.sh` — pilot sweep (5 × 1 × 20K)
- `scripts/run_main.sh` — main sweep (5 × 3 × 100K, sequential)
- `scripts/disk_monitor.py` — 학습 중 디스크 사용량 alert
- `scripts/generate_eval_pairs.py` — matched-noise image pair generator
- `scripts/human_eval_server.py` — Flask 기반 2AFC platform
- `scripts/interpolation_study.py` — slerp + jaggedness
- `scripts/power_analysis.py` — Cohen's d minimum detectable effect
- `scripts/plots.py` — 모든 publication figure 생성

**Tests:**
- `tests/test_branches.py` — schedule 함수 unit tests
- `tests/test_dataset.py` — latent loader tests
- `tests/test_metrics.py` — sharpness/AUC/jaggedness 테스트
- `tests/test_stats.py` — 4-way reporting 테스트

**Documentation:**
- `docs/prereg.md` — pre-registration (frozen, `prereg-v1` 태그)
- `docs/BASELINE.md` — exact baseline configuration
- `docs/EVAL_PROTOCOL.md` — full evaluation methodology
- `README.md` — reproduction instructions

### 수정 대상 (REPA fork의 기존 파일)

Phase 2 Task 2.1 (inventory) 에서 정확한 경로 확인. 예상 수정 대상:
- REPA의 main training script (loss 계산 함수, dataset 인스턴스화, checkpoint 저장 함수)
- REPA의 SiT model wrapper (class conditioning 제거 위치)
- REPA의 sampling/inference utility (10K sample 생성 시 호출)

### 영구 저장 안 됨

- `data/ffhq256_latents/{split}/{idx}.pt` — Phase 1에서 streaming으로 생성, manifest로 추적, `.gitignore` 에 등록
- `exps/{branch}_{seed}/checkpoints/ema_{step}.pt` — 학습 중 생성 (§4.4 EMA-only bf16)
- `exps/{branch}_{seed}/eval_{step}/preview/*.png` — 시각화용 16-64장만

---

## Phase 0 — Repo & Environment Setup (Day 1)

### Task 0.1: REPA fork & clone, working branch

**Files:**
- Create: `repa-reeval/` (working directory)

- [ ] **Step 1**: GitHub에서 `sihyun-yu/REPA` 를 본인 계정으로 fork (manual web action).

- [ ] **Step 2**: Local clone + upstream remote.

```bash
cd ~/projects/dl
git clone https://github.com/<your-username>/REPA.git repa-reeval
cd repa-reeval
git remote add upstream https://github.com/sihyun-yu/REPA.git
git fetch upstream
```

- [ ] **Step 3**: Working branch 생성.

```bash
git checkout -b reeval
```

- [ ] **Step 4**: 작업 디렉토리 scaffold.

```bash
mkdir -p docs scripts tests data exps train/data train/eval
touch docs/.gitkeep scripts/.gitkeep tests/.gitkeep train/data/.gitkeep train/eval/.gitkeep
echo "data/ffhq256_latents/" >> .gitignore
echo "exps/" >> .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
git add docs scripts tests train .gitignore
git commit -m "scaffold: empty dirs and gitignore for reeval workflow"
```

### Task 0.2: Python 환경

**Files:**
- Create: `requirements-reeval.txt`

- [ ] **Step 1**: 기존 REPA 의 dependency 명세 확인.

```bash
cat requirements.txt 2>/dev/null || ls *.toml setup.py 2>/dev/null || echo "no spec found"
```

- [ ] **Step 2**: 본 연구 추가 의존성 정리.

`requirements-reeval.txt`:
```
clean-fid==0.1.35
torch-fidelity==0.3.0
datasets>=2.18.0
diffusers>=0.27.0
transformers>=4.40.0
huggingface_hub>=0.22.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
flask>=3.0.0
pyyaml
scikit-image
opencv-python-headless
scipy>=1.11
lpips
pytest>=8.0
```

- [ ] **Step 3**: venv + 설치.

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install -r requirements-reeval.txt
```

- [ ] **Step 4**: GPU 확인.

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0)); print('mem:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"
```

Expected:
```
cuda: True
device: NVIDIA GeForce RTX 4080 Super
mem: 16.0 GB
```

- [ ] **Step 5**: Commit.

```bash
git add requirements-reeval.txt
git commit -m "env: pin reeval-specific dependencies"
```

### Task 0.3: 디스크 가용량 측정 + 자동 alert script

§4.4 storage budget의 self-enforcing을 위한 disk monitor.

**Files:**
- Create: `scripts/disk_monitor.py`

- [ ] **Step 1**: 현재 디스크 가용량 측정.

```bash
df -h ~/projects/dl | tee data/disk_baseline.txt
```

가용량을 기록해두고, plan의 §4.4 ~30 GB 할당이 정말 들어오는지 향후 비교용.

- [ ] **Step 2**: Disk monitor script.

`scripts/disk_monitor.py`:
```python
"""Polls disk free space, logs to data/disk_log.csv, exits non-zero if below threshold."""
import argparse, csv, datetime, shutil, sys, time

def main(args):
    with open(args.log, "a") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp", "free_gb", "used_gb", "alert"])
        while True:
            total, used, free = shutil.disk_usage(args.path)
            free_gb = free / 1e9
            used_gb = used / 1e9
            alert = free_gb < args.min_gb
            w.writerow([datetime.datetime.now().isoformat(), f"{free_gb:.2f}", f"{used_gb:.2f}", int(alert)])
            f.flush()
            if alert:
                print(f"⚠ DISK ALERT: free {free_gb:.1f} GB < threshold {args.min_gb} GB", file=sys.stderr)
                if args.exit_on_alert:
                    sys.exit(1)
            if args.once: break
            time.sleep(args.interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=".")
    p.add_argument("--log", default="data/disk_log.csv")
    p.add_argument("--min_gb", type=float, default=15.0)
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--exit_on_alert", action="store_true")
    main(p.parse_args())
```

- [ ] **Step 3**: 한번 실행해서 정상 동작 + baseline 기록.

```bash
python scripts/disk_monitor.py --once
cat data/disk_log.csv
```

- [ ] **Step 4**: Commit.

```bash
git add scripts/disk_monitor.py data/disk_baseline.txt
git commit -m "ops: disk monitor with auto-alert (per §4.4 storage policy)"
```

### Task 0.4: SD-VAE FFHQ 재구성 sanity check (script만)

§10 risk "FFHQ-SD-VAE 불일치" 의 mitigation. Streaming dataset이 아직 준비되지 않았으므로 script만 작성, 실행은 Task 1.1 완료 후.

**Files:**
- Create: `scripts/sd_vae_sanity.py`

- [ ] **Step 1**: Sanity script 작성.

`scripts/sd_vae_sanity.py`:
```python
"""SD-VAE FFHQ-256 reconstruction sanity check.

Streams N FFHQ-256 images from HF, encodes/decodes via stabilityai/sd-vae-ft-mse,
reports PSNR + LPIPS. Hard-fails if PSNR < 28 dB or LPIPS > 0.10.
"""
import argparse, io, json, sys
import torch, lpips
from datasets import load_dataset
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

# 28 dB matches SD-VAE-ft-mse operating range on face/natural images
# (27-30 dB; original 30 dB threshold was a round-number heuristic).
PSNR_THRESHOLD = 28.0
LPIPS_THRESHOLD = 0.10

def psnr(x, y, eps=1e-10):
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()

def main(args):
    device = "cuda"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    lp = lpips.LPIPS(net="alex").to(device).eval()

    ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
    tx = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    imgs = []
    for i, sample in enumerate(ds):
        if i >= args.n: break
        if "image" in sample:
            img = sample["image"].convert("RGB")
        else:
            key = next(k for k, v in sample.items() if isinstance(v, (bytes, dict)))
            img = Image.open(io.BytesIO(sample[key]) if isinstance(sample[key], bytes) else sample[key]["bytes"]).convert("RGB")
        imgs.append(tx(img))
    x = torch.stack(imgs).to(device)

    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
        x_hat = vae.decode(z / vae.config.scaling_factor).sample

    x_01 = (x.clamp(-1, 1) + 1) / 2
    x_hat_01 = (x_hat.clamp(-1, 1) + 1) / 2

    out = {
        "n_samples": int(args.n),
        "hf_dataset": args.hf_dataset,
        "psnr_db": float(psnr(x_01, x_hat_01)),
        "lpips_alex": float(lp(x.clamp(-1,1), x_hat.clamp(-1,1)).mean().item()),
        "thresholds": {"psnr_db_min": PSNR_THRESHOLD, "lpips_max": LPIPS_THRESHOLD},
    }
    print(json.dumps(out, indent=2))
    with open("data/sd_vae_recon_sanity.json", "w") as f:
        json.dump(out, f, indent=2)

    failed = []
    if out["psnr_db"] < PSNR_THRESHOLD:
        failed.append(f"PSNR {out['psnr_db']:.2f} dB < {PSNR_THRESHOLD}")
    if out["lpips_alex"] > LPIPS_THRESHOLD:
        failed.append(f"LPIPS {out['lpips_alex']:.4f} > {LPIPS_THRESHOLD}")
    if failed:
        print("✗ SANITY FAILED:", "; ".join(failed), file=sys.stderr)
        sys.exit(1)
    print("✓ SD-VAE FFHQ reconstruction sanity check PASSED")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dataset", required=True, help="HF dataset id discovered in Task 1.1")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=64)
    main(p.parse_args())
```

- [ ] **Step 2**: Commit script.

```bash
git add scripts/sd_vae_sanity.py
git commit -m "feat: SD-VAE FFHQ reconstruction sanity check (run after Task 1.1)"
```

---

## Phase 1 — Data Pipeline (Day 2-3)

### Task 1.1: HF Hub FFHQ-256 mirror discovery

**Goal:** 신뢰 가능한 HF mirror 1개를 선택하고 dataset id, revision hash, sample 수를 기록.

**Files:**
- Create: `data/MANIFEST.json` (initial)

- [ ] **Step 1**: HF Hub 검색 (web 또는 `huggingface_hub` API).

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
results = api.list_datasets(search='ffhq', limit=20)
for r in results:
    print(r.id)
"
```

- [ ] **Step 2**: 후보 dataset 1-2개를 streaming으로 1 sample 받아 schema 확인.

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('<candidate-id>', split='train', streaming=True)
sample = next(iter(ds))
print('keys:', list(sample.keys()))
img = sample.get('image') or sample.get('png')
print('type:', type(img).__name__)
" 2>&1 | head -20
```

- [ ] **Step 3**: 선택 기준:
  - 256×256 또는 더 큰 해상도 (resize 가능)
  - PNG 또는 PIL Image 형식
  - 최소 60K sample (FFHQ는 70K)
  - revision hash 안정적 (last_modified 변경 적음)

선택한 mirror id, revision hash, license를 `data/MANIFEST.json` 에 기록:

```json
{
  "dataset": {
    "hf_id": "<chosen-id>",
    "revision": "<git-sha-from-hub>",
    "split_train": "train",
    "expected_total": 70000,
    "license": "<license>",
    "discovered_at": "2026-04-15"
  },
  "splits": {},
  "latents": {}
}
```

- [ ] **Step 4**: SD-VAE sanity check 실행 (Task 0.4 script 재사용).

```bash
python scripts/sd_vae_sanity.py --hf_dataset <chosen-id> --split train --n 64
```

Expected output: `✓ SD-VAE FFHQ reconstruction sanity check PASSED` (PSNR ≥ 28, LPIPS ≤ 0.10).

- [ ] **Step 5**: Commit.

```bash
git add data/MANIFEST.json data/sd_vae_recon_sanity.json
git commit -m "data: choose HF FFHQ-256 mirror, verify SD-VAE reconstruction"
```

### Task 1.2: Streaming SD-VAE latent precomputation script

**Files:**
- Create: `scripts/precompute_latents.py`

- [ ] **Step 1**: Failing test 작성.

`tests/test_dataset.py`:
```python
import os
import torch
import pytest

LATENT_DIR = "data/ffhq256_latents/train"

def test_latent_shape_and_dtype():
    files = sorted(f for f in os.listdir(LATENT_DIR) if f.endswith(".pt"))
    assert len(files) > 0, "no latents present yet"
    latent = torch.load(os.path.join(LATENT_DIR, files[0]))
    assert latent.shape == (4, 32, 32), f"unexpected shape {tuple(latent.shape)}"
    assert latent.dtype == torch.bfloat16, f"unexpected dtype {latent.dtype}"
```

- [ ] **Step 2**: Run test, expect FAIL (no latents yet).

```bash
pytest tests/test_dataset.py::test_latent_shape_and_dtype -v
```

Expected: FAIL with `FileNotFoundError` or `assert len(files) > 0`.

- [ ] **Step 3**: Implement precompute script.

`scripts/precompute_latents.py`:
```python
"""Streams FFHQ-256 from HF Hub, encodes through SD-VAE, saves bf16 latents.

Per §4.3 옵션 A. Raw images are never written to disk.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import torch
from datasets import load_dataset
from diffusers import AutoencoderKL
import torchvision.transforms as T
from PIL import Image
import io

def transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

def get_image(sample):
    if "image" in sample:
        return sample["image"].convert("RGB")
    for k, v in sample.items():
        if isinstance(v, dict) and "bytes" in v:
            return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        if isinstance(v, bytes):
            return Image.open(io.BytesIO(v)).convert("RGB")
    raise RuntimeError(f"no image in sample keys={list(sample.keys())}")

def main(args):
    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = vae.config.scaling_factor
    tx = transform()

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    saved = []
    t0 = time.time()
    for i, sample in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        img = tx(get_image(sample)).unsqueeze(0).to(device)
        with torch.no_grad():
            z = (vae.encode(img).latent_dist.mean * sf).squeeze(0)
        z_bf16 = z.to(torch.bfloat16).cpu()
        path = out_dir / f"{i:06d}.pt"
        torch.save(z_bf16, path)
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        saved.append({"index": i, "path": str(path.relative_to(".")), "sha256_16": h})
        if (i + 1) % 1000 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta_min = (args.limit - i - 1) / rate / 60 if args.limit else 0
            print(f"[{i+1}] rate={rate:.1f} img/s eta={eta_min:.1f} min")

    manifest_path = Path("data/MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    key = f"latents_{args.hf_split}"
    manifest[key] = {"count": len(saved), "files": saved}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✓ saved {len(saved)} latents to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dataset", required=True)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    main(p.parse_args())
```

- [ ] **Step 4**: 작은 limit으로 dry-run.

```bash
python scripts/precompute_latents.py --hf_dataset <chosen-id> --hf_split train --out data/ffhq256_latents/train --limit 10
```

Expected: `✓ saved 10 latents to data/ffhq256_latents/train` 그리고 `data/ffhq256_latents/train/000000.pt` 등 생성.

- [ ] **Step 5**: Test 실행.

```bash
pytest tests/test_dataset.py::test_latent_shape_and_dtype -v
```

Expected: PASS.

- [ ] **Step 6**: Commit script + test.

```bash
git add scripts/precompute_latents.py tests/test_dataset.py
git commit -m "feat: streaming SD-VAE latent precompute (10-sample dry run validated)"
```

### Task 1.3: Full latent precomputation

**Files:**
- Create: `data/ffhq256_latents/train/*.pt` (~600 MB)
- Modify: `data/MANIFEST.json`

- [ ] **Step 1**: Disk pre-flight.

```bash
python scripts/disk_monitor.py --once
df -h .
```

가용량이 ≥ 5 GB 인지 확인 (latent 600 MB + buffer).

- [ ] **Step 2**: Full train precompute.

```bash
python scripts/precompute_latents.py --hf_dataset <chosen-id> --hf_split train --out data/ffhq256_latents/train
```

Expected: rate 약 30-80 img/s (4080S, batch 1, fp32 VAE), 약 15-40 분에 70K 완료. 종료 시 `✓ saved 70000 latents`.

- [ ] **Step 3**: Sanity check.

```bash
ls data/ffhq256_latents/train | wc -l
du -sh data/ffhq256_latents/train
```

Expected: ~70000 files, ~600 MB.

- [ ] **Step 4**: Test.

```bash
pytest tests/test_dataset.py -v
```

Expected: all PASS.

- [ ] **Step 5**: Commit MANIFEST update (latents 자체는 .gitignore).

```bash
git add data/MANIFEST.json
git commit -m "data: precompute 70K FFHQ-256 SD-VAE latents (manifest updated)"
```

### Task 1.4: Train/eval split fix (deterministic, by latent index)

**Files:**
- Create: `data/ffhq256_train.txt` (65K lines)
- Create: `data/ffhq256_eval.txt` (5K lines)
- Modify: `data/MANIFEST.json`

- [ ] **Step 1**: Failing test for split determinism.

`tests/test_dataset.py` (append):
```python
def test_split_determinism():
    train = open("data/ffhq256_train.txt").read().splitlines()
    eval_ = open("data/ffhq256_eval.txt").read().splitlines()
    assert len(train) == 65000
    assert len(eval_) == 5000
    assert set(train).isdisjoint(set(eval_))
    assert set(train) | set(eval_) == set(f"{i:06d}" for i in range(70000))
```

- [ ] **Step 2**: Run, expect FAIL.

```bash
pytest tests/test_dataset.py::test_split_determinism -v
```

- [ ] **Step 3**: Split generator (one-shot).

```bash
python -c "
import random, hashlib, json
random.seed(20260415)
indices = [f'{i:06d}' for i in range(70000)]
random.shuffle(indices)
train = sorted(indices[:65000])
eval_ = sorted(indices[65000:])
open('data/ffhq256_train.txt','w').write('\n'.join(train) + '\n')
open('data/ffhq256_eval.txt','w').write('\n'.join(eval_) + '\n')
m = json.load(open('data/MANIFEST.json'))
m['split'] = {
    'seed': 20260415,
    'train_n': 65000,
    'eval_n': 5000,
    'train_sha256': hashlib.sha256(open('data/ffhq256_train.txt','rb').read()).hexdigest(),
    'eval_sha256': hashlib.sha256(open('data/ffhq256_eval.txt','rb').read()).hexdigest(),
}
json.dump(m, open('data/MANIFEST.json','w'), indent=2)
print('done')
"
```

- [ ] **Step 4**: Test.

```bash
pytest tests/test_dataset.py::test_split_determinism -v
```

Expected: PASS.

- [ ] **Step 5**: Commit.

```bash
git add data/ffhq256_train.txt data/ffhq256_eval.txt data/MANIFEST.json tests/test_dataset.py
git commit -m "data: fix 65K/5K train/eval split (seed 20260415, deterministic)"
```

### Task 1.5: FID reference statistics

**Files:**
- Create: `scripts/precompute_fid_ref.py`
- Create: `data/fid_ref_ffhq256.npz`

- [ ] **Step 1**: Script 작성.

`scripts/precompute_fid_ref.py`:
```python
"""Compute clean-fid Inception statistics for FFHQ-256 eval split.

Decodes 5K eval latents through SD-VAE in memory, never writes raw images to disk.
Output: data/fid_ref_ffhq256.npz with keys 'mu' (2048,) and 'sigma' (2048,2048).
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance

def main(args):
    device = "cuda"
    eval_idx = open("data/ffhq256_eval.txt").read().splitlines()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = vae.config.scaling_factor
    feat = build_feature_extractor("clean", device)

    feats = []
    batch = []
    BS = 32
    for i, idx in enumerate(eval_idx):
        z = torch.load(f"data/ffhq256_latents/train/{idx}.pt").to(device).to(torch.float32)
        batch.append(z)
        if len(batch) == BS or i == len(eval_idx) - 1:
            zs = torch.stack(batch)
            with torch.no_grad():
                imgs = vae.decode(zs / sf).sample
                imgs_uint8 = ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
                f = feat(imgs_uint8)
            feats.append(f.cpu().numpy())
            batch = []
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(eval_idx)}]")

    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    np.savez("data/fid_ref_ffhq256.npz", mu=mu, sigma=sigma, n=len(eval_idx))
    print(f"✓ FID reference computed from {len(eval_idx)} samples → data/fid_ref_ffhq256.npz")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(p.parse_args())
```

- [ ] **Step 2**: 실행.

```bash
python scripts/precompute_fid_ref.py
```

Expected: ~5-10 분, 종료 시 `✓ FID reference computed from 5000 samples`.

- [ ] **Step 3**: Sanity 검증.

```bash
python -c "
import numpy as np
ref = np.load('data/fid_ref_ffhq256.npz')
print('mu shape:', ref['mu'].shape, 'sigma shape:', ref['sigma'].shape, 'n:', int(ref['n']))
assert ref['mu'].shape == (2048,)
assert ref['sigma'].shape == (2048, 2048)
print('✓')
"
```

- [ ] **Step 4**: Commit.

```bash
git add scripts/precompute_fid_ref.py data/fid_ref_ffhq256.npz
git commit -m "data: compute clean-fid Inception ref stats from eval split"
```

### Task 1.6: DINOv2 reference statistics

**Files:**
- Create: `scripts/precompute_dino_ref.py`
- Create: `data/fd_dinov2_ref_ffhq256.pt`

- [ ] **Step 1**: Script 작성.

`scripts/precompute_dino_ref.py`:
```python
"""Compute DINOv2 ViT-B feature mean/cov for FFHQ-256 eval split.
Output: data/fd_dinov2_ref_ffhq256.pt with keys 'mu', 'sigma', 'n'.
"""
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from transformers import AutoImageProcessor, AutoModel

def main():
    device = "cuda"
    eval_idx = open("data/ffhq256_eval.txt").read().splitlines()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = vae.config.scaling_factor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    feats = []
    BS = 32
    batch = []
    for i, idx in enumerate(eval_idx):
        z = torch.load(f"data/ffhq256_latents/train/{idx}.pt").to(device).to(torch.float32)
        batch.append(z)
        if len(batch) == BS or i == len(eval_idx) - 1:
            zs = torch.stack(batch)
            with torch.no_grad():
                imgs = vae.decode(zs / sf).sample
                imgs_01 = (imgs.clamp(-1, 1) + 1) / 2
                inp = proc(images=[T.ToPILImage()(im.cpu()) for im in imgs_01], return_tensors="pt").to(device)
                out = dino(**inp).last_hidden_state[:, 1:].mean(dim=1)  # patch tokens mean (drop CLS)
            feats.append(out.cpu())
            batch = []
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(eval_idx)}]")

    feats = torch.cat(feats, dim=0).to(torch.float64)
    mu = feats.mean(dim=0)
    sigma = torch.cov(feats.T)
    torch.save({"mu": mu, "sigma": sigma, "n": len(eval_idx)}, "data/fd_dinov2_ref_ffhq256.pt")
    print(f"✓ DINOv2 ref ({mu.shape[0]}-dim) from {len(eval_idx)} samples")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: 실행.

```bash
python scripts/precompute_dino_ref.py
```

Expected: ~5-10 분, 종료 시 `✓ DINOv2 ref (768-dim) from 5000 samples`.

- [ ] **Step 3**: Commit.

```bash
git add scripts/precompute_dino_ref.py data/fd_dinov2_ref_ffhq256.pt
git commit -m "data: compute DINOv2 ViT-B ref stats from eval split"
```

### Task 1.7: Reproducibility verification

**Files:**
- Modify: `tests/test_dataset.py`

- [ ] **Step 1**: 추가 test — 동일 latent을 두 번 로드해서 bit-identical인지.

```python
def test_latent_reproducibility():
    a = torch.load("data/ffhq256_latents/train/000000.pt")
    b = torch.load("data/ffhq256_latents/train/000000.pt")
    assert torch.equal(a, b), "two loads of the same latent file differ"
```

- [ ] **Step 2**: 추가 test — manifest의 SHA-256 첫 16자가 실제 file과 일치하는지.

```python
import hashlib, json
def test_manifest_hash_matches_file():
    m = json.load(open("data/MANIFEST.json"))
    rec = m["latents_train"]["files"][0]
    actual = hashlib.sha256(open(rec["path"], "rb").read()).hexdigest()[:16]
    assert actual == rec["sha256_16"]
```

- [ ] **Step 3**: Run all dataset tests.

```bash
pytest tests/test_dataset.py -v
```

Expected: 모두 PASS.

- [ ] **Step 4**: Commit.

```bash
git add tests/test_dataset.py
git commit -m "test: latent reproducibility + manifest hash verification"
```

---

## Phase 2 — Code Modifications (Day 3-4)

### Task 2.1: REPA fork inventory

**Goal:** 수정 대상 파일 경로/이름을 *문서화*. 추측이 아닌 사실 기반.

**Files:**
- Create: `docs/BASELINE.md`

- [ ] **Step 1**: REPA 코드베이스 구조 dump.

```bash
find . -type f -name "*.py" -not -path "./venv/*" -not -path "./.git/*" | head -100 | tee docs/repa_inventory.txt
```

- [ ] **Step 2**: 핵심 파일 위치 확인.

```bash
grep -l "class SiT\|def train_step\|REPA\|projection" *.py models/*.py 2>/dev/null | tee -a docs/repa_inventory.txt
```

- [ ] **Step 3**: `docs/BASELINE.md` 초안 작성 — 발견한 파일과 책임:

```markdown
# REPA Baseline 코드베이스 매핑

원본: `sihyun-yu/REPA` (commit: <git rev-parse HEAD>)

## 핵심 파일

- `train.py` (혹은 `<actual>`) — main training entry
- `models/sit.py` — SiT backbone
- `models/<repa>.py` — REPA projection head
- `<dataset_loader>.py` — 기존 ImageNet loader (교체 대상)

## 우리가 수정할 hook 위치

1. **Loss 계산** (`<file>:<line>`): `λ` 가 적용되는 곳, scheduler 주입.
2. **Dataset 인스턴스화** (`<file>:<line>`): FFHQ latent loader로 교체.
3. **Class label handling** (`<file>:<line>`): null token으로 대체.
4. **Checkpoint 저장** (`<file>:<line>`): EMA-only bf16, eval ckpt만.

## 의도적 deviation (purity principle, §3.1)

원본 REPA에서 변경된 모든 항목:
- Dataset: ImageNet → FFHQ-256 (§4)
- Conditioning: class → unconditional (null token)
- Coefficient: fixed → 5 schedules (§6)
- Checkpoint policy: 매 step → eval ckpt만 EMA bf16 (§4.4)
- Eval suite: REPA paper FID + 우리 추가 metric (§7)
```

`<file>:<line>` 와 같은 placeholder는 inventory dump의 실제 발견으로 채울 것.

- [ ] **Step 4**: Commit.

```bash
git add docs/BASELINE.md docs/repa_inventory.txt
git commit -m "docs: REPA fork inventory + BASELINE.md scaffold"
```

### Task 2.2: FFHQ-256 latent dataset loader

**Files:**
- Create: `train/data/ffhq256_latents.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1**: Failing test.

`tests/test_dataset.py` (append):
```python
def test_dataset_loader_smoke():
    from train.data.ffhq256_latents import FFHQ256LatentDataset
    ds = FFHQ256LatentDataset("data/ffhq256_latents/train", "data/ffhq256_train.txt")
    assert len(ds) == 65000
    sample = ds[0]
    assert "latent" in sample and "label" in sample
    assert sample["latent"].shape == (4, 32, 32)
    assert sample["latent"].dtype == torch.float32
    assert sample["label"].item() == 0  # null class
```

- [ ] **Step 2**: Run, expect FAIL (module missing).

```bash
pytest tests/test_dataset.py::test_dataset_loader_smoke -v
```

- [ ] **Step 3**: Implement.

`train/data/ffhq256_latents.py`:
```python
"""FFHQ-256 SD-VAE latent dataset loader.

Loads bf16 latents from disk, optionally horizontal flip, returns float32 tensors
with a null class token (label 0) for unconditional training.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path

class FFHQ256LatentDataset(Dataset):
    def __init__(self, latent_dir: str, index_file: str, hflip: bool = True):
        self.dir = Path(latent_dir)
        self.indices = open(index_file).read().splitlines()
        self.hflip = hflip

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        z = torch.load(self.dir / f"{idx}.pt", weights_only=True).to(torch.float32)
        if self.hflip and torch.rand(1).item() < 0.5:
            z = torch.flip(z, dims=[-1])
        return {"latent": z, "label": torch.tensor(0, dtype=torch.long)}
```

- [ ] **Step 4**: Run test.

```bash
pytest tests/test_dataset.py::test_dataset_loader_smoke -v
```

Expected: PASS.

- [ ] **Step 5**: Commit.

```bash
git add train/data/ffhq256_latents.py tests/test_dataset.py
git commit -m "feat: FFHQ256LatentDataset loader (bf16 latents → fp32 + null label)"
```

### Task 2.3: Coefficient schedule functions (5 branches)

**Files:**
- Create: `train/branches.py`
- Create: `tests/test_branches.py`

- [ ] **Step 1**: Test 작성 (5 schedules + edge cases).

`tests/test_branches.py`:
```python
import math, pytest
from train.branches import (
    schedule_off, schedule_fixed, schedule_cosine_decay,
    schedule_cosine_warmup, schedule_hard_cutoff, BRANCHES
)

T = 100_000

def test_schedule_off():
    assert schedule_off(0, T) == 0.0
    assert schedule_off(50_000, T) == 0.0
    assert schedule_off(T, T) == 0.0

def test_schedule_fixed():
    assert schedule_fixed(0, T) == 0.5
    assert schedule_fixed(T, T) == 0.5

def test_schedule_cosine_decay():
    assert schedule_cosine_decay(0, T) == pytest.approx(0.5, abs=1e-9)
    assert schedule_cosine_decay(T // 2, T) == pytest.approx(0.25, abs=1e-9)
    assert schedule_cosine_decay(T, T) == pytest.approx(0.0, abs=1e-9)

def test_schedule_cosine_warmup():
    assert schedule_cosine_warmup(0, T) == pytest.approx(0.0, abs=1e-9)
    assert schedule_cosine_warmup(T // 2, T) == pytest.approx(0.25, abs=1e-9)
    assert schedule_cosine_warmup(T, T) == pytest.approx(0.5, abs=1e-9)

def test_schedule_hard_cutoff():
    assert schedule_hard_cutoff(0, T) == 0.5
    assert schedule_hard_cutoff(T // 2 - 1, T) == 0.5
    assert schedule_hard_cutoff(T // 2, T) == 0.0
    assert schedule_hard_cutoff(T, T) == 0.0

def test_branches_registry():
    assert set(BRANCHES.keys()) == {"off", "fixed", "decay", "warmup", "cutoff"}
    for name, fn in BRANCHES.items():
        v = fn(T // 4, T)
        assert 0.0 <= v <= 0.5
```

- [ ] **Step 2**: Run, expect FAIL (module missing).

```bash
pytest tests/test_branches.py -v
```

- [ ] **Step 3**: Implement.

`train/branches.py`:
```python
"""Coefficient schedules for the 5 research branches (§6).

All functions return the projection-loss coefficient λ at training step t,
total steps T. Maximum coefficient is 0.5 (REPA paper SiT-B/2 default).
"""
import math

MAX_COEFF = 0.5

def schedule_off(t: int, T: int) -> float:
    return 0.0

def schedule_fixed(t: int, T: int) -> float:
    return MAX_COEFF

def schedule_cosine_decay(t: int, T: int) -> float:
    return MAX_COEFF * 0.5 * (1.0 + math.cos(math.pi * t / T))

def schedule_cosine_warmup(t: int, T: int) -> float:
    return MAX_COEFF * 0.5 * (1.0 - math.cos(math.pi * t / T))

def schedule_hard_cutoff(t: int, T: int, cutoff_frac: float = 0.5) -> float:
    return MAX_COEFF if t < T * cutoff_frac else 0.0

BRANCHES = {
    "off":    schedule_off,
    "fixed":  schedule_fixed,
    "decay":  schedule_cosine_decay,
    "warmup": schedule_cosine_warmup,
    "cutoff": schedule_hard_cutoff,
}
```

- [ ] **Step 4**: Run tests.

```bash
pytest tests/test_branches.py -v
```

Expected: 6 PASS.

- [ ] **Step 5**: Commit.

```bash
git add train/branches.py tests/test_branches.py
git commit -m "feat: 5 REPA coefficient schedules (off/fixed/decay/warmup/cutoff)"
```

### Task 2.4: Hook scheduler into training loss

**Files:**
- Modify: REPA fork's main training script (path from Task 2.1)

- [ ] **Step 1**: Inventory에서 확인한 loss 함수 위치 열기.

```bash
$EDITOR <repa_train_script>.py
```

- [ ] **Step 2**: 수정 — loss 계산부에서 schedule 적용. 기존 fixed-coefficient 코드 (`loss = denoise_loss + 0.5 * proj_loss` 형태) 를 다음과 같이 변경:

```python
from train.branches import BRANCHES

def compute_loss(model, batch, step, total_steps, branch_name):
    # ... denoise_loss, proj_loss 계산 (REPA 원본 그대로) ...
    coeff = BRANCHES[branch_name](step, total_steps)
    return denoise_loss + coeff * proj_loss, {
        "denoise_loss": denoise_loss.item(),
        "proj_loss": proj_loss.item(),
        "coeff": coeff,
    }
```

- [ ] **Step 3**: CLI 인자 추가 — `--branch {off,fixed,decay,warmup,cutoff}` 와 `--total_steps`. 기본값 fixed, 100000.

- [ ] **Step 4**: Commit (구체 수정 line 수는 inventory 확인 후).

```bash
git add <modified_files>
git commit -m "feat: hook coefficient scheduler into REPA training loss"
```

### Task 2.5: Remove class conditioning, use null token

**Files:**
- Modify: REPA fork's model/training files

- [ ] **Step 1**: Inventory에서 class label 전달 위치 찾기.

```bash
grep -n "class_emb\|num_classes\|y_embed\|labels" <repa>/*.py
```

- [ ] **Step 2**: Dataset에서는 항상 label=0 (Task 2.2 loader가 이미 그렇게 함). 모델/embedding 쪽에서:
  - Embedding table size를 1로 줄이거나
  - 전체 forward에서 class embedding을 zero tensor로 강제

가장 minimal한 수정: embedding lookup에 항상 0을 전달, embedding table은 그대로 두되 쓰이지 않게. (남는 parameter는 §3 purity로 보면 작은 deviation이지만, 제거하면 코드가 더 invasive해짐. trade-off로 수용.)

- [ ] **Step 3**: 변경 후, 한 batch forward로 NaN/error 없음 확인.

```bash
python -c "
import torch
from train.data.ffhq256_latents import FFHQ256LatentDataset
from <repa.models.sit> import SiT  # Task 2.1에서 확인한 import path
ds = FFHQ256LatentDataset('data/ffhq256_latents/train', 'data/ffhq256_train.txt')
batch = ds[0]
model = SiT(...).cuda()  # 기본 config, SiT-B/2
x = batch['latent'].unsqueeze(0).cuda()
y = batch['label'].unsqueeze(0).cuda()
t = torch.tensor([500.0]).cuda()
out = model(x, t, y)
print('forward ok:', out.shape, 'no NaN:', not torch.isnan(out).any().item())
"
```

- [ ] **Step 4**: Commit.

```bash
git add <modified_files>
git commit -m "feat: unconditional FFHQ training (null label, no class conditioning)"
```

### Task 2.6: Equivalence test — `schedule_off` ≡ REPA-disabled

**Files:**
- Create: `tests/test_equivalence_off.py`

이 test는 Phase 0 critique R1 mitigation의 일부: schedule_off가 정말 REPA-disabled와 같은지 확인.

- [ ] **Step 1**: Test 작성.

`tests/test_equivalence_off.py`:
```python
"""Verify schedule_off (λ=0 multiplier) produces the same gradients as
removing the projection loss entirely. Both should produce identical
loss curves over the first 100 steps within bf16 floating-point noise.
"""
import torch, copy
from train.branches import schedule_off
# from <repa.training_loop> import train_step  # Task 2.1 path

def test_off_matches_no_proj_loss():
    torch.manual_seed(42)
    # ... set up two identical models ...
    # ... run 100 training steps with schedule_off vs with proj loss removed ...
    # ... compare loss curves ...
    losses_off = []
    losses_no_proj = []
    # (구체 구현은 Task 2.4 가 끝난 후 train_step의 signature 보고 채움)
    for a, b in zip(losses_off, losses_no_proj):
        assert abs(a - b) < 1e-3, f"loss diverges: off={a:.6f} no_proj={b:.6f}"
```

- [ ] **Step 2**: Run.

```bash
pytest tests/test_equivalence_off.py -v
```

Expected: PASS (동일한 loss curve). FAIL이면 hook 위치 디버그.

- [ ] **Step 3**: Commit.

```bash
git add tests/test_equivalence_off.py
git commit -m "test: schedule_off ≡ REPA-disabled equivalence (100-step loss match)"
```

### Task 2.7: Equivalence test — `schedule_fixed` ≡ original REPA

**Files:**
- Create: `tests/test_equivalence_fixed.py`

- [ ] **Step 1**: Original REPA fork에 자체 fixed-coefficient 학습 1000 step 실행, loss 기록.

```bash
# upstream REPA의 default 설정으로 1000 step
python <upstream_repa_train>.py --steps 1000 --output_dir /tmp/upstream_test
cp /tmp/upstream_test/loss_log.csv tests/fixtures/upstream_loss_log.csv
```

- [ ] **Step 2**: 우리 wrapper로 동일한 1000 step.

```bash
python scripts/train.py --branch fixed --total_steps 1000 --seed 42 --out /tmp/our_fixed
```

- [ ] **Step 3**: Test.

```python
import csv

def test_fixed_matches_upstream():
    upstream = [float(r["loss"]) for r in csv.DictReader(open("tests/fixtures/upstream_loss_log.csv"))][:1000]
    ours = [float(r["loss"]) for r in csv.DictReader(open("/tmp/our_fixed/loss_log.csv"))][:1000]
    assert len(upstream) == len(ours) == 1000
    diffs = [abs(a - b) / (abs(a) + 1e-8) for a, b in zip(upstream, ours)]
    avg_rel_diff = sum(diffs) / len(diffs)
    assert avg_rel_diff < 0.01, f"avg relative diff {avg_rel_diff:.4f} > 0.01"
```

- [ ] **Step 4**: Run.

```bash
pytest tests/test_equivalence_fixed.py -v
```

- [ ] **Step 5**: Commit.

```bash
git add tests/test_equivalence_fixed.py tests/fixtures/upstream_loss_log.csv
git commit -m "test: schedule_fixed ≡ upstream REPA fixed-coeff (1000 steps within 1% relative)"
```

### Task 2.8: Checkpoint 저장 정책 (EMA-only bf16, eval ckpts only)

**Files:**
- Modify: REPA fork's checkpoint saver
- Create: `train/save.py` (helper)

- [ ] **Step 1**: Helper 작성.

`train/save.py`:
```python
"""§4.4 storage policy: save only EMA weights at eval checkpoints, in bf16.
Maintains exactly one rolling latest checkpoint as scratch (deleted at next save).
"""
import torch
from pathlib import Path

EVAL_STEPS = {20_000, 50_000, 80_000, 100_000}

def is_eval_step(step: int) -> bool:
    return step in EVAL_STEPS

def save_ema_bf16(ema_state_dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bf16_sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in ema_state_dict.items()}
    torch.save(bf16_sd, out_path)

def cleanup_rolling(rolling_path):
    p = Path(rolling_path)
    if p.exists():
        p.unlink()
```

- [ ] **Step 2**: REPA training loop의 checkpoint 저장 부분 수정.

```python
from train.save import is_eval_step, save_ema_bf16, cleanup_rolling

# every save_freq:
if is_eval_step(step):
    save_ema_bf16(ema.state_dict(), f"exps/{branch}_{seed}/checkpoints/ema_{step:06d}.pt")
    cleanup_rolling(rolling_path)  # delete the transient latest
elif step % args.rolling_freq == 0:
    save_ema_bf16(ema.state_dict(), rolling_path)  # overwrite single rolling file
```

- [ ] **Step 3**: Test.

```python
def test_save_policy():
    from train.save import is_eval_step
    for s in [20000, 50000, 80000, 100000]:
        assert is_eval_step(s)
    for s in [10000, 30000, 99999]:
        assert not is_eval_step(s)
```

- [ ] **Step 4**: Pytest.

```bash
pytest tests/ -k save_policy -v
```

- [ ] **Step 5**: Commit.

```bash
git add train/save.py tests/
git commit -m "feat: §4.4 checkpoint policy (EMA-only bf16, eval ckpts only, rolling latest)"
```

### Task 2.9: Loss + coefficient logging to CSV

**Files:**
- Create: `train/log.py`
- Modify: REPA training loop

- [ ] **Step 1**: Logger.

`train/log.py`:
```python
"""Append-only CSV logger for training metrics."""
import csv
from pathlib import Path

class CsvLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = None
        self.f = None
        self.w = None

    def log(self, **kv):
        if self.f is None:
            self.fields = list(kv.keys())
            self.f = open(self.path, "a")
            self.w = csv.DictWriter(self.f, fieldnames=self.fields)
            if self.path.stat().st_size == 0:
                self.w.writeheader()
        self.w.writerow({k: kv[k] for k in self.fields})
        self.f.flush()

    def close(self):
        if self.f: self.f.close()
```

- [ ] **Step 2**: Training loop에서 매 step 로그.

```python
from train.log import CsvLogger
logger = CsvLogger(f"exps/{branch}_{seed}/loss_log.csv")
# in the step loop:
logger.log(step=step, denoise_loss=d_loss, proj_loss=p_loss, coeff=coeff, total_loss=total)
```

- [ ] **Step 3**: Commit.

```bash
git add train/log.py <modified train script>
git commit -m "feat: append-only CSV training logger (per-step loss + coeff)"
```

---

## Phase 3 — Evaluation Harness (Day 4)

### Task 3.1: Sample generation utility

**Files:**
- Create: `train/eval/sample.py`
- Create: `tests/test_metrics.py` (initial)

- [ ] **Step 1**: Test (smoke).

`tests/test_metrics.py`:
```python
import torch

def test_sample_generation_smoke():
    from train.eval.sample import generate_samples
    # mock model
    class M(torch.nn.Module):
        def forward(self, x, t, y): return torch.randn_like(x)
    samples = generate_samples(model=M(), n=8, batch_size=4, latent_shape=(4,32,32),
                                steps=10, device="cpu", seed=42)
    assert samples.shape == (8, 3, 256, 256)
    assert samples.dtype == torch.uint8
```

- [ ] **Step 2**: Implement.

`train/eval/sample.py`:
```python
"""Generate samples from EMA model + SD-VAE decode, no disk writes.

Returns uint8 tensor (N, 3, 256, 256) for downstream metric computation.
"""
import torch
from diffusers import AutoencoderKL

def generate_samples(model, n, batch_size, latent_shape, steps, device, seed,
                     vae=None, sampler="ddpm"):
    if vae is None:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = vae.config.scaling_factor
    g = torch.Generator(device=device).manual_seed(seed)
    out_imgs = []
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        bs = min(batch_size, n - b * batch_size)
        z = torch.randn(bs, *latent_shape, device=device, generator=g)
        # 50-step DDPM-style euler: t from 1.0 → 0.0
        for i in range(steps):
            t = torch.full((bs,), 1.0 - (i + 0.5) / steps, device=device)
            with torch.no_grad():
                v = model(z, t, torch.zeros(bs, dtype=torch.long, device=device))
            z = z - v / steps  # simple forward Euler
        with torch.no_grad():
            img = vae.decode(z / sf).sample
        img = ((img.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()
        out_imgs.append(img)
    return torch.cat(out_imgs, dim=0)
```

⚠ DDPM/Euler 구현은 placeholder. REPA fork의 sampling utility를 발견하면 그것을 사용 (일관성). Task 2.1 inventory에서 확인 필요.

- [ ] **Step 3**: Run smoke test.

```bash
pytest tests/test_metrics.py::test_sample_generation_smoke -v
```

- [ ] **Step 4**: Commit.

```bash
git add train/eval/sample.py tests/test_metrics.py
git commit -m "feat: sample generation utility (EMA → SD-VAE → uint8 tensor)"
```

### Task 3.2: clean-fid integration

**Files:**
- Create: `train/eval/fid.py`

- [ ] **Step 1**: Test.

`tests/test_metrics.py` (append):
```python
def test_fid_against_self_is_small():
    import numpy as np
    from train.eval.fid import compute_fid_from_uint8
    ref = np.load("data/fid_ref_ffhq256.npz")
    # generate dummy samples that match ref distribution → FID should be small-ish
    # (this is a smoke test, not a tight bound)
    samples = (torch.rand(32, 3, 256, 256) * 255).to(torch.uint8)
    fid = compute_fid_from_uint8(samples, ref_mu=ref["mu"], ref_sigma=ref["sigma"])
    assert isinstance(fid, float)
    assert fid > 0
```

- [ ] **Step 2**: Implement.

`train/eval/fid.py`:
```python
"""FID via clean-fid Inception features against precomputed reference."""
import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance

_feat_extractor = None

def _get_extractor(device):
    global _feat_extractor
    if _feat_extractor is None:
        _feat_extractor = build_feature_extractor("clean", device)
    return _feat_extractor

def compute_fid_from_uint8(samples_uint8, ref_mu, ref_sigma, device="cuda", batch_size=64):
    extractor = _get_extractor(device)
    feats = []
    for i in range(0, len(samples_uint8), batch_size):
        b = samples_uint8[i:i+batch_size].to(device)
        with torch.no_grad():
            feats.append(extractor(b).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return float(frechet_distance(mu, sigma, ref_mu, ref_sigma))
```

- [ ] **Step 3**: Test.

```bash
pytest tests/test_metrics.py::test_fid_against_self_is_small -v
```

- [ ] **Step 4**: Commit.

```bash
git add train/eval/fid.py tests/test_metrics.py
git commit -m "feat: FID computation against precomputed Inception ref stats"
```

### Task 3.3: Sharpness Wasserstein metric (3 statistics)

**Files:**
- Create: `train/eval/sharpness.py`

- [ ] **Step 1**: Test.

`tests/test_metrics.py` (append):
```python
def test_sharpness_three_keys():
    import torch
    from train.eval.sharpness import sharpness_wasserstein
    gen = (torch.rand(32, 3, 256, 256) * 255).to(torch.uint8)
    ref = (torch.rand(32, 3, 256, 256) * 255).to(torch.uint8)
    out = sharpness_wasserstein(gen, ref)
    assert set(out.keys()) == {"lapvar", "hffreq", "sobel"}
    for v in out.values():
        assert v >= 0.0
```

- [ ] **Step 2**: Implement (spec §7.2 / §12.7 reference).

`train/eval/sharpness.py`:
```python
"""3-statistic sharpness Wasserstein distance (§7.2)."""
import numpy as np
import cv2
from scipy.stats import wasserstein_distance

def _to_gray(img_chw_uint8):
    # img: (3, H, W) uint8 tensor or numpy array
    if hasattr(img_chw_uint8, "numpy"):
        img_chw_uint8 = img_chw_uint8.numpy()
    img_hwc = img_chw_uint8.transpose(1, 2, 0)
    return cv2.cvtColor(img_hwc, cv2.COLOR_RGB2GRAY)

def _laplacian_var(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def _hf_ratio(g, cutoff_frac=0.5):
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float64)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = int(cutoff_frac * min(cy, cx))
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 >= r * r
    return float(mag[mask].sum() / (mag.sum() + 1e-8))

def _sobel_mean(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())

def sharpness_wasserstein(gen_uint8, ref_uint8):
    fns = {"lapvar": _laplacian_var, "hffreq": _hf_ratio, "sobel": _sobel_mean}
    out = {}
    for name, fn in fns.items():
        gen_vals = np.array([fn(_to_gray(im)) for im in gen_uint8])
        ref_vals = np.array([fn(_to_gray(im)) for im in ref_uint8])
        out[name] = float(wasserstein_distance(gen_vals, ref_vals))
    return out
```

- [ ] **Step 3**: Test.

```bash
pytest tests/test_metrics.py::test_sharpness_three_keys -v
```

- [ ] **Step 4**: Commit.

```bash
git add train/eval/sharpness.py tests/test_metrics.py
git commit -m "feat: 3-statistic sharpness Wasserstein metric (lapvar/hffreq/sobel)"
```

### Task 3.4: FD-DINOv2 metric

**Files:**
- Create: `train/eval/fd_dinov2.py`

- [ ] **Step 1**: Implement.

`train/eval/fd_dinov2.py`:
```python
"""DINOv2 feature Frechet distance vs precomputed reference."""
import torch, torchvision.transforms as T
import numpy as np
from transformers import AutoImageProcessor, AutoModel

_proc = None
_dino = None

def _get():
    global _proc, _dino
    if _dino is None:
        _proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
    return _proc, _dino

def compute_fd_dinov2(samples_uint8, ref_path="data/fd_dinov2_ref_ffhq256.pt", batch_size=32):
    proc, dino = _get()
    ref = torch.load(ref_path)
    ref_mu, ref_sigma = ref["mu"].numpy(), ref["sigma"].numpy()

    feats = []
    for i in range(0, len(samples_uint8), batch_size):
        b = samples_uint8[i:i+batch_size]
        pil = [T.ToPILImage()(x) for x in b]
        inp = proc(images=pil, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = dino(**inp).last_hidden_state[:, 1:].mean(dim=1)
        feats.append(out.cpu().numpy())
    feats = np.concatenate(feats, axis=0).astype(np.float64)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    # Frechet distance
    from scipy.linalg import sqrtm
    diff = mu - ref_mu
    covmean = sqrtm(sigma @ ref_sigma)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma + ref_sigma - 2 * covmean))
```

- [ ] **Step 2**: Smoke test.

`tests/test_metrics.py` (append):
```python
def test_fd_dinov2_smoke():
    pytest.importorskip("transformers")
    pytest.importorskip("scipy")
    from train.eval.fd_dinov2 import compute_fd_dinov2
    samples = (torch.rand(16, 3, 256, 256) * 255).to(torch.uint8)
    fd = compute_fd_dinov2(samples)
    assert isinstance(fd, float) and fd >= 0
```

- [ ] **Step 3**: Run.

```bash
pytest tests/test_metrics.py::test_fd_dinov2_smoke -v
```

- [ ] **Step 4**: Commit.

```bash
git add train/eval/fd_dinov2.py tests/test_metrics.py
git commit -m "feat: FD-DINOv2 metric vs precomputed ref"
```

### Task 3.5: Precision/Recall (Kynkäänniemi 2019)

**Files:**
- Create: `train/eval/precision_recall.py`

- [ ] **Step 1**: Implement using existing impl (e.g., `improved-precision-recall-metric` 가능, 또는 직접 k-NN). 직접 구현이 short이므로 간단히:

```python
"""Improved Precision and Recall (Kynkäänniemi et al. 2019).
Uses VGG-16 features and manifold k-NN.
"""
import torch, torch.nn.functional as F
import numpy as np
import torchvision.models as M
import torchvision.transforms as T

_vgg = None

def _get_vgg():
    global _vgg
    if _vgg is None:
        m = M.vgg16(weights=M.VGG16_Weights.DEFAULT).features.to("cuda").eval()
        _vgg = m
    return _vgg

def _features(imgs_uint8, batch_size=32):
    vgg = _get_vgg()
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    feats = []
    for i in range(0, len(imgs_uint8), batch_size):
        b = imgs_uint8[i:i+batch_size].to("cuda").float() / 255.0
        b = norm(b)
        with torch.no_grad():
            f = vgg(b).mean(dim=[2, 3])
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)

def _knn_dist(x, k):
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    return d.topk(k, largest=False).values[:, -1]

def precision_recall(gen_uint8, ref_uint8, k=3):
    g_feat = _features(gen_uint8)
    r_feat = _features(ref_uint8)
    r_radii = _knn_dist(r_feat, k)
    g_radii = _knn_dist(g_feat, k)

    # precision: fraction of gen inside ref manifold
    d_g_to_r = torch.cdist(g_feat, r_feat)
    inside = (d_g_to_r <= r_radii.unsqueeze(0)).any(dim=1)
    precision = inside.float().mean().item()

    # recall: fraction of ref inside gen manifold
    d_r_to_g = torch.cdist(r_feat, g_feat)
    inside_r = (d_r_to_g <= g_radii.unsqueeze(0)).any(dim=1)
    recall = inside_r.float().mean().item()

    return {"precision": float(precision), "recall": float(recall)}
```

- [ ] **Step 2**: Test smoke.

```python
def test_precision_recall_keys():
    from train.eval.precision_recall import precision_recall
    g = (torch.rand(64, 3, 256, 256) * 255).to(torch.uint8)
    r = (torch.rand(64, 3, 256, 256) * 255).to(torch.uint8)
    out = precision_recall(g, r, k=3)
    assert set(out.keys()) == {"precision", "recall"}
    for v in out.values():
        assert 0 <= v <= 1
```

- [ ] **Step 3**: Commit.

```bash
git add train/eval/precision_recall.py tests/test_metrics.py
git commit -m "feat: improved P/R (Kynkäänniemi 2019) with VGG-16 features"
```

### Task 3.6: FID-vs-step AUC (R5 primary metric)

**Files:**
- Create: `train/eval/auc.py`

- [ ] **Step 1**: Test.

`tests/test_metrics.py` (append):
```python
def test_auc_trapezoid():
    from train.eval.auc import fid_step_auc
    steps = [20000, 50000, 80000, 100000]
    fids = [40.0, 20.0, 12.0, 10.0]
    # trapezoid: 0.5 * sum((f_i + f_{i+1}) * (s_{i+1} - s_i))
    # = 0.5 * ((40+20)*30000 + (20+12)*30000 + (12+10)*20000)
    # = 0.5 * (1_800_000 + 960_000 + 440_000) = 1_600_000
    expected = 1_600_000.0
    assert abs(fid_step_auc(steps, fids) - expected) < 1.0
```

- [ ] **Step 2**: Implement.

`train/eval/auc.py`:
```python
"""FID-vs-step trajectory AUC (§2.2, §7.1, §8 Primary Metric)."""

def fid_step_auc(steps, fids):
    """Trapezoid integration over (step, FID) sequence.
    Lower AUC = lower FID across the trajectory = better.
    """
    assert len(steps) == len(fids) and len(steps) >= 2
    pairs = sorted(zip(steps, fids))
    auc = 0.0
    for (s0, f0), (s1, f1) in zip(pairs, pairs[1:]):
        auc += 0.5 * (f0 + f1) * (s1 - s0)
    return float(auc)
```

- [ ] **Step 3**: Run.

```bash
pytest tests/test_metrics.py::test_auc_trapezoid -v
```

- [ ] **Step 4**: Commit.

```bash
git add train/eval/auc.py tests/test_metrics.py
git commit -m "feat: FID-vs-step trapezoid AUC (R5 primary metric per §7.1)"
```

### Task 3.7: `eval.py` orchestration

**Files:**
- Create: `scripts/eval.py`

- [ ] **Step 1**: Script.

`scripts/eval.py`:
```python
"""Evaluate a single (run, checkpoint) — generates 10K samples, computes all metrics, writes JSON.

Usage: python scripts/eval.py --ckpt exps/b1_s42/checkpoints/ema_100000.pt --out results/b1_s42_100000.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from train.eval.sample import generate_samples
from train.eval.fid import compute_fid_from_uint8
from train.eval.fd_dinov2 import compute_fd_dinov2
from train.eval.sharpness import sharpness_wasserstein
from train.eval.precision_recall import precision_recall

def load_model(ckpt_path):
    # Task 2.1 inventory에서 확인한 model class 사용
    from <repa.models.sit> import SiT  # placeholder
    sd = torch.load(ckpt_path, map_location="cuda")
    model = SiT(...).to("cuda").eval()
    model.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    return model

def load_ref_images(n=5000):
    """Decode eval split latents → uint8 image tensor (in memory)."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda").eval()
    sf = vae.config.scaling_factor
    eval_idx = open("data/ffhq256_eval.txt").read().splitlines()[:n]
    imgs = []
    BS = 32
    for i in range(0, len(eval_idx), BS):
        zs = torch.stack([
            torch.load(f"data/ffhq256_latents/train/{idx}.pt").to(torch.float32)
            for idx in eval_idx[i:i+BS]
        ]).to("cuda")
        with torch.no_grad():
            x = vae.decode(zs / sf).sample
        imgs.append(((x.clamp(-1,1) + 1) * 127.5).to(torch.uint8).cpu())
    return torch.cat(imgs, dim=0)

def main(args):
    t0 = time.time()
    model = load_model(args.ckpt)
    samples = generate_samples(
        model=model, n=args.n_samples, batch_size=64,
        latent_shape=(4, 32, 32), steps=50, device="cuda", seed=args.sample_seed,
    )
    ref = np.load("data/fid_ref_ffhq256.npz")
    fid = compute_fid_from_uint8(samples, ref["mu"], ref["sigma"])
    fd_dino = compute_fd_dinov2(samples)
    ref_imgs = load_ref_images(n=5000)
    sharp = sharpness_wasserstein(samples, ref_imgs)
    pr = precision_recall(samples, ref_imgs, k=3)

    # Save preview (16 samples)
    Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
    from torchvision.utils import save_image
    for i in range(min(16, len(samples))):
        save_image(samples[i].float() / 255.0, f"{args.preview_dir}/sample_{i:02d}.png")

    out = {
        "ckpt": args.ckpt,
        "n_samples": args.n_samples,
        "fid": fid,
        "fd_dinov2": fd_dino,
        "sharpness": sharp,
        "precision_recall": pr,
        "wallclock_sec": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--sample_seed", type=int, default=20260415)
    p.add_argument("--preview_dir", default="exps/preview_default")
    main(p.parse_args())
```

- [ ] **Step 2**: Random-init smoke test (high FID expected).

```bash
python -c "
import torch
from <repa.models.sit> import SiT
m = SiT(...)
torch.save({k: v.bfloat16() for k,v in m.state_dict().items()}, '/tmp/random_ckpt.pt')
"
python scripts/eval.py --ckpt /tmp/random_ckpt.pt --out /tmp/random_eval.json --n_samples 256 --preview_dir /tmp/preview
cat /tmp/random_eval.json
```

Expected: pipeline 통과, FID 매우 높음 (~300+).

- [ ] **Step 3**: Commit.

```bash
git add scripts/eval.py
git commit -m "feat: eval.py orchestration (10K samples → all metrics → JSON)"
```

### Task 3.8: `aggregate.py` results aggregator + 4-way reporting

**Files:**
- Create: `scripts/aggregate.py`
- Create: `train/eval/stats.py`

- [ ] **Step 1**: Stats module (§2.3 4-way).

`train/eval/stats.py`:
```python
"""§2.3 4-way reporting for n=3 seeds.

Per (branch, metric, eval_step):
- mean ± std across seeds (descriptive)
- directional consistency (sign agreement of seed-level differences vs reference branch)
- Cohen's d (paired) + bootstrap 95% CI
- Bonferroni-corrected paired t-test p-value (pre-registered)
"""
import numpy as np
from scipy import stats

def cohens_d_paired(diffs):
    diffs = np.asarray(diffs)
    return float(diffs.mean() / (diffs.std(ddof=1) + 1e-12))

def bootstrap_ci(diffs, n_boot=1000, alpha=0.05):
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(20260415)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(cohens_d_paired(sample))
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def directional_consistency(diffs):
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return 0
    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    total = len(diffs)
    return {"pos": pos, "neg": neg, "total": total,
            "majority_frac": max(pos, neg) / total}

def paired_t_bonferroni(diffs, n_comparisons):
    diffs = np.asarray(diffs)
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return {"t": float("nan"), "p_raw": 1.0, "p_bonf": 1.0, "significant": False}
    t, p = stats.ttest_1samp(diffs, 0.0)
    p_bonf = min(p * n_comparisons, 1.0)
    return {"t": float(t), "p_raw": float(p), "p_bonf": float(p_bonf),
            "significant": p_bonf < 0.05}

def four_way_report(branch_a_values, branch_b_values, n_comparisons=6):
    """Inputs: n=3 seed-level metric values for branches A and B.
    Output: descriptive + directional + effect size + hypothesis test.
    """
    a, b = np.asarray(branch_a_values), np.asarray(branch_b_values)
    diffs = a - b
    d = cohens_d_paired(diffs)
    lo, hi = bootstrap_ci(diffs)
    return {
        "descriptive": {
            "branch_a_mean": float(a.mean()),
            "branch_a_std": float(a.std(ddof=1)),
            "branch_b_mean": float(b.mean()),
            "branch_b_std": float(b.std(ddof=1)),
        },
        "directional": directional_consistency(diffs),
        "effect_size": {"cohens_d": d, "ci_lo": lo, "ci_hi": hi},
        "hypothesis_test": paired_t_bonferroni(diffs, n_comparisons),
        "interpretation": _interpret(d),
    }

def _interpret(d):
    abs_d = abs(d)
    if abs_d < 0.2: return "negligible (true null)"
    if abs_d < 0.5: return "small"
    if abs_d < 0.8: return "medium"
    return "large (underpowered null possible if test fails)"
```

- [ ] **Step 2**: Aggregator.

`scripts/aggregate.py`:
```python
"""Aggregate eval JSONs in results/ → results.csv."""
import argparse, csv, json
from pathlib import Path

def main(args):
    in_dir = Path(args.in_dir)
    rows = []
    for f in sorted(in_dir.glob("*.json")):
        d = json.loads(f.read_text())
        # filename pattern: b{branch}_{seed}_{step}.json
        stem = f.stem
        parts = stem.split("_")
        branch, seed, step = parts[0], int(parts[1].lstrip("s")), int(parts[2])
        row = {
            "branch": branch, "seed": seed, "step": step,
            "fid": d["fid"], "fd_dinov2": d["fd_dinov2"],
            "precision": d["precision_recall"]["precision"],
            "recall": d["precision_recall"]["recall"],
            "sharp_lapvar": d["sharpness"]["lapvar"],
            "sharp_hffreq": d["sharpness"]["hffreq"],
            "sharp_sobel": d["sharpness"]["sobel"],
            "wallclock_sec": d["wallclock_sec"],
        }
        rows.append(row)
    if not rows:
        print("no JSONs found")
        return
    fields = list(rows[0].keys())
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✓ wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="results")
    p.add_argument("--out", default="results.csv")
    main(p.parse_args())
```

- [ ] **Step 3**: Test stats.

```python
def test_four_way_report():
    import numpy as np
    from train.eval.stats import four_way_report
    a = [10.0, 11.0, 9.5]
    b = [12.0, 12.5, 11.0]
    out = four_way_report(a, b, n_comparisons=6)
    assert "descriptive" in out and "directional" in out and "effect_size" in out and "hypothesis_test" in out
    assert out["effect_size"]["cohens_d"] < 0  # a < b
    assert out["directional"]["neg"] == 3  # all 3 seeds: a < b
```

- [ ] **Step 4**: Run.

```bash
pytest tests/test_stats.py -v
```

- [ ] **Step 5**: Commit.

```bash
git add scripts/aggregate.py train/eval/stats.py tests/test_stats.py
git commit -m "feat: results aggregator + §2.3 4-way reporting (R1)"
```

---

## Phase 4 — Pre-registration (Day 5)

### Task 4.1: Power analysis script (R1)

**Files:**
- Create: `scripts/power_analysis.py`

- [ ] **Step 1**: Compute minimum detectable Cohen's d for n=3, α=0.05/6, power=0.8.

`scripts/power_analysis.py`:
```python
"""Compute the minimum detectable effect size for the §2.3 power analysis.
Uses scipy noncentral t distribution.
"""
import json
from scipy import stats
import numpy as np

def min_detectable_d(n, alpha, power):
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # solve for d such that P(|T_nc| > t_crit) = power
    # T_nc has noncentrality parameter delta = d * sqrt(n)
    def power_at(d):
        delta = d * np.sqrt(n)
        return 1 - stats.nct.cdf(t_crit, df, delta) + stats.nct.cdf(-t_crit, df, delta)
    lo, hi = 0.01, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_at(mid) < power:
            lo = mid
        else:
            hi = mid
    return mid

def main():
    n = 3
    alpha = 0.05 / 6
    power = 0.80
    d_min = min_detectable_d(n, alpha, power)
    out = {
        "n_seeds": n,
        "alpha_corrected": alpha,
        "power_target": power,
        "min_detectable_cohens_d": float(d_min),
        "interpretation": "Effects with |d| < this value cannot be detected at the pre-registered significance level.",
    }
    print(json.dumps(out, indent=2))
    open("data/power_analysis.json", "w").write(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: Run.

```bash
python scripts/power_analysis.py
```

Expected output: `min_detectable_cohens_d: ~4.5` (확정 수치는 spec §2.3와 일치해야 함).

- [ ] **Step 3**: Commit.

```bash
git add scripts/power_analysis.py data/power_analysis.json
git commit -m "feat: power analysis (n=3 → min Cohen's d ≈ 4.5)"
```

### Task 4.2: Write `docs/prereg.md`

**Files:**
- Create: `docs/prereg.md`

- [ ] **Step 1**: Spec §8 의 7개 항목을 그대로 옮기되, §2.1, §2.2, §2.3 의 내용을 self-contained하게 복사.

`docs/prereg.md`:
```markdown
# Pre-registration: REPA 재평가 연구 (FFHQ-256)

**Frozen at**: <will be filled by tag commit>
**Tag**: `prereg-v1`
**Analyst**: <name>
**Date**: 2026-04-15

## 1. Hypothesis (Spec §2.1 Claim B 복사)

> 고정 계수 REPA는 FFHQ-256에서 수렴 속도와 샘플 sharpness/diversity 사이에 지금까지 특성화되지 않은 trade-off를 보인다. 본 연구는 다섯 가지 계수 스케줄 (off, fixed, decay, warmup, hard cutoff) 에 걸쳐 이 trade-off를 특성화하고, 분포 수준 sharpness metric과 paired human preference를 사용한 diagnostic protocol을 제시한다.

## 2. Pre-registered Outcome Conditions (Spec §2.2 복사)

**Positive**:
- Primary metric (FID-vs-step trajectory AUC, §7.1) 또는 sharpness/diversity metric 중 최소 하나에서 REPA-Fixed가 REPA-Off 대비 통계적으로 유의미하게 (paired t-test, Bonferroni 보정 α=0.05/6) 저하, 그리고
- 스케줄 변형 (decay/warmup/cutoff) 중 최소 하나가 fixed REPA의 수렴 속도 이점을 유지하면서 저하된 축을 부분적으로 회복.

**Null**: 어떤 branch 간에도 통계적 유의미한 차이가 없음.

**Mixed**: 자동 metric ↔ human eval 불일치.

세 경우 모두 publishable.

## 3. Statistical Power (Spec §2.3 복사)

- n=3 seeds, paired t-test, Bonferroni α=0.05/6 ≈ 0.0083.
- Minimum detectable Cohen's d ≈ 4.5 (`scripts/power_analysis.py` 산출).
- 4-way reporting 강제: descriptive / directional / effect size / hypothesis test.
- |d| > 0.8 + non-significant → "underpowered null" 명시. |d| < 0.2 → true null.

## 4. Primary Metric

**FID-vs-step trajectory AUC** over the 4 eval checkpoints (20K, 50K, 80K, 100K), trapezoid integration. Compared via paired t-test (Bonferroni α=0.05/6).

**Secondary**: 100K step의 FID, Precision, Recall, FD-DINOv2, 3 sharpness Wasserstein, Bradley-Terry score, jaggedness.

## 5. Stop Conditions

- Pilot (5 branches × 1 seed × 20K) 에서 어느 metric에서도 식별 가능한 signal이 3+ branches에 걸쳐 없으면 일시 중단, 연구 질문 재평가.
- Compute가 GPU 연속 사용 5일 초과 시 일시 중단, 우선순위 재정의.

## 6. Pre-registered Branch Ranking Expectation (confirmation-bias check)

- Sharpness: B0 > B2 ≈ B4 > B1 > B3
- FID 수렴 속도: B1 ≈ B4 > B2 > B3 > B0

실험자의 사전 가설. 실제 결과가 이와 *완벽히* 일치하면 더 의심.

## 7. Pre-registered Branches (5)

- B0 off: λ(t) = 0
- B1 fixed: λ(t) = 0.5
- B2 cosine decay: λ(t) = 0.5 * 0.5 * (1 + cos(π t / T))
- B3 cosine warmup: λ(t) = 0.5 * 0.5 * (1 - cos(π t / T))
- B4 hard cutoff: λ(t) = 0.5 if t < T/2 else 0

## 8. Strict rule

이 commit 이후 본 문서는 수정 불가. 추가 분석은 paper에서 "exploratory" 로 명시.
```

- [ ] **Step 2**: Self-review.

```bash
$EDITOR docs/prereg.md
# 모든 placeholder (<...>) 를 실제 값으로 대체. <Frozen at> 는 commit 후 태그 시각으로 갱신.
```

- [ ] **Step 3**: Commit (no tag yet).

```bash
git add docs/prereg.md
git commit -m "prereg: pre-registered hypotheses, outcome conditions, power statement"
```

### Task 4.3: Tag `prereg-v1` (frozen)

- [ ] **Step 1**: Tag.

```bash
git tag -a prereg-v1 -m "Pre-registration frozen. No post-hoc modification."
```

- [ ] **Step 2**: Verify tag.

```bash
git show prereg-v1 | head -20
git tag -l
```

- [ ] **Step 3**: Push tag (if remote configured).

```bash
git push origin reeval --tags
```

- [ ] **Step 4**: Verify CI/CD or local check that the tag is immutable.

```bash
echo "Hypothesis frozen at $(git rev-parse prereg-v1)" >> docs/prereg.md
```

⚠ 이 echo는 tag *이후* docs/prereg.md를 수정하므로, 별도 commit이지 prereg-v1 tag 자체에는 없는 변경. tag-after-frozen-fact를 별도 commit으로 처리:

```bash
git add docs/prereg.md
git commit -m "docs: record prereg-v1 commit hash post-tag (informational only)"
```

---

## Phase 5 — Pilot Runs (Day 6-7)

### Task 5.1: Wallclock pilot 실행 (R3 mitigation)

**Files:**
- Create: `data/wallclock_benchmark.json`

- [ ] **Step 1**: Single 5K-step run으로 wallclock 측정.

```bash
time python scripts/train.py --branch fixed --seed 42 --total_steps 5000 --out exps/wallclock_test 2>&1 | tee data/wallclock_log.txt
```

- [ ] **Step 2**: 결과 json화.

```bash
python -c "
import json, re
log = open('data/wallclock_log.txt').read()
m = re.search(r'real\s+(\d+)m([\d.]+)s', log)
sec = int(m.group(1)) * 60 + float(m.group(2))
hours_per_5k = sec / 3600
hours_per_100k = hours_per_5k * 20
hours_per_15_runs = hours_per_100k * 15
out = {
    'wallclock_5k_sec': sec,
    'hours_per_100k': hours_per_100k,
    'hours_per_15_runs': hours_per_15_runs,
    'budget_check': hours_per_15_runs < 100,
    'spec_estimate_hours_per_run': 3.5,
}
print(json.dumps(out, indent=2))
open('data/wallclock_benchmark.json','w').write(json.dumps(out, indent=2))
"
```

- [ ] **Step 3**: 결과 평가:
  - hours_per_100k ≤ 5 시간 → spec 외삽이 적절. 진행.
  - 5-7 시간 → 약간 over. seed를 3→2로 줄이거나 (§10 fallback) eval frequency를 줄여 보완.
  - > 7 시간 → spec 재평가 필요. Stop condition 발동.

- [ ] **Step 4**: Commit.

```bash
git add data/wallclock_benchmark.json data/wallclock_log.txt
git commit -m "data: wallclock benchmark (5K steps → extrapolate to 15 runs)"
```

### Task 5.2: Pilot training sweep (5 branches × 1 seed × 20K)

**Files:**
- Create: `scripts/run_pilot.sh`

- [ ] **Step 1**: Sweep script.

`scripts/run_pilot.sh`:
```bash
#!/bin/bash
set -euo pipefail
SEED=42
STEPS=20000
for BRANCH in off fixed decay warmup cutoff; do
    OUT="exps/pilot_${BRANCH}_s${SEED}"
    if [ -d "$OUT" ] && [ -f "$OUT/done" ]; then
        echo "skip $OUT (already done)"
        continue
    fi
    echo ">>> $BRANCH s${SEED} ${STEPS} steps"
    python scripts/train.py --branch "$BRANCH" --seed "$SEED" --total_steps "$STEPS" --out "$OUT"
    touch "$OUT/done"
done
```

```bash
chmod +x scripts/run_pilot.sh
```

- [ ] **Step 2**: 디스크 monitor를 background로 띄우고 sweep 시작.

```bash
python scripts/disk_monitor.py --min_gb 15 --interval 120 --exit_on_alert &
DISK_PID=$!
bash scripts/run_pilot.sh
kill $DISK_PID
```

Expected: 5 × ~40min ≈ 3-4 GPU-hours.

- [ ] **Step 3**: Commit.

```bash
git add scripts/run_pilot.sh
git commit -m "ops: pilot sweep (5 × 1 × 20K steps)"
```

### Task 5.3: Pilot eval @ 20K + signal check

- [ ] **Step 1**: Eval all 5 pilot checkpoints.

```bash
mkdir -p results/pilot
for B in off fixed decay warmup cutoff; do
    python scripts/eval.py --ckpt exps/pilot_${B}_s42/checkpoints/ema_020000.pt \
        --out results/pilot/${B}_s42_20000.json \
        --preview_dir exps/pilot_${B}_s42/eval_20000/preview
done
```

- [ ] **Step 2**: Aggregate + 빠른 비교.

```bash
python scripts/aggregate.py --in_dir results/pilot --out results/pilot.csv
column -t -s, results/pilot.csv
```

- [ ] **Step 3**: Stop condition 체크 (§8.5 / §10).

판정 규칙:
- 5 branch의 FID가 모두 같은 범위 (max - min < 5%) → signal 부족, *2-3 branches 추가 디버그* 필요
- 1+ branch 의 sample이 시각적으로 손상 (random noise) → bug 의심, debug
- Loss curve가 NaN/Inf 발생 → halt, debug

문제 없으면 main run으로 진행. 문제 있으면 Phase 2 의 equivalence test 재실행.

- [ ] **Step 4**: Commit.

```bash
git add results/pilot.csv results/pilot/
git commit -m "data: pilot eval results (5 branches × 1 seed × 20K)"
```

---

## Phase 6 — Main Training Runs (Day 8-11)

### Task 6.1: Main sweep launcher

**Files:**
- Create: `scripts/run_main.sh`

- [ ] **Step 1**: Script.

`scripts/run_main.sh`:
```bash
#!/bin/bash
set -euo pipefail
STEPS=100000
SEEDS=(42 1337 2024)
BRANCHES=(off fixed decay warmup cutoff)

for SEED in "${SEEDS[@]}"; do
    for BRANCH in "${BRANCHES[@]}"; do
        OUT="exps/${BRANCH}_s${SEED}"
        if [ -f "$OUT/done" ]; then
            echo "skip $OUT (done)"
            continue
        fi
        echo ">>> $BRANCH s${SEED} $STEPS"
        python scripts/train.py --branch "$BRANCH" --seed "$SEED" --total_steps "$STEPS" --out "$OUT"
        touch "$OUT/done"
        df -h .
    done
done
echo "✓ all 15 main runs complete"
```

```bash
chmod +x scripts/run_main.sh
```

- [ ] **Step 2**: Commit.

```bash
git add scripts/run_main.sh
git commit -m "ops: main sweep launcher (15 runs sequential, resumable)"
```

### Task 6.2: Launch with disk monitor

- [ ] **Step 1**: tmux 또는 screen 사용 권장.

```bash
tmux new -s main
python scripts/disk_monitor.py --min_gb 12 --interval 300 --exit_on_alert > data/disk_log_main.csv 2>&1 &
bash scripts/run_main.sh 2>&1 | tee exps/main_run.log
```

- [ ] **Step 2**: 매일 확인.

```bash
tmux attach -t main
ls exps/*/done | wc -l   # 진행률 (0 → 15)
```

- [ ] **Step 3**: 완료 후 commit (results 폴더만, exps는 .gitignore).

```bash
ls exps/*/done | wc -l   # 15 expected
git status   # data/disk_log_main.csv 정도만 있을 것
git add data/disk_log_main.csv exps/main_run.log
git commit -m "data: main 15 runs complete (logs + disk history)"
```

---

## Phase 7 — Automatic Evaluation (Day 12-14)

### Task 7.1: Run eval on all 60 (run × ckpt) pairs

**Files:**
- Create: `scripts/run_eval_all.sh`

- [ ] **Step 1**: Script.

`scripts/run_eval_all.sh`:
```bash
#!/bin/bash
set -euo pipefail
mkdir -p results/main
for SEED in 42 1337 2024; do
    for BRANCH in off fixed decay warmup cutoff; do
        for STEP in 020000 050000 080000 100000; do
            CKPT="exps/${BRANCH}_s${SEED}/checkpoints/ema_${STEP}.pt"
            OUT="results/main/${BRANCH}_s${SEED}_${STEP}.json"
            if [ -f "$OUT" ]; then continue; fi
            python scripts/eval.py --ckpt "$CKPT" --out "$OUT" \
                --preview_dir "exps/${BRANCH}_s${SEED}/eval_${STEP}/preview"
        done
    done
done
echo "✓ all 60 evals complete"
```

```bash
chmod +x scripts/run_eval_all.sh
bash scripts/run_eval_all.sh 2>&1 | tee exps/eval_all.log
```

Expected: 60 × ~25 min ≈ 25 GPU-hours.

- [ ] **Step 2**: Aggregate.

```bash
python scripts/aggregate.py --in_dir results/main --out results.csv
wc -l results.csv  # expect 61 lines (1 header + 60)
```

- [ ] **Step 3**: Commit.

```bash
git add results.csv results/main/ exps/eval_all.log
git commit -m "data: all 60 main eval JSONs aggregated to results.csv"
```

### Task 7.2: Compute primary metric (trajectory AUC)

**Files:**
- Create: `scripts/compute_primary.py`
- Modify: `results.csv`

- [ ] **Step 1**: Script.

`scripts/compute_primary.py`:
```python
"""Compute primary metric (FID-vs-step trajectory AUC) per (branch, seed).
Appends to results.csv as new row group or writes results_primary.csv.
"""
import pandas as pd
from train.eval.auc import fid_step_auc

df = pd.read_csv("results.csv")
rows = []
for (branch, seed), g in df.groupby(["branch", "seed"]):
    g = g.sort_values("step")
    rows.append({
        "branch": branch, "seed": seed,
        "fid_auc": fid_step_auc(g["step"].tolist(), g["fid"].tolist()),
        "fid_at_100k": float(g[g["step"] == 100000]["fid"].iloc[0]),
        "sharp_lapvar_at_100k": float(g[g["step"] == 100000]["sharp_lapvar"].iloc[0]),
        "sharp_hffreq_at_100k": float(g[g["step"] == 100000]["sharp_hffreq"].iloc[0]),
        "sharp_sobel_at_100k": float(g[g["step"] == 100000]["sharp_sobel"].iloc[0]),
        "fd_dinov2_at_100k": float(g[g["step"] == 100000]["fd_dinov2"].iloc[0]),
        "precision_at_100k": float(g[g["step"] == 100000]["precision"].iloc[0]),
        "recall_at_100k": float(g[g["step"] == 100000]["recall"].iloc[0]),
    })
out = pd.DataFrame(rows)
out.to_csv("results_primary.csv", index=False)
print(out)
```

- [ ] **Step 2**: Run.

```bash
python scripts/compute_primary.py
```

- [ ] **Step 3**: Commit.

```bash
git add scripts/compute_primary.py results_primary.csv
git commit -m "data: compute FID-vs-step trajectory AUC per (branch, seed)"
```

### Task 7.3: Preliminary plots

**Files:**
- Create: `scripts/plots.py`

- [ ] **Step 1**: 첫 plot — FID vs step 곡선, branch별 색상.

`scripts/plots.py`:
```python
"""Generate publication figures."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def fid_vs_step(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for branch, g in df.groupby("branch"):
        agg = g.groupby("step")["fid"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["step"], agg["mean"], yerr=agg["std"], label=branch, marker="o")
    ax.set_xlabel("Training step")
    ax.set_ylabel("FID (clean-fid)")
    ax.set_title("FID vs training step (mean ± std over 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"✓ {out}")

def main(args):
    df = pd.read_csv(args.results)
    fid_vs_step(df, "paper/figures/fig_fid_vs_step.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.csv")
    main(p.parse_args())
```

```bash
mkdir -p paper/figures
python scripts/plots.py
```

- [ ] **Step 2**: Commit.

```bash
git add scripts/plots.py paper/figures/fig_fid_vs_step.png
git commit -m "feat: preliminary FID-vs-step plot"
```

---

## Phase 8 — Human Evaluation (Day 15-18)

### Task 8.1: Generate matched-noise image pairs

**Files:**
- Create: `scripts/generate_eval_pairs.py`

- [ ] **Step 1**: Script.

`scripts/generate_eval_pairs.py`:
```python
"""Generate 6 pair-types × 50 image-pairs from matched noise seeds (§7.3)."""
import json
from pathlib import Path
import torch

PAIR_TYPES = [
    ("off", "fixed"), ("off", "decay"), ("off", "warmup"),
    ("fixed", "decay"), ("fixed", "cutoff"), ("decay", "cutoff"),
]

def main():
    from train.eval.sample import generate_samples
    # ... per pair, load both EMA ckpts, share the same noise seeds, decode, save side-by-side ...
    out = Path("human_eval/pairs")
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for a, b in PAIR_TYPES:
        ckpt_a = f"exps/{a}_s42/checkpoints/ema_100000.pt"
        ckpt_b = f"exps/{b}_s42/checkpoints/ema_100000.pt"
        # ... load models ...
        for i in range(50):
            noise_seed = 100000 + i
            # generate with the same seed for both
            # save as out/{a}_{b}/{i:03d}_a.png and {i:03d}_b.png
            pass  # 구체 구현 시 ckpt loading + sampling 호출
        manifest.append({"a": a, "b": b, "n": 50})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: Run.

```bash
python scripts/generate_eval_pairs.py
ls human_eval/pairs/
```

- [ ] **Step 3**: Commit.

```bash
git add scripts/generate_eval_pairs.py human_eval/pairs/manifest.json
git commit -m "feat: generate 300 matched-noise image pairs for human eval"
```

### Task 8.2: Human eval Flask 서버

**Files:**
- Create: `scripts/human_eval_server.py`
- Create: `templates/human_eval.html`

- [ ] **Step 1**: Minimal server.

`scripts/human_eval_server.py`:
```python
"""Minimal Flask 2AFC server for human eval.
Serves random pair from human_eval/pairs/, records response to CSV.
"""
import csv, json, random
from pathlib import Path
from flask import Flask, request, render_template, send_file

app = Flask(__name__, template_folder="../templates")
PAIRS_DIR = Path("human_eval/pairs")
RESPONSES = Path("human_eval/responses.csv")

@app.route("/")
def index():
    manifest = json.loads((PAIRS_DIR / "manifest.json").read_text())
    pair = random.choice(manifest)
    idx = random.randint(0, pair["n"] - 1)
    swap = random.random() < 0.5
    return render_template("human_eval.html",
        pair=pair, idx=idx, swap=swap)

@app.route("/img/<branch_a>_<branch_b>/<int:idx>/<which>")
def img(branch_a, branch_b, idx, which):
    return send_file(PAIRS_DIR / f"{branch_a}_{branch_b}" / f"{idx:03d}_{which}.png")

@app.route("/submit", methods=["POST"])
def submit():
    RESPONSES.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESPONSES.exists()
    with open(RESPONSES, "a") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "branch_a", "branch_b", "idx", "swap",
                        "rater", "q_sharp", "q_real"])
        w.writerow([
            request.form["timestamp"], request.form["branch_a"], request.form["branch_b"],
            request.form["idx"], request.form["swap"], request.form["rater"],
            request.form["q_sharp"], request.form["q_real"],
        ])
    return "OK"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
```

- [ ] **Step 2**: Template.

`templates/human_eval.html`:
```html
<!doctype html>
<html><body>
<h2>2AFC: Which image is sharper / more like a real photo?</h2>
<form id="f">
  <input type="hidden" name="timestamp">
  <input type="hidden" name="branch_a" value="{{ pair.a }}">
  <input type="hidden" name="branch_b" value="{{ pair.b }}">
  <input type="hidden" name="idx" value="{{ idx }}">
  <input type="hidden" name="swap" value="{{ swap }}">
  <label>Rater: <input name="rater" required></label><br>
  <div>
    <img src="/img/{{ pair.a }}_{{ pair.b }}/{{ idx }}/{{ 'b' if swap else 'a' }}" width=320>
    <img src="/img/{{ pair.a }}_{{ pair.b }}/{{ idx }}/{{ 'a' if swap else 'b' }}" width=320>
  </div>
  <p>Sharper: <label><input type="radio" name="q_sharp" value="left" required> Left</label>
              <label><input type="radio" name="q_sharp" value="right"> Right</label></p>
  <p>More real: <label><input type="radio" name="q_real" value="left" required> Left</label>
                <label><input type="radio" name="q_real" value="right"> Right</label></p>
  <button type="submit">Submit & Next</button>
</form>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  fd.set('timestamp', new Date().toISOString());
  await fetch('/submit', {method:'POST', body: fd});
  location.reload();
};
</script>
</body></html>
```

- [ ] **Step 3**: 로컬 테스트.

```bash
python scripts/human_eval_server.py
# 브라우저에서 http://127.0.0.1:5000 접속, 1-2개 응답 입력해서 동작 확인
cat human_eval/responses.csv
```

- [ ] **Step 4**: Commit.

```bash
git add scripts/human_eval_server.py templates/human_eval.html
git commit -m "feat: minimal Flask 2AFC server for human eval (CSV log)"
```

### Task 8.3-8.5: 응답 수집 + Bradley-Terry 분석

- [ ] **Step 1**: Self + 친구 2-3명에게 링크 공유 (또는 SSH tunnel 사용).
- [ ] **Step 2**: 목표 ≥ 1000 응답 (§14 성공 기준).
- [ ] **Step 3**: Bradley-Terry analysis script.

`scripts/bradley_terry.py`:
```python
"""Bradley-Terry preference scores from human_eval/responses.csv.
1000-iter bootstrap 95% CI per branch.
"""
import pandas as pd
import numpy as np
from itertools import combinations

def fit_bt(comparisons, branches, max_iter=200):
    # Standard MM algorithm
    n = len(branches)
    idx = {b: i for i, b in enumerate(branches)}
    scores = np.ones(n)
    for _ in range(max_iter):
        new_scores = np.zeros(n)
        for (i_winner, i_loser), wins in comparisons.items():
            new_scores[i_winner] += wins / (scores[i_winner] + scores[i_loser])
        new_scores = np.where(new_scores > 0, new_scores, 1e-9)
        new_scores /= new_scores.sum()
        if np.allclose(new_scores, scores / scores.sum(), atol=1e-7):
            break
        scores = new_scores
    return scores

def main():
    df = pd.read_csv("human_eval/responses.csv")
    branches = sorted(set(df["branch_a"]).union(df["branch_b"]))
    # ... build comparison dict ...
    # ... bootstrap CI ...
    # ... save to human_eval/bt_scores.csv ...

if __name__ == "__main__":
    main()
```

- [ ] **Step 4**: Run + commit.

```bash
python scripts/bradley_terry.py
git add human_eval/responses.csv human_eval/bt_scores.csv scripts/bradley_terry.py
git commit -m "data: human eval responses + Bradley-Terry analysis"
```

---

## Phase 9 — Interpolation Study (Day 19)

### Task 9.1: Slerp generation + jaggedness

**Files:**
- Create: `scripts/interpolation_study.py`
- Create: `train/eval/jaggedness.py`

- [ ] **Step 1**: Jaggedness module.

`train/eval/jaggedness.py`:
```python
"""§7.4 interpolation jaggedness via DINOv2 patch-token mean features."""
import torch

def slerp(z0, z1, t):
    omega = torch.acos((z0 * z1).sum(dim=-1) / (z0.norm(dim=-1) * z1.norm(dim=-1) + 1e-8))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so).unsqueeze(-1) * z0 + (torch.sin(t * omega) / so).unsqueeze(-1) * z1

def jaggedness(features_per_pair):
    """features_per_pair: list of (n_steps, dim) tensors. Returns mean jaggedness."""
    out = []
    for f in features_per_pair:
        d = (f[1:] - f[:-1]).norm(dim=-1)  # length n_steps - 1
        out.append((d.var() / (d.mean() + 1e-8)).item())
    return float(sum(out) / len(out))
```

- [ ] **Step 2**: Script.

`scripts/interpolation_study.py`:
```python
"""Per branch: 10 noise pairs × 9 slerp steps → DINOv2 features → jaggedness + grid."""
import torch
from train.eval.jaggedness import slerp, jaggedness
# ... per branch: load EMA ckpt, generate 10 × 9 samples, compute features, compute jaggedness ...
# ... save jaggedness to interpolation/{branch}/jaggedness.json ...
# ... save grid PNG to interpolation/{branch}/grid.png ...

def main():
    pass  # 구체 구현은 sample 생성 + DINOv2 feature 추출

if __name__ == "__main__":
    main()
```

- [ ] **Step 3**: Run + commit.

```bash
python scripts/interpolation_study.py
git add scripts/interpolation_study.py train/eval/jaggedness.py interpolation/
git commit -m "feat: interpolation jaggedness + grid (5 branches × 10 pairs × 9 steps)"
```

---

## Phase 10 — Statistical Analysis & Final Plots (Day 20-21)

### Task 10.1: Apply 4-way reporting per §2.3

**Files:**
- Create: `scripts/final_analysis.py`
- Create: `results/final_report.json`

- [ ] **Step 1**: Script.

`scripts/final_analysis.py`:
```python
"""Apply §2.3 4-way reporting to all pre-registered comparisons.
Outputs: results/final_report.json + per-comparison rows in results/comparisons.csv
"""
import json
import pandas as pd
from train.eval.stats import four_way_report

PRIMARY_METRICS = ["fid_auc", "sharp_lapvar_at_100k", "sharp_hffreq_at_100k", "sharp_sobel_at_100k",
                   "fd_dinov2_at_100k", "recall_at_100k"]

PAIRS = [("off","fixed"), ("off","decay"), ("off","warmup"),
         ("fixed","decay"), ("fixed","cutoff"), ("decay","cutoff")]

def main():
    df = pd.read_csv("results_primary.csv")
    # Pre-registered Bonferroni divisor = 6 (pair count, per §8 #3, frozen).
    # Metric-level tests are treated as exploratory family — not separately corrected.
    n_comp_prereg = 6
    out = {
        "per_pair": {},
        "n_comparisons_prereg": n_comp_prereg,
        "metric_family_size_exploratory": len(PRIMARY_METRICS),
        "note": "Bonferroni applied at pair level (6) per pre-registration; metric-level family is exploratory.",
    }
    for a, b in PAIRS:
        out["per_pair"][f"{a}_vs_{b}"] = {}
        for metric in PRIMARY_METRICS:
            va = df[df["branch"] == a][metric].tolist()
            vb = df[df["branch"] == b][metric].tolist()
            out["per_pair"][f"{a}_vs_{b}"][metric] = four_way_report(va, vb, n_comparisons=n_comp_prereg)
    open("results/final_report.json", "w").write(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2**: Run.

```bash
python scripts/final_analysis.py
```

- [ ] **Step 3**: Commit.

```bash
git add scripts/final_analysis.py results/final_report.json
git commit -m "analysis: §2.3 4-way reporting on all 36 (pair × metric) comparisons"
```

### Task 10.2: Auto vs human correlation

**Files:**
- Create: `scripts/auto_human_correlation.py`

- [ ] **Step 1**: Script.

`scripts/auto_human_correlation.py`:
```python
"""Pearson correlation between Bradley-Terry score and each automatic sharpness metric (§7.3)."""
import pandas as pd
from scipy.stats import pearsonr

bt = pd.read_csv("human_eval/bt_scores.csv")  # branch, bt_score
auto = pd.read_csv("results_primary.csv").groupby("branch").mean(numeric_only=True).reset_index()
joined = bt.merge(auto, on="branch")
for col in ["sharp_lapvar_at_100k", "sharp_hffreq_at_100k", "sharp_sobel_at_100k"]:
    r, p = pearsonr(joined["bt_score"], joined[col])
    print(f"{col}: r={r:.3f} p={p:.3f}")
```

- [ ] **Step 2**: Run + commit.

```bash
python scripts/auto_human_correlation.py | tee results/correlations.txt
git add scripts/auto_human_correlation.py results/correlations.txt
git commit -m "analysis: auto sharpness vs human BT score correlation"
```

### Task 10.3: All publication figures

- [ ] **Step 1**: `scripts/plots.py` 확장 — 5개 figure 추가:
  - Fig 1: FID vs step (이미 Task 7.3)
  - Fig 2: Sharpness Wasserstein bar chart, 3 statistics × 5 branches
  - Fig 3: Jaggedness bar chart, 5 branches
  - Fig 4: Bradley-Terry score with 95% CI
  - Fig 5: Auto-vs-human correlation scatter

- [ ] **Step 2**: 각 figure 함수 구현 (matplotlib, ~30 줄씩).

- [ ] **Step 3**: 모두 생성.

```bash
python scripts/plots.py --all
ls paper/figures/
```

- [ ] **Step 4**: Commit.

```bash
git add scripts/plots.py paper/figures/
git commit -m "feat: all 5 publication figures"
```

### Task 10.4: Pre-registered verdict application

**Files:**
- Create: `results/verdict.md`

- [ ] **Step 1**: §2.2 의 positive/null/mixed 규칙을 results/final_report.json에 적용.

```bash
python -c "
import json
fr = json.load(open('results/final_report.json'))

# Positive: REPA-Off (B0) 와 REPA-Fixed (B1) 사이에 primary OR sharpness/diversity 중 하나가 sig
pair = fr['per_pair']['off_vs_fixed']
sig_metrics = [m for m, r in pair.items() if r['hypothesis_test']['significant']]

# 그리고 스케줄 변형 중 하나가 fixed 대비 회복
recovery = []
for sched in ['decay', 'warmup', 'cutoff']:
    p = fr['per_pair'].get(f'fixed_vs_{sched}') or fr['per_pair'].get(f'{sched}_vs_fixed')
    if p:
        recovery_metrics = [m for m, r in p.items() if r['hypothesis_test']['significant']]
        recovery.append((sched, recovery_metrics))

verdict = 'POSITIVE' if (sig_metrics and any(r for _, r in recovery)) else 'NULL'
# (Mixed는 human eval과 결합)

print(f'Verdict: {verdict}')
print(f'Significant B0-B1 metrics: {sig_metrics}')
print(f'Schedule recoveries: {recovery}')

open('results/verdict.md', 'w').write(f'''# Pre-registered Verdict

Date: 2026-04-15+21
Verdict: **{verdict}**

## Significant degradation (B0 vs B1)
{sig_metrics}

## Schedule-based recoveries
{recovery}

## Underpowered null check (§2.3)
- TODO: |d| > 0.8 가 있지만 not significant 인 metric 명시
''')
"
```

- [ ] **Step 2**: 수동으로 verdict.md 보강 — underpowered null vs true null 구분 (§2.3).

- [ ] **Step 3**: Commit.

```bash
git add results/verdict.md
git commit -m "analysis: apply pre-registered verdict per §2.2 + §2.3 underpowered-null check"
```

---

## Self-Review

### Spec coverage

§1 Research Question — Phase 4 (prereg) + Phase 6-7 (data)
§2 Claim B + Pre-reg + Power — Phase 4
§3 Purity Principle — Task 2.1 (BASELINE.md)
§4 FFHQ-256 dataset — Phase 1
§4.4 Storage budget — Task 0.3 + Task 2.8 + scripts/disk_monitor.py 통합
§5 SiT-B/2 + SD-VAE 학습 설정 — Task 2.4-2.7
§6 5 branches — Task 2.3
§7.1 자동 metric — Phase 3 (Tasks 3.1-3.6)
§7.2 sharpness Wasserstein — Task 3.3
§7.3 human eval 2AFC — Phase 8
§7.4 interpolation jaggedness — Phase 9
§8 Pre-registration — Phase 4
§9 Timeline — Phases 0-10 mapped to Days 1-21
§10 Risks — wallclock pilot (Task 5.1) + disk monitor (Task 0.3) + equivalence tests (Tasks 2.6, 2.7) + storage policy (Task 2.8)
§11 Deliverables — 모든 file 생성 task에 mapping
§12 부록 명령어 — 각 Phase의 step별 bash 블록에 분산

### Placeholder scan

발견된 placeholder (구체화 필요):
- Task 2.1: REPA fork inventory 후에 정확한 file path를 BASELINE.md에 채울 것
- Task 2.4, 2.5: 위와 동일 (수정 대상 file path는 inventory 후 확정)
- Task 2.6, 2.7: train_step / loss curve 비교 코드는 actual training loop signature를 보고 채움
- Task 3.1: DDPM/Euler sampling — REPA fork의 sampling utility 발견 시 그것 사용 (consistency)
- Task 3.7: SiT model class import path — Task 2.1 inventory 결과 반영
- Task 8.1: matched-noise pair generation — model loading + sampling을 Task 2.1 / 3.1 결과 기반으로 구체화
- Task 9.1: interpolation_study.py — 동일

이 placeholder들은 모두 *Task 2.1 inventory* 라는 명확한 single dependency에 묶여 있음. Inventory 결과를 보고 한꺼번에 fill-in.

### Type consistency

- `BRANCHES` dict (Task 2.3) ↔ `compute_loss(branch_name)` (Task 2.4) ↔ `--branch` CLI (Task 2.4) — naming consistent: `off/fixed/decay/warmup/cutoff`
- Latent shape (4, 32, 32) — 일관 (Task 1.2, 2.2, 3.1, 3.7)
- Sample uint8 tensor (N, 3, 256, 256) — 일관 (Task 3.1, 3.2, 3.3, 3.4, 3.5)
- Eval step set {20K, 50K, 80K, 100K} — 일관 (Task 2.8, 7.1)
- Bonferroni n_comparisons = 6 — Task 4.2 prereg에서 정의, Task 3.8 stats에서 사용, Task 10.1 final_analysis에서 36 (pair × metric) 으로 확장 — **불일치**.

**Fix**: §8 의 Bonferroni 분모는 6 (pair 비교 수) 이지만 Task 10.1은 metric 6개도 곱함. Task 10.1 의 n_comparisons 계산은 spec과 일치하도록 검증 필요. Pre-reg은 "6 comparisons" 으로 frozen이므로 Task 10.1도 n_comparisons=6 사용.

`scripts/final_analysis.py` 의 line `n_comp = out["n_comparisons"]  # for Bonferroni` 를 `n_comp = 6` (pre-registered) 로 fix.

```python
# scripts/final_analysis.py 의 다음 수정 필요:
out["n_comparisons"] = 6  # pre-registered (§8 #3, frozen)
```

이 fix를 Phase 10 Task 10.1 step 1에 반영. (이 plan 의 self-review 결과는 plan을 fix하라는 것이지 spec을 fix하라는 것이 아님.)

### Scope check

- Day 1-21 (Phase 0-10): Implementation work — covered in this plan ✓
- Day 22-28 (writing): Out of scope, manual

---

## Execution Handoff

Plan complete and saved to `docs/plan-ko.md`. 두 가지 실행 옵션:

**1. Subagent-Driven (recommended)** — task별로 fresh subagent를 dispatch, 각 task 완료 시 review, 빠른 iteration

**2. Inline Execution** — 이 session에서 executing-plans skill로 batch 실행, checkpoint마다 review

어떤 방식으로 진행하시겠어요?
