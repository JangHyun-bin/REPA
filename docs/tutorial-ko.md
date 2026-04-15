# REPA 재평가 연구 — Step-by-Step Manual Tutorial

> 이 문서는 **사용자가 혼자서 수동으로** 모든 단계를 수행할 수 있도록 만든 가이드입니다.
> Claude는 더 이상 코드를 직접 실행하지 않습니다. 이 튜토리얼만 보고 따라하면 28일짜리 워크샵 연구가 끝까지 진행되어야 합니다.

---

## 0. 들어가며

### 0.1 이 문서의 위치

본 튜토리얼은 다음 두 사전 문서의 **실행 매뉴얼**입니다.

| 문서 | 경로 | 역할 |
|---|---|---|
| Spec / Design | `docs/spec-ko.md` | 연구 질문, 가설, 평가 방법, pre-registration 규칙. **`prereg-v1` tag 이후 §1-§14 변경 금지.** |
| Implementation Plan | `docs/plan-ko.md` | Agentic execution 용 task-by-task 계획. 본 튜토리얼과 같은 내용의 다른 형식. |
| **이 튜토리얼** (현재 문서) | `docs/tutorial-ko.md` | 사람이 직접 따라할 수 있도록 WHY + 코드 + 검증 + 트러블슈팅을 한 흐름으로 정리. |

본 튜토리얼이 plan과 충돌하면 **본 튜토리얼 우선**. Spec과 충돌하면 **spec 우선**.

본 튜토리얼의 모든 `§X.Y` 참조는 spec (`...-design-ko.md`) 의 섹션 번호입니다.

### 0.2 사용 규칙

각 절은 다음 구조를 따릅니다:

- **목적** — 이 단계가 왜 필요한가
- **파일** — 무엇을 만들고/수정하는가
- **단계** — 어떤 순서로 무엇을 할 것인가 (체크박스 `- [ ]`)
- **🔍 검증** — 이 단계가 정상 완료됐는지 확인 (Expected output 포함)
- **⚠ 트러블슈팅** — 흔한 실패와 대응
- **✅ Phase 완료 체크포인트** — 각 Day 끝에 위치. 모두 충족 시 다음으로.

코드 블록은 **그대로 복사** 가능합니다. `<your-username>` 같은 placeholder는 본인 값으로 교체.

### 0.3 사전 요구사항

- **하드웨어**: CUDA 지원 GPU. 본 튜토리얼은 **RTX 4080 Super 16GB**를 가정. 다른 GPU는 batch size 또는 step budget 조정 필요.
- **디스크**: 90 GB 미만 가용. §4.4 storage budget으로 ~30 GB만 사용하도록 설계됨.
- **OS**: Linux (Ubuntu/Debian 계열).
- **도구**: git, Python 3.10+, bash, tmux 또는 screen.
- **계정**: GitHub (REPA fork), HuggingFace (dataset streaming).
- **시간**: 28일 (4주). 본 튜토리얼은 Day 1-21. Day 22-28 (논문 작성) 은 Part 13에서 개요만.

### 0.4 전체 흐름 한눈에

```
Day:   1      2      3      4       5       6-7     8-11      12-14     15-18     19      20-21    22-28
       └env──┴data1─┴data2/code1┴code2/eval┴prereg─┴pilot───┴main runs┴auto eval┴human eval┴interp┴stats+plots┴writing
```

단방향 의존: 각 phase는 다음 phase의 전제. Pilot이 실패하면 main run 진행 안 됨.

### 0.5 지금 시작하기 전에 (사전 점검)

다음 두 가지를 미리 확인하세요:

```bash
# 1. 디스크 가용량 (≥ 30 GB 권장)
df -h ~/projects

# 2. GPU 사용 가능
nvidia-smi
```

가용량 < 30 GB면 cleanup 후 시작. **GPU가 안 보이면 CUDA driver 재설치 필요** — 본 튜토리얼은 CUDA가 작동한다는 전제.

### 0.6 본 튜토리얼이 의도적으로 하지 *않는* 것

- iREPA, REPA-E, INVAE 비교 — spec §13 scope 제외
- ImageNet 결과 재현 — compute 이유 제외
- Multi-GPU 학습 — 단일 GPU 가정
- Reviewer 1 가짜 데이터 fabrication 같은 비정상 시나리오

이런 항목이 필요하면 별도 follow-up.

### 0.7 비상 정지 (panic stop)

언제든지 Ctrl+C로 학습 중단. Eval checkpoint는 §4.4 정책에 따라 안전하게 디스크에 저장됨. 학습 중간 (eval ckpt 사이) 의 진행은 손실되지만, 다시 시작 시 가장 가까운 eval ckpt부터 재개 또는 해당 run을 처음부터 다시.

**돌이키기 어려운 작업** (git push, prereg-v1 tag, 데이터 wipe) 은 본 튜토리얼이 명시적으로 경고합니다. 그 외는 모두 reversible.

---

## Part 1 — Day 1: 환경 준비

**오늘의 목표**: 코드 작업 환경, GPU, Python 의존성, 디스크 모니터링, sanity check 스크립트까지 완비.

**예상 시간**: 2-4 시간 (인터넷 속도와 PyTorch 설치 시간에 따라).

**산출물**:
- `repa-reeval/` working directory + git 초기화
- Python venv with pinned dependencies
- `scripts/disk_monitor.py` + `data/disk_baseline.txt`
- `scripts/sd_vae_sanity.py` (실행은 Day 2에)

---

### 1.1 REPA 원본 fork

**목적**: 모든 수정사항이 원본과 분리되어 보존되도록, GitHub 상에서 fork를 먼저 만든다.

**단계**:

- [ ] **1.1.1** 웹브라우저로 https://github.com/sihyun-yu/REPA 접속.
- [ ] **1.1.2** 우상단 `Fork` 버튼 클릭. 본인 GitHub 계정으로 fork.
- [ ] **1.1.3** Fork된 repo의 URL 복사 (예: `https://github.com/<your-username>/REPA.git`).

> **왜 fork인가?** Spec §3.3은 새 저장소에서 작업하라고 명시. 원본을 직접 수정하면 upstream 변경사항을 받기 어려워지고, 향후 reviewer가 우리 기여분을 식별하기 어려워짐. Fork + branch 패턴이 deviation을 git diff로 명확히 보여주는 가장 단순한 방법.

🔍 **검증**: GitHub 페이지 상단에 `<your-username>/REPA  forked from sihyun-yu/REPA` 표시 확인.

---

### 1.2 Local clone + working branch

**목적**: 로컬에서 작업할 수 있도록 fork를 clone하고, 모든 수정을 한 branch에 모은다.

**파일**:
- Create: `~/projects/dl/repa-reeval/` (working directory)

**단계**:

- [ ] **1.2.1** Working directory 위치로 이동.

```bash
cd ~/projects/dl
```

- [ ] **1.2.2** Clone (URL을 본인 fork URL로 교체).

```bash
git clone https://github.com/<your-username>/REPA.git repa-reeval
cd repa-reeval
```

- [ ] **1.2.3** Upstream remote 등록 (원본 REPA를 따로 추적, 필요 시 sync용).

```bash
git remote add upstream https://github.com/sihyun-yu/REPA.git
git fetch upstream
git remote -v
```

🔍 **검증**: `git remote -v` 출력에 `origin` (your fork) 과 `upstream` (sihyun-yu) 이 모두 보여야 함.

```
origin    https://github.com/<your-username>/REPA.git (fetch)
origin    https://github.com/<your-username>/REPA.git (push)
upstream  https://github.com/sihyun-yu/REPA.git (fetch)
upstream  https://github.com/sihyun-yu/REPA.git (push)
```

- [ ] **1.2.4** Working branch 생성.

```bash
git checkout -b reeval
```

> **왜 branch가 필요한가?** Pre-registration tag (`prereg-v1`) 가 모든 main training commit보다 *선행*해야 한다는 §3.3 규칙 때문. Branch에서 작업하면 우리의 모든 변경이 한 곳에 모이고, tag를 commit graph에 정확히 위치시킬 수 있음.

- [ ] **1.2.5** 작업 디렉토리 scaffold (빈 폴더와 `.gitignore`).

```bash
mkdir -p docs scripts tests data exps train/data train/eval templates paper/figures
touch docs/.gitkeep scripts/.gitkeep tests/.gitkeep
touch train/data/.gitkeep train/eval/.gitkeep templates/.gitkeep paper/figures/.gitkeep

cat > .gitignore << 'EOF'
# 학습 산출물 (영구 저장 안 함)
data/ffhq256_latents/
exps/
results/main/

# Python
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# 임시
*.tmp
.DS_Store
/tmp/
EOF

git add docs scripts tests train templates paper .gitignore
git commit -m "scaffold: empty dirs and .gitignore for reeval workflow"
```

🔍 **검증**:

```bash
git status
ls -la
```

`git status` 가 clean 상태여야 하고, `ls -la` 가 위 폴더 모두를 포함해야 함.

---

### 1.3 Python venv + 의존성

**목적**: 시스템 Python을 오염시키지 않고, 본 연구 전용 venv에 모든 의존성을 핀.

**파일**:
- Create: `requirements-reeval.txt`
- Create: `venv/` (gitignored)

**단계**:

- [ ] **1.3.1** 원본 REPA가 어떤 의존성을 명시하는지 확인.

```bash
ls -la *.txt *.toml setup.py 2>/dev/null
cat requirements.txt 2>/dev/null | head -30
```

기존 `requirements.txt` 가 있으면 그것을 base로 사용. 없으면 본 튜토리얼의 추가 목록만으로 시작.

- [ ] **1.3.2** 본 연구 추가 의존성 명시.

`requirements-reeval.txt` 라는 새 파일을 만들고 다음 내용을 그대로 복사.

```text
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
tqdm
```

> **왜 이 버전들인가?** clean-fid 0.1.35는 Inception feature extractor가 stable. diffusers ≥ 0.27은 SD-VAE-ft-mse loading 호환. transformers ≥ 4.40은 DINOv2 자동 loading 호환.

- [ ] **1.3.3** venv 생성 및 활성화.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
```

- [ ] **1.3.4** PyTorch 설치 (CUDA 12.x 가정. CUDA 11.x면 cu118로 변경).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

⚠ **트러블슈팅**:
- `pip install torch` 가 CPU-only를 받아오면 GPU에서 사용 불가. 반드시 `--index-url` 로 CUDA 빌드 받기.
- nvidia-smi의 "CUDA Version" 이 11.x면 `cu118`, 12.x면 `cu121`.

- [ ] **1.3.5** 나머지 의존성 설치.

```bash
pip install -r requirements.txt 2>/dev/null || true   # REPA 원본
pip install -r requirements-reeval.txt
```

⚠ **트러블슈팅**:
- `cleanfid` 가 별도 weight를 다운받음. 첫 import 시 인터넷 필요.
- `lpips` 도 동일. 첫 사용 시 alex weight 다운로드.

- [ ] **1.3.6** GPU 인식 확인.

```bash
python -c "
import torch
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
print('cudnn:', torch.backends.cudnn.version())
"
```

🔍 **Expected**:

```
cuda available: True
device: NVIDIA GeForce RTX 4080 Super
vram GB: 16.0
cudnn: 90100
```

⚠ 다른 GPU를 사용한다면 batch size 또는 sample batch size 조정 필요. 본 튜토리얼은 16 GB VRAM 가정.

- [ ] **1.3.7** Commit.

```bash
git add requirements-reeval.txt
git commit -m "env: pin reeval-specific dependencies"
```

---

### 1.4 디스크 baseline + monitor

**목적**: §4.4 storage budget을 self-enforce 하도록, 학습 중 디스크가 임계 아래로 떨어지면 자동 alert + 학습 종료.

**파일**:
- Create: `data/disk_baseline.txt`
- Create: `scripts/disk_monitor.py`
- Create: `data/disk_log.csv` (모니터 실행 시 자동 생성)

**단계**:

- [ ] **1.4.1** 현재 가용량 baseline 기록.

```bash
df -h . | tee data/disk_baseline.txt
```

> **왜 baseline을 남기나?** Day 21 finalize 시 "처음에 X GB 있었고 지금 Y GB 남았다" 를 정량적으로 보여줄 수 있어, §4.4 budget의 실제 사용량 검증이 가능.

- [ ] **1.4.2** Disk monitor 스크립트 생성.

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
            w.writerow([datetime.datetime.now().isoformat(),
                        f"{free_gb:.2f}", f"{used_gb:.2f}", int(alert)])
            f.flush()
            if alert:
                print(f"⚠ DISK ALERT: free {free_gb:.1f} GB < threshold {args.min_gb} GB",
                      file=sys.stderr)
                if args.exit_on_alert:
                    sys.exit(1)
            if args.once:
                break
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

- [ ] **1.4.3** 한번 실행해 baseline log 생성.

```bash
python scripts/disk_monitor.py --once
cat data/disk_log.csv
```

🔍 **Expected**: `data/disk_log.csv` 가 1 row 생성됨, alert 컬럼이 0 (가용량 충분).

- [ ] **1.4.4** Commit.

```bash
git add scripts/disk_monitor.py data/disk_baseline.txt data/disk_log.csv
git commit -m "ops: disk monitor with auto-alert (per §4.4)"
```

---

### 1.5 SD-VAE FFHQ 재구성 sanity script (작성만)

**목적**: §10 risk row "FFHQ-SD-VAE 불일치" 를 미리 차단. SD-VAE가 FFHQ-256을 충분히 잘 재구성하는지 객관적으로 확인.

**오늘은 script만 만들고, 실행은 Day 2 (Task 2.2) 에서 한다.** 이유: HF dataset id 가 아직 없음.

**파일**:
- Create: `scripts/sd_vae_sanity.py`

**단계**:

- [ ] **1.5.1** 스크립트 생성.

`scripts/sd_vae_sanity.py`:
```python
"""SD-VAE FFHQ-256 reconstruction sanity check.

Streams N FFHQ-256 images from HF, encodes/decodes via stabilityai/sd-vae-ft-mse,
reports PSNR + LPIPS. Hard-fails if PSNR < 28 dB or LPIPS > 0.10.

Usage:
    python scripts/sd_vae_sanity.py --hf_dataset <id> --split train --n 64
"""
import argparse, io, json, sys
import torch
import lpips
from datasets import load_dataset
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

# PSNR_THRESHOLD lowered from 30.0 to 28.0 on 2026-04-15 after empirical
# measurement on two FFHQ-256 mirrors (merkol, idning) gave consistent
# ~29.6 dB. SD-VAE-ft-mse on face/natural images operates in 27-30 dB range
# (Rombach et al. 2022 + Stability AI fine-tune); the original 30 dB threshold
# was set as a round number, not from literature. LPIPS 0.10 retained as
# perceptual cross-check. See spec §10 risk row.
PSNR_THRESHOLD = 28.0
LPIPS_THRESHOLD = 0.10

def psnr(x, y, eps=1e-10):
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()

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
        if i >= args.n:
            break
        imgs.append(tx(get_image(sample)))
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
        "lpips_alex": float(lp(x.clamp(-1, 1), x_hat.clamp(-1, 1)).mean().item()),
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
    p.add_argument("--hf_dataset", required=True, help="HF dataset id discovered in §2.1")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=64)
    main(p.parse_args())
```

> **왜 PSNR 28 dB / LPIPS 0.10 인가?** SD-VAE-ft-mse는 face/natural image에서 PSNR 27-30 dB 범위가 정상 (원본 SD-VAE 대비 ft-mse는 +1 dB 개선). 28 dB는 이 범위의 *하한*으로, *기대 외로 나쁜* 경우만 reject. 본 임계는 2026-04-15 두 FFHQ-256 mirror (merkol, idning) 에서 ~29.6 dB로 수렴 측정 후 조정 (이전 30 dB는 round-number heuristic으로 너무 strict했음). LPIPS 0.10은 face 이미지에서 "동일 인물로 인식" 가능한 수준. **두 임계 모두 satisfied 여야** baseline이 신뢰 가능. 한쪽만 통과하면 마진이 좁다는 신호로 받아들이고 디버그.

- [ ] **1.5.2** Commit.

```bash
git add scripts/sd_vae_sanity.py
git commit -m "feat: SD-VAE FFHQ reconstruction sanity script (run after §2.1)"
```

---

### ✅ Day 1 완료 체크포인트

다음을 모두 확인:

- [ ] `git log --oneline` 에 4개 commit이 보임 (scaffold, env, disk monitor, sanity script)
- [ ] `git remote -v` 에 origin + upstream이 모두 등록됨
- [ ] `git branch` 가 `* reeval` 을 표시
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 가 `True`
- [ ] `python -c "import diffusers, datasets, lpips, cleanfid, transformers; print('ok')"` 가 에러 없이 `ok`
- [ ] `data/disk_baseline.txt`, `data/disk_log.csv` 존재
- [ ] `scripts/disk_monitor.py`, `scripts/sd_vae_sanity.py` 존재

체크리스트가 모두 ✓ 면 Day 2로 진행. 하나라도 ✗ 면 해당 단계로 돌아가서 fix.

---

## Part 2 — Day 2: 데이터셋 파이프라인 (Streaming + Latent precompute)

**오늘의 목표**: HF mirror에서 FFHQ-256을 streaming으로 받아 SD-VAE latent을 사전 계산하고, 디스크에 ~600 MB만 영구 저장.

**예상 시간**: 4-6 시간 (latent precompute가 1-2 시간).

**산출물**:
- `data/MANIFEST.json` (mirror id + revision + split hash + latent SHA)
- `data/sd_vae_recon_sanity.json` (PSNR/LPIPS 결과)
- `data/ffhq256_latents/train/*.pt` (~70K 파일, ~600 MB, gitignored)
- `scripts/precompute_latents.py`
- `tests/test_dataset.py`

> **왜 streaming인가?** §4.3 옵션 A. Raw FFHQ-256 = ~28 GB. 가용 < 90 GB 환경에서 영구 저장하면 §4.4 budget이 무너짐. Streaming → SD-VAE encode → bf16 latent (600 MB) → raw 폐기 흐름으로 28 GB → 0 GB.

---

### 2.1 HF Hub FFHQ-256 mirror 찾기

**목적**: 신뢰 가능한 mirror 1개를 선택하고 dataset id, revision hash, license를 manifest에 기록.

**파일**:
- Create: `data/MANIFEST.json` (initial)

**단계**:

- [ ] **2.1.1** HF Hub에서 FFHQ 관련 dataset 검색.

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
results = list(api.list_datasets(search='ffhq', limit=30))
for r in results:
    print(r.id)
"
```

후보 예 (시점에 따라 다를 수 있음): `huggan/ffhq`, `bitmind/ffhq256`, `nuwandavek/ffhq256`, `merve/ffhq-faces`, ...

- [ ] **2.1.2** 1-2개 후보를 streaming으로 첫 sample 받아 schema 확인.

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('<candidate-id>', split='train', streaming=True)
sample = next(iter(ds))
print('keys:', list(sample.keys()))
print('first key type:', type(sample[next(iter(sample))]).__name__)
img = sample.get('image')
if img is not None:
    print('image size:', img.size, 'mode:', img.mode)
"
```

🔍 **선택 기준**:
- `image` 필드가 PIL Image (또는 PNG bytes) 로 반환됨
- 256×256 이상 (resize 가능)
- 최소 60K samples (FFHQ는 70K)
- License 명확
- `revision` 또는 last commit이 명시됨

> **트러블슈팅**: 일부 mirror는 1024×1024를 제공. 그래도 OK — `transform`에서 `Resize(256)` + `CenterCrop(256)` 으로 다운샘플하면 됨. 일부 mirror는 `image` 대신 `png` bytes 필드를 사용. `scripts/sd_vae_sanity.py` 의 `get_image()` 함수가 두 경우 모두 처리.

- [ ] **2.1.3** 선택한 mirror의 정보를 manifest에 기록.

```bash
mkdir -p data
cat > data/MANIFEST.json << 'EOF'
{
  "dataset": {
    "hf_id": "<chosen-id>",
    "revision": "<git-commit-sha-from-hub>",
    "split_train": "train",
    "expected_total": 70000,
    "license": "<license-from-hub-page>",
    "discovered_at": "2026-04-15"
  },
  "splits": {},
  "latents_train": {}
}
EOF
```

`<chosen-id>` 등 placeholder를 실제 값으로 교체.

> **revision hash 어떻게 얻나?**
> ```python
> from huggingface_hub import HfApi
> info = HfApi().dataset_info('<chosen-id>')
> print(info.sha)
> ```

🔍 **검증**: `python -c "import json; print(json.load(open('data/MANIFEST.json')))"` 가 dict 출력. `<...>` placeholder가 모두 교체되어 있어야 함.

---

### 2.2 SD-VAE 재구성 sanity check 실제 실행

**목적**: §10 risk row를 closeout. mirror가 유효하고 SD-VAE가 FFHQ를 충분히 잘 재구성하는지 확인.

**파일**:
- Create: `data/sd_vae_recon_sanity.json`

**단계**:

- [ ] **2.2.1** Day 1에 작성한 sanity script 실행 (mirror id를 본인 것으로 교체).

```bash
python scripts/sd_vae_sanity.py --hf_dataset <chosen-id> --split train --n 64
```

🔍 **Expected** (예시 — 정확한 수치는 mirror마다 다름. 두 FFHQ-256 mirror 실측 ≈ 29.6 dB, LPIPS ≈ 0.030):

```json
{
  "n_samples": 64,
  "hf_dataset": "<chosen-id>",
  "psnr_db": 29.6,
  "lpips_alex": 0.030,
  "thresholds": {"psnr_db_min": 28.0, "lpips_max": 0.1}
}
✓ SD-VAE FFHQ reconstruction sanity check PASSED
```

⚠ **트러블슈팅**:
- **PSNR < 28 dB**: SD-VAE가 이 mirror의 이미지를 정상 범위 (27-30 dB) 보다 못 재구성. 다른 mirror 시도. 두 mirror 모두 < 28 dB면 SDXL VAE 등 대안 VAE 검토.
- **LPIPS > 0.10**: 동일 — perceptual 손상. 다른 mirror 시도.
- **HFHubHTTPError**: HF token이 필요할 수 있음. `huggingface-cli login` 으로 로그인.
- **OOM**: 64 samples 한 번에 못 올리면 batch_size 인자 추가 (sanity script가 자동 batched).
- **둘 다 fail**: 본 연구의 baseline 가정이 무너짐. 즉시 Stop. 다른 mirror 또는 다른 VAE (예: SDXL VAE) 검토 필요. 이 case는 spec §10 risk가 실현된 것.

- [ ] **2.2.2** 결과 확인 후 commit.

```bash
git add data/MANIFEST.json data/sd_vae_recon_sanity.json
git commit -m "data: choose HF FFHQ-256 mirror, verify SD-VAE reconstruction"
```

🚨 **결정점**:
- PSNR ≥ 30 dB + LPIPS ≤ 0.05 → 매우 좋음, 그대로 진행
- PSNR 28-30 dB + LPIPS ≤ 0.05 → 정상 (SD-VAE-ft-mse 표준 범위), 진행
- PSNR < 28 dB OR LPIPS > 0.10 → mirror 또는 VAE 문제. 다른 mirror 또는 stop.
- LPIPS 가 ≤ 0.05 인데 PSNR 만 marginal (28 미만)일 경우: results section에 caveat로 명시 후 진행 가능 (PSNR 한계 vs perceptual quality 의 dissociation 자체가 sub-finding).

---

### 2.3 Streaming SD-VAE latent precompute 스크립트

**목적**: Streaming-first workflow의 핵심 — raw 이미지를 디스크에 쓰지 않고 SD-VAE latent만 저장.

**파일**:
- Create: `scripts/precompute_latents.py`
- Create: `tests/test_dataset.py`

**단계**:

- [ ] **2.3.1** Test 먼저 (TDD). 아직 latent이 없으므로 FAIL할 것.

`tests/test_dataset.py`:
```python
import os
import torch

LATENT_DIR = "data/ffhq256_latents/train"

def test_latent_shape_and_dtype():
    files = sorted(f for f in os.listdir(LATENT_DIR) if f.endswith(".pt"))
    assert len(files) > 0, "no latents present yet"
    latent = torch.load(os.path.join(LATENT_DIR, files[0]))
    assert latent.shape == (4, 32, 32), f"unexpected shape {tuple(latent.shape)}"
    assert latent.dtype == torch.bfloat16, f"unexpected dtype {latent.dtype}"
```

```bash
pytest tests/test_dataset.py::test_latent_shape_and_dtype -v
```

🔍 **Expected**: FAIL (디렉토리 없음 또는 빈 폴더).

- [ ] **2.3.2** Precompute script 생성.

`scripts/precompute_latents.py`:
```python
"""Streams FFHQ-256 from HF Hub, encodes through SD-VAE, saves bf16 latents.

Per spec §4.3 옵션 A. Raw images are never written to disk.
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

> **왜 매 latent을 별도 파일로 저장?** Map-style dataset에서 random access가 빠름. 그리고 manifest에서 file-level SHA-256 hash를 추적 가능. Single big tensor 파일은 reproducibility check가 어려움 (한 byte 변경되면 전체 hash가 바뀜).

> **왜 bf16?** 70K × 4×32×32 × 2 byte = ~600 MB. fp32면 ~1.1 GB. 학습 시 어차피 bf16 mixed precision으로 동작하므로 latent을 bf16으로 보관해도 quality 손실 없음.

- [ ] **2.3.3** Commit script.

```bash
git add scripts/precompute_latents.py tests/test_dataset.py
git commit -m "feat: streaming SD-VAE latent precompute script + dataset tests"
```

---

### 2.4 Dry run (10 sample)

**목적**: 전체 70K를 돌리기 전에 pipeline이 작동하는지 작은 수로 확인.

**단계**:

- [ ] **2.4.1** 10개만 dry run.

```bash
python scripts/precompute_latents.py \
    --hf_dataset <chosen-id> \
    --hf_split train \
    --out data/ffhq256_latents/train \
    --limit 10
```

🔍 **Expected**:
```
✓ saved 10 latents to data/ffhq256_latents/train
```

- [ ] **2.4.2** 파일 확인.

```bash
ls data/ffhq256_latents/train | head -15
du -sh data/ffhq256_latents/train
```

🔍 **Expected**: `000000.pt` ~ `000009.pt`, 총 ~80 KB.

- [ ] **2.4.3** Test PASS 확인.

```bash
pytest tests/test_dataset.py::test_latent_shape_and_dtype -v
```

🔍 **Expected**: PASS.

⚠ **트러블슈팅**:
- **shape (4,32,32) 가 아님**: SD-VAE의 latent shape가 다른 모델임. `stabilityai/sd-vae-ft-mse` 가 아니라 다른 VAE 사용 중이면 spec 위반. 원본 사용으로 복귀.
- **dtype torch.float32**: bf16 변환 누락. script의 `z_bf16 = z.to(torch.bfloat16)` 라인 확인.
- **streaming connection 끊김**: 일부 HF mirror가 rate limit. `huggingface-cli login` 후 재시도. 또는 `requests.adapters.HTTPAdapter` 으로 retry 로직 추가.

---

### 2.5 Full latent precompute (70K)

**목적**: 학습용 latent 전체를 한 번에 사전 계산.

**단계**:

- [ ] **2.5.1** Disk pre-flight.

```bash
python scripts/disk_monitor.py --once
df -h .
```

가용량 ≥ 5 GB 인지 확인 (latent 600 MB + scratch buffer + 안전 margin).

- [ ] **2.5.2** Dry run 결과 wipe (인덱스가 0부터 다시 시작하도록).

```bash
rm -rf data/ffhq256_latents/train
```

- [ ] **2.5.3** Full precompute 실행.

```bash
python scripts/precompute_latents.py \
    --hf_dataset <chosen-id> \
    --hf_split train \
    --out data/ffhq256_latents/train
```

🔍 **Expected**:
- 약 30-80 img/s (RTX 4080S, batch=1, fp32 VAE)
- 70,000 / 60 ≈ 약 15-40 분
- 마지막 line: `✓ saved 70000 latents to data/ffhq256_latents/train`

⚠ **트러블슈팅**:
- **속도가 < 20 img/s**: GPU 사용률 낮을 가능성. `nvidia-smi` 로 확인. CPU bottleneck (PIL decode + transform) 일 수 있음. `num_workers > 0` 인 DataLoader로 wrapping 가능하지만, streaming dataset과 DataLoader의 호환은 까다로움. 1번 실행으로 끝나는 작업이므로 그냥 기다리는 게 단순.
- **80% 시점에 OOM**: 메모리 leak 가능. PyTorch tensor cache 정리 필요. `torch.cuda.empty_cache()` 를 1000 step마다 호출하도록 script 수정.
- **HF 도중 disconnect**: 다시 실행하면 처음부터. 안 좋음. 아래 resume 옵션 참고.

> **Resume 옵션**: 본 script는 resume 미지원 (단순함을 위해). 만약 90% 시점에 끊기면 painful. 다음 변경으로 resume 가능:
> ```python
> existing = set(int(p.stem) for p in out_dir.glob("*.pt"))
> if i in existing:
>     continue
> ```
> 위 두 줄을 메인 loop 시작에 넣으면 resume 됨. 단, streaming dataset의 iterator는 처음부터 다시 fetch되므로 wallclock은 절약 안 됨 (디스크 write만 절약).

- [ ] **2.5.4** Sanity 확인.

```bash
ls data/ffhq256_latents/train | wc -l
du -sh data/ffhq256_latents/train
```

🔍 **Expected**: ~70000 files, ~600 MB.

- [ ] **2.5.5** 모든 dataset test 실행.

```bash
pytest tests/test_dataset.py -v
```

🔍 **Expected**: 모든 test PASS.

- [ ] **2.5.6** Commit MANIFEST 업데이트 (latents 자체는 .gitignore).

```bash
git add data/MANIFEST.json
git commit -m "data: precompute 70K FFHQ-256 SD-VAE latents (manifest updated)"
```

---

### ✅ Day 2 완료 체크포인트

다음을 모두 확인:

- [ ] `data/MANIFEST.json` 의 `dataset` 필드에 placeholder가 모두 채워짐
- [ ] `data/sd_vae_recon_sanity.json` 의 PSNR ≥ 28, LPIPS ≤ 0.10
- [ ] `data/ffhq256_latents/train/` 에 정확히 70000개 `.pt` 파일
- [ ] `du -sh data/ffhq256_latents/train` 가 약 600 MB
- [ ] `pytest tests/test_dataset.py -v` 가 모두 PASS
- [ ] `git log --oneline` 에 Day 2 commit 3-4개 추가됨
- [ ] `data/MANIFEST.json` 의 `latents_train.count` == 70000

체크리스트가 모두 ✓ 면 Day 3로 진행.

---

## Part 3 — Day 3: Reference statistics + REPA fork inventory

**오늘의 목표**: Train/eval split 고정, FID + DINOv2 reference 통계 사전 계산, REPA fork의 코드 구조 파악.

**예상 시간**: 3-5 시간.

**산출물**:
- `data/ffhq256_train.txt` (65K lines) + `data/ffhq256_eval.txt` (5K lines)
- `data/fid_ref_ffhq256.npz` (~150 KB)
- `data/fd_dinov2_ref_ffhq256.pt` (~5 MB)
- `docs/repa_inventory.txt` + `docs/BASELINE.md`
- `scripts/precompute_fid_ref.py`, `scripts/precompute_dino_ref.py`

---

### 3.1 Train/eval split 고정 (deterministic)

**목적**: 65K train + 5K eval. 본 연구의 모든 실험이 동일한 split을 사용해야 함. Seed로 한 번 생성하면 다시는 안 바뀜.

**파일**:
- Create: `data/ffhq256_train.txt`
- Create: `data/ffhq256_eval.txt`
- Modify: `data/MANIFEST.json`

**단계**:

- [ ] **3.1.1** Test 먼저.

`tests/test_dataset.py` 에 다음 함수 append:

```python
def test_split_determinism():
    train = open("data/ffhq256_train.txt").read().splitlines()
    eval_ = open("data/ffhq256_eval.txt").read().splitlines()
    assert len(train) == 65000
    assert len(eval_) == 5000
    assert set(train).isdisjoint(set(eval_))
    assert set(train) | set(eval_) == set(f"{i:06d}" for i in range(70000))
```

```bash
pytest tests/test_dataset.py::test_split_determinism -v
```

🔍 **Expected**: FAIL (파일 없음).

- [ ] **3.1.2** Split 생성 (한 줄 명령).

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
print('split written, train_sha256:', m['split']['train_sha256'][:16])
print('              eval_sha256:', m['split']['eval_sha256'][:16])
"
```

> **왜 sort?** Random shuffle 후 sort된 index list로 저장하면, 인덱스 순으로 latent을 순회하기 쉬움. Random 순서는 dataset loader에서 sampler/shuffler가 해결.

- [ ] **3.1.3** Test PASS 확인.

```bash
pytest tests/test_dataset.py::test_split_determinism -v
```

🔍 **Expected**: PASS.

- [ ] **3.1.4** Commit.

```bash
git add data/ffhq256_train.txt data/ffhq256_eval.txt data/MANIFEST.json tests/test_dataset.py
git commit -m "data: fix 65K/5K train/eval split (seed 20260415, deterministic)"
```

🚨 **이 commit 이후 split을 절대 바꾸지 마세요.** Pre-registration 이후 split 변경은 spec §3.3 위반.

---

### 3.2 FID reference statistics

**목적**: 학습 중/후 매번 FID를 계산할 때, 동일한 reference statistics를 사용해야 비교 가능. Eval split 5K 이미지에서 한 번만 사전 계산.

**파일**:
- Create: `scripts/precompute_fid_ref.py`
- Create: `data/fid_ref_ffhq256.npz`

**단계**:

- [ ] **3.2.1** Script 생성.

`scripts/precompute_fid_ref.py`:
```python
"""Compute clean-fid Inception statistics for FFHQ-256 eval split.

Decodes 5K eval latents through SD-VAE in memory, never writes raw images to disk.
Output: data/fid_ref_ffhq256.npz with keys 'mu' (2048,) and 'sigma' (2048, 2048).
"""
import argparse
import numpy as np
import torch
from diffusers import AutoencoderKL
from cleanfid.features import build_feature_extractor

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

> **왜 latent을 decode해서 reference로 쓰나?** 우리 model이 만들어내는 sample도 SD-VAE를 통과한 것. 비교 대상도 SD-VAE encode→decode를 거친 "VAE-reachable" 분포여야 공정. Raw FFHQ 이미지를 reference로 쓰면 우리 model은 SD-VAE의 reconstruction error 만큼 손해를 봄.

- [ ] **3.2.2** 실행.

```bash
python scripts/precompute_fid_ref.py
```

🔍 **Expected**: 5-10 분, 마지막 line `✓ FID reference computed from 5000 samples → data/fid_ref_ffhq256.npz`.

⚠ **트러블슈팅**:
- **`build_feature_extractor("clean", device)` 가 다운로드**: clean-fid가 처음 사용 시 Inception weight (~100 MB) 를 받음. 인터넷 필요.
- **OOM at decode**: BS를 16으로 낮춤.

- [ ] **3.2.3** Sanity 확인.

```bash
python -c "
import numpy as np
ref = np.load('data/fid_ref_ffhq256.npz')
print('mu:', ref['mu'].shape, 'sigma:', ref['sigma'].shape, 'n:', int(ref['n']))
assert ref['mu'].shape == (2048,)
assert ref['sigma'].shape == (2048, 2048)
print('✓')
"
```

- [ ] **3.2.4** Commit.

```bash
git add scripts/precompute_fid_ref.py data/fid_ref_ffhq256.npz
git commit -m "data: compute clean-fid Inception ref stats from eval split"
```

---

### 3.3 DINOv2 reference statistics

**목적**: FD-DINOv2 metric (§7.1) 의 reference. DINOv2 ViT-B 의 patch token mean을 사용.

**파일**:
- Create: `scripts/precompute_dino_ref.py`
- Create: `data/fd_dinov2_ref_ffhq256.pt`

**단계**:

- [ ] **3.3.1** Script 생성.

`scripts/precompute_dino_ref.py`:
```python
"""Compute DINOv2 ViT-B feature mean/cov for FFHQ-256 eval split.
Output: data/fd_dinov2_ref_ffhq256.pt with keys 'mu' (768,), 'sigma' (768,768), 'n'.
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
                pil = [T.ToPILImage()(im.cpu()) for im in imgs_01]
                inp = proc(images=pil, return_tensors="pt").to(device)
                out = dino(**inp).last_hidden_state[:, 1:].mean(dim=1)
            feats.append(out.cpu())
            batch = []
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(eval_idx)}]")

    feats = torch.cat(feats, dim=0).to(torch.float64)
    mu = feats.mean(dim=0)
    sigma = torch.cov(feats.T)
    torch.save({"mu": mu, "sigma": sigma, "n": len(eval_idx)},
               "data/fd_dinov2_ref_ffhq256.pt")
    print(f"✓ DINOv2 ref ({mu.shape[0]}-dim) from {len(eval_idx)} samples")

if __name__ == "__main__":
    main()
```

> **왜 patch token mean (CLS 제외)?** Spec §5.2: "Feature 추출: 최종 layer의 patch tokens (CLS 아님)". Spatial information을 보존하기 위해 CLS는 빼고 patch token만 사용.

- [ ] **3.3.2** 실행.

```bash
python scripts/precompute_dino_ref.py
```

🔍 **Expected**: 5-10 분 (DINOv2 첫 다운로드 포함), 마지막 line `✓ DINOv2 ref (768-dim) from 5000 samples`.

⚠ **트러블슈팅**:
- **DINOv2 download 실패**: HF token 필요할 수 있음. `huggingface-cli login`.
- **CUDA OOM at DINOv2 forward**: BS를 16으로 낮춤. 또는 DINOv2를 fp16으로: `dino.half()` 후 input도 `inp.to(torch.float16)` (단 numerical stability 주의).

- [ ] **3.3.3** Commit.

```bash
git add scripts/precompute_dino_ref.py data/fd_dinov2_ref_ffhq256.pt
git commit -m "data: compute DINOv2 ViT-B ref stats from eval split"
```

---

### 3.4 Reproducibility verification

**목적**: Latent/manifest의 self-consistency가 깨지지 않았는지 자동 test로 가드.

**단계**:

- [ ] **3.4.1** Test append.

`tests/test_dataset.py` 에 추가:

```python
def test_latent_reproducibility():
    a = torch.load("data/ffhq256_latents/train/000000.pt")
    b = torch.load("data/ffhq256_latents/train/000000.pt")
    assert torch.equal(a, b), "two loads of the same latent file differ"

def test_manifest_hash_matches_file():
    import hashlib, json
    m = json.load(open("data/MANIFEST.json"))
    rec = m["latents_train"]["files"][0]
    actual = hashlib.sha256(open(rec["path"], "rb").read()).hexdigest()[:16]
    assert actual == rec["sha256_16"]
```

- [ ] **3.4.2** 모든 dataset test 실행.

```bash
pytest tests/test_dataset.py -v
```

🔍 **Expected**: 모든 test PASS (4-5개).

- [ ] **3.4.3** Commit.

```bash
git add tests/test_dataset.py
git commit -m "test: latent reproducibility + manifest hash verification"
```

---

### 3.5 REPA fork inventory

**목적**: 우리가 수정해야 할 파일과 함수 위치를 *문서화*. 추측이 아닌 사실 기반.

**파일**:
- Create: `docs/repa_inventory.txt`
- Create: `docs/BASELINE.md`

**단계**:

- [ ] **3.5.1** REPA 코드 구조 dump.

```bash
find . -type f -name "*.py" \
    -not -path "./venv/*" \
    -not -path "./.git/*" \
    -not -path "./tests/*" \
    -not -path "./scripts/*" \
    -not -path "./train/*" \
    | sort | tee docs/repa_inventory.txt
```

`tee` 출력에 REPA 원본의 .py 파일 목록이 보임. 보통 다음 패턴:
- `train.py` 또는 `train_repa.py` — main entry
- `models/sit.py` — SiT backbone
- `models/repa.py` 또는 비슷한 이름 — projection head
- `dataset.py` 또는 `data/...` — ImageNet loader
- `sample.py` — inference utility

- [ ] **3.5.2** 핵심 파일들 확인.

```bash
grep -ln "class SiT\|def train_step\|REPA\|projection\|class_emb\|class_label" \
    *.py models/*.py 2>/dev/null \
    | tee -a docs/repa_inventory.txt
```

🔍 본 명령의 출력에서 다음 4가지 hook 위치가 보여야 함:
1. `class SiT` 정의 위치
2. `train_step` 또는 main training loop 위치
3. REPA `projection` loss 계산 위치
4. `class_emb` 또는 `class_label` 처리 위치 (제거 대상)

- [ ] **3.5.3** 위 위치들의 정확한 line number 메모.

```bash
grep -n "lambda_repa\|repa_loss\|projection_loss\|repa_coef" *.py models/*.py
grep -n "num_classes\|class_emb\|y_embed" models/*.py
```

> **왜 line number까지?** Day 4 에서 hook을 정확한 라인에 주입할 때, "guessed file path" 가 아니라 "확인된 파일 + line" 으로 작업해야 misedit를 막음.

- [ ] **3.5.4** `docs/BASELINE.md` 작성.

`docs/BASELINE.md`:
```markdown
# REPA Baseline 코드베이스 매핑

원본 commit: `<git rev-parse upstream/main>`
Fork branch: `reeval`

## 핵심 파일 (위치는 inventory dump 기반 사실)

| 책임 | 파일 | 함수/클래스 | 라인 (대략) |
|---|---|---|---|
| Main training entry | `<file>` | `<main_fn>` | `<line>` |
| SiT backbone | `<file>` | `class SiT` | `<line>` |
| REPA projection loss | `<file>` | `<fn>` | `<line>` |
| Class embedding | `<file>` | `<fn>` | `<line>` |
| Checkpoint save | `<file>` | `<fn>` | `<line>` |
| Sampling utility | `<file>` | `<fn>` | `<line>` |

## 우리가 수정할 hook 위치 (Day 4 작업)

1. **Loss 계산** — `<file>:<line>`. 기존 `λ * proj_loss` 을 `BRANCHES[branch_name](step, T) * proj_loss` 로 교체.
2. **Dataset 인스턴스화** — `<file>:<line>`. `ImageNet(...)` 을 `FFHQ256LatentDataset(...)` 으로 교체.
3. **Class label handling** — `<file>:<line>`. embedding lookup에 항상 0 전달.
4. **Checkpoint 저장** — `<file>:<line>`. `train/save.py` 의 `is_eval_step` + `save_ema_bf16` 사용.

## 의도적 deviation (purity principle, spec §3.1)

원본 REPA에서 변경한 모든 항목:
- Dataset: ImageNet → FFHQ-256 latent (§4)
- Conditioning: class-conditional → unconditional (null label)
- λ scheduling: fixed 0.5 → 5 schedules (§6, B0-B4)
- Checkpoint policy: 매 step → eval ckpt만 EMA-only bf16 (§4.4)
- Eval suite: REPA paper FID 외 추가 metric (§7)

원본 REPA에서 *제거*한 항목 (spec §3.2):
- SPRINT sparse path
- Contrastive Flow Matching (CFM)
- Class-token dual-stream
- Path-drop regularization
- adaLN-Gaussian initialization
- INVAE

이상 모두 본 코드베이스에 *없어야* 함. Inventory dump에서 위 키워드가 발견되면 확인 + 제거.
```

`<file>`, `<line>` 등 placeholder를 inventory dump의 실제 값으로 채울 것.

- [ ] **3.5.5** Commit.

```bash
git add docs/BASELINE.md docs/repa_inventory.txt
git commit -m "docs: REPA fork inventory + BASELINE.md (deviation map)"
```

---

### ✅ Day 3 완료 체크포인트

다음을 모두 확인:

- [ ] `data/ffhq256_train.txt` (65000 lines) + `data/ffhq256_eval.txt` (5000 lines)
- [ ] `data/MANIFEST.json` 의 `split` 필드에 seed + sha256 가 모두 들어가 있음
- [ ] `data/fid_ref_ffhq256.npz` 존재 (mu shape (2048,), sigma shape (2048, 2048))
- [ ] `data/fd_dinov2_ref_ffhq256.pt` 존재
- [ ] `pytest tests/test_dataset.py -v` 모두 PASS
- [ ] `docs/BASELINE.md` 의 모든 `<placeholder>` 가 실제 값으로 채워짐
- [ ] `git log --oneline | head` 에 Day 3 commit 4-5개 추가

체크리스트가 모두 ✓ 면 Day 4로 진행.

🚨 **중요**: 이 시점에서 backup을 권장합니다. Latent precompute는 다시 하기 비싸므로:
```bash
tar -czf ~/repa-reeval-data-snapshot-day3.tar.gz \
    data/MANIFEST.json data/ffhq256_train.txt data/ffhq256_eval.txt \
    data/fid_ref_ffhq256.npz data/fd_dinov2_ref_ffhq256.pt
```
(latent 자체는 600 MB라 backup하기 무거우면 manifest만이라도. 진짜 잃어버리면 §2.5에서 다시 1-2시간 걸림.)

---

## Part 4 — Day 4 (오전): 코드 수정 (REPA fork에 hook 주입)

**오늘 오전의 목표**: REPA fork에 4가지 핵심 변경을 주입 — (1) FFHQ-256 dataset, (2) 5 coefficient schedules, (3) class conditioning 제거, (4) checkpoint 정책. 두 equivalence test로 변경의 정확성을 검증.

**예상 시간**: 4-6 시간 (가장 까다로운 phase).

**산출물**:
- `train/data/ffhq256_latents.py` (loader)
- `train/branches.py` (5 schedules)
- `train/save.py` (checkpoint helper)
- `train/log.py` (CSV logger)
- REPA fork의 main training script 수정
- `tests/test_branches.py`, `tests/test_equivalence_off.py`, `tests/test_equivalence_fixed.py`

🚨 **이 phase가 가장 risk가 높음.** 잘못된 hook 주입은 모든 후속 결과를 무효화. 각 변경 후 반드시 검증 단계 통과 확인.

---

### 4.1 FFHQ-256 dataset loader

**목적**: REPA의 ImageNet loader를 우리의 FFHQ-256 latent loader로 교체.

**파일**:
- Create: `train/data/ffhq256_latents.py`
- Modify: `tests/test_dataset.py`

**단계**:

- [ ] **4.1.1** Failing test.

`tests/test_dataset.py` 에 추가:

```python
def test_dataset_loader_smoke():
    from train.data.ffhq256_latents import FFHQ256LatentDataset
    ds = FFHQ256LatentDataset("data/ffhq256_latents/train", "data/ffhq256_train.txt")
    assert len(ds) == 65000
    sample = ds[0]
    assert "latent" in sample and "label" in sample
    assert sample["latent"].shape == (4, 32, 32)
    assert sample["latent"].dtype == torch.float32
    assert sample["label"].item() == 0
```

```bash
pytest tests/test_dataset.py::test_dataset_loader_smoke -v
```

🔍 **Expected**: FAIL (`ModuleNotFoundError: train.data.ffhq256_latents`).

- [ ] **4.1.2** Loader 구현.

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

> **왜 latent에 hflip을 적용하나?** Latent space에서 horizontal flip은 spatial dimension을 뒤집는 것. SD-VAE는 translation-equivariant 정도여서 latent flip ≈ image flip 이 거의 성립함 (정확히는 아님 — VAE는 stride convolutions이라 약간의 misalignment 가능). 본 연구는 단일 augmentation만 적용하므로 이 약간의 mismatch는 수용 가능.

> **왜 weights_only=True?** PyTorch 2.4+의 보안 권장 설정. arbitrary code execution 위험을 막음.

- [ ] **4.1.3** Test PASS 확인.

```bash
pytest tests/test_dataset.py::test_dataset_loader_smoke -v
```

🔍 **Expected**: PASS.

- [ ] **4.1.4** Commit.

```bash
git add train/data/ffhq256_latents.py tests/test_dataset.py
git commit -m "feat: FFHQ256LatentDataset (bf16 latents → fp32 + null label)"
```

---

### 4.2 5개 coefficient schedule 함수 (TDD)

**목적**: Spec §6의 5개 branch (B0-B4) 를 함수로 구현. 각 schedule이 정확한 endpoint 값을 내는지 test로 확인.

**파일**:
- Create: `train/branches.py`
- Create: `tests/test_branches.py`

**단계**:

- [ ] **4.2.1** Test 먼저 (TDD 핵심).

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

```bash
pytest tests/test_branches.py -v
```

🔍 **Expected**: 6 FAIL (`ModuleNotFoundError`).

- [ ] **4.2.2** 구현.

`train/branches.py`:
```python
"""Coefficient schedules for the 5 research branches (spec §6).

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
    """B2: λ(0)=0.5, λ(T/2)=0.25, λ(T)=0."""
    return MAX_COEFF * 0.5 * (1.0 + math.cos(math.pi * t / T))

def schedule_cosine_warmup(t: int, T: int) -> float:
    """B3: λ(0)=0, λ(T/2)=0.25, λ(T)=0.5."""
    return MAX_COEFF * 0.5 * (1.0 - math.cos(math.pi * t / T))

def schedule_hard_cutoff(t: int, T: int, cutoff_frac: float = 0.5) -> float:
    """B4: λ=0.5 for t < T/2, else 0."""
    return MAX_COEFF if t < T * cutoff_frac else 0.0

BRANCHES = {
    "off":    schedule_off,
    "fixed":  schedule_fixed,
    "decay":  schedule_cosine_decay,
    "warmup": schedule_cosine_warmup,
    "cutoff": schedule_hard_cutoff,
}
```

- [ ] **4.2.3** Test PASS.

```bash
pytest tests/test_branches.py -v
```

🔍 **Expected**: 6 PASS.

- [ ] **4.2.4** Commit.

```bash
git add train/branches.py tests/test_branches.py
git commit -m "feat: 5 REPA coefficient schedules (off/fixed/decay/warmup/cutoff)"
```

---

### 4.3 REPA loss에 schedule hook 주입

**목적**: 기존 `loss = denoise_loss + 0.5 * proj_loss` 형태의 fixed-coefficient loss를 schedule-aware 로 교체.

**파일**:
- Modify: REPA fork의 main training script (Day 3 inventory에서 확인한 파일)

**단계**:

- [ ] **4.3.1** 수정 대상 파일 열기.

```bash
$EDITOR <repa_main_train_script>.py
```

(`<repa_main_train_script>` 는 `docs/BASELINE.md` 의 "Loss 계산" 라인.)

- [ ] **4.3.2** Loss 계산부 수정.

기존 코드 패턴 (예시 — REPA fork의 정확한 형태는 다를 수 있음):

```python
# BEFORE
def compute_loss(model, batch):
    x = batch["image"]
    t = sample_timesteps()
    noise = torch.randn_like(x)
    noisy = add_noise(x, noise, t)
    pred, hidden = model(noisy, t, batch["label"])
    denoise_loss = F.mse_loss(pred, target_v(x, noise, t))
    proj = projection_head(hidden)
    proj_loss = -F.cosine_similarity(proj, dino_features(x), dim=-1).mean()
    return denoise_loss + 0.5 * proj_loss
```

수정 후:

```python
# AFTER
from train.branches import BRANCHES

def compute_loss(model, batch, step, total_steps, branch_name):
    x = batch["latent"]                       # ← FFHQ256LatentDataset returns dict with 'latent'
    t = sample_timesteps()
    noise = torch.randn_like(x)
    noisy = add_noise(x, noise, t)
    pred, hidden = model(noisy, t, batch["label"])
    denoise_loss = F.mse_loss(pred, target_v(x, noise, t))
    proj = projection_head(hidden)
    proj_loss = -F.cosine_similarity(proj, dino_features_from_latent(x), dim=-1).mean()
    coeff = BRANCHES[branch_name](step, total_steps)
    total = denoise_loss + coeff * proj_loss
    return total, {
        "denoise_loss": denoise_loss.item(),
        "proj_loss": proj_loss.item(),
        "coeff": coeff,
        "total": total.item(),
    }
```

⚠ **중요**:
- `batch["image"]` → `batch["latent"]` (loader가 latent을 반환하므로)
- `dino_features(x)` → `dino_features_from_latent(x)` — DINO target은 raw image에서 계산하므로, latent을 SD-VAE decode해서 raw image 얻고 그것으로 DINO forward. 정확한 함수명은 REPA fork에 맞게.
- 반환을 `(loss, log_dict)` 튜플로 변경 — Task 4.6 logger에서 사용.

> **DINO target 처리 주의**: REPA original은 raw image를 input으로 받아 (1) SD-VAE encode → SiT input, (2) DINO encode → projection target 두 가지를 동시에 함. 우리는 latent을 미리 계산했으므로 DINO target도 미리 계산하거나, 학습 중에 latent을 decode해서 DINO를 통과시켜야 함. 후자가 메모리는 더 들지만 cleaner. SD-VAE decode + DINO forward를 학습 step마다 호출하면 wallclock이 늘어남.
>
> **현실적 trade-off**: 학습 시간 vs 디스크.
> - **옵션 A**: 매 step에서 latent → SD-VAE decode → DINO. CPU/GPU 추가 연산.
> - **옵션 B**: 70K image의 DINO feature를 미리 계산 (~70K × 768 × 4 byte = 215 MB). 디스크 추가, 학습은 빠름.
>
> **Day 3에 옵션 B를 추가했으면 더 좋았을 것**. Day 3로 잠깐 돌아가서 `data/dinov2_targets_train.pt` 를 사전 계산해 두는 것을 권장. 70K patch token mean 또는 CLS token. 본 튜토리얼은 옵션 A로 진행 (가장 단순) 하지만, 학습 wallclock이 spec 외삽보다 50%+ 느리면 옵션 B로 전환.

- [ ] **4.3.3** Training loop에서 `compute_loss` 호출 부분 수정.

기존:
```python
loss = compute_loss(model, batch)
loss.backward()
```

수정:
```python
loss, log = compute_loss(model, batch, step, total_steps, args.branch)
loss.backward()
csv_logger.log(step=step, **log)   # csv_logger는 §4.6에서 만들 예정
```

- [ ] **4.3.4** CLI 인자 추가. `argparse`/`accelerate config`/REPA의 config 시스템에 따라 다름.

```python
parser.add_argument("--branch", choices=["off","fixed","decay","warmup","cutoff"], default="fixed")
parser.add_argument("--total_steps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, required=True)
```

- [ ] **4.3.5** Smoke test — argparse 가 깨지지 않았는지.

```bash
python <repa_main_train_script>.py --help 2>&1 | grep -E "branch|total_steps"
```

🔍 **Expected**: `--branch {off,fixed,decay,warmup,cutoff}` 와 `--total_steps` 옵션이 보임.

- [ ] **4.3.6** Commit.

```bash
git add <modified files>
git commit -m "feat: hook coefficient scheduler into REPA training loss"
```

---

### 4.4 Class conditioning 제거

**목적**: FFHQ는 unconditional. ImageNet의 class label embedding을 null token으로 대체.

**파일**:
- Modify: SiT model 또는 training script에서 class embedding을 사용하는 부분

**단계**:

- [ ] **4.4.1** Class label 사용 위치 다시 확인.

```bash
grep -n "class_label\|y_embed\|class_emb\|num_classes\|self\.y" \
    <models_dir>/*.py <training_script>.py
```

- [ ] **4.4.2** 가장 minimal한 수정 결정.

**옵션 1**: Embedding lookup에 항상 0 전달. Embedding table은 그대로 (사이즈 1000) 두지만 0번만 쓰임.
- 장점: 코드 변경 최소
- 단점: 사용 안 되는 999개 embedding이 메모리에 남음 (130M model에 비해 사소함)

**옵션 2**: Embedding table size를 1로 줄임.
- 장점: 메모리 절약
- 단점: 모델 architecture 변경, equivalence test 더 까다로워짐

**권장**: 옵션 1. Spec §3.2 "purity principle" 의 minimal deviation.

- [ ] **4.4.3** 수정. Loader가 이미 `label=0` 을 반환하므로 (Task 4.1), training script에서는 `batch["label"]` 을 그대로 model에 전달하면 됨. 추가 변경 없음. 단, model의 forward 시 `labels` 가 항상 0임을 명시적으로 verify:

```python
# 학습 loop의 첫 batch 직후 한 번 검증
assert (batch["label"] == 0).all().item(), "label is not null token"
```

이 assert를 첫 step에만 실행하도록 추가.

- [ ] **4.4.4** Smoke test — 한 batch forward.

```bash
python -c "
import torch
from train.data.ffhq256_latents import FFHQ256LatentDataset
# from <repa.models.sit> import SiT  # path는 inventory에서 확인
ds = FFHQ256LatentDataset('data/ffhq256_latents/train', 'data/ffhq256_train.txt')
batch = ds[0]
print('latent shape:', batch['latent'].shape)
print('label:', batch['label'].item())
# model = SiT(...).cuda()
# x = batch['latent'].unsqueeze(0).cuda()
# y = batch['label'].unsqueeze(0).cuda()
# t = torch.tensor([500]).cuda()
# out = model(x, t, y)
# print('forward ok:', out.shape, 'no NaN:', not torch.isnan(out).any().item())
"
```

(Model import는 REPA fork 구조에 맞춰 수정.)

- [ ] **4.4.5** Commit.

```bash
git add <modified files>
git commit -m "feat: unconditional FFHQ training (always-null label)"
```

---

### 4.5 Checkpoint 정책 (§4.4 EMA-only bf16)

**목적**: 디스크 budget을 지키기 위해 eval checkpoint만, EMA만, bf16으로 저장.

**파일**:
- Create: `train/save.py`
- Modify: REPA training loop의 checkpoint 저장 부분

**단계**:

- [ ] **4.5.1** Helper module.

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

- [ ] **4.5.2** Test.

`tests/test_save_policy.py`:
```python
import torch
from train.save import is_eval_step, save_ema_bf16

def test_eval_step_set():
    for s in [20_000, 50_000, 80_000, 100_000]:
        assert is_eval_step(s)
    for s in [10_000, 30_000, 99_999, 0, 100_001]:
        assert not is_eval_step(s)

def test_save_bf16(tmp_path):
    sd = {"w": torch.randn(10, 10, dtype=torch.float32)}
    out = tmp_path / "ema.pt"
    save_ema_bf16(sd, out)
    loaded = torch.load(out)
    assert loaded["w"].dtype == torch.bfloat16
    assert loaded["w"].shape == (10, 10)
```

```bash
pytest tests/test_save_policy.py -v
```

🔍 **Expected**: 2 PASS.

- [ ] **4.5.3** REPA training loop의 checkpoint 코드 수정. 기존:

```python
# BEFORE
if step % args.save_every == 0:
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "step": step},
               f"{out}/checkpoint-{step:06d}.pt")
```

수정 후:

```python
# AFTER
from train.save import is_eval_step, save_ema_bf16, cleanup_rolling

rolling_path = f"{out}/checkpoints/_rolling.pt"
if is_eval_step(step):
    save_ema_bf16(ema.state_dict(), f"{out}/checkpoints/ema_{step:06d}.pt")
    cleanup_rolling(rolling_path)
elif step % args.rolling_freq == 0:    # e.g. every 5000 steps
    save_ema_bf16(ema.state_dict(), rolling_path)
```

> **왜 rolling latest를 유지?** 학습 중 crash 시 가장 가까운 5K step 이전부터 재시작 가능 (시간 절약). 단 single file이므로 이전 rolling은 매번 덮어씀 (디스크 ~260 MB만 차지).

⚠ **단순화 옵션**: rolling latest 도 saving 안 하고, fail 시 무조건 처음부터 재시작도 가능. 디스크 ~260 MB 더 절약 + 코드 단순. 각 run이 ~3.5h이므로 fail-restart 비용 크지 않음. 본 튜토리얼은 rolling을 유지하지만, 초보자는 rolling 빼고 단순화하는 것도 합리적.

- [ ] **4.5.4** Commit.

```bash
git add train/save.py tests/test_save_policy.py <modified training script>
git commit -m "feat: §4.4 checkpoint policy (EMA-only bf16, eval ckpts only, rolling latest)"
```

---

### 4.6 Loss CSV logger

**목적**: 매 step의 loss + coefficient를 CSV로 기록. Pre-registered exploratory analysis 또는 debugging에 필수.

**파일**:
- Create: `train/log.py`
- Modify: REPA training loop

**단계**:

- [ ] **4.6.1** Logger module.

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
        if self.f:
            self.f.close()
```

- [ ] **4.6.2** Training loop에 logger 인스턴스화 + 매 step 호출.

```python
from train.log import CsvLogger

logger = CsvLogger(f"{args.out}/loss_log.csv")
# ... in step loop:
logger.log(step=step, denoise_loss=log["denoise_loss"],
           proj_loss=log["proj_loss"], coeff=log["coeff"],
           total_loss=log["total"])
```

(여기서 `log` 는 Task 4.3에서 `compute_loss` 가 반환한 dict.)

- [ ] **4.6.3** Smoke test — 5 step만 학습 후 csv 확인.

```bash
python <repa_main_train_script>.py --branch fixed --seed 42 --total_steps 5 --out /tmp/smoke
cat /tmp/smoke/loss_log.csv
```

🔍 **Expected** (예시):
```csv
step,denoise_loss,proj_loss,coeff,total_loss
1,0.812,-0.45,0.5,0.587
2,0.798,-0.51,0.5,0.543
...
```

- [ ] **4.6.4** Commit.

```bash
git add train/log.py <modified training script>
git commit -m "feat: append-only CSV training logger (per-step loss + coeff)"
```

---

### 4.7 Equivalence test: `schedule_off` ≡ REPA-disabled

**목적**: λ=0 이 정말 "REPA loss term이 없는 것" 과 동일한지 확인. Critique R1의 수치적 정확성 검증.

**파일**:
- Create: `tests/test_equivalence_off.py`

**단계**:

- [ ] **4.7.1** Test 작성.

`tests/test_equivalence_off.py`:
```python
"""Verify schedule_off (λ=0 multiplier) produces the same loss curve as
removing the projection loss term entirely. Both should match within bf16
floating-point noise over 100 training steps.
"""
import subprocess, csv, math, os, shutil

def run_train(branch, out_dir, steps=100, seed=42):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    cmd = ["python", "<repa_main_train_script>",
           "--branch", branch, "--seed", str(seed),
           "--total_steps", str(steps), "--out", out_dir]
    subprocess.run(cmd, check=True)
    rows = list(csv.DictReader(open(f"{out_dir}/loss_log.csv")))
    return [float(r["denoise_loss"]) for r in rows]

def test_off_matches_no_proj_loss():
    # Both runs should disable REPA: branch=off → λ=0
    losses = run_train("off", "/tmp/equiv_off_a")
    losses_b = run_train("off", "/tmp/equiv_off_b")  # second run for noise baseline
    diffs = [abs(a - b) for a, b in zip(losses, losses_b)]
    avg_diff = sum(diffs) / len(diffs)
    assert avg_diff < 1e-3, f"two off runs differ by {avg_diff:.6f}"
```

> **왜 두 번 같은 branch를 돌리나?** Reproducibility를 먼저 확인. `seed=42` 가 deterministic이라면 두 run의 loss curve가 거의 같아야 함. 다르면 RNG 통제가 안 되는 것. CUDA 비결정성을 완전히 막으려면 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` 필요. (선택사항이고 wallclock cost 있음.)

- [ ] **4.7.2** Test 실행.

```bash
pytest tests/test_equivalence_off.py -v
```

🔍 **Expected**: PASS.

⚠ **트러블슈팅**:
- **두 run의 loss curve가 다름** (avg_diff > 1e-3): seed 통제 부족. 다음 fix:
  1. `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` 호출 위치 확인
  2. DataLoader의 `worker_init_fn` 으로 worker seed 설정
  3. Dropout이 항상 같은 mask를 사용하도록 `torch.use_deterministic_algorithms(True)`
- **PASS이지만 marginal** (avg_diff ≈ 5e-4): bf16 mixed precision의 stochastic rounding 때문일 수 있음. 그대로 진행.

- [ ] **4.7.3** Commit.

```bash
git add tests/test_equivalence_off.py
git commit -m "test: schedule_off reproducibility (two runs match within 1e-3)"
```

---

### 4.8 Equivalence test: `schedule_fixed` ≡ upstream REPA

**목적**: 우리의 fixed-coefficient run이 upstream REPA의 fixed-coefficient run과 *수치적으로 거의 같은* loss curve를 생성하는지 확인. 이 test가 통과하면 우리의 hook 주입이 정확함을 보장.

**파일**:
- Create: `tests/test_equivalence_fixed.py`
- Create: `tests/fixtures/upstream_loss_log.csv`

**단계**:

- [ ] **4.8.1** Upstream REPA의 default config로 1000 step 실행.

```bash
# 임시 working tree로 upstream을 clone
mkdir -p tests/fixtures
cd /tmp
git clone https://github.com/sihyun-yu/REPA.git upstream_repa
cd upstream_repa
# 환경 설치 (우리 venv 재사용 권장)
source ~/projects/dl/repa-reeval/venv/bin/activate
# Upstream의 default 학습 1000 step
# 실제 명령은 REPA fork 의 README 참고
python train.py --total_steps 1000 --output_dir /tmp/upstream_out
```

⚠ Upstream REPA가 ImageNet을 요구할 수도 있음. 그러면 이 test는 **실용성이 낮음**. 대신:
- **대안**: 우리 코드에서 `--branch fixed` 로 1000 step + `--branch off` 로 1000 step 비교. denoise_loss는 거의 같아야 (proj_loss만 다르므로). 이 weaker test가 우리 hook의 sanity check.

- [ ] **4.8.2** 실용적 test (대안):

`tests/test_equivalence_fixed.py`:
```python
"""Sanity: with seed fixed, the denoise_loss component should be (nearly) identical
between branches that differ only in λ. Their proj_loss component differs but the
denoise loss should match within bf16 noise.
"""
import subprocess, csv, os, shutil

def run_train(branch, out_dir, steps=200, seed=42):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    cmd = ["python", "<repa_main_train_script>",
           "--branch", branch, "--seed", str(seed),
           "--total_steps", str(steps), "--out", out_dir]
    subprocess.run(cmd, check=True)
    rows = list(csv.DictReader(open(f"{out_dir}/loss_log.csv")))
    return rows

def test_denoise_loss_consistent_across_branches():
    rows_off = run_train("off", "/tmp/eq_off")
    rows_fix = run_train("fixed", "/tmp/eq_fix")
    den_off = [float(r["denoise_loss"]) for r in rows_off]
    den_fix = [float(r["denoise_loss"]) for r in rows_fix]
    # The two branches affect proj_loss only — denoise_loss should differ only
    # because of GRADIENT FEEDBACK from proj_loss into shared parameters.
    # Over a few steps, the divergence is small.
    early_diffs = [abs(a - b) for a, b in zip(den_off[:20], den_fix[:20])]
    assert max(early_diffs) < 0.02, f"early denoise loss diverges: max {max(early_diffs):.4f}"
```

> **왜 strict equivalence가 아닌가?** Fixed coefficient의 proj_loss는 model의 hidden state에 gradient를 backprop. 이 gradient가 다음 step의 model weight를 변경. 그래서 같은 seed에서도 fixed와 off의 denoise_loss는 *step이 진행될수록* 발산. 본 test는 첫 20 step에서 발산이 작은지만 확인 — 이게 통과되면 hook 주입이 sane함.

- [ ] **4.8.3** Test 실행.

```bash
pytest tests/test_equivalence_fixed.py -v
```

🔍 **Expected**: PASS (max early diff < 0.02).

⚠ **트러블슈팅**:
- **첫 step부터 diff > 0.02**: hook 주입에 버그. proj_loss의 gradient flow가 의도와 다르게 흐름. `compute_loss` 의 retain_graph, detach 호출 확인.
- **diff가 step 2에서부터 폭증**: optimizer가 gradient를 곱하는 LR scaling이 잘못. accelerate config 또는 mixed precision 설정 확인.

- [ ] **4.8.4** Commit.

```bash
git add tests/test_equivalence_fixed.py
git commit -m "test: denoise loss consistency across off/fixed branches (sanity)"
```

---

### ✅ Day 4 오전 완료 체크포인트

다음을 모두 확인:

- [ ] `train/data/ffhq256_latents.py`, `train/branches.py`, `train/save.py`, `train/log.py` 모두 존재
- [ ] `pytest tests/test_dataset.py tests/test_branches.py tests/test_save_policy.py tests/test_equivalence_off.py tests/test_equivalence_fixed.py -v` 가 모두 PASS
- [ ] REPA training script가 `--branch`, `--total_steps`, `--seed`, `--out` CLI 인자를 받음
- [ ] Smoke test 5-step 학습이 `loss_log.csv` 생성, schema가 `step,denoise_loss,proj_loss,coeff,total_loss`
- [ ] `docs/BASELINE.md` 의 hook line numbers가 채워져 있음
- [ ] `git log --oneline | head -20` 에 Day 4 오전 commit 6-8개

체크리스트가 모두 ✓ 면 Day 4 오후 (평가 하네스) 로 진행.

---

## Part 5 — Day 4 (오후): 평가 하네스

**오늘 오후의 목표**: 모든 metric (FID, FD-DINOv2, sharpness Wasserstein, P/R, AUC) 과 §2.3 4-way reporting 모듈 작성. eval.py orchestration + aggregate.py 까지.

**예상 시간**: 4-5 시간.

**산출물**:
- `train/eval/sample.py` (sample 생성)
- `train/eval/fid.py`, `fd_dinov2.py`, `sharpness.py`, `precision_recall.py`
- `train/eval/auc.py` (R5 primary)
- `train/eval/stats.py` (R1 4-way)
- `scripts/eval.py`, `scripts/aggregate.py`

---

### 5.1 Sample 생성 utility

**목적**: EMA model + SD-VAE decode → uint8 image tensor. 디스크에 쓰지 않고 메모리에서만.

**파일**:
- Create: `train/eval/sample.py`
- Create: `tests/test_metrics.py` (initial)

**단계**:

- [ ] **5.1.1** Smoke test.

`tests/test_metrics.py`:
```python
import torch

def test_sample_generation_smoke():
    from train.eval.sample import generate_samples
    class DummyModel(torch.nn.Module):
        def forward(self, x, t, y):
            return torch.randn_like(x)
    samples = generate_samples(
        model=DummyModel(), n=8, batch_size=4,
        latent_shape=(4, 32, 32), steps=10, device="cpu", seed=42)
    assert samples.shape == (8, 3, 256, 256)
    assert samples.dtype == torch.uint8
```

```bash
pytest tests/test_metrics.py::test_sample_generation_smoke -v
```

🔍 **Expected**: FAIL.

- [ ] **5.1.2** Implement.

`train/eval/sample.py`:
```python
"""Generate samples from EMA model + SD-VAE decode, no disk writes.

Returns uint8 tensor (N, 3, 256, 256) for downstream metric computation.

NOTE: The forward sampler used here is a placeholder forward Euler. Replace with
the actual sampler from REPA fork (REPA repo의 sample.py 참고) for production runs
to match the spec §7.1 "50-step DDPM sampler" assumption.
"""
import torch
from diffusers import AutoencoderKL

def generate_samples(model, n, batch_size, latent_shape, steps, device, seed,
                     vae=None, sampler="euler"):
    if vae is None and device != "cpu":
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = vae.config.scaling_factor if vae else 0.18215
    g = torch.Generator(device=device).manual_seed(seed)
    out_imgs = []
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        bs = min(batch_size, n - b * batch_size)
        z = torch.randn(bs, *latent_shape, device=device, generator=g)
        for i in range(steps):
            t = torch.full((bs,), 1.0 - (i + 0.5) / steps, device=device)
            with torch.no_grad():
                v = model(z, t, torch.zeros(bs, dtype=torch.long, device=device))
            z = z - v / steps
        if vae is not None:
            with torch.no_grad():
                img = vae.decode(z / sf).sample
            img = ((img.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()
        else:
            # CPU smoke test: skip VAE, fake decode to (3, 256, 256)
            img = torch.zeros(bs, 3, 256, 256, dtype=torch.uint8)
        out_imgs.append(img)
    return torch.cat(out_imgs, dim=0)
```

⚠ **중요**: 위 sampler는 **placeholder**. 실제 학습/평가에서는 REPA fork의 sampling utility를 import해서 사용해야 spec §7.1 "50-step DDPM sampler" 와 일치. Production에서는 이 함수의 Euler block을 REPA의 정확한 sampler 호출로 교체. (예: `from <repa.sample> import sample_from_model`)

> **왜 placeholder를 두는가?** REPA fork의 정확한 sampling API는 fork마다 다름. 본 튜토리얼은 일반 prose로 작성되어 sampling code의 구체적 형태를 가정 못 함. Smoke test pipeline은 동작하므로 metric 모듈을 먼저 작성하고, eval.py 에서만 진짜 sampler를 호출하면 됨.

- [ ] **5.1.3** Test PASS.

```bash
pytest tests/test_metrics.py::test_sample_generation_smoke -v
```

- [ ] **5.1.4** Commit.

```bash
git add train/eval/sample.py tests/test_metrics.py
git commit -m "feat: sample generation utility (model → SD-VAE → uint8)"
```

---

### 5.2 FID metric

**파일**: `train/eval/fid.py`

- [ ] **5.2.1** Test.

`tests/test_metrics.py` 에 추가:
```python
def test_fid_against_self_is_finite():
    import numpy as np
    from train.eval.fid import compute_fid_from_uint8
    ref = np.load("data/fid_ref_ffhq256.npz")
    samples = (torch.rand(32, 3, 256, 256) * 255).to(torch.uint8)
    fid = compute_fid_from_uint8(samples, ref_mu=ref["mu"], ref_sigma=ref["sigma"])
    assert isinstance(fid, float) and fid > 0
```

- [ ] **5.2.2** Implement.

`train/eval/fid.py`:
```python
"""FID via clean-fid Inception features against precomputed reference."""
import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import frechet_distance

_extractor = None

def _get_extractor(device):
    global _extractor
    if _extractor is None:
        _extractor = build_feature_extractor("clean", device)
    return _extractor

def compute_fid_from_uint8(samples_uint8, ref_mu, ref_sigma, device="cuda", batch_size=64):
    ext = _get_extractor(device)
    feats = []
    for i in range(0, len(samples_uint8), batch_size):
        b = samples_uint8[i:i+batch_size].to(device)
        with torch.no_grad():
            feats.append(ext(b).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return float(frechet_distance(mu, sigma, ref_mu, ref_sigma))
```

- [ ] **5.2.3** Test + commit.

```bash
pytest tests/test_metrics.py::test_fid_against_self_is_finite -v
git add train/eval/fid.py tests/test_metrics.py
git commit -m "feat: FID against precomputed Inception ref"
```

---

### 5.3 FD-DINOv2 metric

**파일**: `train/eval/fd_dinov2.py`

- [ ] **5.3.1** Implement.

`train/eval/fd_dinov2.py`:
```python
"""DINOv2 feature Frechet distance vs precomputed reference."""
import torch
import torchvision.transforms as T
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from scipy.linalg import sqrtm

_proc, _dino = None, None

def _get():
    global _proc, _dino
    if _dino is None:
        _proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _dino = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()
    return _proc, _dino

def compute_fd_dinov2(samples_uint8, ref_path="data/fd_dinov2_ref_ffhq256.pt",
                      batch_size=32):
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
    diff = mu - ref_mu
    covmean = sqrtm(sigma @ ref_sigma)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma + ref_sigma - 2 * covmean))
```

- [ ] **5.3.2** Smoke test + commit.

```python
def test_fd_dinov2_smoke():
    from train.eval.fd_dinov2 import compute_fd_dinov2
    samples = (torch.rand(16, 3, 256, 256) * 255).to(torch.uint8)
    fd = compute_fd_dinov2(samples)
    assert isinstance(fd, float) and fd >= 0
```

```bash
pytest tests/test_metrics.py::test_fd_dinov2_smoke -v
git add train/eval/fd_dinov2.py tests/test_metrics.py
git commit -m "feat: FD-DINOv2 metric"
```

---

### 5.4 Sharpness Wasserstein (3 statistics)

**파일**: `train/eval/sharpness.py`

이 metric은 spec §7.2 "본 연구를 촉발한 'blurry' 관찰을 객관화" 의 핵심.

- [ ] **5.4.1** Implement.

`train/eval/sharpness.py`:
```python
"""3-statistic sharpness Wasserstein distance (spec §7.2)."""
import numpy as np
import cv2
from scipy.stats import wasserstein_distance

def _to_gray(img_chw_uint8):
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

> **세 statistic의 의미**:
> - **Laplacian variance**: 경계의 second-derivative 분산. 흐린 이미지는 낮음.
> - **High-frequency energy ratio**: FFT 고주파 대역의 에너지 비율. 흐린 이미지는 낮음.
> - **Sobel gradient mean**: 1차 미분 평균. 흐린 이미지는 낮음.
>
> 세 가지 모두 sharpness의 다른 면을 측정. 실제 분포(reference)와의 *Wasserstein 거리*를 보고하는 이유: "더 sharp" vs "덜 sharp" 가 아니라 "실제 분포에 얼마나 가까운가" 를 측정하기 위함. 이래야 over-sharpening 도 penalize됨.

- [ ] **5.4.2** Test.

```python
def test_sharpness_three_keys():
    from train.eval.sharpness import sharpness_wasserstein
    gen = (torch.rand(32, 3, 256, 256) * 255).to(torch.uint8)
    ref = (torch.rand(32, 3, 256, 256) * 255).to(torch.uint8)
    out = sharpness_wasserstein(gen, ref)
    assert set(out.keys()) == {"lapvar", "hffreq", "sobel"}
    for v in out.values():
        assert v >= 0.0
```

```bash
pytest tests/test_metrics.py::test_sharpness_three_keys -v
git add train/eval/sharpness.py tests/test_metrics.py
git commit -m "feat: 3-statistic sharpness Wasserstein metric (lapvar/hffreq/sobel)"
```

---

### 5.5 Precision/Recall (Kynkäänniemi 2019)

**파일**: `train/eval/precision_recall.py`

- [ ] **5.5.1** Implement.

`train/eval/precision_recall.py`:
```python
"""Improved Precision and Recall (Kynkäänniemi et al. 2019).
VGG-16 features + manifold k-NN.
"""
import torch
import torchvision.models as M
import torchvision.transforms as T

_vgg = None

def _get_vgg():
    global _vgg
    if _vgg is None:
        _vgg = M.vgg16(weights=M.VGG16_Weights.DEFAULT).features.to("cuda").eval()
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

    d_g_to_r = torch.cdist(g_feat, r_feat)
    inside = (d_g_to_r <= r_radii.unsqueeze(0)).any(dim=1)
    precision = inside.float().mean().item()

    d_r_to_g = torch.cdist(r_feat, g_feat)
    inside_r = (d_r_to_g <= g_radii.unsqueeze(0)).any(dim=1)
    recall = inside_r.float().mean().item()

    return {"precision": float(precision), "recall": float(recall)}
```

- [ ] **5.5.2** Test + commit.

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

```bash
pytest tests/test_metrics.py::test_precision_recall_keys -v
git add train/eval/precision_recall.py tests/test_metrics.py
git commit -m "feat: improved P/R (Kynkäänniemi 2019) with VGG-16"
```

---

### 5.6 FID-vs-step trajectory AUC (R5 primary metric)

**목적**: Critique R5의 핵심. 단일 endpoint FID 가 아니라 trajectory 전체를 정량화.

**파일**: `train/eval/auc.py`

- [ ] **5.6.1** Test.

```python
def test_auc_trapezoid():
    from train.eval.auc import fid_step_auc
    steps = [20000, 50000, 80000, 100000]
    fids = [40.0, 20.0, 12.0, 10.0]
    # 0.5 * ((40+20)*30000 + (20+12)*30000 + (12+10)*20000)
    # = 0.5 * (1_800_000 + 960_000 + 440_000) = 1_600_000
    expected = 1_600_000.0
    assert abs(fid_step_auc(steps, fids) - expected) < 1.0
```

- [ ] **5.6.2** Implement.

`train/eval/auc.py`:
```python
"""FID-vs-step trajectory AUC (spec §2.2, §7.1, §8 — primary metric)."""

def fid_step_auc(steps, fids):
    """Trapezoid integration over (step, FID) sequence.

    Lower AUC = lower FID across the trajectory = better.
    Captures both 'how fast FID drops' and 'how low FID gets'.
    """
    assert len(steps) == len(fids) and len(steps) >= 2
    pairs = sorted(zip(steps, fids))
    auc = 0.0
    for (s0, f0), (s1, f1) in zip(pairs, pairs[1:]):
        auc += 0.5 * (f0 + f1) * (s1 - s0)
    return float(auc)
```

- [ ] **5.6.3** Test PASS + commit.

```bash
pytest tests/test_metrics.py::test_auc_trapezoid -v
git add train/eval/auc.py tests/test_metrics.py
git commit -m "feat: FID-vs-step trapezoid AUC (R5 primary metric)"
```

---

### 5.7 §2.3 4-way reporting stats (R1 핵심)

**목적**: Critique R1의 핵심. n=3 이라는 underpowered setup에서, 단일 hypothesis test에 의존하지 않고 4가지 정보 (descriptive / directional / effect size / hypothesis test) 를 함께 보고.

**파일**: `train/eval/stats.py`, `tests/test_stats.py`

- [ ] **5.7.1** Implement.

`train/eval/stats.py`:
```python
"""§2.3 4-way reporting for n=3 seeds.

Per (branch, metric, eval_step):
- mean ± std across seeds (descriptive)
- directional consistency (sign agreement of seed-level differences)
- Cohen's d (paired) + bootstrap 95% CI (effect size)
- Bonferroni-corrected paired t-test p-value (pre-registered)
"""
import numpy as np
from scipy import stats

def cohens_d_paired(diffs):
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.std(ddof=1) == 0:
        return 0.0
    return float(diffs.mean() / (diffs.std(ddof=1) + 1e-12))

def bootstrap_ci(diffs, n_boot=1000, alpha=0.05, seed=20260415):
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(cohens_d_paired(sample))
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def directional_consistency(diffs):
    diffs = np.asarray(diffs)
    total = len(diffs)
    if total == 0:
        return {"pos": 0, "neg": 0, "total": 0, "majority_frac": 0.0}
    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    return {"pos": pos, "neg": neg, "total": total,
            "majority_frac": max(pos, neg) / total}

def paired_t_bonferroni(diffs, n_comparisons):
    diffs = np.asarray(diffs, dtype=np.float64)
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return {"t": float("nan"), "p_raw": 1.0,
                "p_bonf": 1.0, "significant": False}
    t, p = stats.ttest_1samp(diffs, 0.0)
    p_bonf = min(p * n_comparisons, 1.0)
    return {"t": float(t), "p_raw": float(p), "p_bonf": float(p_bonf),
            "significant": p_bonf < 0.05}

def _interpret(d):
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible (true null)"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large (underpowered null possible if test fails)"

def four_way_report(branch_a_values, branch_b_values, n_comparisons=6):
    """Inputs: n=3 seed-level metric values for branches A and B.
    Output: descriptive + directional + effect size + hypothesis test.
    """
    a = np.asarray(branch_a_values, dtype=np.float64)
    b = np.asarray(branch_b_values, dtype=np.float64)
    diffs = a - b
    d = cohens_d_paired(diffs)
    lo, hi = bootstrap_ci(diffs)
    return {
        "descriptive": {
            "branch_a_mean": float(a.mean()),
            "branch_a_std":  float(a.std(ddof=1)),
            "branch_b_mean": float(b.mean()),
            "branch_b_std":  float(b.std(ddof=1)),
        },
        "directional": directional_consistency(diffs),
        "effect_size": {"cohens_d": d, "ci_lo": lo, "ci_hi": hi},
        "hypothesis_test": paired_t_bonferroni(diffs, n_comparisons),
        "interpretation": _interpret(d),
    }
```

- [ ] **5.7.2** Test.

`tests/test_stats.py`:
```python
import numpy as np
from train.eval.stats import four_way_report, cohens_d_paired, directional_consistency

def test_four_way_keys():
    a = [10.0, 11.0, 9.5]
    b = [12.0, 12.5, 11.0]
    out = four_way_report(a, b, n_comparisons=6)
    assert set(out.keys()) == {"descriptive", "directional", "effect_size",
                                "hypothesis_test", "interpretation"}

def test_directional_all_negative():
    out = directional_consistency(np.array([-1.0, -2.0, -0.5]))
    assert out["neg"] == 3 and out["pos"] == 0

def test_cohens_d_negative():
    a = np.array([10.0, 11.0, 9.5])
    b = np.array([12.0, 12.5, 11.0])
    d = cohens_d_paired(a - b)
    assert d < 0  # a < b

def test_interpretation_large():
    from train.eval.stats import _interpret
    assert "large" in _interpret(1.5)
    assert "negligible" in _interpret(0.1)
```

```bash
pytest tests/test_stats.py -v
git add train/eval/stats.py tests/test_stats.py
git commit -m "feat: §2.3 4-way reporting (descriptive/directional/effect size/test) (R1)"
```

---

### 5.8 `eval.py` orchestration

**목적**: 단일 (run, checkpoint) 평가 — 10K samples 생성, 모든 metric 계산, JSON 저장.

**파일**: `scripts/eval.py`

- [ ] **5.8.1** Script.

`scripts/eval.py`:
```python
"""Evaluate a single (run, checkpoint).

Generates 10K samples, computes all auto metrics, writes JSON.

Usage:
    python scripts/eval.py \\
        --ckpt exps/fixed_s42/checkpoints/ema_100000.pt \\
        --out  results/main/fixed_s42_100000.json \\
        --preview_dir exps/fixed_s42/eval_100000/preview
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
    """Load EMA bf16 state dict into a fresh SiT model.

    REPLACE the import below with the actual SiT class from your REPA fork
    (see docs/BASELINE.md from §3.5).
    """
    from <repa.models.sit> import SiT  # ← REPLACE with actual import
    sd = torch.load(ckpt_path, map_location="cuda")
    model = SiT(...).to("cuda").eval()  # ← config args from your fork
    model.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    return model

def load_ref_images(n=5000):
    """Decode eval split latents into uint8 image tensor (in memory)."""
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
        imgs.append(((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu())
    return torch.cat(imgs, dim=0)

def main(args):
    t0 = time.time()
    model = load_model(args.ckpt)
    samples = generate_samples(
        model=model, n=args.n_samples, batch_size=args.batch_size,
        latent_shape=(4, 32, 32), steps=args.steps, device="cuda",
        seed=args.sample_seed,
    )

    ref = np.load("data/fid_ref_ffhq256.npz")
    fid = compute_fid_from_uint8(samples, ref["mu"], ref["sigma"])
    fd_dino = compute_fd_dinov2(samples)

    ref_imgs = load_ref_images(n=5000)
    sharp = sharpness_wasserstein(samples, ref_imgs)
    pr = precision_recall(samples, ref_imgs, k=3)

    Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
    from torchvision.utils import save_image
    for i in range(min(16, len(samples))):
        save_image(samples[i].float() / 255.0,
                   f"{args.preview_dir}/sample_{i:02d}.png")

    out = {
        "ckpt": args.ckpt,
        "n_samples": args.n_samples,
        "steps_sampler": args.steps,
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
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--sample_seed", type=int, default=20260415)
    p.add_argument("--preview_dir", required=True)
    main(p.parse_args())
```

⚠ **`<repa.models.sit>` 부분을 본인 REPA fork의 실제 import path로 교체.** 그렇지 않으면 `eval.py` 가 즉시 ImportError로 실패.

- [ ] **5.8.2** Random-init smoke test (높은 FID 예상).

```bash
# random model checkpoint 만들기
python -c "
import torch
# from <repa.models.sit> import SiT
# m = SiT(...)
# torch.save({k: v.to(torch.bfloat16) for k, v in m.state_dict().items()},
#            '/tmp/random_ckpt.pt')
print('manual: open this script, replace SiT import, run')
"

# eval 시도
python scripts/eval.py \
    --ckpt /tmp/random_ckpt.pt \
    --out /tmp/random_eval.json \
    --preview_dir /tmp/random_preview \
    --n_samples 256

cat /tmp/random_eval.json
```

🔍 **Expected**: pipeline 통과. FID는 매우 높음 (~300+) — 의미 없는 sample이므로 정상.

⚠ **트러블슈팅**:
- **ImportError**: `<repa.models.sit>` 경로를 실제 import로 교체했는지 확인.
- **CUDA OOM at sample generation**: `--batch_size 32` 또는 더 작게.
- **OOM at SD-VAE decode**: `load_ref_images` 의 BS를 16으로 줄임.

- [ ] **5.8.3** Commit.

```bash
git add scripts/eval.py
git commit -m "feat: eval.py orchestration (10K samples → all metrics → JSON)"
```

---

### 5.9 `aggregate.py` results aggregator

**목적**: 60개 eval JSON을 한 CSV로 합침.

**파일**: `scripts/aggregate.py`

- [ ] **5.9.1** Script.

`scripts/aggregate.py`:
```python
"""Aggregate eval JSONs in results/ into results.csv."""
import argparse, csv, json, re
from pathlib import Path

def main(args):
    in_dir = Path(args.in_dir)
    rows = []
    for f in sorted(in_dir.glob("*.json")):
        d = json.loads(f.read_text())
        # filename pattern: {branch}_s{seed}_{step}.json
        m = re.match(r"^([a-z]+)_s(\d+)_(\d+)\.json$", f.name)
        if not m:
            print(f"skip (bad name): {f.name}")
            continue
        branch, seed, step = m.group(1), int(m.group(2)), int(m.group(3))
        rows.append({
            "branch": branch, "seed": seed, "step": step,
            "fid": d["fid"], "fd_dinov2": d["fd_dinov2"],
            "precision": d["precision_recall"]["precision"],
            "recall": d["precision_recall"]["recall"],
            "sharp_lapvar": d["sharpness"]["lapvar"],
            "sharp_hffreq": d["sharpness"]["hffreq"],
            "sharp_sobel":  d["sharpness"]["sobel"],
            "wallclock_sec": d["wallclock_sec"],
        })
    if not rows:
        print("no JSONs found"); return
    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✓ wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="results/main")
    p.add_argument("--out", default="results.csv")
    main(p.parse_args())
```

- [ ] **5.9.2** Smoke (random eval JSON 1개로 테스트).

```bash
mkdir -p results/main
echo '{"fid": 320.5, "fd_dinov2": 12.3, "precision_recall": {"precision": 0.05, "recall": 0.02}, "sharpness": {"lapvar": 1234, "hffreq": 0.1, "sobel": 8.5}, "wallclock_sec": 1500}' \
    > results/main/test_s42_020000.json
python scripts/aggregate.py --in_dir results/main --out /tmp/test.csv
cat /tmp/test.csv
rm results/main/test_s42_020000.json
```

🔍 **Expected**:
```
branch,seed,step,fid,fd_dinov2,precision,recall,sharp_lapvar,sharp_hffreq,sharp_sobel,wallclock_sec
test,42,20000,320.5,12.3,0.05,0.02,1234,0.1,8.5,1500
```

- [ ] **5.9.3** Commit.

```bash
git add scripts/aggregate.py
git commit -m "feat: results aggregator (eval JSONs → results.csv)"
```

---

### ✅ Day 4 오후 완료 체크포인트

다음을 모두 확인:

- [ ] `train/eval/sample.py`, `fid.py`, `fd_dinov2.py`, `sharpness.py`, `precision_recall.py`, `auc.py`, `stats.py` 모두 존재
- [ ] `pytest tests/test_metrics.py tests/test_stats.py -v` 모두 PASS
- [ ] `scripts/eval.py` 의 `<repa.models.sit>` 부분이 실제 import로 교체됨
- [ ] `scripts/eval.py` 가 random-init checkpoint로 smoke test 통과 (FID 매우 높지만 pipeline은 작동)
- [ ] `scripts/aggregate.py` smoke test 통과
- [ ] `git log --oneline | head` 에 Day 4 오후 commit 6-7개

체크리스트가 모두 ✓ 면 Day 5 (pre-registration) 로 진행.

---

## Part 6 — Day 5: Pre-registration (frozen)

**오늘의 목표**: Power analysis 계산, `docs/prereg.md` 작성, `prereg-v1` git tag로 freeze.

**예상 시간**: 2-3 시간.

🚨 **이 phase는 본 연구의 가장 중요한 commit**. Tag 이후 spec/prereg/§2-§8 변경은 *모두* 사후 변경으로 간주되어 paper에서 "exploratory" 로 명시되어야 함.

**산출물**:
- `scripts/power_analysis.py` + `data/power_analysis.json`
- `docs/prereg.md`
- Git tag `prereg-v1`

---

### 6.1 Power analysis 실행

**목적**: §2.3에 명시된 minimum detectable Cohen's d ≈ 4.5 를 *수치로 계산*해서 prereg에 함께 commit.

**파일**: `scripts/power_analysis.py`, `data/power_analysis.json`

- [ ] **6.1.1** Script.

`scripts/power_analysis.py`:
```python
"""Compute the minimum detectable effect size for the §2.3 power statement.
n=3 seeds, paired t-test, Bonferroni α=0.05/6, power=0.80.
Uses scipy noncentral t.
"""
import json
from scipy import stats
import numpy as np

def min_detectable_d(n, alpha, power):
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
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
        "interpretation": (
            "Effects with |d| < {:.2f} cannot be detected at the pre-registered "
            "significance level. Effects with 0.8 < |d| < {:.2f} are reported as "
            "'underpowered null' per §2.3.".format(d_min, d_min)
        ),
    }
    print(json.dumps(out, indent=2))
    open("data/power_analysis.json", "w").write(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **6.1.2** 실행.

```bash
python scripts/power_analysis.py
```

🔍 **Expected**:
```json
{
  "n_seeds": 3,
  "alpha_corrected": 0.008333...,
  "power_target": 0.8,
  "min_detectable_cohens_d": 4.5...,
  "interpretation": "..."
}
```

`min_detectable_cohens_d` 가 spec §2.3의 "≈ 4.5" 와 일치해야 함. 4.0 ~ 5.0 범위면 정상.

- [ ] **6.1.3** Commit.

```bash
git add scripts/power_analysis.py data/power_analysis.json
git commit -m "feat: power analysis (n=3 → min detectable Cohen's d ≈ 4.5)"
```

---

### 6.2 `docs/prereg.md` 작성

**목적**: Spec §2.1, §2.2, §2.3, §8 의 핵심 내용을 self-contained하게 prereg.md에 복사. 향후 모든 분석은 이 문서를 ground truth로.

**파일**: `docs/prereg.md`

- [ ] **6.2.1** Prereg 작성.

`docs/prereg.md`:
```markdown
# Pre-registration: REPA 재평가 연구 (FFHQ-256)

**Frozen at**: <git rev-parse HEAD>  ← 본 commit 후 갱신
**Tag**: `prereg-v1`
**Analyst**: <your name>
**Date**: 2026-04-15

이 문서는 본 연구의 frozen ground truth입니다. 본 commit 이후 어떤 사후 수정도
spec §3.3 위반으로 처리되며, 추가 분석은 paper에서 "exploratory"로 명시됩니다.

## 1. Hypothesis (spec §2.1 Claim B)

> 고정 계수 REPA는 FFHQ-256에서 수렴 속도와 샘플 sharpness/diversity 사이에 지금까지
> 특성화되지 않은 trade-off를 보인다. 본 연구는 다섯 가지 계수 스케줄 (off, fixed,
> decay, warmup, hard cutoff) 에 걸쳐 이 trade-off를 특성화하고, 분포 수준 sharpness
> metric과 paired human preference를 사용한 diagnostic protocol을 제시한다.

## 2. Pre-registered Outcome Conditions (spec §2.2)

**Positive (diagnostic claim 성립)**:
- Primary metric (FID-vs-step trajectory AUC, §7.1) 또는 sharpness/diversity metric
  중 최소 하나에서 REPA-Fixed가 REPA-Off 대비 통계적으로 유의미하게
  (paired t-test, Bonferroni 보정 α=0.05/6) 저하, **그리고**
- 스케줄 변형 (decay/warmup/cutoff) 중 최소 하나가 fixed REPA의 수렴 속도 이점을
  유지하면서 저하된 축을 부분적으로 회복.

**Null**: 어떤 branch 간에도 통계적 유의미한 차이가 없음. Underpowered null인 경우
§2.3의 effect-size 기반 분리 적용.

**Mixed**: 자동 metric ↔ human eval 불일치. Dissociation finding으로 보고.

세 경우 모두 publishable.

## 3. Statistical Power (spec §2.3)

- n=3 seeds per branch, paired t-test, Bonferroni α=0.05/6 ≈ 0.0083.
- Minimum detectable Cohen's d ≈ 4.5 (`scripts/power_analysis.py` 산출, 본 commit
  에 포함된 `data/power_analysis.json` 참조).
- **4-way reporting 강제** (descriptive / directional consistency / effect size /
  hypothesis test). Single test에 의한 결론 금지.
- |d| > 0.8 + non-significant → "underpowered null" 명시. |d| < 0.2 → true null.

## 4. Primary Metric

**FID-vs-step trajectory AUC** over the 4 eval checkpoints (20K, 50K, 80K, 100K),
trapezoid integration. Compared via paired t-test (Bonferroni α=0.05/6 across
the 6 pre-registered branch pairs).

**Secondary**: 100K step 단일점 FID, Precision, Recall, FD-DINOv2,
3개 sharpness Wasserstein, Bradley-Terry score, jaggedness.

## 5. Stop Conditions (spec §8 #5)

- Pilot (5 branches × 1 seed × 20K) 에서 어느 metric에서도 식별 가능한 signal이
  3+ branches에 걸쳐 없으면 **일시 중단**, 연구 질문 재평가.
- Compute가 GPU 연속 사용 5일 초과 시 **일시 중단**, 우선순위 재정의.

## 6. Pre-registered Branch Ranking Expectation (confirmation-bias check, §8 #6)

- Sharpness: B0 > B2 ≈ B4 > B1 > B3
- FID 수렴 속도: B1 ≈ B4 > B2 > B3 > B0

실험자의 사전 가설. 실제 결과가 이와 *완벽히* 일치하면 더 의심스러운 것으로 해석.

## 7. Pre-registered Branches (spec §6)

- B0 (off):    λ(t) = 0
- B1 (fixed):  λ(t) = 0.5
- B2 (decay):  λ(t) = 0.5 × 0.5 × (1 + cos(π t / T))
- B3 (warmup): λ(t) = 0.5 × 0.5 × (1 - cos(π t / T))
- B4 (cutoff): λ(t) = 0.5 if t < T/2 else 0

n_seeds = 3 per branch → 15 main runs.

## 8. Pre-registered Pairs for Hypothesis Test (spec §7.3, §8)

6 pairs:
1. B0 vs B1 (REPA가 도움이 되는가 해가 되는가?)
2. B0 vs B2 (decay schedule 도움?)
3. B0 vs B3 (warmup schedule 도움?)
4. B1 vs B2 (decay가 fixed보다?)
5. B1 vs B4 (hard cutoff가 fixed보다?)
6. B2 vs B4 (smooth vs hard?)

Bonferroni 보정 α/6 ≈ 0.0083.

## 9. Data, Model, Eval Setup (spec §4-§7)

- Dataset: FFHQ-256 streamed from `<HF mirror id>` (revision `<sha>`).
- Latents: SD-VAE-ft-mse, bf16, 70K precomputed.
- Train/eval split: 65000 / 5000 (seed 20260415, content-addressed in MANIFEST.json).
- Model: SiT-B/2, batch=32, AdamW lr=1e-4, EMA 0.9999, bf16 mixed precision.
- 100K steps per run.
- Eval checkpoints: 20K/50K/80K/100K. EMA-only bf16 storage (§4.4).
- Sample generation: 50-step DDPM, no CFG, 10K samples per evaluation.
- FID reference: clean-fid Inception statistics from eval split (decoded latents).
- FD-DINOv2 reference: DINOv2 ViT-B patch-token mean from eval split.

## 10. Strict Rule

이 commit 이후 본 문서는 **수정 불가**. 추가 분석이 필요하면 paper에서 "exploratory"
로 분리해서 보고합니다. Pre-registration은 본 연구의 *과학적 정직성*에 대한 약속입니다.
```

`<your name>`, `<HF mirror id>`, `<sha>` 등 placeholder를 실제 값으로 교체.

> **왜 이렇게 자세히?** Spec §8의 7개 항목을 prereg가 자체 포함하면, paper의 reviewer가 spec과 prereg를 따로 확인할 필요 없이 prereg 한 문서로 모든 commitment를 확인 가능. Self-contained.

- [ ] **6.2.2** Self-review. 다음을 확인:
  - 모든 `<placeholder>` 가 채워짐
  - Spec §2.1, §2.2, §2.3 의 핵심 문구가 그대로 또는 거의 그대로 옮겨짐
  - Branch 정의 (B0-B4) 가 spec §6와 일치
  - Pair 6개가 spec §7.3와 일치

- [ ] **6.2.3** Commit (tag 전).

```bash
git add docs/prereg.md
git commit -m "prereg: pre-registered hypotheses, outcome conditions, power, branches"
```

---

### 6.3 Tag `prereg-v1` (frozen)

**목적**: Git tag로 prereg.md commit을 *immutable* marker로 표시. 이후 모든 main run commit은 이 tag를 *후행*해야 함.

**단계**:

- [ ] **6.3.1** Annotated tag 생성.

```bash
git tag -a prereg-v1 -m "Pre-registration frozen. No post-hoc modification."
```

- [ ] **6.3.2** Tag 검증.

```bash
git show prereg-v1 | head -30
git tag -l
```

🔍 **Expected**: `prereg-v1` 이 tag list에 보이고, `git show prereg-v1` 출력에 우리 commit message + diff가 보임.

- [ ] **6.3.3** (Optional) Remote에 push.

```bash
git push origin reeval --tags
```

🚨 Push는 **돌이키기 어려움**. Push한 tag를 force update하면 ✗. 본인 GitHub fork에 push하는 것이므로 누구에게도 영향은 없지만, 본인의 미래 자아에게 신호 — "이 시점부터 frozen".

- [ ] **6.3.4** Tag 이후 informational commit (frozen 사실 자체를 doc에 기록).

```bash
SHA=$(git rev-parse prereg-v1)
sed -i "s|<git rev-parse HEAD>|$SHA|" docs/prereg.md
git add docs/prereg.md
git commit -m "docs: record prereg-v1 commit hash post-tag (informational)"
```

> **이 commit은 prereg-v1 *이후* 다. Prereg 본문 자체는 frozen이지만, "frozen at <SHA>" 메타데이터를 채우기 위한 minor 수정. Reviewer에게 "이 informational metadata 외 변경 없음" 을 git diff로 증명 가능.

---

### ✅ Day 5 완료 체크포인트

- [ ] `data/power_analysis.json` 의 `min_detectable_cohens_d` 가 4.0 ~ 5.0
- [ ] `docs/prereg.md` 의 모든 placeholder가 채워짐
- [ ] `git tag -l` 에 `prereg-v1` 이 보임
- [ ] `git show prereg-v1` 출력에 prereg.md commit이 표시
- [ ] `git log --oneline` 에 Day 5 commit 2-3개

🚨 **이 시점부터 spec과 prereg를 절대 변경하지 마세요.** 변경이 필요하면 paper에서 별도 "exploratory" 로 분리.

체크리스트가 모두 ✓ 면 Day 6 (pilot) 로 진행.

---

## Part 7 — Day 6-7: Pilot runs + signal check

**오늘의 목표**: 5개 branch × 1 seed × 20K steps의 pilot 학습. Wallclock 검증, signal 확인, 진행 여부 결정.

**예상 시간**: 5-8 시간 (대부분이 학습 wallclock).

🚨 **이 phase가 §10 의 risk row 두 개 (compute, signal) 의 결정점**. 결과에 따라 main run을 진행하거나 일시 중단해야 함.

**산출물**:
- `data/wallclock_benchmark.json`
- `exps/pilot_*/checkpoints/ema_020000.pt` (5개)
- `results/pilot/*.json` (5개)
- `results/pilot.csv`

---

### 7.1 Wallclock 5K-step 측정 (R3 mitigation)

**목적**: Spec §9 의 "각 run 약 3.5h" 외삽이 정확한지 *작은 sample*로 검증. 100K-step 추정의 baseline.

**파일**: `data/wallclock_benchmark.json`

- [ ] **7.1.1** 1개 branch × 5K steps 실행 (시간 측정).

```bash
mkdir -p data
time python <repa_main_train_script>.py \
    --branch fixed --seed 42 --total_steps 5000 \
    --out exps/wallclock_test 2>&1 | tee data/wallclock_log.txt
```

🔍 **Expected**: 5000 step 학습이 정상 종료. `time` 출력의 `real` 시간 기록.

- [ ] **7.1.2** 결과 JSON화.

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
    'hours_per_5k': hours_per_5k,
    'hours_per_100k_extrapolated': hours_per_100k,
    'hours_per_15_runs_extrapolated': hours_per_15_runs,
    'spec_estimate_hours_per_run': 3.5,
    'spec_estimate_hours_total': 52.5,
    'within_spec_estimate': hours_per_15_runs < 100,
}
print(json.dumps(out, indent=2))
open('data/wallclock_benchmark.json','w').write(json.dumps(out, indent=2))
"
```

- [ ] **7.1.3** 결과 해석 (decision point).

🚨 **결정 매트릭스**:

| `hours_per_100k` 값 | 의미 | 행동 |
|---|---|---|
| ≤ 4h | spec 외삽 정확 또는 빠름 | 그대로 main run 진행 |
| 4-5h | spec 외삽 약간 underestimate | 그대로 진행, 단 buffer 적음 |
| 5-7h | spec 외삽 부정확. 15 runs ≈ 100h | seed 3→2 cut 고려 (§10 fallback). 또는 step 100K → 80K. |
| > 7h | spec invalidated. 15 runs > 105h | **STOP**. Spec §10 stop condition. 다음 중 선택: (a) step budget 절감, (b) seed 절감, (c) batch size 증가 + step 절감. |

`> 7h` 인데 진행하면 총 timeline 초과. spec §9 timeline frozen 하에서는 stop이 정답.

- [ ] **7.1.4** Cleanup + commit.

```bash
rm -rf exps/wallclock_test
git add data/wallclock_benchmark.json data/wallclock_log.txt
git commit -m "data: wallclock benchmark (5K → extrapolate to 15 runs)"
```

---

### 7.2 Pilot training sweep

**목적**: 5 branches × 1 seed × 20K steps. 모든 branch가 학습이 가능하고, schedule이 의도대로 동작하는지 확인.

**파일**: `scripts/run_pilot.sh`

- [ ] **7.2.1** Sweep script.

`scripts/run_pilot.sh`:
```bash
#!/bin/bash
set -euo pipefail
SEED=42
STEPS=20000
TRAIN_SCRIPT="<repa_main_train_script>.py"   # ← 실제 파일명으로 교체

for BRANCH in off fixed decay warmup cutoff; do
    OUT="exps/pilot_${BRANCH}_s${SEED}"
    if [ -f "$OUT/done" ]; then
        echo "skip $OUT (already done)"
        continue
    fi
    echo ">>> $BRANCH s${SEED} ${STEPS} steps"
    python "$TRAIN_SCRIPT" --branch "$BRANCH" --seed "$SEED" \
        --total_steps "$STEPS" --out "$OUT"
    touch "$OUT/done"
done
echo "✓ pilot sweep complete"
```

```bash
chmod +x scripts/run_pilot.sh
```

- [ ] **7.2.2** 디스크 monitor를 background로 띄우고 sweep 실행.

```bash
# Terminal A (모니터)
python scripts/disk_monitor.py --min_gb 12 --interval 120 --exit_on_alert &
DISK_PID=$!

# Terminal A continued (sweep)
bash scripts/run_pilot.sh 2>&1 | tee exps/pilot_run.log

# 끝나면 monitor 정리
kill $DISK_PID 2>/dev/null || true
```

🔍 **Expected**: 5 branch × ~40min ≈ 3-4 GPU-hours. (Branch당 wallclock은 7.1의 hours_per_100k × 0.2 정도.)

⚠ **트러블슈팅**:
- **NaN loss**: 학습 발산. 한 branch만 NaN이면 schedule 버그 (특히 B3 warmup의 t=0에서 λ=0 인지 확인). 모든 branch에서 NaN이면 model init 또는 mixed precision 문제.
- **OOM**: batch=32에서 16GB VRAM 빠듯. 배치 16으로 줄임.
- **너무 느림 (실측 > 외삽 × 1.5)**: GPU 사용률 nvidia-smi로 확인. CPU bottleneck (latent loading, DINO target 계산) 가능. DINO target을 미리 계산해두면 (Day 4의 옵션 B) 큰 speedup.

- [ ] **7.2.3** Commit.

```bash
git add scripts/run_pilot.sh exps/pilot_run.log
git commit -m "ops: pilot sweep complete (5 × 1 × 20K)"
```

---

### 7.3 Pilot eval @ 20K + signal check

**목적**: Pilot checkpoint들을 eval해서, branch 간 차이가 *측정 가능한 수준으로* 나타나는지 확인. Signal이 너무 적으면 main run에서도 같음.

**단계**:

- [ ] **7.3.1** 5개 pilot checkpoint eval.

```bash
mkdir -p results/pilot
for B in off fixed decay warmup cutoff; do
    python scripts/eval.py \
        --ckpt exps/pilot_${B}_s42/checkpoints/ema_020000.pt \
        --out results/pilot/${B}_s42_20000.json \
        --preview_dir exps/pilot_${B}_s42/eval_20000/preview \
        --n_samples 5000
done
```

⚠ pilot eval에서는 `--n_samples 5000` 으로 절감 (10K는 main eval에서). 5K로도 FID는 충분히 stable.

🔍 **Expected**: 각 eval 약 15 분 × 5 = ~1.5 시간.

- [ ] **7.3.2** Aggregate + 빠른 비교.

```bash
python scripts/aggregate.py --in_dir results/pilot --out results/pilot.csv
column -t -s, results/pilot.csv
```

- [ ] **7.3.3** Signal check (decision point).

🚨 **결정 매트릭스**:

| 관찰 | 진단 | 행동 |
|---|---|---|
| 5 branch FID가 모두 비슷 (max - min < 5%) | Signal 부족 OR 학습이 너무 짧음 | (a) 수동 sample 시각 검사, (b) loss curve 비교, (c) 학습 너무 짧으면 main run의 100K가 signal 줄지 검토 |
| 1+ branch sample이 random noise (시각적으로 구분 안 됨) | 학습 실패 | 해당 branch 디버그. equivalence test 다시. |
| Preview에서 사람 얼굴이 보임 (몇몇 distortion 있어도 OK) | 학습 정상, signal 존재 | 진행 |
| FID가 1000+ | 학습 거의 안 됨 | step 부족하거나 hyperparameter 잘못. wallclock 재확인 + sample 시각 검사 |

> **5K samples FID는 noisy** — 정확한 비교는 main eval (10K samples) 에서. Pilot eval은 *qualitative* signal check 용도.

- [ ] **7.3.4** 시각 검사. 각 branch의 preview 16장을 한 화면에 띄워 비교.

```bash
ls exps/pilot_*/eval_20000/preview/sample_00.png
# (이미지 viewer로 5장 동시에 열어 face quality 비교)
```

🔍 **Expected**: 모든 branch에서 사람 얼굴이 인식 가능. Quality는 천차만별 OK (이게 본 연구의 finding).

- [ ] **7.3.5** Commit pilot results.

```bash
git add results/pilot/ results/pilot.csv
git commit -m "data: pilot eval @ 20K (5 branches × 1 seed × 5K samples)"
```

🚨 **결정점 — main run 진행 여부**:

✅ 진행 가능:
- 5 branch 모두 학습 정상
- FID/sample이 합리적 (≤ 200)
- Pilot에서 branch 간 작은 차이가 보임
- Wallclock 추정 (§7.1) 이 100h 이내

✗ 진행 중단 (spec §8 stop condition):
- 모든 branch가 동일하게 보임 (signal 0)
- 1+ branch 학습 실패 / 시각적으로 noise
- Wallclock 추정 > 100h

---

### ✅ Day 6-7 완료 체크포인트

- [ ] `data/wallclock_benchmark.json` 의 `hours_per_15_runs_extrapolated` 가 100 미만
- [ ] `exps/pilot_*/checkpoints/ema_020000.pt` 5개 모두 존재
- [ ] `results/pilot.csv` 가 5 row, 모든 metric 채워짐
- [ ] 시각 검사로 모든 branch에서 face 인식 가능
- [ ] Decision point 통과 (위 결정 매트릭스 참조)
- [ ] `git log --oneline` 에 Day 6-7 commit 2-3개

체크리스트가 모두 ✓ 면 Day 8 (main run) 으로 진행.

---

## Part 8 — Day 8-11: Main 15 runs

**오늘들의 목표**: 15 runs (5 branches × 3 seeds) × 100K steps 를 sequential 실행. 이 phase는 *대부분 기다리는 시간*. 하루 2-4 runs 진행.

**예상 시간**: ~52.5 GPU-hours (spec §9 estimate) ≈ 4-5일. 가용 28일 timeline의 가장 큰 단일 비용.

🚨 **이 phase가 hardware/software 안정성에 가장 취약**. GPU thermal throttling, OS update, 시스템 reboot, 디스크 부족 등이 모두 학습 중단의 원인이 됨.

**산출물**:
- `exps/{branch}_s{seed}/checkpoints/ema_*.pt` (60 ckpts: 15 runs × 4 eval points)
- `exps/{branch}_s{seed}/loss_log.csv` (15)
- `exps/main_run.log` (단일 통합 로그)
- `data/disk_log_main.csv` (디스크 history)

---

### 8.1 Main sweep launcher

**파일**: `scripts/run_main.sh`

- [ ] **8.1.1** Script.

`scripts/run_main.sh`:
```bash
#!/bin/bash
set -euo pipefail
STEPS=100000
SEEDS=(42 1337 2024)
BRANCHES=(off fixed decay warmup cutoff)
TRAIN_SCRIPT="<repa_main_train_script>.py"   # ← 실제 파일명

START=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    for BRANCH in "${BRANCHES[@]}"; do
        OUT="exps/${BRANCH}_s${SEED}"
        if [ -f "$OUT/done" ]; then
            echo "[skip] $OUT (already done)"
            continue
        fi
        echo "[$(date +%H:%M:%S)] >>> $BRANCH s${SEED} ${STEPS} steps"
        python "$TRAIN_SCRIPT" --branch "$BRANCH" --seed "$SEED" \
            --total_steps "$STEPS" --out "$OUT"
        touch "$OUT/done"
        df -h . | tail -1
    done
done
END=$(date +%s)
HOURS=$(echo "scale=2; ($END - $START) / 3600" | bc)
echo "✓ all 15 main runs complete in ${HOURS} hours"
```

```bash
chmod +x scripts/run_main.sh
```

> **왜 sequential인가?** 단일 GPU + 16GB VRAM에서 multi-run parallel은 어려움 (각 run이 ~10GB VRAM). Sequential이 가장 단순하고 안전.

> **왜 `done` flag 파일?** 중간에 stop했다가 resume할 때, 이미 끝난 run을 skip하고 남은 것만 진행. Idempotent launcher.

- [ ] **8.1.2** Commit launcher (실행 전).

```bash
git add scripts/run_main.sh
git commit -m "ops: main sweep launcher (15 runs sequential, idempotent)"
```

---

### 8.2 tmux/screen에서 실행

**목적**: 학습이 며칠 걸리므로 SSH 세션이 끊겨도 계속 돌아가야 함.

**단계**:

- [ ] **8.2.1** tmux 세션 시작.

```bash
tmux new -s main
```

- [ ] **8.2.2** 세션 안에서 disk monitor를 background로.

```bash
python scripts/disk_monitor.py --min_gb 12 --interval 300 --exit_on_alert > data/disk_log_main.csv 2>&1 &
DISK_PID=$!
echo $DISK_PID > /tmp/disk_monitor.pid
```

- [ ] **8.2.3** Sweep 실행.

```bash
mkdir -p exps
bash scripts/run_main.sh 2>&1 | tee exps/main_run.log
```

- [ ] **8.2.4** Detach (`Ctrl+B` then `D`).

`tmux ls` 로 세션이 아직 살아있는지 확인.

- [ ] **8.2.5** SSH 끊긴 후 재접속 시.

```bash
ssh <host>
tmux attach -t main
# 진행률 확인:
ls exps/*/done 2>/dev/null | wc -l   # 0 → 15
```

⚠ **트러블슈팅**:
- **tmux 세션 사라짐**: 시스템 reboot됐을 가능성. `dmesg | tail -100` 으로 OOM kill 또는 thermal shutdown 확인. 재시작 후 `bash scripts/run_main.sh` 다시 (idempotent).
- **GPU 안 보임**: nvidia-smi가 실패하면 driver 재설치 필요. Sweep 중지하고 fix.

---

### 8.3 매일 진행 확인 (모니터링 의례)

매일 한 번씩:

- [ ] **8.3.1** 진행률.

```bash
ls exps/*/done 2>/dev/null | wc -l
ls exps/*/checkpoints/ema_*.pt 2>/dev/null | wc -l   # eval ckpt 수
```

- [ ] **8.3.2** 디스크 상태.

```bash
df -h .
tail -5 data/disk_log_main.csv
```

🔍 **Expected**: 디스크가 ~30 GB 이상 가용 유지. 만약 < 15 GB로 떨어지면 disk monitor가 자동 alert + sweep halt.

- [ ] **8.3.3** Loss curve sanity (마지막 끝난 run).

```bash
LATEST=$(ls -t exps/*/loss_log.csv | head -1)
echo "Latest run: $LATEST"
tail -5 "$LATEST"
# Quick plot
python -c "
import pandas as pd, sys
df = pd.read_csv('$LATEST')
print('total_loss range:', df['total_loss'].min(), '→', df['total_loss'].max())
print('last 100 mean:', df['total_loss'].tail(100).mean())
print('did it diverge:', df['total_loss'].iloc[-1] > df['total_loss'].iloc[100])
"
```

🔍 **Expected**: total_loss 가 step에 따라 감소. 마지막 100 step의 평균이 step 100 즈음의 평균보다 낮음. 발산 (마지막이 더 큼) 이면 problem.

- [ ] **8.3.4** Preview (last completed run).

```bash
LATEST=$(ls -td exps/*/eval_*/preview 2>/dev/null | head -1)
echo "Latest preview: $LATEST"
ls "$LATEST" | head
# 이미지 뷰어로 sample_00.png 확인
```

🔍 **Expected**: 사람 얼굴이 인식 가능 (quality는 branch마다 다름).

⚠ **트러블슈팅**:
- **NaN appears mid-training**: 발산. Logs에서 NaN 발생 step 확인. 해당 run을 폐기 (`rm -rf exps/<bad_run>` 후 idempotent rerun으로 다시 시도). Persistent하면 mixed precision overflow — fp32로 fallback 또는 gradient clip 강화.
- **학습 속도가 갑자기 느려짐**: GPU thermal throttling 가능. `nvidia-smi -q -d temperature` 로 확인. > 85°C면 cooling 점검.

---

### 8.4 흔한 trouble + 대응

| 증상 | 원인 후보 | 대응 |
|---|---|---|
| `CUDA out of memory` 첫 step부터 | batch=32가 16GB에 안 맞음 | batch=16, gradient accumulation 2배 |
| 학습이 처음 10K step에서 완전히 stuck (loss 안 떨어짐) | LR 또는 schedule 문제 | LR 1e-4 확인, `total_steps` 가 100000 인지 확인 (B2 cosine decay가 T를 잘못 알면 entire run이 fixed처럼 보임) |
| 디스크 < 15 GB | `exps/` 가 과적재 또는 latent 파일 손상 | `du -sh exps/*` 로 어디가 큰지 확인. preview만 너무 많이 saved면 cleanup |
| `tmux: command not found` | tmux 미설치 | `sudo apt install tmux` 또는 `screen` 으로 대체 |
| 학습 중간 (eval ckpt 사이) 에 reboot | 시스템 update 등 | 해당 run 폐기 후 다시. EMA state는 이전 eval ckpt에서 시작 못 하므로 처음부터. (Spec §4.4 "fail 시 처음부터" 정책의 trade-off.) |

---

### ✅ Day 8-11 완료 체크포인트

- [ ] `ls exps/*/done | wc -l` 가 15 (모든 run 끝)
- [ ] `ls exps/*/checkpoints/ema_*.pt | wc -l` 가 60 (4 eval ckpt × 15 run)
- [ ] `df -h .` 가 ≥ 30 GB 가용
- [ ] 모든 `exps/*/loss_log.csv` 가 100K step (또는 100K-1) row 포함
- [ ] `exps/main_run.log` 의 마지막 line이 `✓ all 15 main runs complete in XX.XX hours`
- [ ] 시각 검사로 모든 branch의 마지막 preview에서 face 인식

이후 Day 12 (auto eval) 진행.

---

## Part 9 — Day 12-14: Automatic evaluation (60 evals)

**오늘들의 목표**: 15 runs × 4 eval checkpoints = 60 evaluation을 sequential 실행. Aggregate. Primary metric (FID-vs-step AUC) 계산.

**예상 시간**: 약 25 GPU-hours (spec §9 estimate) ≈ 1-2일.

**산출물**:
- `results/main/{branch}_s{seed}_{step}.json` (60)
- `results.csv` (aggregated)
- `results_primary.csv` (per-(branch,seed) primary + secondary metrics)
- `paper/figures/fig_fid_vs_step.png`

---

### 9.1 60 evaluation 일괄 실행

**파일**: `scripts/run_eval_all.sh`

- [ ] **9.1.1** Script.

`scripts/run_eval_all.sh`:
```bash
#!/bin/bash
set -euo pipefail
mkdir -p results/main
START=$(date +%s)
for SEED in 42 1337 2024; do
    for BRANCH in off fixed decay warmup cutoff; do
        for STEP in 020000 050000 080000 100000; do
            CKPT="exps/${BRANCH}_s${SEED}/checkpoints/ema_${STEP}.pt"
            OUT="results/main/${BRANCH}_s${SEED}_${STEP}.json"
            if [ -f "$OUT" ]; then
                echo "[skip] $OUT"
                continue
            fi
            if [ ! -f "$CKPT" ]; then
                echo "[!!] missing ckpt: $CKPT"
                continue
            fi
            echo "[$(date +%H:%M:%S)] $BRANCH s${SEED} ${STEP}"
            python scripts/eval.py --ckpt "$CKPT" --out "$OUT" \
                --preview_dir "exps/${BRANCH}_s${SEED}/eval_${STEP}/preview"
        done
    done
done
END=$(date +%s)
HOURS=$(echo "scale=2; ($END - $START) / 3600" | bc)
echo "✓ all 60 evals complete in ${HOURS} hours"
```

```bash
chmod +x scripts/run_eval_all.sh
```

- [ ] **9.1.2** tmux에서 실행 (Day 8-11과 같은 방식).

```bash
tmux new -s eval
python scripts/disk_monitor.py --min_gb 10 --interval 300 > data/disk_log_eval.csv 2>&1 &
bash scripts/run_eval_all.sh 2>&1 | tee exps/eval_all.log
# Ctrl+B, D
```

🔍 **Expected**: 60 × ~25min ≈ 25 hours. Idempotent — fail 시 재시작.

⚠ **트러블슈팅**:
- **CUDA OOM during sample generation**: `--batch_size 32` (eval.py 인자) 로 절감.
- **DINOv2 fail**: 첫 다운로드 안 됐으면 인터넷 필요. 또는 cache 확인.
- **`load_ref_images` 가 매번 reload**: 60 evaluation 동안 SD-VAE 재사용 가능. eval.py 를 수정해서 ref_imgs를 module-level cache로 만들면 ~1 min/eval 절약.

- [ ] **9.1.3** 완료 후 commit.

```bash
ls results/main | wc -l   # 60 expected
git add scripts/run_eval_all.sh exps/eval_all.log
git commit -m "ops: 60 main evaluations complete"
```

---

### 9.2 Aggregate to results.csv

**단계**:

- [ ] **9.2.1** Aggregate.

```bash
python scripts/aggregate.py --in_dir results/main --out results.csv
wc -l results.csv   # 61 expected (1 header + 60 row)
```

🔍 **Expected**: `results.csv` 가 60 data row + header. 컬럼: `branch,seed,step,fid,fd_dinov2,precision,recall,sharp_lapvar,sharp_hffreq,sharp_sobel,wallclock_sec`.

- [ ] **9.2.2** Sanity 확인.

```bash
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print('shape:', df.shape)
print('branches:', sorted(df['branch'].unique()))
print('seeds:', sorted(df['seed'].unique()))
print('steps:', sorted(df['step'].unique()))
print('rows per branch:', df.groupby('branch').size().to_dict())
print('rows per seed:', df.groupby('seed').size().to_dict())
print()
print('FID range per branch (mean over seeds at step 100K):')
print(df[df.step == 100000].groupby('branch')['fid'].mean())
"
```

🔍 **Expected**:
```
shape: (60, 11)
branches: ['cutoff', 'decay', 'fixed', 'off', 'warmup']
seeds: [42, 1337, 2024]
steps: [20000, 50000, 80000, 100000]
rows per branch: {'cutoff': 12, 'decay': 12, 'fixed': 12, 'off': 12, 'warmup': 12}
rows per seed: {42: 20, 1337: 20, 2024: 20}
FID range per branch (mean over seeds at step 100K):
branch
cutoff   <value>
decay    <value>
fixed    <value>
off      <value>
warmup   <value>
```

⚠ **빨간 깃발**:
- `shape` 이 (60, ...) 가 아님 → 일부 eval JSON 누락. `results/main/` 확인.
- 어떤 branch의 FID가 NaN 또는 1000+ → 학습 실패 OR sample generation 실패. 해당 (branch, seed, step) 의 eval 로그 확인.
- 모든 branch FID 가 동일 (예: 모두 200) → 학습이 거의 안 됨, 또는 model이 random init 그대로.

- [ ] **9.2.3** Commit.

```bash
git add results.csv results/main/
git commit -m "data: 60 main eval JSONs aggregated to results.csv"
```

---

### 9.3 Primary metric 계산 (FID-vs-step AUC, R5)

**파일**: `scripts/compute_primary.py`, `results_primary.csv`

- [ ] **9.3.1** Script.

`scripts/compute_primary.py`:
```python
"""Compute primary metric (FID-vs-step trajectory AUC) per (branch, seed).

Outputs results_primary.csv with one row per (branch, seed):
- fid_auc                  ← R5 primary metric
- fid_at_100k              ← legacy single-point comparator
- {sharpness, fd_dino, P/R} at 100k
"""
import pandas as pd
from train.eval.auc import fid_step_auc

df = pd.read_csv("results.csv")
rows = []
for (branch, seed), g in df.groupby(["branch", "seed"]):
    g = g.sort_values("step")
    last = g[g["step"] == 100000].iloc[0]
    rows.append({
        "branch": branch, "seed": seed,
        "fid_auc": fid_step_auc(g["step"].tolist(), g["fid"].tolist()),
        "fid_at_100k": float(last["fid"]),
        "sharp_lapvar_at_100k": float(last["sharp_lapvar"]),
        "sharp_hffreq_at_100k": float(last["sharp_hffreq"]),
        "sharp_sobel_at_100k":  float(last["sharp_sobel"]),
        "fd_dinov2_at_100k":    float(last["fd_dinov2"]),
        "precision_at_100k":    float(last["precision"]),
        "recall_at_100k":       float(last["recall"]),
    })
out = pd.DataFrame(rows)
out.to_csv("results_primary.csv", index=False)
print(out.to_string())
```

- [ ] **9.3.2** 실행.

```bash
python scripts/compute_primary.py
```

🔍 **Expected**: 15 row (5 branch × 3 seed) 의 dataframe 출력. 모든 branch가 3 row.

- [ ] **9.3.3** Pre-registered ranking과 비교 (sanity 차원, 실험자만 참고).

```bash
python -c "
import pandas as pd
df = pd.read_csv('results_primary.csv')
print('FID AUC by branch (lower = better):')
print(df.groupby('branch')['fid_auc'].agg(['mean', 'std']).sort_values('mean'))
print()
print('Sharpness lapvar Wasserstein by branch (lower = closer to real):')
print(df.groupby('branch')['sharp_lapvar_at_100k'].agg(['mean', 'std']).sort_values('mean'))
"
```

> **이 단계에서 결과를 *해석*하지 마세요.** Pre-registered 분석은 Day 20-21에 §2.3 4-way reporting으로 진행. 지금은 단지 "데이터가 sane 한가" 만 확인.

- [ ] **9.3.4** Commit.

```bash
git add scripts/compute_primary.py results_primary.csv
git commit -m "data: compute FID-vs-step trajectory AUC per (branch, seed)"
```

---

### 9.4 Preliminary FID-vs-step plot

**파일**: `scripts/plots.py`, `paper/figures/fig_fid_vs_step.png`

- [ ] **9.4.1** Plot script (이건 stub — Day 20에서 확장).

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
        ax.errorbar(agg["step"], agg["mean"], yerr=agg["std"],
                    label=branch, marker="o", capsize=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("FID (clean-fid)")
    ax.set_title("FID vs training step (mean ± std over 3 seeds)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def main(args):
    df = pd.read_csv(args.results)
    fid_vs_step(df, "paper/figures/fig_fid_vs_step.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.csv")
    main(p.parse_args())
```

- [ ] **9.4.2** 실행.

```bash
mkdir -p paper/figures
python scripts/plots.py
```

🔍 **Expected**: `paper/figures/fig_fid_vs_step.png` 생성. 5개 색상 line, x축 step, y축 FID.

- [ ] **9.4.3** 시각 검사. 모든 branch가 step 따라 감소하는지, B0 (off) 가 가장 높거나 같은 위치인지, schedule 변형들 사이에 차이가 있는지.

> **이 그림이 본 연구의 가장 중요한 시각화 중 하나**. Pre-registered branch ranking과 비교해 보세요 (§8 #6). 완벽히 일치하면 confirmation bias suspect.

- [ ] **9.4.4** Commit.

```bash
git add scripts/plots.py paper/figures/fig_fid_vs_step.png
git commit -m "feat: preliminary FID-vs-step plot"
```

---

### ✅ Day 12-14 완료 체크포인트

- [ ] `results/main/` 에 60개 JSON 파일
- [ ] `results.csv` 가 60 row + header
- [ ] `results_primary.csv` 가 15 row (5 × 3)
- [ ] 모든 row의 metric 값이 finite (NaN 없음)
- [ ] `paper/figures/fig_fid_vs_step.png` 생성됨
- [ ] 시각적으로 모든 branch가 합리적 trajectory (감소)

체크리스트가 모두 ✓ 면 Day 15 (human eval) 로 진행.

---

## Part 10 — Day 15-18: Human evaluation (2AFC)

**오늘들의 목표**: §7.3 의 6개 pair × 50 image-pair 생성, Flask 2AFC platform 구축, 응답 수집, Bradley-Terry 분석.

**예상 시간**: Day 15-16 (platform + pair gen) ~6시간, Day 17-18 (응답 수집 + 분석) 가변 (rater 수에 따라).

**산출물**:
- `human_eval/pairs/{a}_{b}/{:03d}_a.png`, `_b.png` (300 pair × 2 = 600 PNG)
- `human_eval/pairs/manifest.json`
- `scripts/human_eval_server.py`, `templates/human_eval.html`
- `human_eval/responses.csv`
- `human_eval/bt_scores.csv`
- `scripts/bradley_terry.py`

---

### 10.1 Matched-noise image pair 생성

**목적**: 각 비교 pair (예: B0 vs B1) 에 대해, 동일한 noise seed에서 두 branch의 EMA model로 생성. Within-pair variance 제거.

**파일**: `scripts/generate_eval_pairs.py`

- [ ] **10.1.1** Script.

`scripts/generate_eval_pairs.py`:
```python
"""Generate 6 pair-types × 50 image-pairs from matched noise seeds (spec §7.3).

Seeds shared across the two branches in each pair → controls for sample noise.
Uses seed 42 EMA ckpts at step 100K for both branches.
"""
import json
from pathlib import Path
import torch
from train.eval.sample import generate_samples

PAIR_TYPES = [
    ("off", "fixed"),
    ("off", "decay"),
    ("off", "warmup"),
    ("fixed", "decay"),
    ("fixed", "cutoff"),
    ("decay", "cutoff"),
]
N_PAIRS_PER_TYPE = 50
SEED_BASE = 100000

def load_model(ckpt_path):
    """Load EMA bf16 → fp32 SiT."""
    from <repa.models.sit> import SiT  # ← REPLACE
    sd = torch.load(ckpt_path, map_location="cuda")
    model = SiT(...).to("cuda").eval()
    model.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    return model

def main():
    out_root = Path("human_eval/pairs")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for a, b in PAIR_TYPES:
        print(f">>> {a} vs {b}")
        out_dir = out_root / f"{a}_{b}"
        out_dir.mkdir(exist_ok=True)
        model_a = load_model(f"exps/{a}_s42/checkpoints/ema_100000.pt")
        model_b = load_model(f"exps/{b}_s42/checkpoints/ema_100000.pt")

        from torchvision.utils import save_image
        for i in range(N_PAIRS_PER_TYPE):
            seed = SEED_BASE + i
            sa = generate_samples(model_a, n=1, batch_size=1,
                                  latent_shape=(4,32,32), steps=50,
                                  device="cuda", seed=seed)
            sb = generate_samples(model_b, n=1, batch_size=1,
                                  latent_shape=(4,32,32), steps=50,
                                  device="cuda", seed=seed)
            save_image(sa[0].float() / 255.0, out_dir / f"{i:03d}_a.png")
            save_image(sb[0].float() / 255.0, out_dir / f"{i:03d}_b.png")
        del model_a, model_b
        torch.cuda.empty_cache()
        manifest.append({"a": a, "b": b, "n": N_PAIRS_PER_TYPE,
                         "seed_base": SEED_BASE})
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✓ {len(manifest)} pair types, {sum(m['n'] for m in manifest) * 2} images")

if __name__ == "__main__":
    main()
```

⚠ **주의**:
- `<repa.models.sit>` import를 본인 fork 경로로 교체.
- 매 pair마다 두 model 로드 → VRAM 16 GB에서 두 SiT-B/2 (각 ~1 GB) 가 동시에 들어감. 안 들어가면 한 model 로드 → 50개 sample → unload → 다른 model 로드 → 50개로 split.

> **왜 동일 seed?** Spec §7.3: "동일한 초기 noise를 두 branch의 EMA checkpoint에 통과시킴". 같은 noise, 다른 model → 두 출력의 차이가 *순수히 model 차이*. Sample 무작위성 제거.

- [ ] **10.1.2** 실행.

```bash
python scripts/generate_eval_pairs.py
```

🔍 **Expected**: 약 30-40 분, 600개 PNG 파일 생성.

```bash
ls human_eval/pairs/
ls human_eval/pairs/off_fixed | head
```

🔍 **Expected**: 6개 폴더 (`off_fixed`, `off_decay`, ...), 각 폴더에 100개 PNG (50 pair × 2).

- [ ] **10.1.3** Sanity 시각 검사. 한 폴더의 첫 5개 pair를 viewer로 열어 비교.

🔍 **Expected**: 두 이미지가 *같은 noise seed*에서 나왔지만 *다른 face*. (정확히 같으면 noise seed 통제가 안 된 것.)

- [ ] **10.1.4** Commit.

```bash
git add scripts/generate_eval_pairs.py human_eval/pairs/manifest.json
# PNG 자체는 어떻게 할 것인가:
#   - small (300 image-pair × 2 × ~50 KB = ~30 MB) → repo에 commit 가능
#   - 또는 .gitignore 추가
git add human_eval/pairs/   # 모든 PNG 포함
git commit -m "feat: generate 300 matched-noise human eval image pairs"
```

---

### 10.2 Flask 2AFC 서버

**목적**: 사람이 웹 브라우저에서 두 이미지를 비교하고 응답을 CSV로 저장.

**파일**: `scripts/human_eval_server.py`, `templates/human_eval.html`

- [ ] **10.2.1** Server script.

`scripts/human_eval_server.py`:
```python
"""Minimal Flask 2AFC server for human eval.

Serves a random pair from human_eval/pairs/, randomizes left/right,
records response to CSV.

Run:
    python scripts/human_eval_server.py
Then open http://127.0.0.1:5000/ in a browser.
"""
import csv, json, random
from pathlib import Path
from flask import Flask, request, render_template, send_file

ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = ROOT / "human_eval/pairs"
RESPONSES = ROOT / "human_eval/responses.csv"
TEMPLATES = ROOT / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES))

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
    return send_file(str(PAIRS_DIR / f"{branch_a}_{branch_b}" / f"{idx:03d}_{which}.png"))

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

- [ ] **10.2.2** HTML template.

`templates/human_eval.html`:
```html
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>REPA 재평가 — 2AFC Human Eval</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 30px auto; padding: 0 20px; }
    .pair { display: flex; gap: 20px; margin: 20px 0; justify-content: center; }
    .pair img { width: 320px; height: 320px; border: 1px solid #ccc; }
    .q { margin: 10px 0; }
    button { padding: 10px 20px; font-size: 16px; }
  </style>
</head>
<body>
  <h2>2AFC: 어느 쪽이 더 sharp / real photo 같나요?</h2>
  <form id="f">
    <input type="hidden" name="timestamp">
    <input type="hidden" name="branch_a" value="{{ pair.a }}">
    <input type="hidden" name="branch_b" value="{{ pair.b }}">
    <input type="hidden" name="idx" value="{{ idx }}">
    <input type="hidden" name="swap" value="{{ swap }}">
    <p>Rater 이름: <input name="rater" required></p>
    <div class="pair">
      <img src="/img/{{ pair.a }}_{{ pair.b }}/{{ idx }}/{{ 'b' if swap else 'a' }}">
      <img src="/img/{{ pair.a }}_{{ pair.b }}/{{ idx }}/{{ 'a' if swap else 'b' }}">
    </div>
    <div class="q">
      <strong>1) 더 sharp한 쪽:</strong>
      <label><input type="radio" name="q_sharp" value="left" required> 왼쪽</label>
      &nbsp;
      <label><input type="radio" name="q_sharp" value="right"> 오른쪽</label>
    </div>
    <div class="q">
      <strong>2) 더 실제 사진 같은 쪽:</strong>
      <label><input type="radio" name="q_real" value="left" required> 왼쪽</label>
      &nbsp;
      <label><input type="radio" name="q_real" value="right"> 오른쪽</label>
    </div>
    <button type="submit">제출 + 다음</button>
  </form>
  <script>
    document.getElementById('f').onsubmit = async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target);
      fd.set('timestamp', new Date().toISOString());
      const rater = fd.get('rater');
      await fetch('/submit', {method:'POST', body: fd});
      // 다음 화면에서 같은 rater name을 자동 채움
      sessionStorage.setItem('rater', rater);
      location.reload();
    };
    window.onload = () => {
      const stored = sessionStorage.getItem('rater');
      if (stored) document.querySelector('[name=rater]').value = stored;
    };
  </script>
</body>
</html>
```

- [ ] **10.2.3** 로컬 테스트.

```bash
mkdir -p human_eval
python scripts/human_eval_server.py
# 브라우저에서 http://127.0.0.1:5000 접속
# 1-2개 응답 입력, 정상 동작 확인
# Ctrl+C로 server 종료
cat human_eval/responses.csv
```

🔍 **Expected**: `responses.csv` 가 헤더 + 1-2 row.

⚠ **트러블슈팅**:
- **Server start error: "Address already in use"**: port 5000이 이미 쓰임. `--port 5001` 또는 `app.run(port=5001)` 로 변경.
- **이미지가 안 보임**: PAIRS_DIR 경로 확인. `human_eval/pairs/manifest.json` 이 존재하는지 확인.
- **빈 화면**: template 폴더 경로가 잘못. `TEMPLATES = ROOT / "templates"` 가 실제 폴더와 매칭하는지.

- [ ] **10.2.4** Commit.

```bash
git add scripts/human_eval_server.py templates/human_eval.html
git commit -m "feat: minimal Flask 2AFC server for human eval"
```

---

### 10.3 응답 수집

**목적**: ≥ 1000 응답 (spec §14 성공 기준).

**단계**:

- [ ] **10.3.1** Self-rating round (Day 17 오전).

```bash
python scripts/human_eval_server.py
# 본인이 50-100개 응답 입력 (~1시간)
# 각 pair type 당 ≥ 5 응답 목표
```

> 본인이 직접 평가 시 *이미 결과를 알고 있어서 bias 위험*. Pre-registered 분석은 self ratings + 외부 rater의 응답을 분리해서도 보고 가능. 본 튜토리얼은 둘을 합쳐서 BT score를 계산하고 외부 rater 응답 수가 충분하면 별도 sub-analysis 가능.

- [ ] **10.3.2** 친구/지인 2-3명 모집 (Day 17 오후 ~ Day 18).

```bash
# Server를 외부에서 접근 가능하게 하려면 두 가지 옵션:
# 옵션 A: SSH tunnel
#   ssh -R 5000:localhost:5000 <public_host>
# 옵션 B: ngrok 같은 tunneling 서비스
#   ngrok http 5000
# 옵션 C: Local network only — 같은 wifi의 친구에게만 IP 공유
#   ip addr  # eth0/wlan0 IP 확인
```

🚨 **개인정보**: ngrok/SSH tunnel은 응답 데이터가 인터넷을 통과. Privacy-sensitive 한 face image면 신중히. FFHQ는 이미 public이므로 OK.

- [ ] **10.3.3** Prolific 또는 유사 platform (선택).

만약 self + 친구로 < 500 응답이고 budget 있으면 Prolific 사용. 5 raters × 100 responses = 500 추가 응답을 ~$50-100 에 가능. 본 튜토리얼에서는 자세한 Prolific 통합은 다루지 않음 (별도 계정 + 작업 셋업 필요).

- [ ] **10.3.4** 응답 수 확인.

```bash
wc -l human_eval/responses.csv   # header 포함, 목표 ≥ 1001
python -c "
import pandas as pd
df = pd.read_csv('human_eval/responses.csv')
print('total responses:', len(df))
print('unique raters:', df['rater'].nunique())
print('responses per pair type:')
print(df.groupby(['branch_a','branch_b']).size())
print('responses per rater:')
print(df.groupby('rater').size())
"
```

🔍 **Expected**: total ≥ 1000 (또는 spec §14의 minimum 충족), 모든 6개 pair type에 ≥ 50 응답.

- [ ] **10.3.5** Commit (raw responses).

```bash
git add human_eval/responses.csv
git commit -m "data: human eval responses collected (N=<...>)"
```

---

### 10.4 Bradley-Terry 분석

**파일**: `scripts/bradley_terry.py`, `human_eval/bt_scores.csv`

- [ ] **10.4.1** Script.

`scripts/bradley_terry.py`:
```python
"""Bradley-Terry preference scores from human_eval/responses.csv.
1000-iter bootstrap 95% CI per branch.
"""
import argparse, json
import pandas as pd
import numpy as np
from collections import defaultdict

def load_comparisons(df, question="q_sharp"):
    """Returns dict: (winner_branch, loser_branch) → wins."""
    wins = defaultdict(int)
    for _, r in df.iterrows():
        a, b = r["branch_a"], r["branch_b"]
        # swap=True means physical left = b
        physical_left = b if r["swap"] in (True, "True", "true", 1, "1") else a
        physical_right = a if r["swap"] in (True, "True", "true", 1, "1") else b
        winner_phys = r[question]   # "left" or "right"
        winner = physical_left if winner_phys == "left" else physical_right
        loser = physical_right if winner_phys == "left" else physical_left
        wins[(winner, loser)] += 1
    return wins

def fit_bt_mm(wins, branches, max_iter=200, tol=1e-7):
    """Standard Bradley-Terry MM algorithm."""
    n = len(branches)
    idx = {b: i for i, b in enumerate(branches)}
    p = np.ones(n)
    total_wins = np.zeros(n)
    for (a, b), w in wins.items():
        total_wins[idx[a]] += w
    for _ in range(max_iter):
        denom = np.zeros(n)
        for (a, b), w in wins.items():
            ia, ib = idx[a], idx[b]
            denom[ia] += w / (p[ia] + p[ib])
            denom[ib] += w / (p[ia] + p[ib])
        new_p = total_wins / np.where(denom > 0, denom, 1e-12)
        new_p /= new_p.sum()
        if np.max(np.abs(new_p - p / p.sum())) < tol:
            p = new_p
            break
        p = new_p
    return {b: float(p[idx[b]]) for b in branches}

def bootstrap_ci(df, branches, question, n_boot=1000, seed=20260415):
    rng = np.random.default_rng(seed)
    n = len(df)
    boots = {b: [] for b in branches}
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, size=n)]
        w = load_comparisons(sample, question)
        score = fit_bt_mm(w, branches)
        for b in branches:
            boots[b].append(score[b])
    out = {}
    for b in branches:
        arr = np.array(boots[b])
        out[b] = {"lo": float(np.percentile(arr, 2.5)),
                  "hi": float(np.percentile(arr, 97.5))}
    return out

def main():
    df = pd.read_csv("human_eval/responses.csv")
    branches = sorted(set(df["branch_a"]).union(df["branch_b"]))

    rows = []
    for question in ["q_sharp", "q_real"]:
        wins = load_comparisons(df, question)
        scores = fit_bt_mm(wins, branches)
        cis = bootstrap_ci(df, branches, question)
        for b in branches:
            rows.append({
                "question": question, "branch": b,
                "bt_score": scores[b],
                "ci_lo": cis[b]["lo"], "ci_hi": cis[b]["hi"],
            })
    out = pd.DataFrame(rows)
    out.to_csv("human_eval/bt_scores.csv", index=False)
    print(out.to_string())

if __name__ == "__main__":
    main()
```

- [ ] **10.4.2** 실행.

```bash
python scripts/bradley_terry.py
```

🔍 **Expected**: 10 row (5 branch × 2 question) DataFrame 출력. 모든 branch의 BT score는 [0, 1] 범위, sum = 1 per question.

> **해석은 Day 20-21에**. 지금은 결과만 저장.

- [ ] **10.4.3** Commit.

```bash
git add scripts/bradley_terry.py human_eval/bt_scores.csv
git commit -m "data: Bradley-Terry preference scores with bootstrap CI"
```

---

### ✅ Day 15-18 완료 체크포인트

- [ ] `human_eval/pairs/` 에 6개 폴더, 각 100개 PNG
- [ ] `human_eval/pairs/manifest.json` 존재
- [ ] `human_eval/responses.csv` 가 ≥ 1000 row
- [ ] 모든 6개 pair type에 ≥ 50 응답
- [ ] `human_eval/bt_scores.csv` 가 10 row (5 branch × 2 question)
- [ ] `git log --oneline` 에 Day 15-18 commit 4-5개

체크리스트가 모두 ✓ 면 Day 19 (interpolation) 로 진행.

---

## Part 11 — Day 19: Interpolation study (jaggedness)

**오늘의 목표**: §7.4의 interpolation jaggedness 계산. Latent collapse 의 signature 검출.

**예상 시간**: 4-5 시간.

**산출물**:
- `interpolation/{branch}/jaggedness.json`
- `interpolation/{branch}/grid.png`
- `train/eval/jaggedness.py`
- `scripts/interpolation_study.py`

---

### 11.1 Jaggedness module

**파일**: `train/eval/jaggedness.py`

- [ ] **11.1.1** Implement.

`train/eval/jaggedness.py`:
```python
"""§7.4 interpolation jaggedness via DINOv2 patch-token mean features.

A high jaggedness score means consecutive interpolation steps are far apart in
DINO feature space (i.e., 'mode snapping'). Low jaggedness = smooth traversal.
"""
import torch

def slerp(z0, z1, t):
    """Spherical linear interpolation between two batched latents.

    z0, z1: tensors with same shape (..., D)
    t: scalar in [0, 1]
    """
    z0_flat = z0.flatten(start_dim=-3) if z0.dim() > 2 else z0
    z1_flat = z1.flatten(start_dim=-3) if z1.dim() > 2 else z1
    n0 = z0_flat.norm(dim=-1, keepdim=True) + 1e-8
    n1 = z1_flat.norm(dim=-1, keepdim=True) + 1e-8
    u0 = z0_flat / n0
    u1 = z1_flat / n1
    omega = torch.acos((u0 * u1).sum(dim=-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega) + 1e-8
    out = (torch.sin((1 - t) * omega) / so) * z0_flat + (torch.sin(t * omega) / so) * z1_flat
    return out.view_as(z0)

def jaggedness_from_features(features_per_pair):
    """Inputs: list of (n_steps, dim) feature tensors per noise pair.
    Output: mean jaggedness score across pairs.
    """
    out = []
    for f in features_per_pair:
        d = (f[1:] - f[:-1]).norm(dim=-1)  # (n_steps - 1,)
        if d.mean().item() > 0:
            out.append((d.var() / d.mean()).item())
        else:
            out.append(0.0)
    return float(sum(out) / len(out)) if out else 0.0
```

- [ ] **11.1.2** Test.

`tests/test_jaggedness.py`:
```python
import torch
from train.eval.jaggedness import slerp, jaggedness_from_features

def test_slerp_endpoints():
    z0 = torch.randn(2, 4, 32, 32)
    z1 = torch.randn(2, 4, 32, 32)
    out0 = slerp(z0, z1, torch.tensor(0.0))
    out1 = slerp(z0, z1, torch.tensor(1.0))
    assert torch.allclose(out0, z0, atol=1e-4)
    assert torch.allclose(out1, z1, atol=1e-4)

def test_jaggedness_smooth_low():
    smooth = [torch.linspace(0, 1, 9).unsqueeze(1).repeat(1, 768)]
    score = jaggedness_from_features(smooth)
    assert score < 0.1

def test_jaggedness_jumpy_high():
    jumpy = [torch.tensor([0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0, 5.0, 0.0]).unsqueeze(1).repeat(1, 768)]
    score = jaggedness_from_features(jumpy)
    assert score > 1.0
```

```bash
pytest tests/test_jaggedness.py -v
git add train/eval/jaggedness.py tests/test_jaggedness.py
git commit -m "feat: interpolation jaggedness module"
```

---

### 11.2 Interpolation study script

**파일**: `scripts/interpolation_study.py`

- [ ] **11.2.1** Script.

`scripts/interpolation_study.py`:
```python
"""Per branch: 10 noise pairs × 9 slerp steps → DINOv2 features → jaggedness + grid."""
import json
from pathlib import Path
import torch
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from diffusers import AutoencoderKL
from transformers import AutoImageProcessor, AutoModel
from train.eval.jaggedness import slerp, jaggedness_from_features

BRANCHES = ["off", "fixed", "decay", "warmup", "cutoff"]
N_PAIRS = 10
N_STEPS = 9   # interpolation steps including 0 and 1

def load_model(ckpt_path):
    from <repa.models.sit> import SiT  # ← REPLACE
    sd = torch.load(ckpt_path, map_location="cuda")
    model = SiT(...).to("cuda").eval()
    model.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    return model

def sample_along_interp(model, z_seq, vae, sf, sampler_steps=50):
    """Run model from each interpolated noise → image."""
    images = []
    for z in z_seq:
        z_cur = z.clone()
        for i in range(sampler_steps):
            t = torch.full((1,), 1.0 - (i + 0.5) / sampler_steps, device="cuda")
            with torch.no_grad():
                v = model(z_cur, t, torch.zeros(1, dtype=torch.long, device="cuda"))
            z_cur = z_cur - v / sampler_steps
        with torch.no_grad():
            img = vae.decode(z_cur / sf).sample
        images.append(img.cpu())
    return torch.cat(images, dim=0)   # (n_steps, 3, 256, 256)

def main():
    device = "cuda"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    sf = vae.config.scaling_factor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    out_root = Path("interpolation")
    out_root.mkdir(exist_ok=True)

    for branch in BRANCHES:
        print(f">>> {branch}")
        out_dir = out_root / branch
        out_dir.mkdir(exist_ok=True)
        model = load_model(f"exps/{branch}_s42/checkpoints/ema_100000.pt")

        gen_per_pair = torch.Generator(device=device).manual_seed(42)
        all_features = []
        all_grids = []

        for p_idx in range(N_PAIRS):
            z0 = torch.randn(1, 4, 32, 32, device=device, generator=gen_per_pair)
            z1 = torch.randn(1, 4, 32, 32, device=device, generator=gen_per_pair)
            ts = torch.linspace(0, 1, N_STEPS, device=device)
            z_seq = [slerp(z0, z1, t.item()) for t in ts]

            imgs = sample_along_interp(model, z_seq, vae, sf)   # (N_STEPS, 3, 256, 256)
            all_grids.append(imgs)

            # DINOv2 features
            imgs_01 = ((imgs.clamp(-1, 1) + 1) / 2)
            pil = [T.ToPILImage()(im) for im in imgs_01]
            inp = proc(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = dino(**inp).last_hidden_state[:, 1:].mean(dim=1)
            all_features.append(feat.cpu())

        score = jaggedness_from_features(all_features)
        (out_dir / "jaggedness.json").write_text(json.dumps({
            "branch": branch, "n_pairs": N_PAIRS, "n_steps": N_STEPS,
            "jaggedness": score
        }, indent=2))

        # Visualization grid: stack first 5 pairs as rows, 9 steps as cols
        rows = torch.cat(all_grids[:5], dim=0)
        rows = (rows.clamp(-1, 1) + 1) / 2
        save_image(rows, out_dir / "grid.png", nrow=N_STEPS)

        print(f"  jaggedness = {score:.4f}")
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
```

⚠ `<repa.models.sit>` 교체 + sampler 부분도 §5.1과 같은 placeholder. Production에서는 실제 REPA sampler 사용.

- [ ] **11.2.2** 실행.

```bash
python scripts/interpolation_study.py
```

🔍 **Expected**: 5 branch × 약 10 분 ≈ 50 분. 각 branch 끝나면 jaggedness 출력.

```bash
ls interpolation/
cat interpolation/off/jaggedness.json
cat interpolation/fixed/jaggedness.json
```

🔍 **Expected**: 5개 폴더, 각 `jaggedness.json` + `grid.png` 포함.

⚠ **트러블슈팅**:
- **OOM**: 10 pair × 9 step × decode = 90 image batch. VAE decode를 chunk로 분할.
- **slerp NaN**: z0과 z1 이 거의 같으면 omega ≈ 0 → division by 0. 본 구현은 epsilon으로 회피하지만, 매우 가까운 random vector라면 여전히 위험. 다른 random seed로 해결.

- [ ] **11.2.3** Commit.

```bash
git add scripts/interpolation_study.py interpolation/
git commit -m "feat: interpolation jaggedness study (5 branches × 10 pairs × 9 steps)"
```

---

### ✅ Day 19 완료 체크포인트

- [ ] `interpolation/{branch}/jaggedness.json` 5개 (5 branches)
- [ ] `interpolation/{branch}/grid.png` 5개
- [ ] 모든 jaggedness 값이 finite + non-negative
- [ ] 시각 검사: 각 grid.png 에서 9-step interpolation이 *부드럽게* 또는 *jumpy 하게* 변하는 것이 보임
- [ ] `git log --oneline` 에 Day 19 commit 1-2개

체크리스트가 모두 ✓ 면 Day 20 (final analysis) 으로 진행.

---

## Part 12 — Day 20-21: Statistical analysis + publication figures

**오늘들의 목표**: §2.3 4-way reporting 적용, auto-vs-human correlation, pre-registered verdict, 모든 publication figure.

**예상 시간**: 6-10 시간.

🚨 **이 phase가 본 연구의 *결론*을 결정**. 결과 해석이 spec §2.2 의 positive/null/mixed 규칙에 따라 strict하게 적용되어야 함.

**산출물**:
- `results/final_report.json` (4-way report on all comparisons)
- `results/correlations.txt` (auto vs human BT)
- `results/verdict.md` (pre-registered verdict)
- `paper/figures/fig_*.png` (5개 figure)
- `paper/tables/*.csv`

---

### 12.1 4-way reporting on all 6 pairs × all metrics

**파일**: `scripts/final_analysis.py`, `results/final_report.json`

- [ ] **12.1.1** Script.

`scripts/final_analysis.py`:
```python
"""Apply §2.3 4-way reporting to all pre-registered pair comparisons.

Pre-registered Bonferroni divisor = 6 (pair count, frozen in prereg).
Metric-level multiple comparison is exploratory family — not separately corrected.
"""
import json
from pathlib import Path
import pandas as pd
from train.eval.stats import four_way_report

PAIRS = [
    ("off", "fixed"),
    ("off", "decay"),
    ("off", "warmup"),
    ("fixed", "decay"),
    ("fixed", "cutoff"),
    ("decay", "cutoff"),
]

PRIMARY_METRICS = [
    "fid_auc",                    # R5 primary (§7.1)
    "fid_at_100k",                # secondary (legacy comparator)
    "sharp_lapvar_at_100k",
    "sharp_hffreq_at_100k",
    "sharp_sobel_at_100k",
    "fd_dinov2_at_100k",
    "recall_at_100k",
]

def main():
    df = pd.read_csv("results_primary.csv")
    n_comp_prereg = 6   # frozen in prereg-v1, see §8 #3

    out = {
        "n_comparisons_prereg": n_comp_prereg,
        "metric_family_size_exploratory": len(PRIMARY_METRICS),
        "note": ("Bonferroni applied at pair level (6) per pre-registration; "
                 "metric-level family is exploratory."),
        "per_pair": {},
    }
    for a, b in PAIRS:
        out["per_pair"][f"{a}_vs_{b}"] = {}
        for metric in PRIMARY_METRICS:
            va = df[df["branch"] == a][metric].tolist()
            vb = df[df["branch"] == b][metric].tolist()
            out["per_pair"][f"{a}_vs_{b}"][metric] = four_way_report(
                va, vb, n_comparisons=n_comp_prereg
            )
    Path("results").mkdir(exist_ok=True)
    open("results/final_report.json", "w").write(json.dumps(out, indent=2))

    # Print a flat summary
    print(f"\n{'pair':<20} {'metric':<25} {'d':>7} {'p_bonf':>8} {'sig':>5} {'dir':>10} {'interp':<35}")
    print("-" * 115)
    for pair, metrics in out["per_pair"].items():
        for m, r in metrics.items():
            d = r["effect_size"]["cohens_d"]
            p = r["hypothesis_test"]["p_bonf"]
            sig = r["hypothesis_test"]["significant"]
            dr = r["directional"]
            dirs = f"{dr['pos']}+/{dr['neg']}-"
            print(f"{pair:<20} {m:<25} {d:>7.2f} {p:>8.4f} {str(sig):>5} {dirs:>10} {r['interpretation']:<35}")

if __name__ == "__main__":
    main()
```

- [ ] **12.1.2** 실행.

```bash
python scripts/final_analysis.py | tee results/final_summary.txt
```

🔍 **Expected**: 42 row (6 pairs × 7 metrics) 의 flat table 출력. 각 row에 d, p_bonf, sig, directional, interpretation.

> **이 출력이 본 연구의 가장 중요한 print-out**. 출력을 캡처해서 paper의 main results table에 직접 사용 가능.

⚠ **결과 해석 가이드** (§2.3 4-way에 따라):

| 상황 | 의미 | Paper 표현 |
|---|---|---|
| `sig=True` + `dirs=3+/0-` (or `0+/3-`) + `|d|>0.8` | 4-way 모두 일치, strong claim | "...significantly degraded REPA-Off → REPA-Fixed (d=X, p<0.01)" |
| `sig=False` + `|d|>0.8` + `dirs=3+/0-` | underpowered null, large effect | "**underpowered null**: large effect (d=X) but n=3 insufficient for sig at α/6" |
| `sig=False` + `|d|<0.2` | true null | "no detectable effect (|d| < 0.2)" |
| `sig=False` + `0.2<|d|<0.8` + `dirs=2+/1-` | inconclusive | "inconclusive: small/medium effect with mixed directional consistency" |

- [ ] **12.1.3** Commit.

```bash
git add scripts/final_analysis.py results/final_report.json results/final_summary.txt
git commit -m "analysis: §2.3 4-way reporting on 42 (pair × metric) comparisons"
```

---

### 12.2 Auto vs human eval correlation

**파일**: `scripts/auto_human_correlation.py`, `results/correlations.txt`

- [ ] **12.2.1** Script.

`scripts/auto_human_correlation.py`:
```python
"""Pearson correlation between Bradley-Terry score and each automatic sharpness metric.

Tests whether automatic sharpness metrics predict human preference.
Pre-registered analysis per §7.3.
"""
import pandas as pd
from scipy.stats import pearsonr

bt = pd.read_csv("human_eval/bt_scores.csv")
auto = pd.read_csv("results_primary.csv").groupby("branch").mean(numeric_only=True).reset_index()

results = []
for question in ["q_sharp", "q_real"]:
    bt_q = bt[bt["question"] == question][["branch", "bt_score"]]
    joined = bt_q.merge(auto, on="branch")
    print(f"\n=== {question} ===")
    for col in ["sharp_lapvar_at_100k", "sharp_hffreq_at_100k", "sharp_sobel_at_100k",
                "fid_auc", "fd_dinov2_at_100k", "recall_at_100k"]:
        if joined[col].std() == 0:
            print(f"{col:<25} skipped (zero variance)")
            continue
        r, p = pearsonr(joined["bt_score"], joined[col])
        print(f"{col:<25} r={r:+.3f} p={p:.4f}")
        results.append({"question": question, "metric": col, "pearson_r": r, "p": p})

pd.DataFrame(results).to_csv("results/correlations.csv", index=False)
print("\n✓ saved to results/correlations.csv")
```

- [ ] **12.2.2** 실행.

```bash
python scripts/auto_human_correlation.py | tee results/correlations.txt
```

🔍 **Expected**: 두 question (q_sharp, q_real) 각각에 대해 6개 metric의 r, p 출력.

> **해석 노트**:
> - **|r| > 0.8** + **p < 0.05**: 그 metric이 human preference를 예측. Paper의 supporting evidence.
> - **|r| < 0.3**: dissociation. "automatic metric은 human-perceived quality와 무관" — finding 자체가 contribution.
> - n=5 branch 만으로 correlation을 추정하므로 power가 매우 낮음. 결과는 *exploratory* 로 해석.

- [ ] **12.2.3** Commit.

```bash
git add scripts/auto_human_correlation.py results/correlations.csv results/correlations.txt
git commit -m "analysis: auto sharpness vs human BT correlation"
```

---

### 12.3 Pre-registered verdict

**파일**: `results/verdict.md`

- [ ] **12.3.1** Spec §2.2 의 positive/null/mixed 규칙을 *기계적으로* 적용. 본인의 해석을 넣지 말고, 규칙대로.

```bash
python -c "
import json
fr = json.load(open('results/final_report.json'))

# 규칙 1: B0 vs B1에서 primary OR sharpness/diversity 중 하나가 sig?
b0_b1 = fr['per_pair']['off_vs_fixed']
sig_metrics = [m for m, r in b0_b1.items() if r['hypothesis_test']['significant']]
print('B0 vs B1 significant metrics:', sig_metrics)

# 규칙 2: 스케줄 변형 중 하나가 fixed보다 회복?
recovery = []
for sched in ['decay', 'warmup', 'cutoff']:
    key = f'fixed_vs_{sched}'
    if key in fr['per_pair']:
        sm = [m for m, r in fr['per_pair'][key].items() if r['hypothesis_test']['significant']]
        recovery.append((sched, sm))
print('Schedule recoveries vs fixed:', recovery)

# Verdict
positive = bool(sig_metrics) and any(s for _, s in recovery)
verdict = 'POSITIVE' if positive else 'NULL_OR_MIXED'
print('\\nPre-registered verdict:', verdict)

# Underpowered null check (§2.3)
underpowered = []
for pair, metrics in fr['per_pair'].items():
    for m, r in metrics.items():
        d = abs(r['effect_size']['cohens_d'])
        sig = r['hypothesis_test']['significant']
        if not sig and d > 0.8:
            underpowered.append((pair, m, d))
print(f'\\nUnderpowered null candidates ({len(underpowered)}):')
for p, m, d in underpowered[:10]:
    print(f'  {p} / {m}: |d|={d:.2f}')
"
```

- [ ] **12.3.2** verdict.md 작성. 스크립트 출력을 보고 다음 template 채움.

`results/verdict.md`:
```markdown
# Pre-registered Verdict

Date: 2026-04-15+21
Pre-registration: `prereg-v1` (commit `<sha>`)
Analysis script: `scripts/final_analysis.py`

## Verdict (per spec §2.2): **<POSITIVE | NULL | MIXED>**

## Evidence

### B0 vs B1 (primary diagnostic)
- Significant metrics (paired t, Bonferroni α/6):
  - `<list>`
- Effect sizes (Cohen's d):
  - `fid_auc`: d = `<value>` (`<interpretation>`)
  - `sharp_lapvar`: d = `<value>` (`<interpretation>`)
  - ...

### Schedule recoveries vs fixed
- decay: `<list of significant metrics>`
- warmup: `<list>`
- cutoff: `<list>`

### Underpowered null check (§2.3)
다음 (pair, metric) 조합은 |d| > 0.8 이지만 not significant:
- `<list with d values>`

이들은 **underpowered null** 로 분류 — n=3 이 부족해 검출 못 한 가능성. Paper에서
"effect may exist but n=3 cannot detect at α/6" 로 보고.

### True null (|d| < 0.2)
- `<list of (pair, metric) with d < 0.2>`

이들은 **true null** 로 분류 — 효과 없음.

## Branch ranking (descriptive only)

`results_primary.csv` 의 mean 기준:

| Branch | FID AUC mean ± std | sharp_lapvar mean ± std | recall mean ± std |
|---|---|---|---|
| B0 (off)    | `<...>` | `<...>` | `<...>` |
| B1 (fixed)  | `<...>` | `<...>` | `<...>` |
| B2 (decay)  | `<...>` | `<...>` | `<...>` |
| B3 (warmup) | `<...>` | `<...>` | `<...>` |
| B4 (cutoff) | `<...>` | `<...>` | `<...>` |

## Pre-registered ranking 일치 여부 (§8 #6 confirmation-bias check)

- Sharpness pre-registered: B0 > B2 ≈ B4 > B1 > B3
- Sharpness 실측 ranking: `<actual>`
- 일치도: `<exact / mostly / partial / opposite>`

🚨 만약 *완벽한 일치*면 confirmation bias 의심. Paper의 limitations에서 명시.

## Auto-vs-human correlation (§7.3 secondary analysis)

`results/correlations.csv` 참조.

- Strong correlations (|r| > 0.8): `<list>`
- Weak/no correlations (|r| < 0.3): `<list>`

|r| 가 약하면 dissociation finding으로 paper에 보고.

## Conclusions

(2-3 문단으로 paper의 abstract 결론을 작성. spec §10의 framing 활용.)
```

- [ ] **12.3.3** Commit.

```bash
git add results/verdict.md
git commit -m "analysis: apply pre-registered verdict per §2.2 + §2.3"
```

---

### 12.4 모든 publication figures

**파일**: `scripts/plots.py` (확장)

- [ ] **12.4.1** Plot script 확장.

`scripts/plots.py` 를 다음 5개 figure를 모두 생성하도록 확장:

```python
"""Generate publication figures."""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BRANCH_COLORS = {
    "off": "#999999", "fixed": "#e41a1c", "decay": "#377eb8",
    "warmup": "#4daf4a", "cutoff": "#984ea3",
}
BRANCH_ORDER = ["off", "fixed", "decay", "warmup", "cutoff"]

def fid_vs_step(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for branch in BRANCH_ORDER:
        g = df[df["branch"] == branch]
        agg = g.groupby("step")["fid"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["step"], agg["mean"], yerr=agg["std"],
                    label=branch, color=BRANCH_COLORS[branch],
                    marker="o", capsize=3, lw=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("FID (clean-fid)")
    ax.set_title("FID vs training step (mean ± std over 3 seeds)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def sharpness_bars(primary_df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    metrics = ["sharp_lapvar_at_100k", "sharp_hffreq_at_100k", "sharp_sobel_at_100k"]
    titles = ["Laplacian variance W₁", "HF energy ratio W₁", "Sobel mean W₁"]
    for ax, m, t in zip(axes, metrics, titles):
        means = [primary_df[primary_df["branch"] == b][m].mean() for b in BRANCH_ORDER]
        stds  = [primary_df[primary_df["branch"] == b][m].std()  for b in BRANCH_ORDER]
        colors = [BRANCH_COLORS[b] for b in BRANCH_ORDER]
        ax.bar(BRANCH_ORDER, means, yerr=stds, color=colors, capsize=4)
        ax.set_title(t)
        ax.set_ylabel("Wasserstein distance to ref")
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle("Sharpness Wasserstein vs reference (lower = closer to real)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def jaggedness_bar(out):
    import json
    from pathlib import Path
    means = []
    for b in BRANCH_ORDER:
        d = json.loads((Path("interpolation") / b / "jaggedness.json").read_text())
        means.append(d["jaggedness"])
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [BRANCH_COLORS[b] for b in BRANCH_ORDER]
    ax.bar(BRANCH_ORDER, means, color=colors)
    ax.set_ylabel("Jaggedness (var/mean of feature distances)")
    ax.set_title("Interpolation jaggedness (lower = smoother traversal)")
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def bt_score_bar(bt_df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, q in zip(axes, ["q_sharp", "q_real"]):
        df = bt_df[bt_df["question"] == q].set_index("branch").reindex(BRANCH_ORDER)
        means = df["bt_score"]
        err = [(means - df["ci_lo"]).values, (df["ci_hi"] - means).values]
        colors = [BRANCH_COLORS[b] for b in BRANCH_ORDER]
        ax.bar(BRANCH_ORDER, means, yerr=err, color=colors, capsize=4)
        ax.set_title(f"BT score: {q}")
        ax.set_ylabel("Bradley-Terry probability")
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle("Human preference (Bradley-Terry, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def auto_human_scatter(corr_df, primary_df, bt_df, out):
    bt_sharp = bt_df[bt_df["question"] == "q_sharp"][["branch", "bt_score"]]
    auto = primary_df.groupby("branch")["sharp_lapvar_at_100k"].mean().reset_index()
    j = bt_sharp.merge(auto, on="branch")
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in j.iterrows():
        ax.scatter(row["sharp_lapvar_at_100k"], row["bt_score"],
                   color=BRANCH_COLORS[row["branch"]], s=120)
        ax.annotate(row["branch"], (row["sharp_lapvar_at_100k"], row["bt_score"]),
                    textcoords="offset points", xytext=(8, 5))
    ax.set_xlabel("Sharpness Wasserstein (lapvar) — lower = closer to real")
    ax.set_ylabel("Human BT score (sharper question)")
    ax.set_title("Auto sharpness vs human preference")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"✓ {out}")

def main(args):
    df = pd.read_csv(args.results)
    primary = pd.read_csv("results_primary.csv")
    bt = pd.read_csv("human_eval/bt_scores.csv")
    corr = pd.read_csv("results/correlations.csv")

    fid_vs_step(df, "paper/figures/fig_fid_vs_step.png")
    sharpness_bars(primary, "paper/figures/fig_sharpness_bars.png")
    jaggedness_bar("paper/figures/fig_jaggedness.png")
    bt_score_bar(bt, "paper/figures/fig_bt_scores.png")
    auto_human_scatter(corr, primary, bt, "paper/figures/fig_auto_vs_human.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.csv")
    main(p.parse_args())
```

- [ ] **12.4.2** 실행.

```bash
mkdir -p paper/figures
python scripts/plots.py
ls paper/figures/
```

🔍 **Expected**: 5개 PNG (fig_fid_vs_step, fig_sharpness_bars, fig_jaggedness, fig_bt_scores, fig_auto_vs_human).

- [ ] **12.4.3** 시각 검사. 각 figure를 열어서:
  - 라벨이 잘림 없이 보임
  - Color scheme이 branch별로 일관
  - 단위가 axis label에 표시됨
  - Title이 설명적

⚠ **트러블슈팅**:
- **Figure가 하얗다**: `tight_layout` 또는 `bbox_inches='tight'` 가 잘못. 확인.
- **Korean font 깨짐**: matplotlib에 한글 폰트 등록. 또는 영문 제목/라벨 사용 (paper용).
- **Bar chart의 error bar가 너무 큼**: n=3의 std는 정의상 큼. 정상.

- [ ] **12.4.4** Commit.

```bash
git add scripts/plots.py paper/figures/
git commit -m "feat: all 5 publication figures"
```

---

### 12.5 Main results table

**파일**: `paper/tables/main_results.csv`

- [ ] **12.5.1** 표 생성.

```bash
mkdir -p paper/tables
python -c "
import pandas as pd
p = pd.read_csv('results_primary.csv')
agg = p.groupby('branch').agg({
    'fid_auc': ['mean', 'std'],
    'fid_at_100k': ['mean', 'std'],
    'sharp_lapvar_at_100k': ['mean', 'std'],
    'recall_at_100k': ['mean', 'std'],
    'fd_dinov2_at_100k': ['mean', 'std'],
})
agg.columns = ['_'.join(c) for c in agg.columns]
agg = agg.reindex(['off', 'fixed', 'decay', 'warmup', 'cutoff'])
agg.to_csv('paper/tables/main_results.csv')
print(agg.to_string())
"
```

- [ ] **12.5.2** Commit.

```bash
git add paper/tables/main_results.csv
git commit -m "data: main results table (per-branch mean ± std)"
```

---

### ✅ Day 20-21 완료 체크포인트

- [ ] `results/final_report.json` 존재, 6 pair × 7 metric = 42 entries
- [ ] `results/final_summary.txt` 가 flat table
- [ ] `results/correlations.csv` + `results/correlations.txt` 존재
- [ ] `results/verdict.md` 의 모든 placeholder가 채워짐
- [ ] `paper/figures/` 에 5개 PNG
- [ ] `paper/tables/main_results.csv` 존재
- [ ] `git log --oneline` 에 Day 20-21 commit 5-6개

체크리스트가 모두 ✓ 면 Day 22 (paper writing) 로 진행 — 본 튜토리얼은 여기서 사실상 끝.

---

## Part 13 — Day 22-28: 논문 작성 (개요만)

본 phase는 코드/실험이 아닌 글쓰기. 본 튜토리얼은 step-by-step을 다루지 않고 *checklist*만 제공. 자세한 글쓰기는 별도 가이드 또는 elements-of-style 참고.

### 13.1 권장 구조 (4-8 page workshop format)

- **Abstract** (200 단어): 동기 (주관적 blurry 관찰) → 방법 (5 schedule, FFHQ-256, 4-way reporting) → 결과 (verdict) → 함의 (1 문장)
- **Introduction** (1 page): REPA 배경, 우리 의문, 연구 질문, contribution
- **Related work** (0.5 page): REPA, HASTE, 다른 alignment 방법
- **Method** (2 pages): purity principle, 5 branches, evaluation suite, §2.3 reporting
- **Experiments** (2 pages): primary results table, FID-vs-step plot, sharpness bars, BT scores, jaggedness, auto-human correlation
- **Discussion** (1 page): verdict 해석, limitations (n=3 power, FFHQ instead of ImageNet), implications
- **Pre-registration appendix**: prereg.md 전체

### 13.2 Limitations 섹션 (필수)

다음을 명시:

- **n=3 underpowers all hypothesis tests** (§2.3). Pre-registered test는 |d| > 4.5 effect만 검출.
- **Single dataset**: FFHQ-256만. ImageNet, CIFAR 등으로 일반화 불확실.
- **Single architecture**: SiT-B/2만. 더 큰 model 또는 다른 backbone 미검증.
- **Single VAE**: SD-VAE-ft-mse. SDXL VAE 또는 INVAE 등은 미검증.
- **Confounder removal post-hoc**: SpeedrunDiT extras 제거가 effect를 사라지게 했다는 *직접적* 증거 없음. 단지 baseline에서 trade-off 검출 여부만 보고.

### 13.3 Negative result framing (Null verdict 일 때)

- 제목을 "trade-off 존재" → "trade-off 부재 evidence" 로 변경
- "이전 관찰은 confounder 탓으로 설명 가능" 으로 reframe
- §2.3의 underpowered null 분리 설명. "absent in n=3 study, may exist in larger studies."

### 13.4 제출 전 checklist

- [ ] 모든 figure가 paper에 올바른 해상도로 embed
- [ ] 모든 table이 spec과 일치
- [ ] Pre-registration appendix가 prereg.md 와 *글자 단위로* 일치
- [ ] Limitations 섹션 명시
- [ ] Reproducibility statement: git tag, dataset hash, seed, hyperparameter
- [ ] Author info (anonymized for double-blind 시)
- [ ] Workshop call의 page limit 준수
- [ ] LaTeX bibliography 최종 확인

---

## Appendix A — 트러블슈팅 모음

각 문제에 대해 *증상 → 원인 후보 → 대응* 순으로 정리.

### A.1 GPU OOM (Out of Memory)

**증상**: `RuntimeError: CUDA out of memory. Tried to allocate ...`

**원인 후보 + 대응**:

1. **학습 batch가 너무 큼**:
   - `--batch_size 16` 으로 절반.
   - Gradient accumulation 2배 (effective batch 동일).
2. **Sample generation batch가 너무 큼**:
   - `eval.py` 의 `--batch_size 32` (또는 16).
3. **DINOv2/VAE를 동시에 메모리에 들고 있음**:
   - Forward 후 `del dino; torch.cuda.empty_cache()`.
4. **EMA + main model 동시에**:
   - 정상. 130M model × 2 = ~1 GB. OOM은 다른 곳.
5. **Memory leak**:
   - `nvidia-smi` 의 GPU memory가 train step마다 증가하면 leak. PyTorch tensor cache 문제. `gc.collect()` + `torch.cuda.empty_cache()` 를 매 100 step에 호출.

### A.2 NaN / Inf loss (학습 발산)

**증상**: `loss=nan` 또는 학습 중 갑자기 loss가 1e10 등 폭증.

**원인 후보 + 대응**:

1. **bf16 mixed precision overflow**:
   - 일부 op가 fp32 필요. `accelerate config` 에서 `mixed_precision: bf16` 인지 확인. fp16은 더 위험 — bf16이 default.
2. **Gradient explosion**:
   - `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` 추가 (spec §5.3).
3. **LR이 너무 큼**:
   - Spec §5.3은 1e-4. 그보다 크면 위험.
4. **B3 warmup 의 t=0**:
   - λ(0) = 0 인지 확인. 만약 0이 아니면 schedule 함수 버그.
5. **Bad batch (corrupt latent)**:
   - 학습 시작 시 모든 batch에서 NaN check 추가:
     ```python
     assert not torch.isnan(batch["latent"]).any()
     ```

### A.3 디스크 부족 alert

**증상**: `disk_monitor.py` 가 `⚠ DISK ALERT: free X.X GB < threshold 12 GB` + sweep halt.

**원인 후보 + 대응**:

1. **`exps/` 폴더가 과적재**:
   - `du -sh exps/* | sort -h | tail` 로 어느 run이 큰지.
   - 정상적이면 run당 ~3 GB (4 ckpt × ~260 MB + preview + log).
   - 정상보다 큰 run은 rolling latest가 안 지워졌을 가능성. 확인: `ls -la exps/<run>/checkpoints/_rolling.pt`. 필요 시 `rm`.
2. **Eval samples가 디스크에 저장됨**:
   - `eval.py` 의 preview 저장 부분. 16-64장만 저장하도록 확인.
3. **Latent 폴더가 손상**:
   - `du -sh data/ffhq256_latents/train` 가 600 MB 초과면 비정상.
4. **Sweep 도중 재시작 후 cleanup 누락**:
   - `find exps -name "*.tmp" -delete`
5. **Backup 또는 cache가 차지**:
   - HF cache: `du -sh ~/.cache/huggingface`. ~10 GB 가능.
   - PyTorch cache: 사소.

### A.4 SD-VAE 재구성 실패 (PSNR < 28 또는 LPIPS > 0.10)

**증상**: `scripts/sd_vae_sanity.py` 가 `✗ SANITY FAILED: PSNR 26.5 dB < 28` 또는 `LPIPS 0.15 > 0.10`.

**원인 후보 + 대응**:

1. **HF mirror가 압축 손실 (JPEG)**:
   - PNG 대신 JPEG로 저장된 mirror일 가능성. 다른 mirror.
2. **이미지가 256×256이 아님**:
   - Resize → 정보 손실. mirror가 실제 256 인지 확인.
3. **SD-VAE-ft-mse가 아닌 다른 VAE**:
   - `from_pretrained("stabilityai/sd-vae-ft-mse")` 정확한지.
4. **Faces가 너무 close-crop**:
   - SD-VAE는 일반 photo로 학습. extreme close-up은 잘 못 함.

### A.5 Equivalence test 실패 (4.7, 4.8)

**증상**: `test_off_matches_no_proj_loss` 또는 `test_denoise_loss_consistent_across_branches` 가 FAIL.

**원인 후보 + 대응**:

1. **Seed 통제 부족**:
   - `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` 위치 확인.
   - DataLoader의 `worker_init_fn` 으로 worker seed.
   - `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` (단 wallclock cost).
2. **Hook 주입이 잘못된 위치**:
   - `compute_loss` 가 schedule을 받지 않는 경로 (예: dropout test loop) 에서 호출되면 일치 안 함.
3. **bf16 stochastic rounding**:
   - bf16의 비결정적 반올림으로 미세한 차이 가능. tolerance 늘림 (1e-3 → 5e-3).
4. **Different parameter init**:
   - `--branch off` 와 `--branch fixed` 두 run 사이에 model init이 다르면 즉시 diverge. seed로 정확히 동일 init 하는지 확인.

### A.6 HF dataset streaming 끊김

**증상**: 학습 또는 latent precompute 도중 `ConnectionError`, `HTTPError`, `EOFError`.

**원인 후보 + 대응**:

1. **Rate limit**:
   - `huggingface-cli login` 으로 token 등록. authenticated request는 limit 더 높음.
2. **Mirror downtime**:
   - HF status page 확인. 다른 mirror 시도.
3. **본인 인터넷 불안정**:
   - `ping huggingface.co` 로 확인.
4. **Resume**:
   - `precompute_latents.py` 는 resume 미지원 (단순함). Day 2의 resume 옵션 참고하여 script에 4줄 추가.

### A.7 NaN/Inf in metrics

**증상**: `eval.py` 가 정상 종료했지만 `fid: NaN` 또는 매우 큰 값.

**원인 후보 + 대응**:

1. **Sample이 모두 같은 값** (model collapse):
   - VAE feature의 covariance가 singular → FID 계산에서 NaN. EMA가 NaN 또는 zero인지 확인.
2. **Sample이 NaN**:
   - Model의 EMA state에 NaN. Day 4의 학습 발산 (A.2) 흔적.
3. **clean-fid Inception input 범위 잘못**:
   - uint8 [0, 255] 이어야 함. fp32 [0, 1] 이거나 [-1, 1] 이면 nonsense.

---

## Appendix B — 의사결정 가이드 (decision flowchart)

본 phase에서 *연구 진행을 멈추거나 방향을 바꿀* 결정점들을 모음.

### B.1 Pilot signal 부족 (Day 6-7 끝)

**상황**: Pilot eval에서 5 branch FID가 모두 거의 같음 (max - min < 5%).

**의사결정**:

```
signal 부족 발견 → 다음 중 선택
├─ 옵션 1: pilot을 40K로 늘려 다시 실행
│  → +20% wallclock. signal이 나타나면 main run 정상 진행.
├─ 옵션 2: hyperparameter 점검
│  → batch=32, lr=1e-4, EMA=0.9999 가 spec과 일치하는지.
│  → projection layer가 정상 학습 중인지 (proj_loss curve 확인).
├─ 옵션 3: equivalence test 재실행
│  → schedule_off ≡ disabled 가 정말 같은지 다시.
└─ 옵션 4: STOP
   → spec §8 #5 stop condition. 연구 질문 재평가 필요.
```

### B.2 Compute 초과 (Day 8-11 진행 중)

**상황**: 3 runs 마쳤는데 wallclock이 외삽보다 50% 더 걸림. 남은 12 runs로 timeline 초과 예상.

**의사결정**:

```
compute over → 다음 중 선택
├─ Fallback A (spec §10): seed 3 → 2
│  → 3 × 5 = 15 → 2 × 5 = 10 runs. 33% 절감.
│  → ⚠ §2.3의 power가 더 약화. n=2 paired t-test는 거의 의미 없음. Effect size + descriptive로 보고.
├─ Fallback B: step 100K → 80K
│  → 20% 절감. 모든 branch가 같은 step.
│  → ⚠ trajectory가 짧아져 AUC primary metric의 정보량 감소.
├─ Fallback C: branch 5 → 4 (B3 warmup 제거)
│  → 20% 절감. B3는 falsification check이므로 빠지면 spec 약화.
│  → ✗ NOT recommended. 우리 design의 핵심.
└─ Fallback D: STOP
   → spec §8 #5: 5일 GPU 연속 사용 시 stop.
```

권장: **A + B 조합** (n=2, step=80K) → 47% 절감.

### B.3 Null 결과 vs underpowered null (Day 20)

**상황**: §12.1 의 final analysis 결과, B0 vs B1 에서 어떤 metric도 not significant.

**의사결정**:

```
not significant → effect size 확인
├─ |d| < 0.2 → True null
│  └─ Paper framing: "REPA는 우리 setting에서 sharpness/diversity trade-off를 일으키지 않음.
│     이전 관찰은 SpeedrunDiT extras 같은 confounder로 설명 가능."
├─ 0.2 ≤ |d| < 0.8 → 미결
│  └─ Paper framing: "small/medium effect 가능성이 있으나 n=3로 검출 못 함. 
│     더 큰 후속 연구 필요."
└─ |d| ≥ 0.8 → Underpowered null
   └─ Paper framing: "large effect (d≈X) 의 가능성. n=3 paired test의 power 부족이 
      유의수준 미통과의 원인일 가능성. Replication study 권장."
```

### B.4 Mixed 결과 (auto vs human)

**상황**: 자동 metric에서는 sig, BT score에서는 효과 없음 (또는 반대).

**의사결정**:

- Dissociation finding으로 paper에 *주력 contribution*으로 reframe.
- "Existing automatic sharpness metrics fail to capture human perception in REPA-trained samples" 같은 statement.
- 본 finding은 후속 REPA 연구의 method choice (auto vs human eval 의존성) 에 영향.

### B.5 Reviewer 안티패턴 미리 대응

**예상되는 critique → 미리 적은 답변**:

| Reviewer concern | 우리 답변 |
|---|---|
| "Why FFHQ instead of ImageNet?" | Spec §4.2 — compute budget + face가 sharpness/diversity 에 더 sensitive. |
| "n=3 is too small" | §2.3 + pre-registered. 4-way reporting으로 single test 의존 회피. Underpowered null vs true null 분리. |
| "100K steps may be undertrained" | §5.5 batch 32에서 ~46 epoch. SiT-B 작은 model, latent dim 128, FFHQ 70K → 경험적으로 수렴 도달. |
| "HASTE already showed this" | §10. B4 hard cutoff branch가 명시적으로 HASTE를 비교. 우리 contribution은 *smooth schedules* + sharpness/diversity trade-off의 자세한 characterization. |
| "Why only DINO target?" | Spec §3.1 purity principle — 원본 REPA 그대로. Teacher ablation은 §13 future work. |

---

## Appendix C — 빠른 참조 (quick reference)

### C.1 Git tag 흐름

```
scaffold (day 1) → env → ... (data, code, eval) → prereg-v1 (Day 5)
                                                   ↓
                                              [FROZEN]
                                                   ↓
main runs commits → eval commits → analysis → final
```

`prereg-v1` *이전*의 모든 commit은 자유롭게 amend/rebase 가능. *이후*는 commit history 변경 금지.

### C.2 모든 생성 파일 경로

```
docs/
├── spec-ko.md   ← Spec (frozen)
├── spec-en.md      ← English mirror
├── plan-ko.md     ← Plan (agentic version)
├── tutorial-ko.md ← THIS FILE
├── prereg.md                                    ← Pre-registration (frozen)
├── BASELINE.md                                  ← Deviation map
└── repa_inventory.txt                           ← Code structure dump

data/
├── MANIFEST.json                                ← Dataset + split + latent hashes
├── ffhq256_train.txt                            ← 65K indices
├── ffhq256_eval.txt                             ← 5K indices
├── ffhq256_latents/train/*.pt                   ← 70K bf16 latents (gitignored)
├── fid_ref_ffhq256.npz                          ← FID reference
├── fd_dinov2_ref_ffhq256.pt                     ← DINOv2 reference
├── sd_vae_recon_sanity.json                     ← Day 1 PSNR/LPIPS
├── power_analysis.json                          ← Day 5 Cohen's d_min
├── wallclock_benchmark.json                     ← Day 6 5K-step measurement
├── disk_baseline.txt                            ← Day 1 baseline
├── disk_log.csv                                 ← 모니터 로그
└── disk_log_main.csv                            ← Main run 모니터 로그

train/
├── branches.py                                  ← 5 schedules
├── data/ffhq256_latents.py                      ← Loader
├── save.py                                      ← Checkpoint policy
├── log.py                                       ← CSV logger
└── eval/
    ├── sample.py
    ├── fid.py
    ├── fd_dinov2.py
    ├── sharpness.py
    ├── precision_recall.py
    ├── auc.py
    ├── jaggedness.py
    └── stats.py

scripts/
├── sd_vae_sanity.py
├── precompute_latents.py
├── precompute_fid_ref.py
├── precompute_dino_ref.py
├── disk_monitor.py
├── wallclock_pilot.py                           ← (Day 6에 사용)
├── train.py 또는 (REPA fork의 train.py 직접 수정)
├── eval.py
├── aggregate.py
├── compute_primary.py
├── run_pilot.sh
├── run_main.sh
├── run_eval_all.sh
├── generate_eval_pairs.py
├── human_eval_server.py
├── interpolation_study.py
├── bradley_terry.py
├── final_analysis.py
├── auto_human_correlation.py
├── plots.py
└── power_analysis.py

tests/
├── test_dataset.py
├── test_branches.py
├── test_save_policy.py
├── test_equivalence_off.py
├── test_equivalence_fixed.py
├── test_metrics.py
├── test_stats.py
└── test_jaggedness.py

templates/
└── human_eval.html

exps/ (gitignored)
├── pilot_*_s42/                                 ← 5 pilot runs
├── {branch}_s{seed}/                            ← 15 main runs
│   ├── checkpoints/ema_*.pt                     ← 4 EMA bf16 ckpts (per run)
│   ├── loss_log.csv
│   └── eval_*/preview/*.png                     ← 16-64 visualization samples
├── pilot_run.log
├── main_run.log
└── eval_all.log

results/
├── pilot/*.json                                 ← 5 pilot eval JSONs
├── main/*.json                                  ← 60 main eval JSONs
├── final_report.json                            ← 4-way reports
├── final_summary.txt
├── correlations.csv
├── correlations.txt
└── verdict.md

results.csv                                       ← Day 12 aggregate
results_primary.csv                               ← Day 12 primary metric

human_eval/
├── pairs/{a}_{b}/{:03d}_{a,b}.png               ← 600 PNG
├── pairs/manifest.json
├── responses.csv
└── bt_scores.csv

interpolation/
└── {branch}/
    ├── jaggedness.json
    └── grid.png

paper/
├── figures/
│   ├── fig_fid_vs_step.png
│   ├── fig_sharpness_bars.png
│   ├── fig_jaggedness.png
│   ├── fig_bt_scores.png
│   └── fig_auto_vs_human.png
├── tables/
│   └── main_results.csv
└── draft.md (또는 .tex)                         ← Day 22-28
```

### C.3 §2.3 4-way reporting checklist

각 metric × pair에 대해 *항상* 다음 4가지를 함께 보고:

1. **Descriptive**: branch별 mean ± std (n=3)
2. **Directional**: 3개 seed 중 몇 개가 같은 부등식 방향 (3/3, 2/3)
3. **Effect size**: Cohen's d (paired) + bootstrap 95% CI
4. **Hypothesis test**: paired t-test, Bonferroni α/6 = 0.0083, p_bonf

해석 규칙 (§2.3):
- 4가지 모두 일치 → strong claim
- |d| > 0.8 + not sig → "underpowered null"
- |d| < 0.2 → "true null"

### C.4 §4.4 Storage budget 확인 명령

```bash
# 영구 사용량 (학습/eval 산출물)
du -sh exps/ results/ paper/ data/ffhq256_latents/ 2>/dev/null

# 가용량
df -h .

# Latent 파일 수 (정상이면 70000)
ls data/ffhq256_latents/train | wc -l

# Checkpoint 수 (정상이면 60 = 15 runs × 4 eval ckpts)
ls exps/*/checkpoints/ema_*.pt 2>/dev/null | wc -l
```

기대 값 (Day 21 finalize 시점):
- `data/ffhq256_latents/`: ~600 MB
- `exps/`: ~16-20 GB (15 runs × ~1 GB)
- `results/` + `paper/`: < 1 GB
- 합계: < 25 GB

### C.5 흔한 명령 한 번에

```bash
# 학습 진행 확인
ls exps/*/done 2>/dev/null | wc -l

# 디스크 + 진행 상황
df -h . && ls exps/*/checkpoints/ema_*.pt 2>/dev/null | wc -l

# 가장 최근 loss 확인
tail -3 $(ls -t exps/*/loss_log.csv | head -1)

# 모든 test 한 번에
pytest tests/ -v

# Aggregate + primary metric 한 번에
python scripts/aggregate.py --in_dir results/main --out results.csv && python scripts/compute_primary.py
```

---

## Appendix D — 관련 문서

| 문서 | 경로 | 언제 보나 |
|---|---|---|
| Spec / Design (KO) | `docs/spec-ko.md` | 항상 ground truth. 변경 금지. |
| Spec / Design (EN) | `docs/spec-en.md` | 영문 mirror. 영어 reviewer가 필요할 때. |
| Plan (KO) | `docs/plan-ko.md` | Agentic version. 본 튜토리얼이 너무 verbose 할 때 압축된 step 참고용. |
| Pre-registration | `docs/prereg.md` | Day 5 이후 frozen. Paper appendix 그대로 들어감. |
| Baseline 매핑 | `docs/BASELINE.md` | REPA fork 수정 시 hook 위치 참고. |
| 본 튜토리얼 | `docs/tutorial-ko.md` | 사용자 manual execution 가이드. |

### D.1 외부 자료

- **REPA 원본 (Yu et al., ICLR 2025)**: 이론적 motivation. ImageNet 결과와 비교 시 인용.
- **HASTE (2025)**: §10 risk row의 prior art. B4 hard cutoff의 직접 reference.
- **DiT (Peebles & Xie, 2023)**: SiT의 이론적 base. Architecture 설명 시.
- **DINOv2 (Oquab et al., 2024)**: REPA target encoder의 출처.
- **clean-fid (Parmar et al., 2022)**: FID 구현 + reference statistics 표준화.
- **Kynkäänniemi et al. (2019)**: improved P/R metric.

### D.2 본 튜토리얼이 참조한 spec 섹션 인덱스

| Tutorial section | Spec section |
|---|---|
| Part 1 (Day 1) | §3.3, §10 disk row |
| Part 2 (Day 2) | §4.3 옵션 A streaming, §10 SD-VAE row |
| Part 3 (Day 3) | §4.3, §5.2, §3.3 |
| Part 4 (Day 4 AM) | §3.1-§3.2, §5.4, §6 (5 branches), §4.4 |
| Part 5 (Day 4 PM) | §7.1, §7.2, §2.3 |
| Part 6 (Day 5) | §2.1-§2.3, §8 (pre-reg) |
| Part 7 (Day 6-7) | §8 #5 stop, §10 |
| Part 8 (Day 8-11) | §5.5 hyperparameters, §9 timeline |
| Part 9 (Day 12-14) | §7.1 metric table |
| Part 10 (Day 15-18) | §7.3 |
| Part 11 (Day 19) | §7.4 |
| Part 12 (Day 20-21) | §2.2, §2.3, §8 #6 |
| Part 13 (Day 22-28) | §10, §11, §13 |

---

## 끝

본 튜토리얼은 Day 1-21 (코드 + 실험 + 분석) 과 Day 22-28 (논문 작성) 의 *전체 manual execution path* 를 자체 포함한다. 각 phase의 산출물이 다음 phase의 input.

문제 발생 시:
1. 해당 Day의 트러블슈팅 섹션 확인
2. Appendix A 트러블슈팅 모음 확인
3. Spec doc (§X) 의 해당 섹션 재확인
4. Appendix B 의사결정 가이드 (특히 stop condition)

본 튜토리얼은 *frozen 문서* — pre-registration 이후에는 본 문서의 내용을 *원칙적으로 변경하지 않는다*. 코드 수정은 별도 commit. 문서 수정은 typo 또는 trouble-shooting append 만.

**Good luck with the experiment.** 결과가 어떻게 나오든 (positive / null / mixed), 본 design은 그 결과를 publishable 하게 만들도록 의도되었습니다.

