"""SD-VAE FFHQ-256 reconstruction sanity check.

Streams N FFHQ-256 images from HF, encodes/decodes via stabilityai/sd-vae-ft-mse,
reports PSNR + LPIPS. Hard-fails if PSNR < 30 dB or LPIPS > 0.10.

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

PSNR_THRESHOLD = 30.0
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
    p.add_argument("--hf_dataset", required=True, help="HF dataset id discovered in Day 2 §2.1")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=64)
    main(p.parse_args())
