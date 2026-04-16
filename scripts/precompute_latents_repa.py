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