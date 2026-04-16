"""Compute clean-fid Inception statistics for FFHQ-256 eval split.

Loads 5K eval latents (REPA format), decodes via SD-VAE (mean only),
computes Inception features, writes mu/sigma.
Output: data/fid_ref_ffhq256.npz with keys 'mu' (2048,) and 'sigma' (2048,2048).
"""
import argparse
import numpy as np
import torch
from diffusers import AutoencoderKL
from cleanfid.features import build_feature_extractor

LATENTS_SCALE = 0.18215

def main(args):
    device = "cuda"
    eval_idx = open(args.eval_list).read().splitlines()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    extractor = build_feature_extractor("clean", device)

    feats = []
    batch = []
    BS = args.batch_size
    for i, idx in enumerate(eval_idx):
        m = np.load(f"{args.latents_dir}/{idx}.npy")
        mean = torch.from_numpy(m)[0, :4].float()
        batch.append(mean)
        if len(batch) == BS or i == len(eval_idx) - 1:
            zs = torch.stack(batch).to(device)
            with torch.no_grad():
                imgs = vae.decode(zs / LATENTS_SCALE).sample
                imgs_01 = (imgs.clamp(-1, 1) + 1) / 2.0  # [0, 1] float
                imgs_299 = torch.nn.functional.interpolate(
                    imgs_01, size=(299, 299), mode='bilinear', align_corners=False)
                f = extractor(imgs_299)
            feats.append(f.cpu().numpy())
            batch = []
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(eval_idx)}]", flush=True)

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
