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
    i = -1
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
            print(f"[{i+1}] rate={rate:.1f} img/s elapsed={dt/60:.1f} min", flush=True)
    print(f"✓ saved {i+1} images to {out_dir}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dataset", default="merkol/ffhq-256")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="data/ffhq256-prelim/images")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    main(p.parse_args())
