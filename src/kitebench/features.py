"""Precompute frozen VLM features so head-only training fits in 8 GB of VRAM.

Storing ``vlm_features`` / ``aux_visual_inputs`` for every sample costs ~3.5 GB on
disk (fp16) but lets training skip the 3.5 GB Florence2 backbone entirely.

Usage::

    python -m kitebench.features --episodes out/episodes --model <weights> --out out/feats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kitebench.train import load_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="precompute VLM features")
    parser.add_argument("--episodes", type=Path, default=Path("out/episodes"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/feats"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor

    from kitebench.train import collate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_samples(args.episodes, args.limit)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval().to(device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    for p in model.vlm.parameters():
        p.requires_grad_(False)

    args.out.mkdir(parents=True, exist_ok=True)
    feats = None
    aux = None
    for start in range(0, len(samples), args.batch):
        batch = samples[start : start + args.batch]
        input_ids, image_input, image_mask, _, _ = collate(processor, torch, batch, device)
        with torch.no_grad():
            enc = model.forward_vlm(input_ids, image_input, image_mask)
        v = enc["vlm_features"].to(torch.float16).cpu().numpy()
        a = enc["aux_visual_inputs"].to(torch.float16).cpu().numpy()
        if feats is None:
            feats = np.lib.format.open_memmap(
                args.out / "vlm_features.npy", mode="w+", dtype=np.float16,
                shape=(len(samples), *v.shape[1:]),
            )
            aux = np.lib.format.open_memmap(
                args.out / "aux_visual.npy", mode="w+", dtype=np.float16,
                shape=(len(samples), *a.shape[1:]),
            )
        feats[start : start + len(batch)] = v
        aux[start : start + len(batch)] = a
        if (start // args.batch) % 50 == 0:
            print(f"{start + len(batch)}/{len(samples)}", flush=True)

    feats.flush()
    aux.flush()
    (args.out / "index.json").write_text(json.dumps({
        "count": len(samples),
        "vlm_shape": list(feats.shape[1:]),
        "aux_shape": list(aux.shape[1:]),
        "samples": [
            {"image": str(s["image"]), "instruction": s["instruction"]} for s in samples
        ],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.out} for {len(samples)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
