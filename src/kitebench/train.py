"""Head-only fine-tuning for X-VLA on collected episodes.

The Florence2 backbone is frozen (it is pinned to fp32 inside XVLA), so only the
soft-prompted transformer and the action space heads are trained. Training
replicates ``XVLA.forward`` with the encoder features computed under ``no_grad``.

Usage::

    python -m kitebench.train --episodes out/episodes --model <weights> \\
        --epochs 5 --batch 4 --out out/head.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

HORIZON = 30
DIM_ACTION = 20


def load_samples(episodes: Path, limit: int | None = None) -> list[dict]:
    samples: list[dict] = []
    for ep in sorted(p for p in episodes.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        actions = np.load(ep / "action.npy")  # [T, 10]: xyz, rot6d, grip
        for img_path in sorted((ep / "obs").glob("*_front.png")):
            step = int(img_path.name.split("_")[0])
            if step >= len(actions):
                continue
            samples.append({
                "image": img_path,
                "proprio": float_to_proprio(actions[step]),
                "action": build_chunk(actions, step),
                "instruction": meta["instruction"],
            })
        if limit and len(samples) >= limit:
            return samples[:limit]
    return samples


def float_to_proprio(row: np.ndarray) -> np.ndarray:
    """Official WidowX client layout: [ee_pos(3), 1,0,0,1,0,0,0] then zeros."""
    proprio = np.zeros(DIM_ACTION, dtype=np.float32)
    proprio[0:3] = row[0:3]
    proprio[3:10] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return proprio


def build_chunk(actions: np.ndarray, step: int) -> np.ndarray:
    """Future HORIZON actions as an ee6d chunk; pad the tail with the last one."""
    chunk = np.zeros((HORIZON, DIM_ACTION), dtype=np.float32)
    future = actions[step : step + HORIZON]
    chunk[: len(future), 0:10] = future
    if len(future) < HORIZON:
        chunk[len(future) :, 0:10] = future[-1]
    return chunk


def load_feature_store(torch, path: Path, indices: list[int], device: str):
    feats = np.load(path / "vlm_features.npy", mmap_mode="r")
    aux = np.load(path / "aux_visual.npy", mmap_mode="r")
    v = torch.as_tensor(np.asarray(feats[indices]), dtype=torch.float32, device=device)
    a = torch.as_tensor(np.asarray(aux[indices]), dtype=torch.float32, device=device)
    return {"vlm_features": v, "aux_visual_inputs": a}


def collate(processor, torch, samples: list[dict], device: str):
    images = [[Image.open(s["image"]).convert("RGB")] for s in samples]
    instr = [s["instruction"] for s in samples]
    inputs = processor(images=images, language_instruction=instr)
    proprio = torch.as_tensor(
        np.stack([s["proprio"] for s in samples]), dtype=torch.float32, device=device
    )
    action = torch.as_tensor(
        np.stack([s["action"] for s in samples]), dtype=torch.float32, device=device
    )
    return (
        inputs["input_ids"].to(device),
        inputs["image_input"].to(device),
        inputs["image_mask"].to(device),
        proprio,
        action,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="head-only finetuning for X-VLA")
    parser.add_argument("--episodes", type=Path, default=Path("out/episodes"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("out/head.pt"))
    parser.add_argument("--log", type=Path, default=Path("out/train_log.json"))
    parser.add_argument("--train-blocks", type=int, default=0,
                        help="train only the last N transformer blocks (0 = all)")
    parser.add_argument("--features", type=Path, default=None,
                        help="precomputed VLM features; skips the backbone at train time")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")  # tf32 for the fp32 head
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.features:
        # keep the 3.5 GB backbone on the host; only the head goes to the GPU
        model.transformer.to(device)
    else:
        model.to(device)

    for p in model.vlm.parameters():
        p.requires_grad_(False)
    model.vlm.eval()
    model.transformer.train()
    if args.train_blocks:
        blocks = model.transformer.blocks
        keep = len(blocks) - args.train_blocks
        for i, blk in enumerate(blocks):
            for p in blk.parameters():
                p.requires_grad_(i >= keep)
    trainable = [p for p in model.transformer.parameters() if p.requires_grad]
    print(f"device={device} trainable={sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)

    samples = load_samples(args.episodes, args.limit)
    print(f"samples={len(samples)}", flush=True)
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    history: list[dict] = []

    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(len(samples))
        running = {"position_loss": 0.0, "rotate6D_loss": 0.0, "gripper_loss": 0.0}
        steps = 0
        t_epoch = time.perf_counter()
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch].tolist()
            batch = [samples[j] for j in idx]
            if args.features:
                proprio = torch.as_tensor(np.stack([s["proprio"] for s in batch]),
                                          dtype=torch.float32, device=device)
                action = torch.as_tensor(np.stack([s["action"] for s in batch]),
                                         dtype=torch.float32, device=device)
                b = len(batch)
                enc = load_feature_store(torch, args.features, idx, device)
            else:
                inputs = collate(processor, torch, batch, device)
                input_ids, image_input, image_mask, proprio, action = inputs
                b = input_ids.shape[0]
                with torch.no_grad():
                    enc = model.forward_vlm(input_ids, image_input, image_mask)
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - 1e-5)
            noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, noisy_m = model.action_space.preprocess(proprio, noisy)
            pred = model.transformer(
                domain_id=torch.zeros(b, dtype=torch.long, device=device),
                action_with_noise=noisy_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
            losses = model.action_space.compute_loss(pred, action)
            loss = sum(losses.values())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            for k in running:
                running[k] += float(losses[k].detach())
            steps += 1
            if steps % 50 == 0:
                print(f"epoch {epoch} step {steps}/{len(order)//args.batch} "
                      f"loss={float(loss.detach()):.4f} "
                      f"pos={float(losses['position_loss']):.3f} "
                      f"rot={float(losses['rotate6D_loss']):.3f} "
                      f"grip={float(losses['gripper_loss']):.3f}",
                      flush=True)
        row = {"epoch": epoch, "steps": steps,
               **{k: round(v / max(steps, 1), 4) for k, v in running.items()},
               "seconds": round(time.perf_counter() - t_epoch, 1)}
        history.append(row)
        print(f"[epoch {epoch}] {row}", flush=True)
        # checkpoint every epoch so an interrupted run still leaves usable weights
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"transformer": model.transformer.state_dict(), "epoch": epoch,
                    "args": {k: str(v) for k, v in vars(args).items()}}, args.out)
        args.log.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"saved {args.out} (epoch {epoch})", flush=True)

    print(f"done, last checkpoint {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



