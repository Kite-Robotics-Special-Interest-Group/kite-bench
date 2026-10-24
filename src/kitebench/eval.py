"""Perturbation sweep: measure how far model actions drift when the input changes.

The policy is injected, so the harness does not depend on any specific model. A
reference adapter for X-VLA checkpoints is provided and only imports
``transformers``/``torch`` when used.

Usage::

    python -m kitebench.eval --episodes out/episodes --model path/to/weights \\
        --frames 3 --steps 5 --out out/perturb.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from kitebench import metrics, perturb

# family -> list of level arguments (each a dict of kwargs for the function)
LEVELS: dict[str, list[dict]] = {
    "brightness": [{"factor": v} for v in (0.8, 0.9, 1.1, 1.2)],
    "contrast": [{"factor": v} for v in (0.8, 0.9, 1.1, 1.2)],
    "gamma": [{"value": v} for v in (0.7, 0.85, 1.2, 1.5)],
    "noise": [{"sigma": v, "seed": 0} for v in (5.0, 10.0, 15.0, 20.0)],
    "blur": [{"radius": v} for v in (1.0, 2.0, 3.0, 4.0)],
    "jpeg": [{"quality": v} for v in (80, 60, 40, 20)],
    "translate": [{"dx": v} for v in (-0.06, -0.03, 0.03, 0.06)],
    "scale": [{"factor": v} for v in (0.95, 0.9, 1.05, 1.1)],
    "rotate": [{"degrees": v} for v in (-5.0, -2.0, 2.0, 5.0)],
    "downsample": [{"size": v} for v in (192, 168, 140, 112)],
    "occlude_rect": [{"area": a, "seed": i} for i, a in enumerate((0.05, 0.10, 0.20, 0.30))],
    "occlude_arm": [{}],
}


class Policy(Protocol):
    def act(self, image: Image.Image, proprio: np.ndarray, instruction: str) -> np.ndarray:
        """Return a ``[T, 20]`` action chunk."""


@dataclass
class Frame:
    image: Image.Image
    proprio: np.ndarray
    instruction: str
    episode: str
    step: int


def load_frames(root: Path, count: int = 3, stride: int = 7) -> list[Frame]:
    """Pick evenly spaced front-view frames from the collected episodes."""
    episodes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not episodes:
        raise FileNotFoundError(f"no episodes under {root}")
    frames: list[Frame] = []
    per_episode = max(1, count // len(episodes))
    for ep in episodes:
        meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        actions = np.load(ep / "action.npy")
        images = sorted((ep / "obs").glob("*_front.png"))
        picked = images[:: max(1, len(images) // per_episode)][:per_episode]
        for img_path in picked:
            step = int(img_path.name.split("_")[0])
            proprio = np.zeros(20, dtype=np.float32)
            proprio[0:3] = actions[step, 0:3]
            proprio[3:10] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            frames.append(Frame(Image.open(img_path).convert("RGB"), proprio,
                                meta["instruction"], ep.name, step))
        if len(frames) >= count:
            break
    return frames[:count]


class XVLAAdapter:
    """Reference adapter for an X-VLA checkpoint (``ee6d`` WidowX edition)."""

    def __init__(self, model_path: str | Path, steps: int = 5, device: str | None = None,
                 seed: int | None = 0):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # XVLA pins its Florence2 backbone to float32 in __init__, so half precision
        # on the inputs fails with "Input type (Half) and bias type (float)".
        self.dtype = torch.float32
        self.model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=self.dtype
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.steps = steps

    def act(self, image: Image.Image, proprio: np.ndarray, instruction: str) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=[image], language_instruction=instruction)
        if self.seed is not None:
            # the denoiser starts from torch.randn; fixing it isolates input effects
            torch.manual_seed(self.seed)
        with torch.no_grad():
            action = self.model.generate_actions(
                input_ids=inputs["input_ids"].to(self.device),
                image_input=inputs["image_input"].to(self.device, dtype=self.dtype),
                image_mask=inputs["image_mask"].to(self.device),
                domain_id=torch.zeros(1, dtype=torch.long, device=self.device),
                proprio=torch.as_tensor(proprio, dtype=self.dtype, device=self.device).unsqueeze(0),
                steps=self.steps,
            )
        return action[0].float().cpu().numpy()


def run_sweep(policy: Policy, frames: list[Frame], out_md: Path) -> list[dict]:
    clean = [policy.act(f.image, f.proprio, f.instruction) for f in frames]
    rows: list[dict] = []

    # noise floor: the denoiser starts from random noise, so two clean runs differ
    repeat = [
        metrics.action_drift(clean[i], policy.act(f.image, f.proprio, f.instruction))
        for i, f in enumerate(frames)
    ]
    baseline = {"family": "baseline_repeat", "level": "-", **metrics.summarise(repeat)}
    rows.append(baseline)
    print(f"{'baseline_repeat':14s} {'-':22s} pos={baseline['pos_mm_mean']:7.2f}mm "
          f"rot={baseline['rot_deg_mean']:6.2f}° flip={baseline['grip_flip_mean']:.2f}", flush=True)

    registry = perturb.registry()
    for family, levels in LEVELS.items():
        fn = registry[family]
        for kwargs in levels:
            drifts = [
                metrics.action_drift(
                    clean[i],
                    policy.act(fn(f.image, **kwargs), f.proprio, f.instruction),
                )
                for i, f in enumerate(frames)
            ]
            row = {
                "family": family,
                "level": ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "-",
                **metrics.summarise(drifts),
            }
            rows.append(row)
            print(f"{row['family']:14s} {row['level']:22s} "
                  f"pos={row['pos_mm_mean']:7.2f}mm rot={row['rot_deg_mean']:6.2f}° "
                  f"flip={row['grip_flip_mean']:.2f}", flush=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(to_markdown(rows, len(frames)), encoding="utf-8")
    return rows


def to_markdown(rows: list[dict], n_frames: int) -> str:
    lines = [
        f"# 输入扰动 → 动作漂移（{n_frames} 帧均值）",
        "",
        "| 扰动 | 档位 | Δpos (mm) | Δrot (°) | 夹爪翻转率 | robust_score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['family']} | {r['level']} | {r['pos_mm_mean']} | {r['rot_deg_mean']} | "
            f"{r['grip_flip_mean']} | {metrics.robust_score(r['pos_mm_mean']):.2f} |"
        )
    lines.append("")
    lines.append("> Δpos 是绝对末端位置目标的漂移（mm）；同一动作空间内横向可比，")
    lines.append("> 不可与增量动作的漂移混报。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="input perturbation sweep")
    parser.add_argument("--episodes", type=Path, default=Path("out/episodes"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("out/perturb.md"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="fixed denoiser seed; use -1 for the stochastic baseline")
    args = parser.parse_args()

    frames = load_frames(args.episodes, args.frames)
    print(f"frames: {[(f.episode, f.step) for f in frames]}", flush=True)
    policy = XVLAAdapter(args.model, steps=args.steps,
                         seed=None if args.seed < 0 else args.seed)
    rows = run_sweep(policy, frames, args.out)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
