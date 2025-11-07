#!/usr/bin/env python3
"""
run.py — convenience launcher for engagement_hybrid_tabs.py

Usage examples:
  python run.py
  python run.py --cam 0 --size 224 --ema 0.25
  python run.py --only resnet18 deit         # only load models whose filenames match these tokens
  python run.py --models_dir models          # default is ./models
"""

import argparse
import os
import sys
from pathlib import Path
import shutil
import subprocess

APP = "engagement_hybrid_tabs.py"


# Heuristics to infer torch arch from filename
def infer_arch_from_name(name: str) -> str | None:
    n = name.lower()
    if "resnet18" in n or "resnet-18" in n or "rn18" in n:
        return "resnet18"
    if (
        "deit" in n
        or "vit_tiny" in n
        or "vit-tiny" in n
        or "vittiny" in n
        or "vit" in n
        and "tiny" in n
    ):
        return "deit-tiny"
    # If user put "minix" or "mini-xception" as .pt (not recommended), there is no built-in torch arch in the hybrid.
    # Suggest they export Mini-X to ONNX instead.
    if "minix" in n or "mini-x" in n or "mini_x" in n or "xception" in n:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models_dir",
        default="models",
        help="Folder containing model files (.onnx or .pt)",
    )
    ap.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    ap.add_argument(
        "--size", type=int, default=224, help="Model input size (default 224)"
    )
    ap.add_argument(
        "--ema", type=float, default=0.25, help="EMA smoothing alpha (default 0.25)"
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Load only models whose filename contains any of these tokens (e.g. resnet18 deit onnx)",
    )
    args = ap.parse_args()

    app_path = Path(__file__).with_name(APP)
    if not app_path.exists():
        sys.exit(
            f"[run.py] Could not find {APP} next to run.py. Make sure it’s in the same folder."
        )

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        sys.exit(f"[run.py] Models folder not found: {models_dir.resolve()}")

    # Gather candidate model files
    all_files = sorted(
        [p for p in models_dir.iterdir() if p.suffix.lower() in (".onnx", ".pt")]
    )
    if args.only:
        tokens = [t.lower() for t in args.only]

        def keep(p: Path) -> bool:
            n = p.name.lower()
            return any(tok in n for tok in tokens)

        model_files = [p for p in all_files if keep(p)]
    else:
        model_files = all_files

    if not model_files:
        sys.exit(
            f"[run.py] No model files found in {models_dir.resolve()} "
            f"(looking for .onnx or .pt). Add files and try again."
        )

    # Build command to launch the app
    cmd = [
        sys.executable,
        str(app_path),
        "--cam",
        str(args.cam),
        "--size",
        str(args.size),
        "--ema",
        str(args.ema),
    ]

    # Add models with proper backend/arch flags
    torch_needed = False
    onnx_needed = False
    skipped = []

    for p in model_files:
        if p.suffix.lower() == ".onnx":
            cmd += ["--backend", "onnx", "--weights", str(p)]
            onnx_needed = True
        elif p.suffix.lower() == ".pt":
            arch = infer_arch_from_name(p.name)
            if arch is None:
                # Unknown arch in filename — try a checkpoint-backed model (will attempt to
                # load a pickled nn.Module from the .pt). If the checkpoint only contains a
                # state_dict we may still need an explicit arch or an ONNX export.
                arch = "ckpt"
            cmd += ["--backend", "torch", "--arch", arch, "--weights", str(p)]
            torch_needed = True
    # Note: files with unknown arch tokens will be launched with --arch ckpt which
    # tells the app to attempt loading a pickled model object from the checkpoint.
    # If that fails (only a state_dict present and no known arch), the app will
    # report an error explaining next steps (export to ONNX or provide a supported arch).

    # Quick dependency checks (optional but friendly)
    if onnx_needed and not _module_available("onnxruntime"):
        print(
            "[run.py] WARNING: onnxruntime not installed; ONNX models will fail. Install: pip install onnxruntime"
        )
    if torch_needed:
        if not _module_available("torch"):
            print(
                "[run.py] WARNING: torch not installed; PyTorch models will fail. Install: pip install torch torchvision"
            )
        if not _module_available("timm"):
            print(
                "[run.py] WARNING: timm not installed; required for PyTorch model creation. Install: pip install timm"
            )

    # Show final command for transparency
    print("\n[run.py] Launch command:")
    print(" ", " ".join(map(shell_quote, cmd)), "\n")

    # Launch
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[run.py] App exited with non-zero status: {e.returncode}")


def _module_available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def shell_quote(s: str) -> str:
    if os.name == "nt":
        # Windows: keep simple
        return f'"{s}"' if (" " in s or "\t" in s) else s
    else:
        # POSIX: use single quotes if needed
        return f"'{s}'" if any(c in s for c in " \t!$&()[]{};") else s


if __name__ == "__main__":
    main()
