# Student Engagement Monitor — UI Demo

A small demo that renders a webcam-based student engagement UI. A `--demo` mode is provided so you can preview the UI without a camera or models — it draws a representative static frame and writes a PNG.

## Features

- Demo mode to render a representative UI frame to an image file (no camera or models required).
- Cross-platform usage examples (PowerShell and POSIX shells).
- Formatting tooling configured via `pyproject.toml` (Black / isort / Ruff).

## Prerequisites

- Python 3.8+ recommended
- Optional: virtual environment (venv, virtualenv)
- Developer tools (optional): black, isort, ruff

## Installation (recommended)

1. Create and activate a virtual environment (optional but recommended)

   - Windows (PowerShell)
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS / Linux
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies
   - If the repo contains a requirements file:
     ```bash
     pip install -r requirements.txt
     ```
   - Otherwise, install the packages you need (example):
     ```bash
     pip install opencv-python numpy pillow
     ```

## Running the demo

The demo mode writes a static representative UI frame to a PNG file and does not load ML models or use the camera.

- PowerShell

  ````powershell
  # Student Engagement Monitor

  This repository contains a responsive demo UI and runtime for a Student Engagement Monitor.
  It runs a facial-expression model (ONNX or PyTorch) to predict seven emotions, maps those
  emotions to a three-class engagement label (Engaged / Neutral / Disengaged), and computes
  a single interpretable 1–10 engagement score for visualization and analytics.

  ## Key features
  - Live webcam UI with responsive layout and export options (CSV / PNG / JSON).
  - Demo mode (`--demo`) that renders a static representative UI frame without models or camera.
  - Support for ONNX and PyTorch backends; flexible handling of various model checkpoint formats.
  - Utility script: `scripts/inspect_checkpoint.py` to examine `.pt` files locally.

  ## Quick start (Windows PowerShell)
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt   # if present
  pip install opencv-python numpy mediapipe
  # Optional: for PyTorch models
  pip install torch torchvision timm
  # Optional: for ONNX
  pip install onnxruntime
  ````

  Run the demo (no camera, no models):

  ```powershell
  python engagement_hybrid_tabs.py --demo
  python engagement_hybrid_tabs.py --demo --demo-out .\exports\demo_ui.png
  ```

  Run the app with discovered models (from `models/`):

  ```powershell
  python run.py
  ```

  ## Models: ONNX vs PyTorch `.pt`

  - ONNX: portable and recommended for sharing models across environments. `run.py` will load `.onnx` files directly.
  - PyTorch `.pt`: may be a pickled Module (torch.save(model, path)) or a `state_dict` (torch.save(model.state_dict(), path)). Pickled Modules require the original class definition to be importable when loading.

  ### Common checkpoint issue: "Can't get attribute 'HybridMiniXMobile'"

  - Symptom: torch.load fails with an AttributeError about `HybridMiniXMobile` when launching the app.
  - Cause: the `.pt` file is a pickled model instance whose class is not available/importable in this runtime (pickle needs the class definition to reconstruct the object).
  - Fixes:
    1. Run the app in the same project/environment where `HybridMiniXMobile` is defined, or copy the class source into this repo so it can be imported.
    2. Re-save the model as a `state_dict` in the original environment and then instantiate and load it here:
       ```python
       torch.save(model.state_dict(), 'model_state.pt')
       model.load_state_dict(torch.load('model_state.pt', map_location='cpu'))
       ```
    3. Export the model to ONNX in the original environment and drop the `.onnx` file into `models/`.

  Use the inspect helper to decide which route to take:

  ```powershell
  python scripts/inspect_checkpoint.py models\raf_affectnet_balanced_HybridMiniXMobile.pt
  ```

  ## How emotions are mapped to engagement (brief)

  - The model predicts 7 emotion probabilities p7 for: surprise, fear, disgust, happiness, sadness, anger, neutral.
  - A fixed 7×3 matrix `EMO2ENG` maps p7 → p3 (Engaged, Neutral, Disengaged) via matrix multiplication and normalization.
  - The label is `argmax(p3)`. The numeric score is `eng10 = clip(W_ENG10 · p7, 1, 10)`. Both are smoothed with an EMA to reduce flicker.

  ## Configuration & development

  - Tweak `EMO2ENG` or `W_ENG10` at the top of `engagement_hybrid_tabs.py` to change mapping or scoring.
  - Adjust smoothing with the `--ema` flag when launching the app.
  - Formatting: use Black / isort / Ruff for consistent style.

  ## Support

  - If you'd like, I can add: ONNX export helpers, a minimal shim for `HybridMiniXMobile` for testing, or CI/pre-commit config. Tell me which and I will prepare changes.

  ## License

  Add your preferred license (e.g., MIT) or keep the existing `LICENSE` file.

  ***

  Generated README — tell me if you want a shorter tutorial version or a slide-ready summary.
