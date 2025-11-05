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
  ```powershell
  # activate virtualenv (if created)
  .\.venv\Scripts\Activate.ps1

  # write demo UI to exports/demo_ui.png
  python engagement_hybrid_tabs.py --demo

  # specify a custom output path
  python engagement_hybrid_tabs.py --demo --demo-out .\exports\my_demo.png
  ```

- macOS / Linux (bash)
  ```bash
  # activate virtualenv (if created)
  source .venv/bin/activate

  # write demo UI to exports/demo_ui.png
  python engagement_hybrid_tabs.py --demo

  # specify a custom output path
  python engagement_hybrid_tabs.py --demo --demo-out ./exports/my_demo.png
  ```

Notes
- Demo mode does not load models or use the camera; it simply draws a static representative UI frame and writes a PNG.
- Ensure the `exports/` directory exists or supply an existing path for `--demo-out`.

## Formatting and linting
This repository was configured to use Black / isort (via `pyproject.toml`). To format and lint the codebase locally:

```bash
pip install black isort ruff
black .
isort .
ruff check .
```

You can optionally add a `pre-commit` configuration to run these automatically on commit. If you'd like, I can add a `pre-commit` config and install instructions.

## Contributing
- Feel free to open issues for bugs or feature requests.
- If you want me to update the README or add CI/pre-commit configuration, tell me what to include and I will prepare a PR.

## License
Add your preferred license here (e.g., MIT). If you already have one in the repository, keep that.

## Contact
Maintained by Eugenelcc — open an issue or create a PR to propose changes.
