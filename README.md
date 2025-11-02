# Student Engagement Monitor — UI Demo

This repository contains a simple student engagement demo that runs a webcam-based UI. I added a `--demo` mode so you can preview the UI without a camera.

How to run the demo (PowerShell):

```powershell
# activate provided virtualenv (optional)
.\fer_env\Scripts\Activate.ps1

# write demo UI to exports/demo_ui.png
python engagement_hybrid_tabs.py --demo

# specify a custom output path
python engagement_hybrid_tabs.py --demo --demo-out .\exports\my_demo.png
```

Notes

- The demo mode does not load models or use the camera; it simply draws a static representative UI frame and writes a PNG.
- I also configured Black/isort via `pyproject.toml` earlier. To format the file(s):

```powershell
pip install black isort ruff
black .
isort .
```

If you'd like, I can add a `pre-commit` config to run those automatically on commit.
