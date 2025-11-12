# Student Engagement Monitor — UI Demo

Facial expression recognition for a student engagement system (CSCI 323).

This repository contains code and notebooks for a Student Engagement Monitor. It includes:
- a live webcam UI and export utilities (CSV / PNG / JSON),
- helper scripts to inspect model checkpoints,
- and a `--demo` mode that renders a static representative UI frame to a PNG without requiring a camera or ML models.

Repository summary
- Description: Facial expression recognition for student engagement system for CSCI 323
- Primary languages: Jupyter Notebook, Python

## Overview (short)
- Demo mode (`--demo`) to preview the UI without models or webcam.
- Three model notebooks are included in the repository to train / inspect / evaluate models (see the notebooks or root directory for their .ipynb files).
- Focused instructions on creating a virtual environment, installing dependencies, and running the demo or notebooks locally.

## Prerequisites
- Python 3.8+ recommended
- Optional: virtual environment (venv / virtualenv)
- Optional developer tools: black, isort, ruff, jupyterlab

## Quickstart — create & activate a venv
Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies
1. If the repository includes a requirements.txt:
```bash
pip install -r requirements.txt
```

2. If there is no requirements.txt, install the likely needed packages (examples):
```bash
pip install opencv-python numpy pillow mediapipe jupyterlab
# Optional: if you work with PyTorch models
pip install torch torchvision timm
```

3. (Optional) Create a requirements.txt from your environment for repeatable installs:
```bash
pip freeze > requirements.txt
```

## Running the demo (no camera, no models)
The demo mode draws a static representative UI frame and writes a PNG file. It is intended to preview the UI and verify export paths.

Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
python engagement_hybrid_tabs.py --demo
python engagement_hybrid_tabs.py --demo --demo-out .\exports\demo_ui.png
```

macOS / Linux
```bash
source .venv/bin/activate
python engagement_hybrid_tabs.py --demo
python engagement_hybrid_tabs.py --demo --demo-out ./exports/demo_ui.png
```

Notes:
- Demo mode does not load ML models or open a webcam — it only writes a static PNG.
- Ensure the `exports/` directory exists or provide an existing path via `--demo-out`.

## Running the app with camera / models
- Place model files (e.g., `.pt`) into the `models/` directory if present.
- Launch the application:
```bash
python run.py
```
- For PyTorch `.pt` files: prefer saving and loading `state_dict`s for portability. If you run into torch.load pickling errors, re-export the checkpoint as a `state_dict` in the environment where the model was trained.

## Notebooks / model files
This repo contains three model notebooks (.ipynb) that cover training, evaluation, and/or inference workflows. Open them with JupyterLab or Jupyter Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Tips:
- Run cells in order; if a notebook expects model checkpoints, put them in the `models/` directory or update the notebook paths.
- To convert a notebook to a script:
```bash
jupyter nbconvert --to script path/to/notebook.ipynb
```

If you'd like the README to list the exact filenames of the three notebooks, tell me their paths and I'll add them.

## How emotions map to engagement (brief)
- The model predicts probabilities for 7 emotion classes (surprise, fear, disgust, happiness, sadness, anger, neutral).
- A mapping in the code converts the 7-class probabilities to a 3-class engagement label (Engaged / Neutral / Disengaged) and computes a 1–10 engagement score. Both label and score are smoothed to reduce flicker.
- Tweak `EMO2ENG`, `W_ENG10`, or smoothing parameters in `engagement_hybrid_tabs.py` if you want different mappings or smoothing.

## Formatting & development workflow
Use Black / isort / Ruff for consistent style:
```bash
pip install black isort ruff
black .
isort .
ruff check .
```
Consider adding a `pre-commit` config to enforce formatting on commit.

## Suggested repo updates (I can prepare these)
- Add a `requirements.txt` listing runtime dependencies.
- Add a `pre-commit` configuration to enforce formatting automatically.
- Add a CONTRIBUTING.md describing the development workflow.
- Add a LICENSE (e.g., MIT) if you want to make licensing explicit.

## Contributing
- Open issues for bugs or feature requests.
- Create pull requests describing changes and testing steps.
- If you want me to add the exact notebook filenames, a requirements.txt, or a pre-commit config, tell me and I will prepare the changes.

## License
Add your preferred license file (e.g., MIT) to the repository if you haven't already.

## Contact
Maintained by @Eugenelcc — open an issue or create a PR to propose changes.# Student Engagement Monitor — UI Demo

Facial expression recognition for a student engagement system (CSCI 323).

This repository contains code and notebooks for a Student Engagement Monitor. It includes:
- a live webcam UI and export utilities (CSV / PNG / JSON),
- helper scripts to inspect model checkpoints,
- and a `--demo` mode that renders a static representative UI frame to a PNG without requiring a camera or ML models.

Repository summary
- Description: Facial expression recognition for student engagement system for CSCI 323
- Primary languages: Jupyter Notebook, Python

## Overview (short)
- Demo mode (`--demo`) to preview the UI without models or webcam.
- Three model notebooks are included in the repository to train / inspect / evaluate models (see the notebooks or root directory for their .ipynb files).
- Focused instructions on creating a virtual environment, installing dependencies, and running the demo or notebooks locally.

## Prerequisites
- Python 3.8+ recommended
- Optional: virtual environment (venv / virtualenv)
- Optional developer tools: black, isort, ruff, jupyterlab

## Quickstart — create & activate a venv
Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies
1. If the repository includes a requirements.txt:
```bash
pip install -r requirements.txt
```

2. If there is no requirements.txt, install the likely needed packages (examples):
```bash
pip install opencv-python numpy pillow mediapipe jupyterlab
# Optional: if you work with PyTorch models
pip install torch torchvision timm
```

3. (Optional) Create a requirements.txt from your environment for repeatable installs:
```bash
pip freeze > requirements.txt
```

## Running the demo (no camera, no models)
The demo mode draws a static representative UI frame and writes a PNG file. It is intended to preview the UI and verify export paths.

Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
python engagement_hybrid_tabs.py --demo
python engagement_hybrid_tabs.py --demo --demo-out .\exports\demo_ui.png
```

macOS / Linux
```bash
source .venv/bin/activate
python engagement_hybrid_tabs.py --demo
python engagement_hybrid_tabs.py --demo --demo-out ./exports/demo_ui.png
```

Notes:
- Demo mode does not load ML models or open a webcam — it only writes a static PNG.
- Ensure the `exports/` directory exists or provide an existing path via `--demo-out`.

## Running the app with camera / models
- Place model files (e.g., `.pt`) into the `models/` directory if present.
- Launch the application:
```bash
python run.py
```
- For PyTorch `.pt` files: prefer saving and loading `state_dict`s for portability. If you run into torch.load pickling errors, re-export the checkpoint as a `state_dict` in the environment where the model was trained.

## Notebooks / model files
This repo contains three model notebooks (.ipynb) that cover training, evaluation, and/or inference workflows. Open them with JupyterLab or Jupyter Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Tips:
- Run cells in order; if a notebook expects model checkpoints, put them in the `models/` directory or update the notebook paths.
- To convert a notebook to a script:
```bash
jupyter nbconvert --to script path/to/notebook.ipynb
```

If you'd like the README to list the exact filenames of the three notebooks, tell me their paths and I'll add them.

## How emotions map to engagement (brief)
- The model predicts probabilities for 7 emotion classes (surprise, fear, disgust, happiness, sadness, anger, neutral).
- A mapping in the code converts the 7-class probabilities to a 3-class engagement label (Engaged / Neutral / Disengaged) and computes a 1–10 engagement score. Both label and score are smoothed to reduce flicker.
- Tweak `EMO2ENG`, `W_ENG10`, or smoothing parameters in `engagement_hybrid_tabs.py` if you want different mappings or smoothing.

## Formatting & development workflow
Use Black / isort / Ruff for consistent style:
```bash
pip install black isort ruff
black .
isort .
ruff check .
```
Consider adding a `pre-commit` config to enforce formatting on commit.

## Suggested repo updates (I can prepare these)
- Add a `requirements.txt` listing runtime dependencies.
- Add a `pre-commit` configuration to enforce formatting automatically.
- Add a CONTRIBUTING.md describing the development workflow.
- Add a LICENSE (e.g., MIT) if you want to make licensing explicit.

## Contributing
- Open issues for bugs or feature requests.
- Create pull requests describing changes and testing steps.
- If you want me to add the exact notebook filenames, a requirements.txt, or a pre-commit config, tell me and I will prepare the changes.

## License
MIT License 

## Contact
Maintained by @Eugenelcc — open an issue or create a PR to propose changes.
