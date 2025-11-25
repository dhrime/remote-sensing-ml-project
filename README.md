# Setup & Running the Project

## Requirements
- Python 3.12 
- uv 

### Install uv

Windows (PowerShell):
```
iwr https://astral.sh/uv/install.ps1 -useb | iex
```

macOS / Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install Dependencies

Run this in the project root:
```
uv sync
```

This creates `.venv/` and installs all dependencies.

## Running Experiments

Use `uv run` to execute any module inside the `remote_sensing_ml` package.

Example: run the KNN module
```
uv run python -m remote_sensing_ml.train_knn
```

(Always run from the project root, not inside `src/`.)

## Summary
1. Install uv
2. Run `uv sync`
3. Run experiments using:
```
uv run python -m remote_sensing_ml.<script_name>
```
