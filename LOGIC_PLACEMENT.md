# Logic placement

- main.py
  - Application entry point and orchestration only (argument parsing, wiring modules, calling functions).
  - Should import and call functions/classes from model.py and preprocess.py; avoid heavy logic or I/O on import.

- model.py
  - Place model-related logic here: model classes, training/evaluation routines, prediction helpers.
  - Keep functions/classes pure and testable; do not execute training on import (use guarded `if __name__ == "__main__":` for demos).

- preprocess.py
  - Place data-loading, cleaning, transformation, feature engineering, and dataset utilities here.
  - Keep preprocessing pipelines and small helper functions that operate on data structures.

- If project grows
  - Break into a package (e.g., src/ or app/) and create submodules: app/main.py, app/core/model.py, app/core/preprocess.py, app/services/, app/utils/.
  - Keep side-effecting I/O at the edges (main or dedicated I/O module) and put business logic in model/preprocess modules for easier testing.