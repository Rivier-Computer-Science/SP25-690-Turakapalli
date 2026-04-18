18/04/2026 00:05 EDT Fix module import under uv run
- Added repo root insertion into `sys.path` at the top of `main.py`
- Ensures `from models.residual_cnn import ResidualCNN` resolves correctly when executed with `uv run`
- Keeps `main.py` argument validation intact for `--input` and `--output`

18/04/2026 00:00 EDT Fix imports and create models package
- Renamed conflicting root file `models` to `models_placeholder.txt`
- Created `models/` package directory
- Added `models/__init__.py` with exported model classes
- Added `models/baseline_cnn.py` and `models/residual_cnn.py`
- Verified `from models.residual_cnn import ResidualCNN` and `from models.baseline_cnn import BaselineCNN` import successfully