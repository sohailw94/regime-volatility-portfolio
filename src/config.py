from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    path.mkdir(parents=True, exist_ok=True)