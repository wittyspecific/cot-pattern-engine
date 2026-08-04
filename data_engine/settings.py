from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CFTC_DATASETS = {"legacy_futures_only": "6dca-aqww"}
CFTC_BASE = "https://publicreporting.cftc.gov/resource"
MAX_COT_AGE_DAYS = 14
MAX_PRICE_AGE_DAYS = 7
