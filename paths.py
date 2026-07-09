from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
RAW_FILE = DATA / "raw_data.parquet"
INFO_FILE = DATA / "raw_info.parquet"
STOCK_FILE = DATA / "stock_data.parquet"
SECTOR_FILE = DATA / "sector_data.parquet"
TICKERS_FILE = DATA / "company_tickers.json"

def cluster_save( type: str, depth:int, gen:int ) -> str:
    return DATA / type / f"gen{gen}_cluster_data_depth{depth}"