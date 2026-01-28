import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BATCH_DIR = Path("data/processed/daily_features_batch")
OUTPUT_PATH = Path("data/processed/daily_features_full.parquet")

DEDUP_KEYS = ["event_id", "permno", "date"]


def load_all_batches(batch_dir: Path) -> list[pd.DataFrame]:
    parquet_files = sorted(batch_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")

    dfs = []
    for fp in parquet_files:
        logging.info(f"Loading {fp.name}")
        df = pd.read_parquet(fp)
        dfs.append(df)

    return dfs


def validate_schema(dfs: list[pd.DataFrame]) -> None:
    base_cols = set(dfs[0].columns)

    for i, df in enumerate(dfs[1:], start=1):
        if set(df.columns) != base_cols:
            raise ValueError(
                f"Schema mismatch in batch {i}: "
                f"{set(df.columns) ^ base_cols}"
            )


def merge_batches(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    logging.info("Concatenating batches")
    df = pd.concat(dfs, axis=0, ignore_index=True)

    logging.info("Dropping duplicates")
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_KEYS)
    after = len(df)

    logging.info(f"Removed {before - after} duplicate rows")

    logging.info("Sorting by event_id, then date")
    df = df.sort_values(["event_id", "date"]).reset_index(drop=True)

    return df


def main():
    logging.info("Starting daily_feature merge")

    dfs = load_all_batches(BATCH_DIR)
    validate_schema(dfs)

    merged = merge_batches(dfs)

    logging.info(f"Writing merged parquet to {OUTPUT_PATH}")
    merged.to_parquet(OUTPUT_PATH, index=False)

    logging.info("Merge completed successfully")


if __name__ == "__main__":
    main()
