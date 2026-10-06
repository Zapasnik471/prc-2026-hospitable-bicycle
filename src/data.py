"""Loading the competition files."""
import polars as pl

import config as C


def load_training() -> pl.DataFrame:
    """All 12 months of 2025 (train + validation months)."""
    files = sorted(C.DATA_DIR.glob(C.TRAIN_GLOB))
    if not files:
        raise FileNotFoundError(f"no {C.TRAIN_GLOB} in {C.DATA_DIR} (set PRC_DATA_DIR)")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def load_ranking() -> pl.DataFrame:
    """Jan + Jul 2026; departures have BLOCK_TIME and TAXITIME blanked."""
    return pl.read_parquet(C.RANKING_FILE)


def load_template() -> pl.DataFrame:
    return pl.read_parquet(C.TEMPLATE_FILE)


def sec(a: str, b: str) -> pl.Expr:
    """a - b in seconds, timezone stripped on both sides (inputs mix naive and UTC)."""
    return (pl.col(a).dt.replace_time_zone(None) - pl.col(b).dt.replace_time_zone(None)).dt.total_seconds()
