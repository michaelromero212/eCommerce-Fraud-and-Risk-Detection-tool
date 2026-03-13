"""
src/data_pipeline.py
--------------------
ETL module for EFRiskEngine.

Responsibilities:
  - Load raw CSV files into Pandas DataFrames
  - Normalise data types, parse timestamps, fill/drop nulls
  - Persist processed records to a local SQLite database via SQLAlchemy

Typical usage:
    from src.data_pipeline import run_etl
    users_df, txns_df = run_etl()
"""

import os
import logging

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
USERS_CSV = os.path.join(DATA_DIR, "sample_users.csv")
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "sample_transactions.csv")
DB_PATH = os.path.join(DATA_DIR, "efrisk.db")
DB_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_users(path: str = USERS_CSV) -> pd.DataFrame:
    """
    Load raw user behaviour log from CSV.

    Args:
        path: Absolute path to sample_users.csv.

    Returns:
        Raw DataFrame with original column names.

    Raises:
        FileNotFoundError: If the CSV does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Users CSV not found at {path!r}. "
            "Run `python generate_sample_data.py` first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d user records from %s", len(df), path)
    return df


def load_transactions(path: str = TRANSACTIONS_CSV) -> pd.DataFrame:
    """
    Load raw transaction log from CSV.

    Args:
        path: Absolute path to sample_transactions.csv.

    Returns:
        Raw DataFrame with original column names.

    Raises:
        FileNotFoundError: If the CSV does not exist at *path*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Transactions CSV not found at {path!r}. "
            "Run `python generate_sample_data.py` first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d transaction records from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def normalize_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw users DataFrame.

    Transformations applied:
      - Strip leading/trailing whitespace from string columns.
      - Cast `account_age_days`, `failed_login_count`, `password_resets` to int.
      - Parse `signup_date` as datetime.
      - Fill missing `country` with 'UNKNOWN'.

    Args:
        df: Raw users DataFrame from :func:`load_users`.

    Returns:
        Cleaned and type-normalised DataFrame.
    """
    df = df.copy()

    # String columns — strip whitespace
    str_cols = ["user_id", "username", "email", "country", "device_fingerprint",
                "ip_address", "browser", "device_type"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Numeric casts
    for col in ["account_age_days", "failed_login_count", "password_resets", "is_fraudulent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Date parse
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # Fill nulls (guard: column may not exist in sparse test DataFrames)
    if "country" in df.columns:
        df["country"] = df["country"].replace("nan", "UNKNOWN").fillna("UNKNOWN")

    logger.info("Normalized %d user records.", len(df))
    return df


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw transactions DataFrame.

    Transformations applied:
      - Parse `timestamp` as datetime.
      - Cast `purchase_amount` to float, clip to 0 minimum.
      - Strip whitespace from string columns.
      - Fill missing `transaction_country` with user_registered_country.

    Args:
        df: Raw transactions DataFrame from :func:`load_transactions`.

    Returns:
        Cleaned and type-normalised DataFrame.
    """
    df = df.copy()

    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Purchase amount
    if "purchase_amount" in df.columns:
        df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce").clip(lower=0)

    # String columns
    str_cols = ["transaction_id", "user_id", "ip_address", "device_fingerprint",
                "payment_method", "transaction_country", "user_registered_country",
                "device_type", "browser"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fill missing country
    if "transaction_country" in df.columns and "user_registered_country" in df.columns:
        mask = df["transaction_country"].isin(["nan", "", "None"])
        df.loc[mask, "transaction_country"] = df.loc[mask, "user_registered_country"]

    if "is_flagged" in df.columns:
        df["is_flagged"] = pd.to_numeric(df["is_flagged"], errors="coerce").fillna(0).astype(int)

    logger.info("Normalized %d transaction records.", len(df))
    return df


def normalize_data(
    users_df: pd.DataFrame,
    txns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate normalisation for both users and transactions.

    Args:
        users_df: Raw users DataFrame.
        txns_df:  Raw transactions DataFrame.

    Returns:
        Tuple of (normalised_users, normalised_transactions).
    """
    return normalize_users(users_df), normalize_transactions(txns_df)


# ---------------------------------------------------------------------------
# SQLite Persistence
# ---------------------------------------------------------------------------

def store_to_sqlite(
    users_df: pd.DataFrame,
    txns_df: pd.DataFrame,
    db_url: str = DB_URL,
) -> None:
    """
    Persist normalised DataFrames to a SQLite database.

    Tables created/replaced:
      - ``users``         — one row per user
      - ``transactions``  — one row per transaction

    Args:
        users_df: Normalised users DataFrame.
        txns_df:  Normalised transactions DataFrame.
        db_url:   SQLAlchemy connection string. Default: ``sqlite:///data/efrisk.db``.
    """
    engine = create_engine(db_url, echo=False)

    # Coerce datetime columns to strings so SQLite accepts them cleanly
    users_out = users_df.copy()
    txns_out = txns_df.copy()
    for col in users_out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        users_out[col] = users_out[col].astype(str)
    for col in txns_out.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        txns_out[col] = txns_out[col].astype(str)

    with engine.begin() as conn:
        users_out.to_sql("users", conn, if_exists="replace", index=False)
        txns_out.to_sql("transactions", conn, if_exists="replace", index=False)

    logger.info(
        "Stored %d users and %d transactions to %s",
        len(users_df), len(txns_df), db_url,
    )


def query_sqlite(sql: str, db_url: str = DB_URL) -> pd.DataFrame:
    """
    Execute a read query against the SQLite database and return results.

    Args:
        sql:    SQL string to execute.
        db_url: SQLAlchemy connection string.

    Returns:
        DataFrame containing query results.

    Example:
        >>> df = query_sqlite("SELECT * FROM transactions WHERE is_flagged = 1")
    """
    engine = create_engine(db_url, echo=False)
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ---------------------------------------------------------------------------
# ETL Orchestrator
# ---------------------------------------------------------------------------

def run_etl(
    users_path: str = USERS_CSV,
    txns_path: str = TRANSACTIONS_CSV,
    db_url: str = DB_URL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ETL pipeline: Load → Normalise → Store.

    Args:
        users_path: Path to sample_users.csv.
        txns_path:  Path to sample_transactions.csv.
        db_url:     SQLAlchemy connection string for the output database.

    Returns:
        Tuple of (normalised_users_df, normalised_transactions_df).
    """
    logger.info("=== EFRiskEngine ETL Pipeline START ===")

    # 1. Extract
    raw_users = load_users(users_path)
    raw_txns = load_transactions(txns_path)

    # 2. Transform
    users, txns = normalize_data(raw_users, raw_txns)

    # 3. Load
    store_to_sqlite(users, txns, db_url)

    logger.info("=== EFRiskEngine ETL Pipeline COMPLETE ===")
    return users, txns


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    users_df, txns_df = run_etl()
    print(f"\nUsers shape:       {users_df.shape}")
    print(f"Transactions shape: {txns_df.shape}")
    print("\nSample users:\n", users_df.head(3).to_string(index=False))
    print("\nSample transactions:\n", txns_df.head(3).to_string(index=False))
