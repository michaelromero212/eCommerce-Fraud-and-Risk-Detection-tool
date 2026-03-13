"""
tests/test_data_pipeline.py
----------------------------
Unit tests for src/data_pipeline.py

Run with:
    pytest tests/test_data_pipeline.py -v
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data_pipeline import (
    load_transactions,
    load_users,
    normalize_transactions,
    normalize_users,
    query_sqlite,
    store_to_sqlite,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal in-memory CSVs for testing
# ---------------------------------------------------------------------------

USERS_COLUMNS = [
    "user_id", "username", "email", "country", "account_age_days",
    "device_fingerprint", "ip_address", "failed_login_count",
    "password_resets", "signup_date", "browser", "device_type", "is_fraudulent",
]

TXNS_COLUMNS = [
    "transaction_id", "user_id", "timestamp", "ip_address", "device_fingerprint",
    "purchase_amount", "payment_method", "transaction_country",
    "user_registered_country", "device_type", "browser", "is_flagged",
]


def _make_users_csv(path: str, n: int = 5) -> None:
    """Write a minimal users CSV to *path*."""
    rows = [
        {
            "user_id": f"U{i:04d}",
            "username": f"user_{i}",
            "email": f"u{i}@test.com",
            "country": "US",
            "account_age_days": 100 + i,
            "device_fingerprint": f"dev{i:04d}",
            "ip_address": f"192.168.0.{i}",
            "failed_login_count": i % 4,
            "password_resets": 0,
            "signup_date": "2024-01-01",
            "browser": "Chrome",
            "device_type": "desktop",
            "is_fraudulent": 0,
        }
        for i in range(1, n + 1)
    ]
    pd.DataFrame(rows, columns=USERS_COLUMNS).to_csv(path, index=False)


def _make_txns_csv(path: str, n: int = 10) -> None:
    """Write a minimal transactions CSV to *path*."""
    rows = [
        {
            "transaction_id": f"T{i:05d}",
            "user_id": f"U{(i % 5) + 1:04d}",
            "timestamp": f"2025-06-{(i % 28) + 1:02d} 10:00:00",
            "ip_address": "192.168.1.1",
            "device_fingerprint": f"dev{(i % 5) + 1:04d}",
            "purchase_amount": 10.0 * i,
            "payment_method": "credit_card",
            "transaction_country": "US",
            "user_registered_country": "US",
            "device_type": "desktop",
            "browser": "Firefox",
            "is_flagged": 0,
        }
        for i in range(1, n + 1)
    ]
    pd.DataFrame(rows, columns=TXNS_COLUMNS).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_csv_dir(tmp_path):
    """Return a temporary directory containing sample users and transactions CSVs."""
    users_path = tmp_path / "sample_users.csv"
    txns_path = tmp_path / "sample_transactions.csv"
    _make_users_csv(str(users_path))
    _make_txns_csv(str(txns_path))
    return tmp_path, str(users_path), str(txns_path)


@pytest.fixture()
def raw_users_df(tmp_csv_dir):
    _, users_path, _ = tmp_csv_dir
    return load_users(users_path)


@pytest.fixture()
def raw_txns_df(tmp_csv_dir):
    _, _, txns_path = tmp_csv_dir
    return load_transactions(txns_path)


# ---------------------------------------------------------------------------
# load_users
# ---------------------------------------------------------------------------

class TestLoadUsers:
    def test_returns_dataframe(self, raw_users_df):
        """load_users should return a DataFrame."""
        assert isinstance(raw_users_df, pd.DataFrame)

    def test_expected_row_count(self, raw_users_df):
        """Should load exactly the number of rows we wrote (5)."""
        assert len(raw_users_df) == 5

    def test_expected_columns(self, raw_users_df):
        """All expected columns must be present."""
        for col in ["user_id", "username", "country", "failed_login_count"]:
            assert col in raw_users_df.columns, f"Missing column: {col}"

    def test_raises_on_missing_file(self, tmp_path):
        """FileNotFoundError should be raised for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_users(str(tmp_path / "nonexistent.csv"))


# ---------------------------------------------------------------------------
# load_transactions
# ---------------------------------------------------------------------------

class TestLoadTransactions:
    def test_returns_dataframe(self, raw_txns_df):
        assert isinstance(raw_txns_df, pd.DataFrame)

    def test_expected_row_count(self, raw_txns_df):
        """Should load exactly 10 rows."""
        assert len(raw_txns_df) == 10

    def test_expected_columns(self, raw_txns_df):
        for col in ["transaction_id", "user_id", "purchase_amount", "timestamp"]:
            assert col in raw_txns_df.columns, f"Missing column: {col}"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_transactions(str(tmp_path / "nonexistent.csv"))


# ---------------------------------------------------------------------------
# normalize_users
# ---------------------------------------------------------------------------

class TestNormalizeUsers:
    def test_failed_login_count_is_int(self, raw_users_df):
        normalised = normalize_users(raw_users_df)
        assert normalised["failed_login_count"].dtype in (int, "int64", "int32")

    def test_signup_date_is_datetime(self, raw_users_df):
        normalised = normalize_users(raw_users_df)
        assert pd.api.types.is_datetime64_any_dtype(normalised["signup_date"])

    def test_country_filled_if_missing(self):
        df = pd.DataFrame({"country": [None, "US", float("nan")]})
        result = normalize_users(df)
        assert result["country"].isnull().sum() == 0

    def test_whitespace_stripped_from_user_id(self):
        df = pd.DataFrame({"user_id": ["  U0001  ", "U0002"]})
        result = normalize_users(df)
        assert result["user_id"].iloc[0] == "U0001"


# ---------------------------------------------------------------------------
# normalize_transactions
# ---------------------------------------------------------------------------

class TestNormalizeTransactions:
    def test_timestamp_is_datetime(self, raw_txns_df):
        normalised = normalize_transactions(raw_txns_df)
        assert pd.api.types.is_datetime64_any_dtype(normalised["timestamp"])

    def test_purchase_amount_is_float(self, raw_txns_df):
        normalised = normalize_transactions(raw_txns_df)
        assert pd.api.types.is_float_dtype(normalised["purchase_amount"])

    def test_purchase_amount_non_negative(self, raw_txns_df):
        normalised = normalize_transactions(raw_txns_df)
        assert (normalised["purchase_amount"] >= 0).all()

    def test_no_negative_amount_after_clipping(self):
        df = pd.DataFrame({
            "purchase_amount": [-10.0, 5.0, None],
            "timestamp": ["2025-01-01 00:00:00"] * 3,
        })
        result = normalize_transactions(df)
        assert (result["purchase_amount"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# store_to_sqlite + query_sqlite (round-trip)
# ---------------------------------------------------------------------------

class TestSQLiteRoundTrip:
    def test_store_and_retrieve_users(self, raw_users_df, raw_txns_df, tmp_path):
        """Data written to SQLite should be retrievable via query_sqlite."""
        db_url = f"sqlite:///{tmp_path}/test_efrisk.db"
        from src.data_pipeline import normalize_users, normalize_transactions
        users = normalize_users(raw_users_df)
        txns = normalize_transactions(raw_txns_df)

        store_to_sqlite(users, txns, db_url=db_url)

        result = query_sqlite("SELECT * FROM users", db_url=db_url)
        assert len(result) == len(users)

    def test_store_and_retrieve_transactions(self, raw_users_df, raw_txns_df, tmp_path):
        db_url = f"sqlite:///{tmp_path}/test_efrisk2.db"
        from src.data_pipeline import normalize_users, normalize_transactions
        users = normalize_users(raw_users_df)
        txns = normalize_transactions(raw_txns_df)

        store_to_sqlite(users, txns, db_url=db_url)

        result = query_sqlite("SELECT * FROM transactions", db_url=db_url)
        assert len(result) == len(txns)

    def test_user_columns_preserved(self, raw_users_df, raw_txns_df, tmp_path):
        db_url = f"sqlite:///{tmp_path}/test_efrisk3.db"
        from src.data_pipeline import normalize_users, normalize_transactions
        users = normalize_users(raw_users_df)
        txns = normalize_transactions(raw_txns_df)
        store_to_sqlite(users, txns, db_url=db_url)

        result = query_sqlite("SELECT * FROM users LIMIT 1", db_url=db_url)
        assert "user_id" in result.columns
        assert "country" in result.columns
