"""
tests/test_risk_engine.py
--------------------------
Unit tests for src/risk_engine.py

Run with:
    pytest tests/test_risk_engine.py -v
"""

import pandas as pd
import pytest

from src.risk_engine import (
    FAILED_LOGIN_THRESHOLD,
    HIGH_VALUE_THRESHOLD,
    compute_transaction_risk,
    compute_user_risk,
    run_risk_engine,
    score_country_mismatch,
    score_failed_logins,
    score_high_value_new_device,
    score_velocity,
    score_ato_profile_change,
    score_threat_intel_match,
    _risk_label,
    ATO_RECENT_CHANGE_HOURS,
    THREAT_INTEL_RISKY_ASNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clean_user() -> pd.Series:
    """A low-risk user with no suspicious attributes."""
    return pd.Series({
        "user_id": "U0001",
        "username": "alice",
        "country": "US",
        "account_age_days": 365,
        "device_fingerprint": "aabbccdd",
        "failed_login_count": 0,
        "password_resets": 0,
        "is_fraudulent": 0,
    })


@pytest.fixture()
def fraud_user() -> pd.Series:
    """A high-risk user who has exceeded failed login threshold."""
    return pd.Series({
        "user_id": "U0002",
        "username": "bob",
        "country": "US",
        "account_age_days": 3,
        "device_fingerprint": "deadbeef",
        "failed_login_count": FAILED_LOGIN_THRESHOLD + 2,
        "password_resets": 4,
        "is_fraudulent": 1,
    })


@pytest.fixture()
def normal_txn() -> pd.Series:
    """A normal, low-risk transaction."""
    return pd.Series({
        "transaction_id": "T00001",
        "user_id": "U0001",
        "timestamp": pd.Timestamp("2025-06-01 12:00:00"),
        "ip_address": "192.168.1.1",
        "device_fingerprint": "aabbccdd",
        "purchase_amount": 49.99,
        "payment_method": "credit_card",
        "transaction_country": "US",
        "user_registered_country": "US",
        "device_type": "desktop",
        "browser": "Chrome",
        "is_flagged": 0,
    })


@pytest.fixture()
def high_value_txn() -> pd.Series:
    """A high-value transaction from an unrecognised device."""
    return pd.Series({
        "transaction_id": "T00002",
        "user_id": "U0002",
        "timestamp": pd.Timestamp("2025-06-01 14:00:00"),
        "ip_address": "10.0.0.1",
        "device_fingerprint": "11223344",  # different from fraud_user's device
        "purchase_amount": HIGH_VALUE_THRESHOLD + 100,
        "payment_method": "crypto",
        "transaction_country": "NG",
        "user_registered_country": "US",
        "device_type": "mobile",
        "browser": "Chrome",
        "is_flagged": 1,
    })


@pytest.fixture()
def small_txns_df(normal_txn, high_value_txn) -> pd.DataFrame:
    """Small DataFrame of two transactions for integration tests."""
    return pd.DataFrame([normal_txn, high_value_txn])


@pytest.fixture()
def small_users_df(clean_user, fraud_user) -> pd.DataFrame:
    """Small DataFrame of two users for integration tests."""
    return pd.DataFrame([clean_user, fraud_user])


# ---------------------------------------------------------------------------
# score_failed_logins
# ---------------------------------------------------------------------------

class TestScoreFailedLogins:
    def test_no_flag_below_threshold(self, clean_user):
        """User with 0 failed logins should not be flagged."""
        score, reason = score_failed_logins(clean_user)
        assert score == 0
        assert reason == ""

    def test_flag_at_threshold(self, fraud_user):
        """User at or above threshold should receive FAILED_LOGIN_SCORE."""
        score, reason = score_failed_logins(fraud_user)
        assert score > 0
        assert "failed login" in reason.lower()

    def test_flag_exact_threshold(self):
        """User with exactly FAILED_LOGIN_THRESHOLD failed logins should be flagged."""
        user = pd.Series({"failed_login_count": FAILED_LOGIN_THRESHOLD})
        score, reason = score_failed_logins(user)
        assert score > 0

    def test_missing_field_defaults_to_zero(self):
        """Missing failed_login_count field should not raise and should return 0."""
        user = pd.Series({})
        score, reason = score_failed_logins(user)
        assert score == 0


# ---------------------------------------------------------------------------
# score_high_value_new_device
# ---------------------------------------------------------------------------

class TestScoreHighValueNewDevice:
    def test_no_flag_low_amount(self, normal_txn, clean_user):
        """Low-value transaction should not be flagged."""
        score, reason = score_high_value_new_device(normal_txn, clean_user)
        assert score == 0

    def test_flag_high_value_different_device(self, high_value_txn, fraud_user):
        """High-value txn with a different device fingerprint should be flagged."""
        score, reason = score_high_value_new_device(high_value_txn, fraud_user)
        assert score > 0
        assert "device" in reason.lower()

    def test_no_flag_same_device(self, fraud_user):
        """High-value txn with the SAME device fingerprint should NOT be flagged."""
        txn = pd.Series({
            "purchase_amount": HIGH_VALUE_THRESHOLD + 50,
            "device_fingerprint": fraud_user["device_fingerprint"],  # same device
        })
        score, reason = score_high_value_new_device(txn, fraud_user)
        assert score == 0

    def test_flag_without_user_data(self):
        """High-value txn with no user reference should still flag."""
        txn = pd.Series({"purchase_amount": HIGH_VALUE_THRESHOLD + 1})
        score, reason = score_high_value_new_device(txn, user_row=None)
        assert score > 0


# ---------------------------------------------------------------------------
# score_country_mismatch
# ---------------------------------------------------------------------------

class TestScoreCountryMismatch:
    def test_no_mismatch(self, normal_txn):
        """Same countries → no flag."""
        score, reason = score_country_mismatch(normal_txn)
        assert score == 0

    def test_mismatch_flagged(self, high_value_txn):
        """Different txn and registered country → flag."""
        score, reason = score_country_mismatch(high_value_txn)
        assert score > 0
        assert "mismatch" in reason.lower()

    def test_empty_countries_no_flag(self):
        """Missing country fields should not produce a false positive."""
        txn = pd.Series({"transaction_country": "", "user_registered_country": ""})
        score, reason = score_country_mismatch(txn)
        assert score == 0


# ---------------------------------------------------------------------------
# score_velocity
# ---------------------------------------------------------------------------

class TestScoreVelocity:
    def test_no_velocity_spike(self, normal_txn, small_txns_df):
        """Single transaction should not trigger velocity spike."""
        single_txn_df = pd.DataFrame([normal_txn])
        score, reason = score_velocity(normal_txn, single_txn_df)
        assert score == 0

    def test_velocity_spike_detected(self):
        """Multiple transactions in a short window → velocity flag."""
        base_ts = pd.Timestamp("2025-01-01 10:00:00")
        # Create 5 transactions within 30 minutes for the same user
        txns = pd.DataFrame([
            {
                "user_id": "U9999",
                "transaction_id": f"T{i}",
                "timestamp": base_ts + pd.Timedelta(minutes=i * 5),
                "risk_score": 0,
            }
            for i in range(5)
        ])
        target_txn = txns.iloc[2]
        score, reason = score_velocity(target_txn, txns)
        assert score > 0
        assert "velocity" in reason.lower()

# ---------------------------------------------------------------------------
# score_ato_profile_change
# ---------------------------------------------------------------------------

class TestScoreATOProfileChange:
    def test_no_flag_if_no_user(self, high_value_txn):
        """Should safely ignore if user data is missing."""
        score, reason = score_ato_profile_change(high_value_txn, None)
        assert score == 0

    def test_no_flag_low_value_txn(self, normal_txn, fraud_user):
        """Even if recent password reset, low value txn is ignored by this specific rule."""
        user = fraud_user.copy()
        user["recent_password_reset_hours"] = 2
        score, reason = score_ato_profile_change(normal_txn, user)
        assert score == 0

    def test_flag_high_value_after_reset(self, high_value_txn, fraud_user):
        """High-value txn shortly after a password reset should flag as ATO."""
        user = fraud_user.copy()
        user["recent_password_reset_hours"] = ATO_RECENT_CHANGE_HOURS - 1
        score, reason = score_ato_profile_change(high_value_txn, user)
        assert score > 0
        assert "ato" in reason.lower()

    def test_no_flag_long_after_reset(self, high_value_txn, fraud_user):
        """High-value txn long after password reset should not trigger ATO rule."""
        user = fraud_user.copy()
        user["recent_password_reset_hours"] = ATO_RECENT_CHANGE_HOURS + 100
        score, reason = score_ato_profile_change(high_value_txn, user)
        assert score == 0

# ---------------------------------------------------------------------------
# score_threat_intel_match
# ---------------------------------------------------------------------------

class TestScoreThreatIntelMatch:
    def test_no_flag_benign_asn(self, normal_txn):
        """Normal ASN should not flag."""
        txn = normal_txn.copy()
        txn["ip_asn"] = "AS12345"
        score, reason = score_threat_intel_match(txn)
        assert score == 0

    def test_flag_risky_asn(self, normal_txn):
        """Transaction from a known risky ASN should flag heavily."""
        txn = normal_txn.copy()
        txn["ip_asn"] = list(THREAT_INTEL_RISKY_ASNS)[0]
        score, reason = score_threat_intel_match(txn)
        assert score > 0
        assert "threat intel match" in reason.lower()

    def test_missing_asn(self, normal_txn):
        """Missing ASN should be safely ignored."""
        txn = normal_txn.copy()
        txn["ip_asn"] = ""
        score, reason = score_threat_intel_match(txn)
        assert score == 0


# ---------------------------------------------------------------------------
# compute_transaction_risk
# ---------------------------------------------------------------------------

class TestComputeTransactionRisk:
    def test_returns_expected_keys(self, high_value_txn, small_txns_df, fraud_user):
        user_lookup = {"U0002": fraud_user}
        result = compute_transaction_risk(high_value_txn, small_txns_df, user_lookup)
        assert "risk_score" in result
        assert "risk_label" in result
        assert "reasons" in result

    def test_score_clamped_to_100(self, high_value_txn, small_txns_df, fraud_user):
        user_lookup = {"U0002": fraud_user}
        result = compute_transaction_risk(high_value_txn, small_txns_df, user_lookup)
        assert 0 <= result["risk_score"] <= 100


# ---------------------------------------------------------------------------
# compute_user_risk
# ---------------------------------------------------------------------------

class TestComputeUserRisk:
    def test_returns_expected_keys(self, fraud_user, small_txns_df):
        scored_txns = small_txns_df.copy()
        scored_txns["risk_score"] = [10, 75]
        result = compute_user_risk(fraud_user, scored_txns)
        assert "user_risk_score" in result
        assert "user_risk_label" in result
        assert "flagged_txn_count" in result

    def test_empty_transactions(self, clean_user):
        empty_df = pd.DataFrame(columns=["user_id", "risk_score"])
        result = compute_user_risk(clean_user, empty_df)
        assert result["max_txn_risk"] == 0


# ---------------------------------------------------------------------------
# run_risk_engine integration
# ---------------------------------------------------------------------------

class TestRunRiskEngine:
    def test_returns_dataframes(self, small_txns_df, small_users_df):
        """run_risk_engine should return two DataFrames."""
        scored, summaries = run_risk_engine(small_txns_df, small_users_df)
        assert isinstance(scored, pd.DataFrame)
        assert isinstance(summaries, pd.DataFrame)

    def test_scored_has_risk_columns(self, small_txns_df, small_users_df):
        scored, _ = run_risk_engine(small_txns_df, small_users_df)
        assert "risk_score" in scored.columns
        assert "risk_label" in scored.columns

    def test_user_summaries_sorted_descending(self, small_txns_df, small_users_df):
        _, summaries = run_risk_engine(small_txns_df, small_users_df)
        scores = summaries["user_risk_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# _risk_label
# ---------------------------------------------------------------------------

class TestRiskLabel:
    @pytest.mark.parametrize("score,expected", [
        (0, "Low"), (24, "Low"),
        (25, "Medium"), (49, "Medium"),
        (50, "High"), (74, "High"),
        (75, "Critical"), (100, "Critical"),
    ])
    def test_bands(self, score, expected):
        assert _risk_label(score) == expected
