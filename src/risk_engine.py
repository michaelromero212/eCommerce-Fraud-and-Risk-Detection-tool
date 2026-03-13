"""
src/risk_engine.py
------------------
Rules-based fraud risk scoring engine for EFRiskEngine.

Each rule function returns a numeric score contribution and a human-readable
reason string. Scores are additive; the final risk score is clamped to [0, 100].

Risk score bands:
  0–24   → Low
  25–49  → Medium
  50–74  → High
  75–100 → Critical

Typical usage:
    from src.risk_engine import run_risk_engine
    from src.data_pipeline import run_etl

    users, txns = run_etl()
    scored_txns, user_summaries = run_risk_engine(txns, users)
"""

import logging
from datetime import timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk thresholds (easily tunable)
# ---------------------------------------------------------------------------
FAILED_LOGIN_THRESHOLD = 3          # ≥ this many failures triggers flag
FAILED_LOGIN_SCORE = 30             # points added to user risk

HIGH_VALUE_THRESHOLD = 500.0        # USD — above this is flagged
HIGH_VALUE_SCORE = 25               # points added when high-value + new device

COUNTRY_MISMATCH_SCORE = 20         # points added on country mismatch

VELOCITY_WINDOW_HOURS = 1           # rolling window for velocity check
VELOCITY_MAX_TRANSACTIONS = 3       # > this many txns in window → flag
VELOCITY_SCORE = 20                 # points added per velocity spike

BOT_ROUND_AMOUNT_SCORE = 10         # extra points for round-number amounts
BOT_BROWSER_KEYWORDS = {"bot", "crawler", "spider", "headless"}

ATO_RECENT_CHANGE_HOURS = 24        # hours since password reset considered risky
ATO_PROFILE_CHANGE_SCORE = 35       # points added for high-value txn + recent pw reset

THREAT_INTEL_RISKY_ASNS = {"AS14061", "AS200000", "AS210000"}  # mock threat intel feed
THREAT_INTEL_ASN_SCORE = 40         # massive penalty for known bad ASNs

RISK_LABEL_BANDS = [
    (75, "Critical"),
    (50, "High"),
    (25, "Medium"),
    (0,  "Low"),
]


def _risk_label(score: float) -> str:
    """Convert a numeric risk score to a human-readable band label."""
    for threshold, label in RISK_LABEL_BANDS:
        if score >= threshold:
            return label
    return "Low"


# ---------------------------------------------------------------------------
# Rule functions — one per fraud signal
# ---------------------------------------------------------------------------

def score_failed_logins(user_row: pd.Series) -> tuple[int, str]:
    """
    Flag users with an excessive number of failed login attempts.

    Rule: failed_login_count >= FAILED_LOGIN_THRESHOLD → +FAILED_LOGIN_SCORE pts.

    Args:
        user_row: A single row from the users DataFrame.

    Returns:
        (score_delta, reason) where reason is empty string when no flag.
    """
    count = int(user_row.get("failed_login_count", 0))
    if count >= FAILED_LOGIN_THRESHOLD:
        reason = f"Excessive failed logins: {count} (threshold={FAILED_LOGIN_THRESHOLD})"
        return FAILED_LOGIN_SCORE, reason
    return 0, ""


def score_high_value_new_device(
    txn_row: pd.Series,
    user_row: Optional[pd.Series] = None,
) -> tuple[int, str]:
    """
    Flag high-value transactions originating from a device not registered to the user.

    Rule: purchase_amount > HIGH_VALUE_THRESHOLD AND
          txn device_fingerprint ≠ user registered device_fingerprint
          → +HIGH_VALUE_SCORE pts.

    Args:
        txn_row:  A single row from the transactions DataFrame.
        user_row: Corresponding user row (optional; skips device check if None).

    Returns:
        (score_delta, reason).
    """
    amount = float(txn_row.get("purchase_amount", 0))
    if amount <= HIGH_VALUE_THRESHOLD:
        return 0, ""

    # If we have user data, compare device fingerprints
    if user_row is not None:
        txn_device = str(txn_row.get("device_fingerprint", ""))
        usr_device = str(user_row.get("device_fingerprint", ""))
        if txn_device and usr_device and txn_device != usr_device:
            reason = (
                f"High-value txn (${amount:.2f}) from unrecognised device "
                f"(registered={usr_device[:8]}…, txn={txn_device[:8]}…)"
            )
            return HIGH_VALUE_SCORE, reason
    else:
        # No user data — flag purely on amount
        reason = f"High-value txn (${amount:.2f}) without device verification"
        return HIGH_VALUE_SCORE, reason

    return 0, ""


def score_country_mismatch(txn_row: pd.Series) -> tuple[int, str]:
    """
    Flag transactions where the transaction country differs from the user's
    registered country, which may indicate stolen credentials or VPN abuse.

    Rule: transaction_country ≠ user_registered_country → +COUNTRY_MISMATCH_SCORE pts.

    Args:
        txn_row: A single row from the transactions DataFrame.

    Returns:
        (score_delta, reason).
    """
    txn_country = str(txn_row.get("transaction_country", "")).strip().upper()
    usr_country = str(txn_row.get("user_registered_country", "")).strip().upper()

    if txn_country and usr_country and txn_country != usr_country:
        reason = (
            f"Country mismatch: transaction from {txn_country!r}, "
            f"user registered in {usr_country!r}"
        )
        return COUNTRY_MISMATCH_SCORE, reason
    return 0, ""


def score_velocity(
    txn_row: pd.Series,
    all_txns: pd.DataFrame,
) -> tuple[int, str]:
    """
    Flag transaction velocity spikes: many transactions for the same user
    within a short rolling window.

    Rule: count of txns for same user within ±VELOCITY_WINDOW_HOURS
          > VELOCITY_MAX_TRANSACTIONS → +VELOCITY_SCORE pts.

    Args:
        txn_row:  The transaction being evaluated.
        all_txns: Full transactions DataFrame (needed for context lookup).

    Returns:
        (score_delta, reason).
    """
    uid = txn_row.get("user_id")
    ts = txn_row.get("timestamp")

    if uid is None or ts is None or pd.isna(ts):
        return 0, ""

    # Filter to same user
    user_txns = all_txns[all_txns["user_id"] == uid]

    # Ensure timestamps are datetime
    timestamps = pd.to_datetime(user_txns["timestamp"], errors="coerce")
    ts_dt = pd.to_datetime(ts)

    window_start = ts_dt - timedelta(hours=VELOCITY_WINDOW_HOURS)
    window_end   = ts_dt + timedelta(hours=VELOCITY_WINDOW_HOURS)

    in_window = ((timestamps >= window_start) & (timestamps <= window_end)).sum()

    if in_window > VELOCITY_MAX_TRANSACTIONS:
        reason = (
            f"Velocity spike: {in_window} transactions in "
            f"{VELOCITY_WINDOW_HOURS * 2}h window (max={VELOCITY_MAX_TRANSACTIONS})"
        )
        return VELOCITY_SCORE, reason
    return 0, ""


def score_bot_behaviour(txn_row: pd.Series) -> tuple[int, str]:
    """
    Flag bot-like behaviour signals: known bot browsers and round-number
    purchase amounts that suggest automated purchasing.

    Args:
        txn_row: A single row from the transactions DataFrame.

    Returns:
        (score_delta, reason).
    """
    score = 0
    reasons = []

    # Check browser string
    browser = str(txn_row.get("browser", "")).lower()
    if any(kw in browser for kw in BOT_BROWSER_KEYWORDS):
        score += BOT_ROUND_AMOUNT_SCORE
        reasons.append(f"Bot-like browser: {browser!r}")

    # Check round-number amount
    amount = txn_row.get("purchase_amount", 0)
    try:
        if float(amount) % 100 == 0 and float(amount) > 0:
            score += BOT_ROUND_AMOUNT_SCORE
            reasons.append(f"Round purchase amount: ${amount:.2f}")
    except (TypeError, ValueError):
        pass

    return score, "; ".join(reasons)


def score_ato_profile_change(txn_row: pd.Series, user_row: Optional[pd.Series] = None) -> tuple[int, str]:
    """
    Flag Account Takeover (ATO) attempts: high-value transactions occurring
    shortly after a sensitive profile modification (e.g., password reset).

    Rule: purchase_amount > HIGH_VALUE_THRESHOLD AND
          recent_password_reset_hours <= ATO_RECENT_CHANGE_HOURS
          → +ATO_PROFILE_CHANGE_SCORE pts.
    """
    if user_row is None:
        return 0, ""

    amount = float(txn_row.get("purchase_amount", 0))
    if amount <= HIGH_VALUE_THRESHOLD:
        return 0, ""

    hours_since_reset = float(user_row.get("recent_password_reset_hours", 9999))
    if hours_since_reset <= ATO_RECENT_CHANGE_HOURS:
        reason = (
            f"Possible ATO: High-value txn (${amount:.2f}) only "
            f"{hours_since_reset}h after password reset"
        )
        return ATO_PROFILE_CHANGE_SCORE, reason
    return 0, ""


def score_threat_intel_match(txn_row: pd.Series) -> tuple[int, str]:
    """
    Simulate Threat Intelligence feed integration. Flag transactions originating
    from known risky Autonomous System Numbers (ASNs) like Tor exits or bulletproof hosts.

    Rule: txn ip_asn IN THREAT_INTEL_RISKY_ASNS → +THREAT_INTEL_ASN_SCORE pts.
    """
    asn = str(txn_row.get("ip_asn", "")).strip()
    if asn in THREAT_INTEL_RISKY_ASNS:
        reason = f"Threat Intel Match: Transaction from known risky ASN {asn!r}"
        return THREAT_INTEL_ASN_SCORE, reason
    return 0, ""

# ---------------------------------------------------------------------------
# Per-transaction scoring
# ---------------------------------------------------------------------------

def compute_transaction_risk(
    txn_row: pd.Series,
    all_txns: pd.DataFrame,
    user_lookup: dict[str, pd.Series],
) -> dict:
    """
    Compute total risk score and reasons for a single transaction.

    Applies all rule functions and accumulates their scores.

    Args:
        txn_row:     Single transaction row.
        all_txns:    Full transactions DataFrame (for velocity context).
        user_lookup: Dict mapping user_id → user row Series.

    Returns:
        Dict with keys: transaction_id, user_id, purchase_amount,
        risk_score, risk_label, reasons.
    """
    uid = txn_row.get("user_id", "")
    user_row = user_lookup.get(uid)

    total_score = 0
    all_reasons = []

    # Rule 1: Failed logins (user-level signal applied to each of their txns)
    if user_row is not None:
        s, r = score_failed_logins(user_row)
        total_score += s
        if r:
            all_reasons.append(r)

    # Rule 2: High-value + new device
    s, r = score_high_value_new_device(txn_row, user_row)
    total_score += s
    if r:
        all_reasons.append(r)

    # Rule 3: Country mismatch
    s, r = score_country_mismatch(txn_row)
    total_score += s
    if r:
        all_reasons.append(r)

    # Rule 4: Velocity spike
    s, r = score_velocity(txn_row, all_txns)
    total_score += s
    if r:
        all_reasons.append(r)

    # Rule 5: Bot-like behaviour
    s, r = score_bot_behaviour(txn_row)
    total_score += s
    if r:
        all_reasons.append(r)

    # Rule 6: ATO Profile Modification
    s, r = score_ato_profile_change(txn_row, user_row)
    total_score += s
    if r:
        all_reasons.append(r)

    # Rule 7: Threat Intel ASN Match
    s, r = score_threat_intel_match(txn_row)
    total_score += s
    if r:
        all_reasons.append(r)

    # Clamp score to [0, 100]
    total_score = max(0, min(100, total_score))

    return {
        "transaction_id": txn_row.get("transaction_id"),
        "user_id": uid,
        "timestamp": txn_row.get("timestamp"),
        "purchase_amount": txn_row.get("purchase_amount"),
        "payment_method": txn_row.get("payment_method"),
        "transaction_country": txn_row.get("transaction_country"),
        "device_type": txn_row.get("device_type"),
        "risk_score": total_score,
        "risk_label": _risk_label(total_score),
        "reasons": " | ".join(all_reasons) if all_reasons else "No flags",
    }


# ---------------------------------------------------------------------------
# Per-user risk roll-up
# ---------------------------------------------------------------------------

def compute_user_risk(
    user_row: pd.Series,
    scored_txns: pd.DataFrame,
) -> dict:
    """
    Aggregate transaction-level risk scores into a user-level risk summary.

    Metric: max risk score across all user transactions + standalone
    failed-login score.

    Args:
        user_row:    Single user row.
        scored_txns: DataFrame output of :func:`run_risk_engine` (transaction-level).

    Returns:
        Dict with keys: user_id, max_txn_risk, avg_txn_risk,
        flagged_txn_count, failed_login_count, user_risk_score, user_risk_label.
    """
    uid = user_row.get("user_id", "")
    user_txns = scored_txns[scored_txns["user_id"] == uid]

    if user_txns.empty:
        max_risk = avg_risk = flagged_count = 0
    else:
        max_risk = user_txns["risk_score"].max()
        avg_risk = round(user_txns["risk_score"].mean(), 2)
        flagged_count = int((user_txns["risk_score"] >= 25).sum())

    login_score, _ = score_failed_logins(user_row)

    # User-level score = max of their transaction risk and login score
    user_score = max(int(max_risk), login_score)
    user_score = max(0, min(100, user_score))

    return {
        "user_id": uid,
        "username": user_row.get("username", ""),
        "country": user_row.get("country", ""),
        "failed_login_count": user_row.get("failed_login_count", 0),
        "account_age_days": user_row.get("account_age_days", 0),
        "max_txn_risk": int(max_risk),
        "avg_txn_risk": avg_risk,
        "flagged_txn_count": flagged_count,
        "user_risk_score": user_score,
        "user_risk_label": _risk_label(user_score),
    }


# ---------------------------------------------------------------------------
# Engine Orchestrator
# ---------------------------------------------------------------------------

def run_risk_engine(
    txns_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full rules-based risk engine over all transactions and users.

    Steps:
      1. Build a user_id → user_row lookup dict for O(1) access.
      2. Score each transaction using all five rule functions.
      3. Roll up transaction scores to a per-user risk summary.

    Args:
        txns_df:  Normalised transactions DataFrame.
        users_df: Normalised users DataFrame.

    Returns:
        Tuple of:
          - scored_txns  (DataFrame): One row per transaction with risk_score,
            risk_label, and reasons columns appended.
          - user_summaries (DataFrame): One row per user with aggregated
            risk metrics.
    """
    logger.info("=== Risk Engine START — %d transactions, %d users ===",
                len(txns_df), len(users_df))

    # Build lookup for fast user access
    user_lookup: dict[str, pd.Series] = {
        row["user_id"]: row
        for _, row in users_df.iterrows()
    }

    # Score each transaction
    scored_rows = []
    for _, txn_row in txns_df.iterrows():
        result = compute_transaction_risk(txn_row, txns_df, user_lookup)
        scored_rows.append(result)

    scored_txns = pd.DataFrame(scored_rows)
    logger.info("Transaction scoring complete. High-risk: %d",
                (scored_txns["risk_score"] >= 50).sum())

    # Roll up per user
    user_rows = []
    for _, user_row in users_df.iterrows():
        summary = compute_user_risk(user_row, scored_txns)
        user_rows.append(summary)

    user_summaries = pd.DataFrame(user_rows).sort_values(
        "user_risk_score", ascending=False
    )
    logger.info("User risk roll-up complete. High-risk users: %d",
                (user_summaries["user_risk_score"] >= 50).sum())

    logger.info("=== Risk Engine COMPLETE ===")
    return scored_txns, user_summaries


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_pipeline import run_etl

    users_df, txns_df = run_etl()
    scored_txns, user_summaries = run_risk_engine(txns_df, users_df)

    print("\nTop 5 highest-risk transactions:")
    print(
        scored_txns.sort_values("risk_score", ascending=False)
        .head(5)[["transaction_id", "user_id", "purchase_amount", "risk_score",
                  "risk_label", "reasons"]]
        .to_string(index=False)
    )

    print("\nTop 5 highest-risk users:")
    print(
        user_summaries.head(5)[["user_id", "username", "user_risk_score",
                                "user_risk_label", "flagged_txn_count"]]
        .to_string(index=False)
    )
