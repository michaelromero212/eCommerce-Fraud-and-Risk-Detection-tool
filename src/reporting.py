"""
src/reporting.py
----------------
Report generation module for EFRiskEngine.

Produces:
  - CSV and JSON exports of flagged transactions and high-risk users.
  - Summary statistics (aggregate metrics for dashboard consumption).
  - A standalone HTML summary report.

Typical usage:
    from src.reporting import run_reporting

    run_reporting(scored_txns, user_summaries, output_dir="reports/")
"""

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPORTS_DIR = os.path.join(_ROOT, "reports")


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _now_str() -> str:
    """Return current UTC timestamp as a filename-safe string."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Individual report generators
# ---------------------------------------------------------------------------

def generate_transaction_report(
    scored_txns: pd.DataFrame,
    output_dir: str = DEFAULT_REPORTS_DIR,
    min_risk_score: int = 25,
    fmt: str = "csv",
) -> str:
    """
    Export flagged transactions to CSV or JSON.

    Filters to transactions with risk_score >= *min_risk_score* (Medium+).

    Args:
        scored_txns:    Output DataFrame from :func:`risk_engine.run_risk_engine`.
        output_dir:     Directory to write the report file.
        min_risk_score: Minimum risk score to include (default 25 = Medium+).
        fmt:            Output format: "csv" or "json".

    Returns:
        Absolute path of the written file.
    """
    _ensure_dir(output_dir)
    flagged = scored_txns[scored_txns["risk_score"] >= min_risk_score].copy()
    flagged = flagged.sort_values("risk_score", ascending=False)

    ts = _now_str()
    ext = "json" if fmt == "json" else "csv"
    filename = f"flagged_transactions_{ts}.{ext}"
    path = os.path.join(output_dir, filename)

    if fmt == "json":
        # Convert timestamps for JSON serialisation
        flagged["timestamp"] = flagged["timestamp"].astype(str)
        flagged.to_json(path, orient="records", indent=2)
    else:
        flagged.to_csv(path, index=False)

    logger.info(
        "Transaction report written: %d rows → %s", len(flagged), path
    )
    return path


def generate_user_report(
    user_summaries: pd.DataFrame,
    output_dir: str = DEFAULT_REPORTS_DIR,
    min_risk_score: int = 25,
    fmt: str = "csv",
) -> str:
    """
    Export high-risk users to CSV or JSON.

    Args:
        user_summaries: Output DataFrame from :func:`risk_engine.run_risk_engine`.
        output_dir:     Directory to write the report file.
        min_risk_score: Minimum user risk score to include.
        fmt:            Output format: "csv" or "json".

    Returns:
        Absolute path of the written file.
    """
    _ensure_dir(output_dir)
    high_risk = user_summaries[user_summaries["user_risk_score"] >= min_risk_score].copy()
    high_risk = high_risk.sort_values("user_risk_score", ascending=False)

    ts = _now_str()
    ext = "json" if fmt == "json" else "csv"
    filename = f"high_risk_users_{ts}.{ext}"
    path = os.path.join(output_dir, filename)

    if fmt == "json":
        high_risk.to_json(path, orient="records", indent=2)
    else:
        high_risk.to_csv(path, index=False)

    logger.info(
        "User report written: %d rows → %s", len(high_risk), path
    )
    return path


def summary_stats(
    scored_txns: pd.DataFrame,
    user_summaries: pd.DataFrame,
) -> dict:
    """
    Compute aggregate fraud metrics for the current dataset.

    Args:
        scored_txns:   Transaction-level scored DataFrame.
        user_summaries: User-level risk summary DataFrame.

    Returns:
        Dict containing key metrics suitable for dashboard consumption or logging.

    Metrics computed:
      - total_transactions
      - flagged_transactions (risk_score >= 25)
      - flag_rate_pct
      - avg_risk_score
      - critical_transactions (risk_score >= 75)
      - high_risk_users (user_risk_score >= 50)
      - top_countries_by_flag (top 5)
      - top_devices_by_flag (top 3)
      - total_flagged_amount
    """
    flagged = scored_txns[scored_txns["risk_score"] >= 25]
    critical = scored_txns[scored_txns["risk_score"] >= 75]
    high_risk_users = user_summaries[user_summaries["user_risk_score"] >= 50]

    # Top countries
    if "transaction_country" in flagged.columns:
        top_countries = (
            flagged["transaction_country"]
            .value_counts()
            .head(5)
            .to_dict()
        )
    else:
        top_countries = {}

    # Top devices
    if "device_type" in flagged.columns:
        top_devices = (
            flagged["device_type"]
            .value_counts()
            .head(3)
            .to_dict()
        )
    else:
        top_devices = {}

    total = len(scored_txns)
    flagged_count = len(flagged)

    stats = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "total_transactions": total,
        "flagged_transactions": flagged_count,
        "flag_rate_pct": round(flagged_count / total * 100, 2) if total else 0,
        "avg_risk_score": round(scored_txns["risk_score"].mean(), 2) if total else 0,
        "critical_transactions": len(critical),
        "high_risk_users": len(high_risk_users),
        "top_countries_by_flag": top_countries,
        "top_devices_by_flag": top_devices,
        "total_flagged_amount": round(
            flagged["purchase_amount"].sum() if "purchase_amount" in flagged.columns else 0.0, 2
        ),
    }
    return stats


def generate_summary_report(
    scored_txns: pd.DataFrame,
    user_summaries: pd.DataFrame,
    output_dir: str = DEFAULT_REPORTS_DIR,
) -> str:
    """
    Write a JSON summary metrics file and a human-readable HTML snapshot.

    Args:
        scored_txns:    Transaction-level scored DataFrame.
        user_summaries: User-level risk summary DataFrame.
        output_dir:     Output directory path.

    Returns:
        Absolute path of the written JSON file.
    """
    _ensure_dir(output_dir)
    stats = summary_stats(scored_txns, user_summaries)

    ts = _now_str()
    json_path = os.path.join(output_dir, f"summary_{ts}.json")
    with open(json_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    # Simple HTML snapshot
    html = _render_html_summary(stats, scored_txns, user_summaries)
    html_path = os.path.join(output_dir, f"summary_{ts}.html")
    with open(html_path, "w") as fh:
        fh.write(html)

    logger.info("Summary report written → %s", json_path)
    logger.info("HTML snapshot written  → %s", html_path)
    return json_path


def _render_html_summary(
    stats: dict,
    scored_txns: pd.DataFrame,
    user_summaries: pd.DataFrame,
) -> str:
    """Render a minimal self-contained HTML report page."""
    top_txns = (
        scored_txns.sort_values("risk_score", ascending=False)
        .head(10)
        [["transaction_id", "user_id", "purchase_amount", "risk_score", "risk_label", "reasons"]]
        .to_html(index=False, border=0, classes="table")
    )
    top_users = (
        user_summaries.head(10)
        [["user_id", "username", "country", "user_risk_score", "user_risk_label",
          "flagged_txn_count", "failed_login_count"]]
        .to_html(index=False, border=0, classes="table")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EFRiskEngine — Fraud Summary Report</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ color: #f97316; }}
  h2 {{ color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 0.4rem; }}
  .metrics {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; min-width: 160px; }}
  .card .value {{ font-size: 2rem; font-weight: bold; color: #f97316; }}
  .card .label {{ font-size: 0.8rem; color: #94a3b8; }}
  .table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .table th {{ background: #334155; padding: 0.5rem; text-align: left; }}
  .table td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid #1e293b; }}
  .table tr:hover td {{ background: #1e293b; }}
  small {{ color: #64748b; }}
</style>
</head>
<body>
<h1>🛡 EFRiskEngine — Fraud & Risk Summary</h1>
<small>Generated: {stats['generated_at']}</small>

<div class="metrics">
  <div class="card"><div class="value">{stats['total_transactions']}</div><div class="label">Total Transactions</div></div>
  <div class="card"><div class="value">{stats['flagged_transactions']}</div><div class="label">Flagged (Medium+)</div></div>
  <div class="card"><div class="value">{stats['flag_rate_pct']}%</div><div class="label">Flag Rate</div></div>
  <div class="card"><div class="value">{stats['avg_risk_score']}</div><div class="label">Avg Risk Score</div></div>
  <div class="card"><div class="value">{stats['critical_transactions']}</div><div class="label">Critical Transactions</div></div>
  <div class="card"><div class="value">{stats['high_risk_users']}</div><div class="label">High-Risk Users</div></div>
  <div class="card"><div class="value">${stats['total_flagged_amount']:,.2f}</div><div class="label">Total Flagged Amount</div></div>
</div>

<h2>Top 10 Highest-Risk Transactions</h2>
{top_txns}

<h2>Top 10 Highest-Risk Users</h2>
{top_users}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Reporting orchestrator
# ---------------------------------------------------------------------------

def run_reporting(
    scored_txns: pd.DataFrame,
    user_summaries: pd.DataFrame,
    output_dir: str = DEFAULT_REPORTS_DIR,
    formats: list[str] | None = None,
) -> dict[str, str]:
    """
    Run the full reporting pipeline: export transactions, users, and summary.

    Args:
        scored_txns:    Transaction-level scored output from risk engine.
        user_summaries: User-level risk summary from risk engine.
        output_dir:     Directory to write all report files.
        formats:        List of output formats, e.g. ["csv", "json"].
                        Defaults to ["csv", "json"].

    Returns:
        Dict mapping report_name → absolute file path for each report written.
    """
    if formats is None:
        formats = ["csv", "json"]

    logger.info("=== Reporting Pipeline START ===")
    _ensure_dir(output_dir)
    paths: dict[str, str] = {}

    for fmt in formats:
        paths[f"transactions_{fmt}"] = generate_transaction_report(
            scored_txns, output_dir, fmt=fmt
        )
        paths[f"users_{fmt}"] = generate_user_report(
            user_summaries, output_dir, fmt=fmt
        )

    paths["summary_json"] = generate_summary_report(scored_txns, user_summaries, output_dir)

    logger.info("=== Reporting Pipeline COMPLETE — %d files written ===", len(paths))
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_pipeline import run_etl
    from src.risk_engine import run_risk_engine

    users_df, txns_df = run_etl()
    scored_txns, user_summaries = run_risk_engine(txns_df, users_df)
    paths = run_reporting(scored_txns, user_summaries)

    print("\nReports generated:")
    for name, path in paths.items():
        print(f"  {name:30s} → {path}")

    stats = summary_stats(scored_txns, user_summaries)
    print(f"\nSummary: {stats['flagged_transactions']} / {stats['total_transactions']} "
          f"transactions flagged ({stats['flag_rate_pct']}%)")
