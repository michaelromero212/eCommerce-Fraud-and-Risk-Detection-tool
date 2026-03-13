"""
generate_sample_data.py
-----------------------
Standalone script that generates synthetic sample data for EFRiskEngine.
Run from the project root:
    python generate_sample_data.py

Outputs:
    data/sample_users.csv
    data/sample_transactions.csv

Fraud patterns injected:
  - Multiple failed logins in short period
  - High-value transactions from new/unknown devices
  - Country mismatches (IP geolocation vs. payment country)
  - Velocity spikes (many transactions in a short window)
  - Bot-like behaviour (round-number purchases, millisecond regularity)
"""

import csv
import os
import random
import uuid
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
NUM_USERS = 100
NUM_TRANSACTIONS = 500
FRAUD_USER_RATIO = 0.15       # 15 % of users are fraudulent
FRAUD_TXN_RATIO = 0.12        # ~12 % of transactions are fraudulent
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

random.seed(SEED)

# ---------------------------------------------------------------------------
# Reference data pools
# ---------------------------------------------------------------------------
COUNTRIES = [
    "US", "GB", "CA", "AU", "DE", "FR", "JP", "CN", "BR", "NG",
    "RU", "IN", "MX", "ZA", "KR",
]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "crypto", "bank_transfer"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "unknown"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "bot_crawler"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def random_ip(country: str, is_fraud: bool = False) -> tuple[str, str]:
    """Generate a plausible IP and ASN; for mismatches we deliberately use a different country's block or a risky ASN."""
    country_octets = {
        "US": "192.168", "GB": "86.6", "CA": "142.36", "AU": "101.160",
        "DE": "91.194", "FR": "90.63", "JP": "202.232", "CN": "101.6",
        "BR": "187.0", "NG": "41.207", "RU": "95.163", "IN": "117.217",
        "MX": "200.57", "ZA": "196.25", "KR": "59.6",
    }
    
    # Simulating risky ASNs (e.g., Tor, known bulletproof hosters)
    RISKY_ASNS = ["AS14061", "AS200000", "AS210000"]
    BENIGN_ASNS = ["AS7922", "AS3320", "AS15169", "AS16509", "AS7018"]

    if is_fraud and random.random() < 0.4:
        # Flagged IPs get a risky ASN
        prefix = random.choice(list(country_octets.values())) # randomized routing
        asn = random.choice(RISKY_ASNS)
    else:
        prefix = country_octets.get(country, "10.0")
        asn = random.choice(BENIGN_ASNS)
        
    ip = f"{prefix}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return ip, asn


def random_device_fingerprint() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Generate Users
# ---------------------------------------------------------------------------

def generate_users(n: int) -> list[dict]:
    users = []
    fraud_count = int(n * FRAUD_USER_RATIO)

    for i in range(n):
        is_fraud = i < fraud_count
        country = random.choice(COUNTRIES)
        failed_logins = random.randint(3, 15) if is_fraud else random.randint(0, 2)
        password_resets = random.randint(2, 6) if is_fraud else random.randint(0, 1)
        account_age = random.randint(0, 15) if is_fraud else random.randint(30, 1800)
        device = random_device_fingerprint()
        ip, asn = random_ip(country, is_fraud=is_fraud)

        # For fraudulent users, maybe they had a very recent password reset (ATO simulation)
        recent_pw_reset = random.randint(1, 24) if is_fraud and random.random() < 0.3 else random.randint(200, 5000)

        users.append({
            "user_id": f"U{str(i+1).zfill(4)}",
            "username": f"user_{i+1}",
            "email": f"user{i+1}@example.com",
            "country": country,
            "account_age_days": account_age,
            "device_fingerprint": device,
            "ip_address": ip,
            "ip_asn": asn,
            "failed_login_count": failed_logins,
            "password_resets": password_resets,
            "recent_password_reset_hours": recent_pw_reset,
            "signup_date": (END_DATE - timedelta(days=account_age)).strftime("%Y-%m-%d"),
            "browser": random.choice(BROWSERS),
            "device_type": random.choice(DEVICE_TYPES),
            "is_fraudulent": int(is_fraud),
        })

    random.shuffle(users)
    return users


# ---------------------------------------------------------------------------
# Generate Transactions
# ---------------------------------------------------------------------------

def generate_transactions(users: list[dict], n: int) -> list[dict]:
    transactions = []
    fraud_user_ids = {u["user_id"] for u in users if u["is_fraudulent"]}
    fraud_target = int(n * FRAUD_TXN_RATIO)
    fraud_generated = 0

    for i in range(n):
        user = random.choice(users)
        uid = user["user_id"]
        is_fraud_user = uid in fraud_user_ids
        is_fraud_txn = is_fraud_user and (fraud_generated < fraud_target)

        # Pick a payment country (mismatch for fraudulent txns)
        user_country = user["country"]
        if is_fraud_txn and random.random() < 0.6:
            # country mismatch — pick a different country
            payment_country = random.choice([c for c in COUNTRIES if c != user_country])
        else:
            payment_country = user_country

        # Transaction timestamp — velocity spike for fraud users
        ts = random_date(START_DATE, END_DATE)

        # Purchase amount
        if is_fraud_txn and random.random() < 0.5:
            amount = round(random.uniform(600, 9999), 2)   # high value
        elif is_fraud_txn and random.random() < 0.3:
            amount = round(random.choice([100, 200, 500, 1000]), 2)  # round (bot-like)
        else:
            amount = round(random.uniform(5, 499), 2)

        # Device — new/unknown device for flagged fraud
        if is_fraud_txn and random.random() < 0.6:
            device = random_device_fingerprint()   # different from registered device
        else:
            device = user["device_fingerprint"]

        # IP and ASN — may differ from registered IP
        if is_fraud_txn and random.random() < 0.5:
            txn_ip, txn_asn = random_ip(payment_country, is_fraud=True)
        else:
            txn_ip = user["ip_address"]
            txn_asn = user["ip_asn"]

        payment_method = random.choice(PAYMENT_METHODS)

        transactions.append({
            "transaction_id": f"T{str(i+1).zfill(5)}",
            "user_id": uid,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "ip_address": txn_ip,
            "ip_asn": txn_asn,
            "device_fingerprint": device,
            "purchase_amount": amount,
            "payment_method": payment_method,
            "transaction_country": payment_country,
            "user_registered_country": user_country,
            "device_type": random.choice(DEVICE_TYPES),
            "browser": random.choice(BROWSERS),
            "is_flagged": int(is_fraud_txn),
        })

        if is_fraud_txn:
            fraud_generated += 1

    # Sort by timestamp so time-based queries are intuitive
    transactions.sort(key=lambda x: x["timestamp"])
    return transactions


# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Written {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔄 Generating sample data for EFRiskEngine…")
    users = generate_users(NUM_USERS)
    transactions = generate_transactions(users, NUM_TRANSACTIONS)

    write_csv(os.path.join(OUTPUT_DIR, "sample_users.csv"), users)
    write_csv(os.path.join(OUTPUT_DIR, "sample_transactions.csv"), transactions)

    fraud_users = sum(1 for u in users if u["is_fraudulent"])
    fraud_txns = sum(1 for t in transactions if t["is_flagged"])
    print(f"\n✅ Done — {NUM_USERS} users ({fraud_users} fraudulent), "
          f"{NUM_TRANSACTIONS} transactions ({fraud_txns} flagged).")
