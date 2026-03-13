"""
src/ai_assist.py
----------------
AI-assisted anomaly detection module for EFRiskEngine.

Design:
  - Abstract base class ``LLMProvider`` allows swapping between
    Google Gemini, OpenAI GPT, Anthropic Claude, or a
    mock provider (no API key required).
  - Provider selection is driven by the ``LLM_PROVIDER`` environment
    variable (default: "gemini").
  - API credentials are loaded ONLY from environment variables or a
    .env file — never hardcoded in source code.

Typical usage:
    from src.ai_assist import get_provider, analyze_user_behavior

    provider = get_provider()
    analysis = analyze_user_behavior(user_row, txns_for_user, provider)
"""

import json
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from dotenv import load_dotenv

# Load .env file if present (credentials stay out of source code)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base class — implement this to add a new LLM provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface for any text-generation LLM backend."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Args:
            prompt:     The full prompt string.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's text response as a string.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for logging."""


# ---------------------------------------------------------------------------
# OpenAI provider (optional — set LLM_PROVIDER=openai)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    LLM provider backed by OpenAI's Chat Completions API.

    Environment variables read:
      - OPENAI_API_KEY  (required)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            import openai  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self._key = os.environ.get("OPENAI_API_KEY", "")
        if not self._key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        self._model = model

    @property
    def name(self) -> str:
        return f"OpenAI/{self._model}"

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        import openai  # noqa: PLC0415
        client = openai.OpenAI(api_key=self._key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Anthropic provider (optional — set LLM_PROVIDER=anthropic)
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    LLM provider backed by Anthropic's Messages API.

    Environment variables read:
      - ANTHROPIC_API_KEY  (required)
    """

    def __init__(self, model: str = "claude-3-haiku-20240307") -> None:
        try:
            import anthropic  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        self._key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self._model = model

    @property
    def name(self) -> str:
        return f"Anthropic/{self._model}"

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        import anthropic  # noqa: PLC0415
        client = anthropic.Anthropic(api_key=self._key)
        message = client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# Gemini provider (optional — set LLM_PROVIDER=gemini)
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    LLM provider backed by Google's Gemini API.

    Environment variables read:
      - GEMINI_API_KEY  (required)
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        try:
            from google import genai  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ImportError("Install google-genai: pip install google-genai") from exc

        self._key = os.environ.get("GEMINI_API_KEY", "")
        if not self._key:
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self._model = model

    @property
    def name(self) -> str:
        return f"Gemini/{self._model}"

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        from google import genai  # noqa: PLC0415
        client = genai.Client(api_key=self._key)

        # Use 0.0 temperature for deterministic (consistent) outputs
        config = genai.types.GenerateContentConfig(
            temperature=0.0,
        )

        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            # response.text can be None if the model returned no candidates or
            # the content was blocked — fall back gracefully.
            if response.text is not None:
                return response.text.strip()
            # Try extracting from candidate parts directly
            for candidate in (response.candidates or []):
                for part in (candidate.content.parts if candidate.content else []):
                    if hasattr(part, "text") and part.text:
                        return part.text.strip()
            logger.warning("Gemini returned an empty response (no text in any candidate).")
            return "[Gemini returned an empty response. The content may have been filtered.]"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini API call failed: %s", exc)
            return f"[Gemini API Error] {exc}"


# ---------------------------------------------------------------------------
# Mock provider — no API key required, useful for testing/CI
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """
    Deterministic mock provider that returns pre-canned responses.

    Use this when:
      - Running tests without API credentials.
      - Offline / CI environments.
      - Demonstrating the pipeline without LLM costs.
    """

    @property
    def name(self) -> str:
        return "Mock/offline"

    def complete(self, prompt: str, max_tokens: int = 512) -> str:  # noqa: ARG002
        return (
            "MOCK ANALYSIS — No live LLM call made.\n"
            "Detected signals: Multiple failed logins, high-value transaction "
            "from unrecognised device, country mismatch between registered "
            "location and transaction origin.\n"
            "Recommendation: Flag for manual review. Verify user identity via "
            "secondary authentication before processing transactions."
        )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_provider(provider_name: str | None = None) -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    Provider selection order:
      1. Explicit ``provider_name`` argument.
      2. ``LLM_PROVIDER`` environment variable.
      3. Default: "gemini".

    Available providers:
      - "gemini"       → GeminiProvider (requires GEMINI_API_KEY)
      - "openai"       → OpenAIProvider (requires OPENAI_API_KEY)
      - "anthropic"    → AnthropicProvider (requires ANTHROPIC_API_KEY)
      - "mock"         → MockProvider (no credentials required)

    Args:
        provider_name: Provider key string (case-insensitive).

    Returns:
        Instantiated :class:`LLMProvider`.

    Raises:
        ValueError: If an unrecognised provider name is given.
    """
    name = (provider_name or os.environ.get("LLM_PROVIDER", "gemini")).lower().strip()
    providers: dict[str, Any] = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock": MockProvider,
    }
    if name not in providers:
        raise ValueError(
            f"Unknown LLM provider {name!r}. "
            f"Choose from: {', '.join(providers)}"
        )
    logger.info("Initialising LLM provider: %s", name)
    return providers[name]()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_user_behavior_prompt(
    user_row: pd.Series,
    txns: pd.DataFrame,
    risk_score: int = 0,
) -> str:
    """Build a structured prompt for user-behaviour analysis."""
    recent = txns.sort_values("timestamp", ascending=False).head(10)
    txn_summary = recent[["timestamp", "purchase_amount", "transaction_country",
                           "device_type", "payment_method"]].to_string(index=False)

    return textwrap.dedent(f"""
        You are a fraud analyst AI. Analyse the following user behaviour data and
        provide a concise fraud risk assessment.

        === USER PROFILE ===
        User ID:            {user_row.get('user_id', 'N/A')}
        Country:            {user_row.get('country', 'N/A')}
        Account age (days): {user_row.get('account_age_days', 'N/A')}
        Failed logins:      {user_row.get('failed_login_count', 0)}
        Password resets:    {user_row.get('password_resets', 0)}
        Device type:        {user_row.get('device_type', 'N/A')}
        Rules-based risk:   {risk_score}/100

        === RECENT TRANSACTIONS (last 10) ===
        {txn_summary}

        === TASK ===
        1. Identify the top fraud signals present in the data above.
        2. Assess overall fraud likelihood as: LOW / MEDIUM / HIGH / CRITICAL.
        3. Suggest one concrete next action (e.g., block, step-up auth, manual review).
        4. Keep your response to 150 words or fewer.
    """).strip()


def _build_pattern_detection_prompt(flagged_df: pd.DataFrame) -> str:
    """Build a prompt for emerging fraud pattern detection across flagged transactions."""
    sample = flagged_df.head(20)[["user_id", "purchase_amount", "transaction_country",
                                   "device_type", "payment_method", "risk_score"]]
    table = sample.to_string(index=False)

    return textwrap.dedent(f"""
        You are a fraud intelligence analyst. Review the following flagged transactions
        and identify emerging fraud patterns or coordinated attack vectors.

        === FLAGGED TRANSACTIONS (sample of up to 20) ===
        {table}

        === TASK ===
        1. List up to 3 distinct fraud patterns you observe (e.g., geography cluster,
           device type concentration, payment method abuse).
        2. For each pattern, give a brief description and recommended mitigation.
        3. Note any signals that suggest coordinated/organised fraud rings.
        4. Keep your response to 200 words or fewer.
    """).strip()


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------

def analyze_user_behavior(
    user_row: pd.Series,
    user_txns: pd.DataFrame,
    provider: LLMProvider,
    rules_risk_score: int = 0,
) -> dict:
    """
    Send a user's behavioural profile to the LLM for qualitative fraud analysis.

    Combines structured user data and recent transaction history into a
    prompt and parses the LLM response.

    Args:
        user_row:         Single row from the users DataFrame.
        user_txns:        Transactions belonging to this user.
        provider:         An instantiated :class:`LLMProvider`.
        rules_risk_score: Score from the rules engine (provides context to LLM).

    Returns:
        Dict with keys: user_id, rules_risk_score, ai_analysis, provider_name.
    """
    uid = user_row.get("user_id", "unknown")
    logger.info("Running AI analysis for user %s via %s", uid, provider.name)

    prompt = _build_user_behavior_prompt(user_row, user_txns, rules_risk_score)
    analysis = provider.complete(prompt, max_tokens=256)

    return {
        "user_id": uid,
        "rules_risk_score": rules_risk_score,
        "ai_analysis": analysis,
        "provider_name": provider.name,
    }


def detect_emerging_patterns(
    flagged_txns: pd.DataFrame,
    provider: LLMProvider,
) -> str:
    """
    Prompt the LLM to identify emerging fraud patterns across flagged transactions.

    Args:
        flagged_txns: DataFrame of transactions with elevated risk scores.
        provider:     An instantiated :class:`LLMProvider`.

    Returns:
        Multi-line string containing the LLM's pattern analysis.
    """
    if flagged_txns.empty:
        return "No flagged transactions to analyse."

    logger.info(
        "Detecting emerging patterns across %d flagged transactions via %s",
        len(flagged_txns), provider.name,
    )
    prompt = _build_pattern_detection_prompt(flagged_txns)
    return provider.complete(prompt, max_tokens=400)


def combine_with_rules(
    rules_score: int,
    ai_analysis: str,
    ai_weight: float = 0.25,
) -> dict:
    """
    Combine rules-based score with AI qualitative analysis into a final verdict.

    The AI analysis is parsed for explicit risk keywords to derive an
    AI risk bias which is blended with the rules score.

    Blending formula:
        combined_score = (1 - ai_weight) * rules_score + ai_weight * ai_bias_score

    Args:
        rules_score:  Integer rules-based risk score (0–100).
        ai_analysis:  Text response from the LLM.
        ai_weight:    Weight given to the AI signal (0–1, default 0.25).

    Returns:
        Dict with keys: rules_score, ai_bias_score, combined_score, combined_label.
    """
    # Derive a simple numeric bias from the AI text (keyword-based)
    text = ai_analysis.lower()
    if "critical" in text:
        ai_bias = 90
    elif "high" in text:
        ai_bias = 70
    elif "medium" in text:
        ai_bias = 40
    elif "low" in text:
        ai_bias = 15
    else:
        ai_bias = rules_score  # no signal → trust rules

    combined = (1 - ai_weight) * rules_score + ai_weight * ai_bias
    combined = max(0, min(100, round(combined)))

    from src.risk_engine import _risk_label  # local import to avoid circular deps
    return {
        "rules_score": rules_score,
        "ai_bias_score": ai_bias,
        "combined_score": combined,
        "combined_label": _risk_label(combined),
    }


def run_ai_analysis(
    scored_txns: pd.DataFrame,
    users_df: pd.DataFrame,
    provider: LLMProvider | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Run AI analysis on the top-N highest-risk users and return enriched results.

    Args:
        scored_txns: Transaction-level scores from the risk engine.
        users_df:    Normalised users DataFrame.
        provider:    LLM provider (defaults to get_provider() if None).
        top_n:       Number of top-risk users to analyse.

    Returns:
        DataFrame with AI analysis results merged onto user risk scores.
    """
    if provider is None:
        provider = get_provider()

    # Find top-N users by max transaction risk
    user_max_risk = (
        scored_txns.groupby("user_id")["risk_score"]
        .max()
        .reset_index()
        .rename(columns={"risk_score": "max_risk"})
        .sort_values("max_risk", ascending=False)
        .head(top_n)
    )

    user_lookup = {row["user_id"]: row for _, row in users_df.iterrows()}
    results = []

    for _, row in user_max_risk.iterrows():
        uid = row["user_id"]
        user_row = user_lookup.get(uid, pd.Series({"user_id": uid}))
        user_txns = scored_txns[scored_txns["user_id"] == uid]
        rules_score = int(row["max_risk"])

        analysis = analyze_user_behavior(user_row, user_txns, provider, rules_score)
        combined = combine_with_rules(rules_score, analysis["ai_analysis"])

        results.append({
            "user_id": uid,
            "rules_risk_score": rules_score,
            "ai_bias_score": combined["ai_bias_score"],
            "combined_score": combined["combined_score"],
            "combined_label": combined["combined_label"],
            "ai_analysis": analysis["ai_analysis"],
            "provider": provider.name,
        })

    return pd.DataFrame(results)
