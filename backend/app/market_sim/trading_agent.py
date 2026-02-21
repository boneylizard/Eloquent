"""
AI Trading Agent for Market Simulator.
Believes it manages real money; thinks probabilistically; uses Nate Silver-style data confidence;
generates 10 strategies, runs Monte Carlo tournament, selects winner, executes.
"""
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .market_data_service import market_data_service
from .monte_carlo_engine import run_tournament, STRATEGY_TYPES
from .market_db import market_db

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a portfolio manager at a registered investment advisor. You are managing a real $10,000 portfolio for a client. The money is real. You make actual trading decisions that will be executed.

## Epistemic framework (Nate Silver approach)
- Think probabilistically. Express uncertainty. Use phrases like "60% probability" or "my confidence is medium."
- Start with base rates before adjusting for new information.
- Evaluate source credibility. Apply these confidence scores to data:
  - Analyst ratings: 0.31 (low)
  - Prediction markets: 0.73 (medium-high)
  - Fundamental data: 0.85 (high)
  - News sentiment: 0.38 (low)
  - Insider trading: 0.62 (medium, for buys)
  - Price/volume history: 0.90 (high)
- Acknowledge what you don't know. Show epistemic humility.

## Your workflow
1. You receive market context (current prices, S&P 500, recent data).
2. You generate exactly 10 competing strategies with distinct approaches:
   - aggressive_growth, conservative_value, index_fund, cash, momentum_trade,
   - contrarian, sector_rotation, dca, hedged_position, dividend_focus
   (or similar variants with clear differentiation)
3. A Monte Carlo engine runs 10,000 market scenarios and simulates each strategy.
4. You receive the results: expected value, median return, std dev, Sharpe ratio,
   max drawdown, VaR 95%, win rate, profit factor, performance by regime, probability of beating S&P 500.
5. You select the winning strategy based on multi-criteria analysis:
   - Risk-adjusted returns (Sharpe)
   - Downside protection (max drawdown, VaR)
   - Current market context
   - Win rate and profit factor
   - Performance in bear/crash regimes
6. You output your decision and reasoning in the specified JSON format.

## Response format (strict JSON)
Return ONLY valid JSON in this shape:
{
  "winner_id": "strategy_id",
  "reasoning": "Your multi-criteria analysis and why you chose this strategy. Be specific about tradeoffs.",
  "confidence": 0.0-1.0,
  "trade_action": {
    "action": "buy" | "sell" | "hold",
    "symbol": "AAPL" | "SPY" | "cash" | etc,
    "description": "Human-readable description of what to do"
  },
  "acknowledged_uncertainty": "What you're uncertain about and how it could affect the outcome."
}
"""


def _resolve_llm_endpoint(model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve endpoint: custom endpoint by id, or first enabled, or env."""
    try:
        from ..openai_compat import get_configured_endpoint, is_api_endpoint
        if model_id and is_api_endpoint(model_id):
            return get_configured_endpoint(model_id)
        ep = get_configured_endpoint(None)
        if ep:
            return ep
        env_url = os.getenv("MARKET_SIM_LLM_URL")
        env_model = os.getenv("MARKET_SIM_LLM_MODEL", "gpt-4o-mini")
        if env_url and env_model:
            return {
                "url": env_url.rstrip("/"),
                "model": env_model,
                "api_key": os.getenv("MARKET_SIM_LLM_API_KEY", ""),
                "name": "Market Sim LLM",
            }
        return None
    except Exception:
        return None


async def _call_llm(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    model_manager: Any = None,
    primary_model: Optional[str] = None,
) -> Optional[str]:
    """Call configured LLM for trading decision (async)."""
    try:
        from ..openai_compat import (
            get_configured_endpoint,
            is_api_endpoint,
            prepare_endpoint_request,
            forward_to_configured_endpoint_non_streaming,
        )
        from .. import inference

        # Prefer API endpoint if model_id is an endpoint
        if model_id and is_api_endpoint(model_id):
            endpoint = get_configured_endpoint(model_id)
            if endpoint:
                request_data = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1500,
                }
                endpoint_config, url, prepared_data = prepare_endpoint_request(model_id, request_data)
                result = await forward_to_configured_endpoint_non_streaming(endpoint_config, url, prepared_data)
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content if content else None

        # Fallback: local model
        if model_manager and primary_model and not is_api_endpoint(primary_model or ""):
            prompt = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            reply = await inference.generate_text(
                model_manager=model_manager,
                model_name=primary_model,
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3,
            )
            if isinstance(reply, str):
                return reply
            if isinstance(reply, dict) and reply.get("choices"):
                return (reply.get("choices", [{}])[0].get("text") or "").strip()
            return str(reply) if reply else None
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return None


def generate_strategies() -> List[Dict[str, Any]]:
    """Generate 10 distinct strategies for the tournament."""
    strategies = []
    for i, st in enumerate(STRATEGY_TYPES):
        strategies.append({
            "id": f"strat_{i}",
            "type": st,
            "name": st.replace("_", " ").title(),
        })
    return strategies


async def run_ai_tournament_and_decide(
    market_context: Dict[str, Any],
    model_id: Optional[str] = None,
    model_manager: Any = None,
    primary_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full workflow: generate strategies, Monte Carlo tournament, AI selects winner.
    Returns tournament results + AI decision.
    """
    strategies = generate_strategies()
    tournament = run_tournament(strategies, initial_value=10_000.0)
    results_summary = {}
    for sid, r in tournament["strategy_results"].items():
        results_summary[sid] = r["analysis"]

    prompt = f"""## Market context
{json.dumps(market_context, indent=2)}

## Monte Carlo tournament results (10,000 scenarios)
{json.dumps(results_summary, indent=2)}

## Regime performance (avg final value per regime)
{json.dumps(tournament.get("regime_performance", {}), indent=2)}

## Your task
Select the winning strategy. Consider risk-adjusted returns, downside protection, and current market context.
Output JSON only."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw = await _call_llm(messages, model_id, model_manager, primary_model)
    if not raw:
        # Fallback: pick by Sharpe
        best_id = max(
            results_summary.keys(),
            key=lambda k: results_summary[k].get("sharpe_ratio", -999),
        )
        return {
            "tournament": tournament,
            "winner_id": best_id,
            "reasoning": "LLM unavailable; selected by highest Sharpe ratio.",
            "confidence": 0.5,
            "trade_action": {"action": "hold", "symbol": "SPY", "description": "Hold current position until LLM available."},
            "raw_response": None,
        }

    # Parse JSON from response
    try:
        # Try to extract JSON block
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            parsed = json.loads(m.group(0))
        else:
            parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
        parsed["winner_id"] = list(results_summary.keys())[0]
        parsed["reasoning"] = raw[:500]
        parsed["confidence"] = 0.5
        parsed["trade_action"] = {"action": "hold", "symbol": "SPY", "description": "Parse error; holding."}

    return {
        "tournament": tournament,
        "winner_id": parsed.get("winner_id", list(results_summary.keys())[0]),
        "reasoning": parsed.get("reasoning", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "trade_action": parsed.get("trade_action", {"action": "hold", "symbol": "SPY", "description": "Hold"}),
        "acknowledged_uncertainty": parsed.get("acknowledged_uncertainty", ""),
        "raw_response": raw,
    }
