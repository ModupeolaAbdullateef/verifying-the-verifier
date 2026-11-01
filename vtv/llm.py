"""
llm.py - Model-call wrapper with measurement.

Every call to the language model in the experiment goes through call_llm(),
which returns the parsed JSON response together with the measurements needed
for RQ2: input, output and thinking tokens, latency and cost. Transient API
errors are retried with exponential back-off; retries are recorded but do not
count towards the measured latency of the successful call.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass

from vtv import config

_client = None


@dataclass
class CallRecord:
    """Measurements for one model call, appended to the run trace."""
    node: str
    attempt_retries: int
    latency_s: float
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float
    parse_ok: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _get_client():
    """Create the Gen AI client once, on first use."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(vertexai=True, project=config.PROJECT,
                               location=config.LOCATION)
    return _client


def call_cost(input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    """Token cost in USD at the configured list prices (thinking billed as output)."""
    return (input_tokens * config.PRICE_INPUT_PER_M
            + (output_tokens + thinking_tokens) * config.PRICE_OUTPUT_PER_M) / 1_000_000


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating markdown code fences."""
    if not text:
        return None
    cleaned = _FENCE.sub("", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def call_llm(node: str, system: str, prompt: str, seed: int,
             mock_answer: dict | None = None) -> tuple[dict | None, CallRecord]:
    """
    Send one request and return (parsed_json_or_None, CallRecord).

    All agents request JSON output (response_mime_type) without a response
    schema, so that structural validity remains an observable outcome rather
    than being enforced by the provider.
    """
    if config.MOCK_MODE:
        return _mock_call(node, mock_answer)

    from google.genai import types

    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=config.TEMPERATURE,
        seed=seed,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
    )

    retries, last_error = 0, None
    for attempt in range(config.MAX_API_ATTEMPTS):
        start = time.perf_counter()
        try:
            response = _get_client().models.generate_content(
                model=config.MODEL_ID, contents=prompt, config=gen_config)
        except Exception as error:  # rate limits and transient server errors
            last_error = str(error).splitlines()[0][:300]
            retries += 1
            time.sleep(config.BACKOFF_SECONDS * (2 ** attempt))
            continue
        latency = time.perf_counter() - start
        usage = response.usage_metadata
        in_tok = usage.prompt_token_count or 0
        out_tok = usage.candidates_token_count or 0
        think_tok = getattr(usage, "thoughts_token_count", None) or 0
        parsed = parse_json(response.text or "")
        return parsed, CallRecord(
            node=node, attempt_retries=retries, latency_s=round(latency, 3),
            input_tokens=in_tok, output_tokens=out_tok, thinking_tokens=think_tok,
            cost_usd=round(call_cost(in_tok, out_tok, think_tok), 8),
            parse_ok=parsed is not None,
            error=None if parsed is not None else "response was not valid JSON",
        )

    return None, CallRecord(node=node, attempt_retries=retries, latency_s=0.0,
                            input_tokens=0, output_tokens=0, thinking_tokens=0,
                            cost_usd=0.0, parse_ok=False,
                            error=f"API failed after {retries} attempts: {last_error}")


def _mock_call(node: str, mock_answer: dict | None) -> tuple[dict | None, CallRecord]:
    """Offline test mode: return the supplied answer with nominal measurements."""
    record = CallRecord(node=node, attempt_retries=0, latency_s=0.01,
                        input_tokens=500, output_tokens=150, thinking_tokens=0,
                        cost_usd=round(call_cost(500, 150, 0), 8),
                        parse_ok=mock_answer is not None)
    return (json.loads(json.dumps(mock_answer)) if mock_answer is not None else None), record
