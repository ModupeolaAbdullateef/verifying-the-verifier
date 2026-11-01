"""
smoke_test.py - Environment check for the experiment platform.

Verifies that the local environment can call Gemini models hosted on Google
Cloud (Gemini Enterprise Agent Platform, formerly Vertex AI) using
Application Default Credentials. For each candidate model, one short
extraction request is sent and the script reports availability, end-to-end
latency and token usage (input, output and thinking tokens).

Configuration is read from a .env file:
    GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION

Usage:
    python smoke_test.py
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# Low-cost candidate models, newest first. Models unavailable to the
# project are reported and skipped.
CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
]

PROMPT = (
    "Extract the invoice number and the total due from the text below. "
    "Reply only with JSON using the keys invoice_number and total.\n\n"
    "INVOICE INV-20417\n"
    "Supplier: Northgate Office Supplies Ltd\n"
    "Total due: GBP 1,284.50"
)

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

print(f"Project: {PROJECT} | Location: {LOCATION}\n")

for model in CANDIDATES:
    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=0,
                seed=42,
                max_output_tokens=1024,
            ),
        )
    except Exception as error:  # report and move on to the next model
        first_line = str(error).splitlines()[0][:200]
        print(f"[NOT AVAILABLE] {model}\n     {first_line}\n")
        continue

    elapsed = time.perf_counter() - start
    usage = response.usage_metadata
    text = (response.text or "").strip()

    print(f"[OK] {model}  ({elapsed:.2f} s)")
    print(
        f"     input tokens: {usage.prompt_token_count} | "
        f"output tokens: {usage.candidates_token_count} | "
        f"thinking tokens: {getattr(usage, 'thoughts_token_count', None)}"
    )
    print(f"     response: {text[:200]}\n")

print("Done.")
