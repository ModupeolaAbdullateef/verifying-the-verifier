"""
config.py - Central configuration for the experiment artefact.

All settings that must be held constant across experimental conditions are
defined here so that they are recorded in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Results location; VTV_RESULTS selects a separate folder (for example, for the pilot).
_results = Path(os.environ.get("VTV_RESULTS", "results"))
RESULTS_DIR = _results if _results.is_absolute() else ROOT / _results

# --- Input -----------------------------------------------------------------
# Agents that read the source document (extractor and checker) receive the
# scanned page images; the plain text is used only for ground truth and scoring.
INPUT_MODE = "image"

# --- Model and sampling ----------------------------------------------------
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
MODEL_ID = "gemini-3.1-flash-lite"

# Provider-recommended default for Gemini 3 models. A per-trial seed is
# recorded with every call so that individual runs can be reproduced.
TEMPERATURE = 1.0
SEED_BASE = 20261101
MAX_OUTPUT_TOKENS = 4096

# --- Verification gate -----------------------------------------------------
# Automated mechanisms may request at most one revision of the extraction.
MAX_REVISIONS = 1

# --- Cost accounting (USD per one million tokens, published list price) ----
PRICE_INPUT_PER_M = 0.25
PRICE_OUTPUT_PER_M = 1.50
PRICE_DATE = "2026-11-01"

# --- API resilience --------------------------------------------------------
# Transient API failures (rate limits, server errors) are retried with
# exponential back-off. Retries are logged but are not counted as revisions.
MAX_API_ATTEMPTS = 5
BACKOFF_SECONDS = 2.0

# --- Offline test mode -----------------------------------------------------
# When VTV_MOCK=1, model calls are answered from the ground truth instead of
# the API. Used only to test the plumbing of the workflow at zero cost.
MOCK_MODE = os.environ.get("VTV_MOCK", "0") == "1"
