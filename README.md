# Verifying the Verifier

**Evaluating an Automated Verification Layer for Reducing Errors and Hallucinations in an Enterprise Multi-Agent AI Workflow**

This repository contains the experimental artefact for my MSc Computer Science dissertation at the **University of East London (UEL)**, module CN-7000 (Dissertation), 2026. It is academic work submitted for assessment; please cite it if you build on it (see [Citation](#citation)).

Author: Modupeola Muhammed Abdullateef

---

## What this project does

Enterprise workflows increasingly chain several large language model (LLM) agents together, so an error made by one agent can pass silently into a business decision. This project builds such a workflow and measures how well three kinds of verification catch errors, and at what cost.

The workflow is a three-agent accounts-payable pipeline built in [LangGraph](https://docs.langchain.com/oss/python/langgraph/):

```
Extractor agent ──► [ Verification gate ] ──► Normaliser agent ──► Decision agent
                          │
                          └── issues found: one revision by the extractor
```

The gate runs one of four conditions:

| Condition | Mechanism |
|---|---|
| C0 | No verification (baseline) |
| C1 | Deterministic schema and constraint validation (formats, dates, arithmetic) |
| C2 | A separate LLM checker agent with an explicit rubric |
| C3 | Human-in-the-loop review via a LangGraph interrupt and a blinded review app |

The research questions are:

1. **RQ1:** To what extent does each mechanism reduce error and hallucination rates relative to the baseline?
2. **RQ2:** What operational cost (latency and tokens) does each mechanism impose?
3. **RQ3:** Which mechanism offers the most favourable trade-off?

## Experiments

- **Experiment 1 (natural errors):** 21 synthetic invoices, rendered as scanned page images, × C0–C2 × 30 trials, plus a C3 subset.
- **Experiment 2 (seeded faults):** each record contains one known fault from a five-type taxonomy (omission, substitution, calculation, hallucination, format) or none (control), so each mechanism's detection rate can be measured directly.
- **Replication:** an IT ticket-triage workflow with the same gate.

In total: 3,805 completed runs and 103 blinded human reviews, at a token cost of about US$13.70.

### Headline findings

- Deterministic validation (C1) raised task success on natural errors from 83.0% to 92.7% and removed every wrong decision, at negligible cost.
- The LLM checker (C2) detected 95% of seeded faults but gave no significant gain on natural errors; as a second call to the same model, it appears to share the extractor's misreadings.
- Human review (C3) was the slowest and most expensive mechanism and, in this sample, detected about half of the seeded faults.
- No mechanism could correct errors introduced after the gate.

Full results, statistics and discussion are in the dissertation.

## Repository structure

```
vtv/                    Core package
  config.py             Model, prices, seeds and paths
  llm.py                Measured model-call wrapper (tokens, latency, cost, retries)
  prompts.py            Invoice agent prompts
  mechanisms.py         Verification mechanisms C0–C3 (invoice and ticket)
  workflow.py           Invoice workflow (LangGraph) and human checkpoint
  ticket_workflow.py    Ticket-triage workflow (replication)
  ticket_prompts.py     Ticket agent prompts
  faults.py             Loads the seeded-fault records (Experiment 2)
  tools.py              Supplier search, purchase-order lookup, service catalogue
  traces.py             Trace writing and ground-truth access
  review_queue.py       Queue for human review items
  runner.py             Dispatches runs to the invoice or ticket workflow

generate_data.py        Generates the synthetic dataset and scanned invoice images
generate_faults.py      Generates the seeded-fault records for Experiment 2
run_batch.py            Runs an experimental plan (randomised, resumable, cost-capped)
run_one.py              Runs a single workflow run
review_app.py           Blinded Streamlit review interface for C3
resume_reviews.py       Completes reviewed C3 runs
check_review_queue.py   Detects and removes stale review items
score.py                Scores traces against ground truth (error taxonomy)
analysis.py             Statistical tests, tables and figures
data/                   Generated dataset, ground truth and seeded faults
```

The `results/` folder (traces, scores and review logs) is not tracked in Git.

## Setup

Requirements: Python 3.12, and a Google Cloud project with the Vertex AI / Gemini Enterprise Agent Platform API enabled.

```bash
pip install -r requirements.txt
gcloud auth application-default login
```

Create a `.env` file in the project root (it is excluded from Git):

```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

The experiments used `gemini-3.1-flash-lite` at temperature 1.0, with a recorded seed for every call (see `vtv/config.py`).

## Reproducing the study

```bash
python generate_data.py                 # synthetic dataset and scanned images
python generate_faults.py               # seeded-fault records (Experiment 2)

python run_batch.py --plan main         # Experiment 1
python run_batch.py --plan faults       # Experiment 2
python run_batch.py --plan tickets      # replication

streamlit run review_app.py             # blinded human review (C3)
python resume_reviews.py                # complete the reviewed C3 runs

python score.py                         # score all traces
python analysis.py                      # statistics, tables and figures
```

The data generators are deterministic: the same seeds produce byte-identical files.

### Offline test mode

To test the whole pipeline without any API calls or cost, set `VTV_MOCK=1`. Every model call then returns the ground-truth answer:

```bash
VTV_MOCK=1 python run_batch.py --plan pilot
```

(On Windows PowerShell: `$env:VTV_MOCK="1"`.)

## Versions

| Tag | Contents | Results produced |
|---|---|---|
| `v1.0-main-run` | Invoice workflow, mechanisms, generators, harness, scorer, review app | Experiments 1 and 2 |
| `v1.1-tickets` | Ticket-triage replication | Replication |
| `v1.2-final` | Final version submitted with the dissertation | All results |

Later versions add to earlier ones without changing the behaviour of the invoice workflow, so `v1.2-final` reproduces every result in the dissertation.

## Citation

Abdullateef, M. M. (2026). *Verifying the verifier: Evaluating an automated verification layer for reducing errors and hallucinations in an enterprise multi-agent AI workflow* [MSc dissertation and software]. University of East London.

## Notice

All documents, tickets and reference data in this repository are synthetic. No real customer, employee or organisational data was used, and no BT Group systems or data were involved. This repository is shared for academic transparency and reproducibility; it is not production software.
