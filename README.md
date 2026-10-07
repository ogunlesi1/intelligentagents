# LLM-Powered Academic Research Planning Agent

A multi-agent planning system in a single, well-commented file (`agent.py`).
Given a research goal, it decomposes the goal into search steps, retrieves
scholarly records from open academic APIs, extracts structured fields from
them, validates every extracted claim against its source, and exports a
comparison table only after human approval.

Implements the design proposed in the Unit 6 team submission (Group E).

## Why one file

The system is six cooperating roles sharing one typed state object. At this
size, one file that can be read top to bottom is easier to review and defend
than a package of a dozen files with the same amount of code — the trade-off
is explained in the module docstring at the top of `agent.py`.

Numbered section banners inside the file mark the read order: configuration,
state schema, LLM gateway, source clients, orchestrator (the core routing
logic), worker agents, validator/gate/storage, graph assembly, CLI.

## What it does

The distinguishing property: **no claim reaches the output unless the span of
source text supporting it can be located in the retrieved abstract.**
Unverifiable claims trigger a bounded retry (own budget for relevance
failures vs accuracy failures); exhausted retries escalate to a human
reviewer with the result flagged as unverified rather than silently dropped.

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add HF_TOKEN and CONTACT_EMAIL
```

## Running

**Offline demonstration** — no keys, no network, deterministic output:

```bash
python agent.py --demo --auto-approve "Compare ML methods for literature screening"
```

**Live run** against Crossref, OpenAlex and Hugging Face inference:

```bash
export $(grep -v '^#' .env | xargs)
python agent.py "Compare ML methods for literature screening"
```

The run pauses at the approval gate and prints a summary; answering `y`
releases the export, anything else withholds it.

**Evaluation set** (the ≥90% error-free target from the design proposal):

```bash
python agent.py --eval
```

Artefacts land in `runs/outputs/<run_id>.md` and `.json`. Structured
JSON-line logs go to stderr (`2> runs/<run_id>.log` to capture them).

**Offline / no hosted API key**: if `HF_TOKEN` is unset, `--eval` and the
`--live` path both fall back automatically to a local model rather than
failing. See "Local model fallback" below for exactly what that means.

### What "90% error-free" means

`--eval` doesn't report one pass/fail number — it reports four explicitly
defined measures, read directly off state the graph already produces (see
`run_eval()` in `agent.py` §9):

| Measure | Definition | Where it comes from |
|---|---|---|
| `retrieval_error_rate` | a step never reached `min_papers_per_step` and the run escalated via a replan | `escalation_reason == "replan budget exhausted"` |
| `mean_hallucination_rate` | untraceable claims ÷ claims checked | the Validator's own `ValidationReport`, not a second judgement |
| `citation_validity_rate` | fraction of exported rows whose `source_id` is a resolvable DOI shape | regex check against `verified` extractions |
| `error_free_rate` | exported, not escalated, 0% hallucination, valid citations | strict AND of the above |

The strict `error_free_rate` is what's measured against the 90% target —
escalation is correct designed behaviour, but a run that escalates could not
independently reach a verified answer, so it does not count as error-free.

### Local model fallback

Used when `HF_TOKEN` is unset or hosted inference is unreachable:

| | |
|---|---|
| Model | Llama-3.2-3B-Instruct |
| Quantisation | Q4_K_M (GGUF), via `llama.cpp` |
| File size | ~2.0 GB on disk |
| RAM required | ~4 GB resident (model + KV cache) |
| Context limit | 8192 tokens (not the architecture's full 128K — see `agent.py` for why) |
| Throughput | ~8–12 tok/s on an 8-core CPU; ~40–60 tok/s with partial GPU offload on 8 GB VRAM |

Chosen over the 8B model used for hosted inference specifically to fit a
typical student laptop's RAM budget; the trade-off is weaker JSON-formatting
reliability, which is why `parse_json()` tolerates fenced/prose-wrapped
output rather than assuming a clean response. Requires `llama-cpp-python`
(not in `requirements.txt` by default, since most runs use hosted inference)
and the GGUF file at `LOCAL_MODEL_PATH` (default `models/llama-3.2-3b-instruct-q4_k_m.gguf`).

## Testing

Two files:

- **`test_agent.py`** — 21 tests. The submission-facing core suite: one test
  per distinct behaviour, chosen to demonstrate the properties the design
  claims (the termination/escalation guarantee, independent retry budgets,
  accuracy rejecting a fabricated span, resilience to a dead source) rather
  than to maximise a count.
- **`test_agent_extended.py`** — 22 tests. Additional edge cases and
  alternate inputs per agent (fenced JSON, whitespace-tolerant provenance
  matching, de-duplication across sources, the remaining routing
  destinations, the local-model fallback selection, and so on) that back up
  the core claims more broadly.

```bash
pytest test_agent.py -v                    # core suite only
pytest -v                                  # both files, 43 tests total
pytest --cov=agent --cov-report=term-missing
```

Fully deterministic — the model is stubbed with `ScriptedLLM` and the
sources with `FixtureSource`, so nothing can fail because an API was slow.
Shared fixtures live in `conftest.py`, loaded automatically by pytest.

## A real defect found during development

`decide_next()` is the routing function; the node wrapper originally called
it without forwarding the run's `Settings`, so routing silently used
module-level defaults while every other agent used the injected `Settings`.
A functional test configured with a tight retry cap observed escalation
firing later than configured, which exposed it. `make_orchestrator(settings)`
exists specifically to fix this — see the comment above it in `agent.py`, and
`test_orchestrator_node_uses_injected_settings` in `test_agent.py`, which
guards against it recurring.

## Known limitations

- Relevance judgement is itself LLM-based, so it is a check, not a guarantee.
  Accuracy is mechanical and therefore stronger.
- Crossref abstracts are frequently absent, which reduces extractable
  evidence for older records; OpenAlex partially compensates.
- The retry cap means some runs return flagged partial results by design.
- Only abstracts are processed, not full texts.

## Acknowledgement of sources

External services and libraries: Crossref REST API (Crossref, 2026); OpenAlex
(OpenAlex, 2026); Hugging Face Inference Providers (Hugging Face, 2026);
LangGraph (LangChain, 2026); Pydantic; pytest; OpenAI ChatGPT. 

Design decisions in the code comments draw on: Padgham and Winikoff (2004)
for the agent-role decomposition; Müller and Pischel (1993) for escalation in
place of inter-layer competition; Simon (1956) for bounded rationality as the
justification for retry caps; Yao et al. (2023) for the plan-and-execute
versus ReAct comparison.

Full references in the accompanying presentation and the Unit 6 design
proposal.
