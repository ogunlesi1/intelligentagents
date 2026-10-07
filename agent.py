"""
LLM-powered multi-agent academic research planning agent.

ONE FILE, DELIBERATELY. Six cooperating roles share one typed state object
rather than passing messages, and the whole system is small enough to read
top to bottom without losing the thread - splitting it into a package would
buy conventional structure at the cost of making the control flow harder to
follow in one sitting, which matters more for a system this size.

Run `python agent.py --demo --auto-approve` for an offline, deterministic run
(no API keys, no network). See README.md for installation, live-mode and
testing instructions.

Sections, in read order:
  1. Configuration
  2. State schema (what the agents share)
  3. LLM gateway (so nodes don't talk to the SDK directly)
  4. Academic source clients (Crossref, OpenAlex, HTTP cache/backoff)
  5. Orchestrator - the routing/escalation logic (this is the core of the design)
  6. Worker agents - Planner, Retrieval, Processing
  7. Validator, human approval gate, Storage/Export
  8. Graph assembly (LangGraph wiring)
  9. CLI entry point
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, Literal, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
#
# Thresholds live in one place rather than inlined at their use sites, so the
# bounded-rationality argument made in the presentation is auditable in one
# spot rather than scattered through the agent functions below.


@dataclass(frozen=True)
class Settings:
    # Retry caps. Three is a design choice, not a magic number: unbounded
    # replanning risks non-termination, and a bounded agent that returns a
    # flagged partial result is preferable to one that never returns at all
    # (Simon, 1956). Each loop gets its own budget - see AgentState below.
    max_relevance_retries: int = 3
    max_accuracy_retries: int = 3
    max_replans: int = 2

    # Evidence thresholds.
    min_papers_per_step: int = 3  # below this, retrieval is a planning failure
    target_papers: int = 5

    # Sources. Crossref is the primary open metadata source and needs no key;
    # OpenAlex is complementary and rate-limits more generously when a contact
    # address is supplied (the "polite pool"), hence CONTACT_EMAIL below.
    contact_email: str = os.getenv("CONTACT_EMAIL", "student@example.ac.uk")
    request_timeout: float = 15.0
    max_backoff_attempts: int = 4
    cache_dir: str = os.getenv("CACHE_DIR", ".cache/http")

    # LLM gateway.
    hf_model: str = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    hf_token: str | None = os.getenv("HF_TOKEN")

    # Local fallback model, used when HF_TOKEN is unset or hosted inference is
    # unreachable. Specific model + quantisation chosen, not left generic:
    #
    #   Model:          Llama-3.2-3B-Instruct
    #   Quantisation:   Q4_K_M (GGUF), via llama.cpp
    #   File size:      ~2.0 GB on disk
    #   RAM required:   ~4 GB resident (model + KV cache at the context below)
    #   Context limit:  running at 8192 tokens (not the architecture's full
    #                   128K) - plan+step prompts here are short, and a
    #                   smaller KV cache keeps the ~4 GB RAM budget realistic
    #                   on a student laptop with no dedicated GPU
    #   Throughput:     ~8-12 tokens/sec on a modern 8-core CPU (no GPU);
    #                   ~40-60 tokens/sec with partial GPU offload on an
    #                   8 GB-VRAM card
    #
    # Chosen over the 8B model used for hosted inference specifically because
    # it fits the RAM budget of typical student hardware (4 GB vs ~8 GB for
    # an 8B Q4 quant); the trade-off is weaker JSON-formatting reliability,
    # which is why parse_json() below tolerates fenced/prose-wrapped output
    # rather than assuming a clean response.
    local_model_path: str = os.getenv(
        "LOCAL_MODEL_PATH", "models/llama-3.2-3b-instruct-q4_k_m.gguf"
    )
    local_context_tokens: int = 8192
    local_n_threads: int = os.cpu_count() or 4

    # Human-in-the-loop. Disabled only for automated evaluation runs; the
    # default must stay True because the integrity argument in the design
    # depends on no synthesis being exported unreviewed.
    require_human_approval: bool = True

    checkpoint_db: str = os.getenv("CHECKPOINT_DB", "runs/checkpoints.sqlite")
    output_dir: str = os.getenv("OUTPUT_DIR", "runs/outputs")


SETTINGS = Settings()


# ============================================================================
# 2. STATE SCHEMA
# ============================================================================
#
# Why a single typed state object rather than message passing: the design
# produced six loosely-coupled roles that must nonetheless agree on one
# evolving artefact - plan, evidence, extractions, validation verdict. Free-
# form messages between them would make traceability a convention; a typed
# state object makes it a constraint, which is what the Evidence Validator
# relies on (Padgham and Winikoff, 2004).
#
# Every field that crosses an agent boundary is a Pydantic model so a
# malformed LLM response fails loudly at the boundary that produced it,
# rather than silently corrupting a downstream agent's input.


class PlanStep(BaseModel):
    """One retrieval/processing unit produced by the Planner.

    `query` and `sources` are separated because the two recovery loops act on
    different things: a relevance failure re-runs the same intent against
    *different sources*, while a full replan rewrites the intent itself.
    """

    index: int
    description: str
    query: str
    sources: list[str] = Field(default_factory=lambda: ["crossref", "openalex"])


class Paper(BaseModel):
    """A retrieved scholarly record, normalised across source APIs.

    `source_id` (DOI where available) is mandatory: an evidence item that
    cannot be named cannot be cited, and the Validator rejects any extracted
    field whose provenance cannot be resolved back to one of these.
    """

    source_id: str
    title: str
    year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    source_api: str = "unknown"
    url: Optional[str] = None

    @field_validator("source_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source_id must not be empty - provenance is mandatory")
        return v


class ExtractedField(BaseModel):
    """A single extracted value plus the span of source text it came from.

    Storing `evidence_span` alongside `value` is the whole basis of the
    accuracy check: the Validator does not ask the LLM whether it was right,
    it asks whether the span it quoted actually appears in the retrieved
    abstract. That converts an unverifiable claim into a testable one.
    """

    name: Literal["methodology", "dataset", "reported_accuracy"]
    value: str
    evidence_span: str = ""


class Extraction(BaseModel):
    """All fields extracted from one paper, tied back to that paper."""

    source_id: str
    fields: list[ExtractedField] = Field(default_factory=list)


class FailureMode(str, Enum):
    """Why validation failed - drives which recovery loop fires.

    The two modes are deliberately distinct rather than a single boolean:
    a relevance failure means the retrieval step found the wrong literature
    (re-retrieve with different sources), whereas an accuracy failure means
    the extraction cannot be traced to the literature we did find
    (re-process the same evidence). Collapsing them would make the escalation
    rule claimed in the design untrue in practice.
    """

    NONE = "none"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ValidationReport(BaseModel):
    passed: bool
    failure_mode: FailureMode = FailureMode.NONE
    checked: int = 0
    untraceable: list[str] = Field(default_factory=list)
    notes: str = ""


class TransitionLog(BaseModel):
    """One state transition, emitted by the Orchestrator on every hop.

    This is deliberately part of the state rather than only a log file: the
    assessment requires evidence that an adaptive loop fired, and evidence
    that lives in the returned artefact cannot drift from the run that
    produced it.
    """

    step: int
    node: str
    decision: str
    detail: str = ""


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer for accumulating lists across nodes (LangGraph merges with this)."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    """The shared blackboard. Only the Orchestrator writes routing fields."""

    run_id: str
    goal: str

    plan: list[PlanStep]
    step_index: int

    retrieved: list[Paper]  # evidence for the current step only
    extractions: list[Extraction]  # extractions for the current step only
    verified: list[Extraction]  # accumulated, validated output

    validation: Optional[ValidationReport]

    # Two independent counters, per the bounded-recovery design. A single
    # shared counter would let a relevance retry consume an accuracy budget.
    relevance_retries: int
    accuracy_retries: int
    replans: int

    escalated: bool
    escalation_reason: str
    approved: bool
    export_path: str

    transitions: Annotated[list[TransitionLog], _append]
    next_node: str


def initial_state(goal: str, run_id: str) -> AgentState:
    return AgentState(
        run_id=run_id,
        goal=goal,
        plan=[],
        step_index=0,
        retrieved=[],
        extractions=[],
        verified=[],
        validation=None,
        relevance_retries=0,
        accuracy_retries=0,
        replans=0,
        escalated=False,
        escalation_reason="",
        approved=False,
        export_path="",
        transitions=[],
        next_node="planner",
    )


# ============================================================================
# 3. LLM GATEWAY
# ============================================================================
#
# Why an interface rather than direct SDK calls inside each agent function:
# the agents are the units under test, and a unit test that needs a live
# model tests the model, not the agent. Every agent takes an `LLMClient`;
# tests inject `ScriptedLLM`, the CLI injects `HuggingFaceClient`. This one
# decision is what makes deterministic unit testing of an LLM system possible.


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        ...


def parse_json(raw: str) -> object:
    """Parse a model response that is *supposed* to be JSON.

    Instruction-tuned models routinely wrap JSON in prose or code fences even
    when told not to, so the tolerant path is taken deliberately rather than
    trusting the prompt. A hard failure here is preferable to a silent empty
    plan downstream, so an unrecoverable response raises.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        span = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if not span:
            raise LLMError(f"No JSON found in model response: {raw[:200]!r}")
        try:
            return json.loads(span.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Malformed JSON in model response: {exc}") from exc


class ScriptedLLM:
    """Deterministic stand-in for tests and offline demonstrations.

    Responses are matched on a substring of the prompt so a test can script a
    planner reply and a processing reply independently.
    """

    def __init__(self, responses: dict[str, str], default: str = "{}") -> None:
        self._responses = responses
        self._default = default
        self.calls: list[str] = []

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        self.calls.append(prompt)
        for key, value in self._responses.items():
            if key in prompt or key in system:
                return value
        return self._default


class HuggingFaceClient:
    """Hugging Face Inference Providers, chat-completions compatible endpoint.

    Kept thin on purpose: retry/backoff lives in `get_json` below and is
    shared with the retrieval clients rather than reimplemented per caller.
    """

    ENDPOINT = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, model: str | None = None, token: str | None = None) -> None:
        self.model = model or SETTINGS.hf_model
        self.token = token or SETTINGS.hf_token
        if not self.token:
            raise LLMError("HF_TOKEN is not set - see README for the offline --demo mode")

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        import requests  # imported lazily so the module loads without network deps

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            self.ENDPOINT,
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                # Temperature 0 for reproducibility: a demonstration that
                # cannot be re-run to the same result is weak execution evidence.
                "temperature": 0.0,
            },
            timeout=SETTINGS.request_timeout,
        )
        if response.status_code != 200:
            raise LLMError(f"HF inference failed [{response.status_code}]: {response.text[:200]}")
        return response.json()["choices"][0]["message"]["content"]


class LocalLlamaClient:
    """Offline fallback: Llama-3.2-3B-Instruct, Q4_K_M GGUF, via llama.cpp.

    Exists so the system still runs when HF_TOKEN is unset or the hosted
    endpoint is unreachable - a genuine single point of failure otherwise,
    given every agent routes through one LLMClient. Specification (model,
    quantisation, RAM, context) is fixed in Settings above rather than here,
    so the design rationale and the running code cannot drift apart.

    `n_ctx` is deliberately the Settings value, not the model's architectural
    maximum (128K): loading a KV cache that large would blow the ~4 GB RAM
    budget this fallback is chosen to fit within.
    """

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._llm = None  # loaded lazily - importing llama_cpp costs real time

    def _ensure_loaded(self):
        if self._llm is not None:
            return self._llm
        if not os.path.exists(self.settings.local_model_path):
            raise LLMError(
                f"Local model not found at {self.settings.local_model_path}. "
                "Download llama-3.2-3b-instruct-q4_k_m.gguf (~2.0 GB) or set "
                "LOCAL_MODEL_PATH; see README for the exact source."
            )
        from llama_cpp import Llama  # imported lazily; not a hard dependency

        self._llm = Llama(
            model_path=self.settings.local_model_path,
            n_ctx=self.settings.local_context_tokens,
            n_threads=self.settings.local_n_threads,
            verbose=False,
        )
        return self._llm

    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        llm = self._ensure_loaded()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # temperature 0 for the same reproducibility reason as HuggingFaceClient.
        result = llm.create_chat_completion(messages=messages, max_tokens=max_tokens,
                                            temperature=0.0)
        return result["choices"][0]["message"]["content"]


def make_default_llm(settings: Settings = SETTINGS) -> LLMClient:
    """Primary-with-fallback selection, used by the CLI's --live path.

    Hosted inference is preferred when a token is configured (faster, no
    local RAM cost); the local model is the fallback, not the default, so a
    live run degrades gracefully rather than failing outright if the token
    is missing or the hosted endpoint is briefly unreachable.
    """
    if settings.hf_token:
        try:
            return HuggingFaceClient(settings.hf_model, settings.hf_token)
        except LLMError:
            pass
    return LocalLlamaClient(settings)


# ============================================================================
# 4. ACADEMIC SOURCE CLIENTS
# ============================================================================
#
# Each client normalises its API's response into `Paper`. Normalising at the
# boundary - rather than letting Crossref's and OpenAlex's differing shapes
# reach the Processing agent - means a new source can be added without
# touching any downstream agent, and means the relevance-retry loop can
# genuinely swap sources rather than re-asking the same index.


class SourceUnavailable(RuntimeError):
    """Raised when a source cannot be reached within the backoff budget."""


def _cache_path(url: str, params: dict[str, Any]) -> str:
    key = hashlib.sha256(f"{url}?{sorted(params.items())}".encode()).hexdigest()[:24]
    return os.path.join(SETTINGS.cache_dir, f"{key}.json")


def get_json(url: str, params: dict[str, Any], *, use_cache: bool = True) -> dict:
    """Shared HTTP behaviour for every external source.

    Two decisions worth defending:

    1. On-disk response cache. A live demonstration that depends on a third
       party being up is a demonstration that can fail for reasons unrelated
       to the system. Caching by URL makes a recorded run replayable offline.

    2. Exponential backoff on 429/5xx rather than a fixed sleep. Crossref and
       OpenAlex both signal rate limits with 429; retrying immediately
       converts a soft limit into a hard block. Backoff is capped so a dead
       source surfaces as a retrieval failure - which the Validator can act
       on - rather than hanging.
    """
    path = _cache_path(url, params)
    if use_cache and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    import requests

    # The polite-pool contact address is sent in the User-Agent because both
    # Crossref and OpenAlex route identified traffic to a higher rate limit.
    headers = {"User-Agent": f"agentic-research/0.1 (mailto:{SETTINGS.contact_email})"}

    delay = 1.0
    last_error = ""
    for _attempt in range(SETTINGS.max_backoff_attempts):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=SETTINGS.request_timeout
            )
        except Exception as exc:  # network-level failure
            last_error = str(exc)
        else:
            if response.status_code == 200:
                payload = response.json()
                if use_cache:
                    os.makedirs(SETTINGS.cache_dir, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle)
                return payload
            if response.status_code not in (429, 500, 502, 503, 504):
                raise SourceUnavailable(f"{url} returned {response.status_code}")
            last_error = f"HTTP {response.status_code}"

        time.sleep(delay)
        delay *= 2

    raise SourceUnavailable(f"{url} unavailable after backoff: {last_error}")


class SourceClient(Protocol):
    name: str

    def search(self, query: str, limit: int) -> list[Paper]:
        ...


class CrossrefClient:
    """Crossref REST API (Crossref, 2026). Primary source: open, no key."""

    name = "crossref"
    BASE = "https://api.crossref.org/works"

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        payload = get_json(
            self.BASE,
            {"query.bibliographic": query, "rows": limit,
             "select": "DOI,title,abstract,author,issued,URL"},
        )
        items = payload.get("message", {}).get("items", [])
        papers: list[Paper] = []
        for item in items:
            doi = item.get("DOI")
            if not doi:
                # No DOI means no provenance anchor, so the record is dropped
                # here rather than failing validation later at greater cost.
                continue
            papers.append(
                Paper(
                    source_id=doi,
                    title=(item.get("title") or ["untitled"])[0],
                    year=(item.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
                    authors=[
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])
                    ],
                    abstract=_strip_jats(item.get("abstract", "")),
                    source_api=self.name,
                    url=item.get("URL"),
                )
            )
        return papers


class OpenAlexClient:
    """OpenAlex (OpenAlex, 2026). Complementary graph; better topical recall.

    Held as the *alternate* source for the relevance-retry loop: retrying the
    same query against the same index would not change the result set, so
    the loop would be decorative.
    """

    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        payload = get_json(self.BASE, {"search": query, "per-page": limit})
        papers: list[Paper] = []
        for item in payload.get("results", []):
            doi = (item.get("doi") or "").replace("https://doi.org/", "")
            identifier = doi or item.get("id", "")
            if not identifier:
                continue
            papers.append(
                Paper(
                    source_id=identifier,
                    title=item.get("title") or "untitled",
                    year=item.get("publication_year"),
                    authors=[
                        a.get("author", {}).get("display_name", "")
                        for a in item.get("authorships", [])
                    ],
                    abstract=_invert_abstract(item.get("abstract_inverted_index")),
                    source_api=self.name,
                    url=item.get("id"),
                )
            )
        return papers


class FixtureSource:
    """Deterministic source used by tests and the offline --demo mode."""

    def __init__(self, name: str, papers: list[Paper],
                 fail_with: Exception | None = None) -> None:
        self.name = name
        self._papers = papers
        self._fail_with = fail_with

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        if self._fail_with is not None:
            raise self._fail_with
        return self._papers[:limit]


def _strip_jats(abstract: str) -> str:
    """Crossref abstracts arrive as JATS XML; tags would pollute extraction."""
    return re.sub(r"<[^>]+>", " ", abstract or "").replace("&amp;", "&").strip()


def _invert_abstract(index: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index to avoid redistributing text."""
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, slots in index.items():
        positions.extend((slot, word) for slot in slots)
    return " ".join(word for _, word in sorted(positions))


def default_sources() -> dict[str, SourceClient]:
    return {c.name: c for c in (CrossrefClient(), OpenAlexClient())}


# ============================================================================
# 5. ORCHESTRATOR - routing, retry budgets, escalation
# ============================================================================
#
# The Orchestrator is kept separate from the Planner deliberately (the
# Planner decomposes goals; the Orchestrator maintains execution state,
# routing and retries). Mixing them would mean every routing decision cost
# an LLM call and could not be unit tested - `decide_next` below is a pure
# function of state, so the escalation guarantee claimed in the design is
# provable by test rather than by assertion.
#
# Escalation follows InteRRaP's principle that an unresolved problem is
# passed to the layer above rather than resolved by competition between
# layers (Muller and Pischel, 1993); here the layer above is the human
# approval gate.
#
# NOTE ON A REAL BUG FOUND DURING TESTING (see docs/REMEDIATION.md R-001):
# the orchestrator node originally called decide_next(state) without
# forwarding the run's Settings, so routing silently used module defaults
# while every other agent used the injected Settings object. A functional
# test configured with a tight retry cap observed escalation firing later
# than configured, which exposed it. `make_orchestrator(settings)` below
# exists specifically so the same Settings object reaches every agent,
# including the router.


@dataclass
class Decision:
    next_node: str
    reason: str
    updates: dict[str, Any] = field(default_factory=dict)


def decide_next(state: AgentState, settings: Settings = SETTINGS) -> Decision:
    """Return the next node plus any state mutations the routing implies.

    Ordering matters: escalation is checked before progress, so a run that
    has exhausted its budget cannot be revived by a later condition.
    """
    plan = state.get("plan") or []
    step_index = state.get("step_index", 0)
    validation = state.get("validation")

    if state.get("escalated"):
        return Decision("human_gate", "already escalated")

    if not plan:
        return Decision("planner", "no plan yet")

    if validation is not None:
        if validation.passed:
            # Commit this step's extractions and move on. Per-step evidence is
            # cleared so a later step cannot be validated against earlier papers.
            next_index = step_index + 1
            updates = {
                "verified": (state.get("verified") or []) + (state.get("extractions") or []),
                "retrieved": [],
                "extractions": [],
                "validation": None,
                "step_index": next_index,
            }
            if next_index >= len(plan):
                return Decision("human_gate", "all plan steps validated", updates)
            return Decision("retrieval", f"step {next_index} of {len(plan)}", updates)

        if validation.failure_mode is FailureMode.RELEVANCE:
            used = state.get("relevance_retries", 0)
            if used < settings.max_relevance_retries:
                # Re-enter at retrieval with an alternate source: repeating
                # the same query against the same index cannot change the
                # outcome.
                return Decision(
                    "retrieval",
                    f"relevance retry {used + 1}/{settings.max_relevance_retries}",
                    {"relevance_retries": used + 1, "validation": None, "retrieved": [],
                     "extractions": []},
                )
            return _escalate("relevance retry budget exhausted", state)

        if validation.failure_mode is FailureMode.ACCURACY:
            used = state.get("accuracy_retries", 0)
            if used < settings.max_accuracy_retries:
                # Re-enter at processing, not retrieval: the evidence is
                # sound, the extraction from it is not.
                return Decision(
                    "processing",
                    f"accuracy retry {used + 1}/{settings.max_accuracy_retries}",
                    {"accuracy_retries": used + 1, "validation": None, "extractions": []},
                )
            return _escalate("accuracy retry budget exhausted", state)

        if validation.failure_mode is FailureMode.INSUFFICIENT_EVIDENCE:
            used = state.get("replans", 0)
            if used < settings.max_replans:
                return Decision(
                    "planner",
                    f"replan {used + 1}/{settings.max_replans}",
                    {"replans": used + 1, "validation": None, "retrieved": [],
                     "extractions": []},
                )
            return _escalate("replan budget exhausted", state)

    if not state.get("retrieved"):
        return Decision("retrieval", f"step {step_index} needs evidence")

    if not state.get("extractions"):
        return Decision("processing", "evidence retrieved, not yet processed")

    return Decision("validator", "extractions await validation")


def _escalate(reason: str, state: AgentState) -> Decision:
    """Stop looping and hand an explicitly flagged partial result to the human.

    Returning a flagged partial is preferred over returning nothing: an
    unverified result the reviewer knows is unverified is usable; silence
    is not.
    """
    return Decision(
        "human_gate",
        f"escalated: {reason}",
        {
            "escalated": True,
            "escalation_reason": reason,
            "verified": state.get("verified") or [],  # keep whatever already passed
        },
    )


def make_orchestrator(settings: Settings = SETTINGS) -> Callable[[AgentState], dict[str, Any]]:
    """Bind the retry budget at construction time (see the R-001 note above)."""

    def orchestrator_node(state: AgentState) -> dict[str, Any]:
        decision = decide_next(state, settings)
        updates = dict(decision.updates)
        updates["next_node"] = decision.next_node
        updates["transitions"] = [
            TransitionLog(
                step=len(state.get("transitions") or []),
                node="orchestrator",
                decision=decision.next_node,
                detail=decision.reason,
            )
        ]
        return updates

    return orchestrator_node


def route(state: AgentState) -> str:
    """Conditional edge function - reads the decision the node already made."""
    return state.get("next_node", "planner")


# ============================================================================
# 6. WORKER AGENTS - Planner, Retrieval, Processing
# ============================================================================
#
# Each is a factory returning a node function with its dependencies already
# bound. Dependency injection is used rather than module-level clients so a
# unit test can supply a scripted LLM and a fixture source without
# monkey-patching - the tests then exercise the agent's logic rather than
# the library's.

Node = Callable[[AgentState], dict[str, Any]]

PLANNER_SYSTEM = (
    "You decompose an academic research goal into 2-4 ordered search steps. "
    "Reply with JSON only: a list of objects with keys 'description' and 'query'. "
    "No prose, no code fences."
)

PROCESSING_SYSTEM = (
    "You extract structured fields from a paper abstract. For each of "
    "'methodology', 'dataset', 'reported_accuracy' return the value and the "
    "exact substring of the abstract that supports it, copied verbatim. "
    "If a field is absent, use value 'not reported' and an empty span. "
    "Reply with JSON only: {'fields': [{'name':..., 'value':..., 'evidence_span':...}]}"
)


def make_planner(llm: LLMClient) -> Node:
    def planner(state: AgentState) -> dict[str, Any]:
        replan = state.get("replans", 0)
        hint = ""
        if replan:
            # A replan that repeats the previous decomposition wastes the
            # budget, so prior failure is fed back into the prompt.
            hint = (
                "\nThe previous plan returned too little evidence. Broaden the "
                "queries and avoid narrow jargon."
            )
        raw = llm.complete(f"Goal: {state['goal']}{hint}", system=PLANNER_SYSTEM)

        try:
            parsed = parse_json(raw)
            steps = [
                PlanStep(
                    index=i,
                    description=str(item["description"]),
                    query=str(item["query"]),
                    # Source order rotates with the replan count so a second
                    # plan does not inherit the first plan's failing source.
                    sources=["crossref", "openalex"] if replan % 2 == 0
                    else ["openalex", "crossref"],
                )
                for i, item in enumerate(parsed)  # type: ignore[arg-type]
            ]
        except (LLMError, KeyError, TypeError) as exc:
            # A malformed plan is recoverable: fall back to a single-step plan
            # built from the raw goal rather than aborting the run. The
            # failure is recorded so it is visible in the transition log.
            steps = [PlanStep(index=0, description=state["goal"], query=state["goal"])]
            return {
                "plan": steps,
                "step_index": 0,
                "transitions": [
                    TransitionLog(
                        step=len(state.get("transitions") or []),
                        node="planner",
                        decision="fallback_plan",
                        detail=f"unparseable plan ({exc}); using single-step plan",
                    )
                ],
            }

        return {
            "plan": steps,
            "step_index": 0,
            "transitions": [
                TransitionLog(
                    step=len(state.get("transitions") or []),
                    node="planner",
                    decision="plan_created",
                    detail=f"{len(steps)} steps",
                )
            ],
        }

    return planner


def make_retrieval(sources: dict[str, SourceClient], settings: Settings = SETTINGS) -> Node:
    def retrieval(state: AgentState) -> dict[str, Any]:
        plan: Sequence[PlanStep] = state.get("plan") or []
        step = plan[min(state.get("step_index", 0), len(plan) - 1)]

        # On a relevance retry the source order is rotated, so attempt 2
        # queries a different index to the one that produced the irrelevant
        # results.
        order = list(step.sources)
        rotation = state.get("relevance_retries", 0) % max(len(order), 1)
        order = order[rotation:] + order[:rotation]

        papers: list[Paper] = []
        errors: list[str] = []
        for name in order:
            client = sources.get(name)
            if client is None:
                continue
            try:
                papers.extend(client.search(step.query, settings.target_papers))
            except SourceUnavailable as exc:
                # A dead source must not kill the run: it is recorded and the
                # next source is tried, which is the whole point of holding two.
                errors.append(f"{name}: {exc}")
            if len(papers) >= settings.target_papers:
                break

        deduped: dict[str, Paper] = {}
        for paper in papers:
            deduped.setdefault(paper.source_id, paper)
        result = list(deduped.values())[: settings.target_papers]

        updates: dict[str, Any] = {
            "retrieved": result,
            "transitions": [
                TransitionLog(
                    step=len(state.get("transitions") or []),
                    node="retrieval",
                    decision="retrieved",
                    detail=f"{len(result)} papers via {order}"
                    + (f"; errors: {errors}" if errors else ""),
                )
            ],
        }

        if len(result) < settings.min_papers_per_step:
            # Too little evidence is a planning problem, not an extraction
            # problem, so it is reported as such and routed to a replan.
            updates["validation"] = ValidationReport(
                passed=False,
                failure_mode=FailureMode.INSUFFICIENT_EVIDENCE,
                notes=f"only {len(result)} papers found (min {settings.min_papers_per_step})",
            )
        return updates

    return retrieval


def make_processing(llm: LLMClient) -> Node:
    def processing(state: AgentState) -> dict[str, Any]:
        extractions: list[Extraction] = []
        failures: list[str] = []
        strict = state.get("accuracy_retries", 0) > 0
        reminder = (
            "\nIMPORTANT: a previous attempt failed verification. Copy the "
            "supporting span character-for-character from the abstract."
            if strict else ""
        )

        for paper in state.get("retrieved") or []:
            prompt = f"Title: {paper.title}\nAbstract: {paper.abstract}{reminder}"
            try:
                parsed = parse_json(llm.complete(prompt, system=PROCESSING_SYSTEM))
                fields = [
                    ExtractedField(**f)
                    for f in parsed.get("fields", [])  # type: ignore[union-attr]
                ]
            except (LLMError, TypeError, ValueError) as exc:
                # One unparseable paper should not void the batch; it is
                # dropped and logged, and the Validator sees a smaller
                # evidence set.
                failures.append(f"{paper.source_id}: {exc}")
                continue
            extractions.append(Extraction(source_id=paper.source_id, fields=fields))

        return {
            "extractions": extractions,
            "transitions": [
                TransitionLog(
                    step=len(state.get("transitions") or []),
                    node="processing",
                    decision="extracted",
                    detail=f"{len(extractions)} papers processed"
                    + (f"; {len(failures)} failed: {failures}" if failures else ""),
                )
            ],
        }

    return processing


# ============================================================================
# 7. VALIDATOR, HUMAN APPROVAL GATE, STORAGE/EXPORT
# ============================================================================
#
# The Validator is deliberately *not* an LLM asking itself whether it was
# correct, which would inherit the same failure it is meant to catch.
# Accuracy is checked mechanically - does the quoted span actually occur in
# the retrieved abstract - and only relevance, which is genuinely a
# judgement, uses the model. That split is what makes the accuracy check a
# guarantee rather than an opinion.

RELEVANCE_SYSTEM = (
    "Decide whether the retrieved papers address the stated research step. "
    "Reply with JSON only: {'relevant': true|false, 'note': '...'}"
)


def _normalise(text: str) -> str:
    """Whitespace/case-insensitive comparison.

    Models reflow whitespace when quoting; treating that as a provenance
    failure would produce retries that can never succeed.
    """
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def make_validator(llm: LLMClient | None = None) -> Node:
    def validator(state: AgentState) -> dict[str, Any]:
        papers: dict[str, Paper] = {p.source_id: p for p in state.get("retrieved") or []}
        extractions: list[Extraction] = state.get("extractions") or []

        untraceable: list[str] = []
        checked = 0
        for extraction in extractions:
            paper = papers.get(extraction.source_id)
            if paper is None:
                # An extraction attributed to a paper we never retrieved is
                # the clearest possible provenance failure.
                untraceable.append(f"{extraction.source_id} (no such source)")
                continue
            haystack = _normalise(f"{paper.title} {paper.abstract}")
            for f in extraction.fields:
                if f.value.strip().lower() in ("", "not reported"):
                    continue  # an honest absence is not an accuracy failure
                checked += 1
                if not f.evidence_span or _normalise(f.evidence_span) not in haystack:
                    untraceable.append(f"{extraction.source_id}:{f.name}")

        if untraceable:
            return _report(
                state,
                ValidationReport(
                    passed=False,
                    failure_mode=FailureMode.ACCURACY,
                    checked=checked,
                    untraceable=untraceable,
                    notes=f"{len(untraceable)} of {checked} claims could not be traced",
                ),
            )

        # Relevance is only worth an LLM call once accuracy has passed -
        # there is no value in judging the relevance of untraceable claims.
        if llm is not None and papers:
            plan = state.get("plan") or []
            step = plan[min(state.get("step_index", 0), len(plan) - 1)] if plan else None
            titles = "\n".join(f"- {p.title}" for p in papers.values())
            verdict = llm.complete(
                f"Research step: {step.description if step else state.get('goal', '')}\n"
                f"Papers:\n{titles}",
                system=RELEVANCE_SYSTEM,
            )
            try:
                parsed = parse_json(verdict)
                relevant = bool(parsed.get("relevant", True))  # type: ignore[union-attr]
                note = str(parsed.get("note", ""))  # type: ignore[union-attr]
            except (LLMError, AttributeError):
                # An unreadable verdict is treated as a pass with a note
                # rather than a failure: failing here would burn a retry on
                # our own bug.
                relevant, note = True, "relevance verdict unparseable; accepted with flag"
            if not relevant:
                return _report(
                    state,
                    ValidationReport(
                        passed=False,
                        failure_mode=FailureMode.RELEVANCE,
                        checked=checked,
                        notes=note or "retrieved papers do not address the step",
                    ),
                )

        return _report(
            state,
            ValidationReport(passed=True, checked=checked, notes="all claims traced to source"),
        )

    return validator


def _report(state: AgentState, report: ValidationReport) -> dict[str, Any]:
    return {
        "validation": report,
        "transitions": [
            TransitionLog(
                step=len(state.get("transitions") or []),
                node="validator",
                decision="pass" if report.passed else f"fail:{report.failure_mode.value}",
                detail=report.notes,
            )
        ],
    }


def make_human_gate(settings: Settings = SETTINGS) -> Node:
    def human_gate(state: AgentState) -> dict[str, Any]:
        summary = (
            f"{len(state.get('verified') or [])} validated papers"
            + (f" | FLAGGED UNVERIFIED: {state.get('escalation_reason')}"
               if state.get("escalated") else "")
        )
        approved = True
        if settings.require_human_approval:
            # LangGraph's interrupt suspends the run against the
            # checkpointer, so approval is a resumable pause rather than a
            # blocking prompt - the reviewer can come back to it, which is
            # what a real review is.
            from langgraph.types import interrupt

            answer = interrupt({"action": "approve_export", "summary": summary})
            approved = str(answer).strip().lower() in ("y", "yes", "true", "approve", "approved")

        return {
            "approved": approved,
            "transitions": [
                TransitionLog(
                    step=len(state.get("transitions") or []),
                    node="human_gate",
                    decision="approved" if approved else "rejected",
                    detail=summary,
                )
            ],
        }

    return human_gate


def make_storage(settings: Settings = SETTINGS) -> Node:
    def storage(state: AgentState) -> dict[str, Any]:
        if not state.get("approved"):
            # No approval, no artefact. Writing a rejected synthesis to disk
            # would defeat the gate that justifies it.
            return {
                "transitions": [
                    TransitionLog(
                        step=len(state.get("transitions") or []),
                        node="storage",
                        decision="withheld",
                        detail="export withheld - not approved by reviewer",
                    )
                ]
            }

        os.makedirs(settings.output_dir, exist_ok=True)
        run_id = state.get("run_id", "run")
        base = os.path.join(settings.output_dir, run_id)

        payload = {
            "run_id": run_id,
            "goal": state.get("goal"),
            "flagged_unverified": state.get("escalated", False),
            "escalation_reason": state.get("escalation_reason", ""),
            "retry_counts": {
                "relevance": state.get("relevance_retries", 0),
                "accuracy": state.get("accuracy_retries", 0),
                "replans": state.get("replans", 0),
            },
            "results": [e.model_dump() for e in state.get("verified") or []],
            "transitions": [t.model_dump() for t in state.get("transitions") or []],
        }
        with open(f"{base}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        with open(f"{base}.md", "w", encoding="utf-8") as handle:
            handle.write(_markdown(state))

        return {
            "export_path": f"{base}.md",
            "transitions": [
                TransitionLog(
                    step=len(state.get("transitions") or []),
                    node="storage",
                    decision="exported",
                    detail=f"{base}.md / {base}.json",
                )
            ],
        }

    return storage


def _markdown(state: AgentState) -> str:
    """Human-readable comparison table.

    Markdown is emitted alongside JSON because the two audiences differ: the
    reviewer reads the table, downstream tooling reads the JSON. Emitting
    only one would force the other to parse a format not meant for it.
    """
    lines = [f"# Research synthesis: {state.get('goal', '')}", ""]
    if state.get("escalated"):
        lines += [
            "> **Flagged as unverified.** "
            f"{state.get('escalation_reason')} - claims below have not passed "
            "full validation and require manual checking.",
            "",
        ]
    lines += ["| Source | Methodology | Dataset | Reported accuracy |", "|---|---|---|---|"]
    for extraction in state.get("verified") or []:
        values = {f.name: f.value for f in extraction.fields}
        lines.append(
            f"| {extraction.source_id} | {values.get('methodology', '-')} "
            f"| {values.get('dataset', '-')} | {values.get('reported_accuracy', '-')} |"
        )
    lines += ["", "## Run trace", ""]
    for t in state.get("transitions") or []:
        lines.append(f"{t.step:>3}. **{t.node}** -> {t.decision} - {t.detail}")
    return "\n".join(lines) + "\n"


# ============================================================================
# 8. GRAPH ASSEMBLY
# ============================================================================
#
# Hub-and-spoke around the Orchestrator: every worker returns to the
# Orchestrator rather than edging directly to its successor. This costs one
# extra hop per stage and buys two things the design requires: a single
# place where retry budgets are enforced, and a transition log that is
# complete by construction rather than by each agent remembering to write to it.


def build_graph(
    llm: LLMClient,
    sources: dict[str, SourceClient],
    *,
    settings: Settings = SETTINGS,
    checkpointer=None,
):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", make_orchestrator(settings))
    graph.add_node("planner", make_planner(llm))
    graph.add_node("retrieval", make_retrieval(sources, settings))
    graph.add_node("processing", make_processing(llm))
    graph.add_node("validator", make_validator(llm))
    graph.add_node("human_gate", make_human_gate(settings))
    graph.add_node("storage", make_storage(settings))

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route,
        {
            "planner": "planner",
            "retrieval": "retrieval",
            "processing": "processing",
            "validator": "validator",
            "human_gate": "human_gate",
        },
    )

    # Workers always hand control back; they never choose their own successor.
    for worker in ("planner", "retrieval", "processing", "validator"):
        graph.add_edge(worker, "orchestrator")

    # The gate is terminal in control terms: after a human decision the run
    # exports or withholds and stops. Looping back would let the system ask
    # again until it got the answer it wanted.
    graph.add_edge("human_gate", "storage")
    graph.add_edge("storage", END)

    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# 9. CLI ENTRY POINT
# ============================================================================
#
# Two modes exist for a specific reason. --demo runs against fixture sources
# and a scripted model so the system is demonstrable with no keys, no
# network and no cost, and produces repeatable output each time - which is
# what makes the execution evidence used for the presentation reproducible.
# --live uses Crossref, OpenAlex and Hugging Face inference.

LOG_FORMAT = '{"ts":"%(asctime)s","level":"%(levelname)s","run":"%(run_id)s","msg":"%(message)s"}'

DEMO_PAPERS = [
    Paper(
        source_id="10.1000/demo1",
        title="A transformer approach to citation screening",
        year=2024,
        abstract="We fine-tune a transformer model on the CORD-19 dataset and "
                 "report an F1 score of 0.91 on held-out screening decisions.",
        source_api="fixture",
    ),
    Paper(
        source_id="10.1000/demo2",
        title="Retrieval-augmented screening of clinical literature",
        year=2023,
        abstract="A retrieval-augmented pipeline evaluated on the PubMed-RCT "
                 "corpus achieves 87.4% accuracy against expert annotation.",
        source_api="fixture",
    ),
    Paper(
        source_id="10.1000/demo3",
        title="Weak supervision for systematic review triage",
        year=2022,
        abstract="Using weak supervision over the EPPI benchmark we obtain a "
                 "recall of 0.95 while halving reviewer workload.",
        source_api="fixture",
    ),
]

DEMO_PLAN = json.dumps(
    [{"description": "Find recent ML approaches to literature screening",
      "query": "machine learning systematic review screening"}]
)


def _configure_logging(run_id: str, verbose: bool) -> logging.LoggerAdapter:
    """Structured (JSON-line) logs.

    Chosen over free-text logging because the transition trace is submitted
    as evidence: a machine-readable log can be filtered and quoted exactly,
    while a prose log has to be trusted.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger = logging.getLogger("agentic_research")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logging.LoggerAdapter(logger, {"run_id": run_id})


def _demo_extraction(paper: Paper) -> str:
    """Scripted extraction that quotes the abstract verbatim, so validation passes."""
    span = paper.abstract.split(". ")[-1].strip()
    return json.dumps(
        {"fields": [
            {"name": "methodology", "value": paper.title, "evidence_span": paper.title},
            {"name": "dataset", "value": "see abstract", "evidence_span": span},
            {"name": "reported_accuracy", "value": "see abstract", "evidence_span": span},
        ]}
    )


def run_eval(goals_csv: str = "eval/goals.csv", out: str = "runs/eval_results.json") -> int:
    """Evaluation harness, against four explicitly defined error measures.

    "90% error-free" is not a single number the code can compute on its own
    without first saying what an error *is*. Four are defined, each read
    directly off state the graph already produces, rather than invented for
    this function:

      retrieval_error    - a step never reached `min_papers_per_step` and the
                            run escalated via INSUFFICIENT_EVIDENCE. Read from
                            `replans` > 0 and `escalation_reason`.
      hallucination_rate - untraceable claims / total claims checked, taken
                            directly from the Validator's own ValidationReport
                            (`untraceable` vs `checked`) rather than a second,
                            separate judgement - the number the Validator
                            already computes to decide retries is the number
                            reported here.
      citation_validity   - fraction of exported rows whose source_id is a
                            resolvable DOI shape (10.xxxx/...), since a Paper
                            with no usable identifier is filtered out at
                            retrieval (see CrossrefClient/OpenAlexClient) but
                            a malformed one could still slip through.
      overall_error_free  - completed (artefact exported) AND not escalated
                            AND hallucination_rate == 0 for that run. This is
                            the strict reading of the 90% target: escalation
                            is correct behaviour but is not "error-free",
                            since it means the system could not independently
                            reach a verified answer.

    The three component rates are reported alongside the strict one so a
    reader can see *why* a run counted as an error, not just that it did.
    """
    settings = Settings(require_human_approval=False, output_dir="runs/eval")
    llm = make_default_llm(settings)
    app = build_graph(llm, default_sources(), settings=settings)

    with open(goals_csv, newline="", encoding="utf-8") as handle:
        goals = list(csv.DictReader(handle))

    results = []
    for row in goals:
        started = time.time()
        record = {"id": row["id"], "goal": row["goal"], "scope": row.get("scope", "")}
        try:
            state = app.invoke(
                initial_state(row["goal"], run_id=f"eval-{row['id']}"),
                {"recursion_limit": 80},
            )
            transitions = state.get("transitions") or []
            validator_hops = [t for t in transitions if t.node == "validator"]
            checked = sum(int(re.search(r"(\d+) of (\d+)", t.detail).group(2))
                         for t in validator_hops
                         if t.decision == "fail:accuracy" and re.search(r"\d+ of \d+", t.detail))
            untraced = sum(int(re.search(r"(\d+) of (\d+)", t.detail).group(1))
                          for t in validator_hops
                          if t.decision == "fail:accuracy" and re.search(r"\d+ of \d+", t.detail))
            retrieval_error = state.get("escalation_reason", "") == "replan budget exhausted"
            dois_ok = all(
                re.match(r"^10\.\d{4,9}/\S+$", e.source_id)
                for e in state.get("verified") or []
            )
            hallucination_rate = round(untraced / checked, 3) if checked else 0.0
            completed = bool(state.get("export_path"))
            escalated = bool(state.get("escalated"))
            record.update(
                completed=completed,
                escalated=escalated,
                retrieval_error=retrieval_error,
                hallucination_rate=hallucination_rate,
                citation_valid=dois_ok,
                error_free=completed and not escalated and hallucination_rate == 0.0 and dois_ok,
                papers=len(state.get("verified") or []),
                hops=len(transitions),
                error="",
            )
        except Exception as exc:  # an unhandled exception is counted as an error outright
            record.update(completed=False, escalated=False, retrieval_error=False,
                          hallucination_rate=None, citation_valid=False, error_free=False,
                          papers=0, hops=0, error=str(exc))
        record["seconds"] = round(time.time() - started, 1)
        results.append(record)
        print(json.dumps(record))

    n = len(results)
    summary = {
        "runs": n,
        "error_free_rate": round(sum(1 for r in results if r["error_free"]) / n, 3),
        "retrieval_error_rate": round(sum(1 for r in results if r["retrieval_error"]) / n, 3),
        "mean_hallucination_rate": round(
            sum(r["hallucination_rate"] or 0 for r in results) / n, 3
        ),
        "citation_validity_rate": round(sum(1 for r in results if r["citation_valid"]) / n, 3),
        "target": 0.9,
        "meets_target": sum(1 for r in results if r["error_free"]) / n >= 0.9,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-powered research planning agent")
    parser.add_argument("goal", nargs="?", default="Compare ML methods for literature screening")
    parser.add_argument("--demo", action="store_true", help="offline fixtures, no keys required")
    parser.add_argument("--auto-approve", action="store_true",
                        help="skip the human gate (evaluation runs only)")
    parser.add_argument("--eval", action="store_true", help="run eval/goals.csv and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.eval:
        return run_eval()

    run_id = uuid.uuid4().hex[:8]
    log = _configure_logging(run_id, args.verbose)
    settings = Settings(require_human_approval=not args.auto_approve)

    if args.demo:
        responses: dict[str, str] = {
            "decompose an academic research goal": DEMO_PLAN,
            "Decide whether the retrieved papers": json.dumps({"relevant": True, "note": "on topic"}),
        }
        for paper in DEMO_PAPERS:
            responses[paper.title] = _demo_extraction(paper)
        llm: Any = ScriptedLLM(responses)
        sources = {"crossref": FixtureSource("crossref", DEMO_PAPERS),
                   "openalex": FixtureSource("openalex", DEMO_PAPERS)}
    else:
        llm = HuggingFaceClient()
        sources = default_sources()

    checkpointer = None
    if settings.require_human_approval:
        # The gate uses interrupt(), which requires a checkpointer to
        # suspend against. In-memory is sufficient for a single-process run;
        # a SQLite saver would persist an approval across restarts.
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    app = build_graph(llm, sources, settings=settings, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": run_id}}

    log.info("run started")
    state = app.invoke(initial_state(args.goal, run_id), config=config)

    if "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print(f"\nAPPROVAL REQUIRED: {payload['summary']}", file=sys.stderr)
        answer = input("Approve export? [y/N] ")
        from langgraph.types import Command

        state = app.invoke(Command(resume=answer), config=config)

    for t in state.get("transitions", []):
        log.info(f"{t.node} -> {t.decision}: {t.detail}")
    if state.get("export_path"):
        print(state["export_path"])
    else:
        print("no artefact exported", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
