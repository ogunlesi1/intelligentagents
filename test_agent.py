"""
Core test suite for agent.py.

Run with: pytest test_agent.py -v

This is the submission-facing suite: one test per distinct behaviour, chosen
to demonstrate the properties the design claims rather than to maximise a
count. It's deliberately smaller than the full regression suite in
test_agent_extended.py (edge cases, alternate inputs, additional agents) -
run both together for full coverage: `pytest -v`.

Each agent gets at least its happy path and its most important failure path.
The orchestrator gets the most attention because it's the one component
making a provable claim (the system always terminates and escalates
correctly), so that claim needs to be demonstrated, not just asserted.
"""

from __future__ import annotations

import json

import pytest

import agent as m
from conftest import llm_for_run, run_graph


# ============================================================================
# Orchestrator - decide_next() is a pure function of state, so these run
# without a model or network. This is the strongest evidence in the suite:
# it proves the escalation/termination guarantee the design relies on.
# ============================================================================


def test_routes_to_planner_when_no_plan(base_state, settings):
    base_state["plan"] = []
    assert m.decide_next(base_state, settings).next_node == "planner"


def test_relevance_failure_re_enters_at_retrieval_not_accuracy_budget(base_state, settings):
    """Proves the two recovery loops are independent, as the design claims."""
    base_state["validation"] = m.ValidationReport(passed=False,
                                                   failure_mode=m.FailureMode.RELEVANCE)
    decision = m.decide_next(base_state, settings)
    assert decision.next_node == "retrieval"
    assert decision.updates["relevance_retries"] == 1
    assert "accuracy_retries" not in decision.updates


def test_accuracy_failure_re_enters_at_processing_keeping_evidence(base_state, settings, papers):
    base_state["retrieved"] = papers
    base_state["validation"] = m.ValidationReport(passed=False,
                                                   failure_mode=m.FailureMode.ACCURACY)
    decision = m.decide_next(base_state, settings)
    assert decision.next_node == "processing"
    assert "retrieved" not in decision.updates  # evidence is kept, only re-processed


def test_insufficient_evidence_triggers_replan(base_state, settings):
    base_state["validation"] = m.ValidationReport(
        passed=False, failure_mode=m.FailureMode.INSUFFICIENT_EVIDENCE)
    assert m.decide_next(base_state, settings).next_node == "planner"


@pytest.mark.parametrize(
    "mode,counter",
    [(m.FailureMode.RELEVANCE, "relevance_retries"),
     (m.FailureMode.ACCURACY, "accuracy_retries"),
     (m.FailureMode.INSUFFICIENT_EVIDENCE, "replans")],
)
def test_exhausted_budget_escalates_to_human(base_state, settings, mode, counter):
    """The termination guarantee, for all three failure modes. Without this
    the central design claim - the system always terminates - is unproven."""
    cap = {"relevance_retries": settings.max_relevance_retries,
           "accuracy_retries": settings.max_accuracy_retries,
           "replans": settings.max_replans}[counter]
    base_state[counter] = cap
    base_state["validation"] = m.ValidationReport(passed=False, failure_mode=mode)
    decision = m.decide_next(base_state, settings)
    assert decision.next_node == "human_gate"
    assert decision.updates["escalated"] is True


def test_orchestrator_node_uses_injected_settings(base_state, settings):
    """Regression test for a real defect (see README): the router must use
    the injected budget, not a module-level default, or escalation fires at
    the wrong point."""
    base_state["relevance_retries"] = settings.max_relevance_retries
    base_state["validation"] = m.ValidationReport(passed=False,
                                                   failure_mode=m.FailureMode.RELEVANCE)
    result = m.make_orchestrator(settings)(base_state)
    assert result["escalated"] is True  # would be False if the default (cap=3) were used


# ============================================================================
# Planner
# ============================================================================


def test_planner_parses_plan(base_state, plan_json):
    node = m.make_planner(m.ScriptedLLM({"decompose": plan_json}))
    result = node(base_state)
    assert [s.query for s in result["plan"]] == ["ml screening", "screening benchmark"]


def test_planner_falls_back_on_malformed_response(base_state):
    """A broken model response degrades to a single-step plan, not a crash."""
    node = m.make_planner(m.ScriptedLLM({"decompose": "I'm sorry, I can't do that."}))
    result = node(base_state)
    assert len(result["plan"]) == 1
    assert result["transitions"][0].decision == "fallback_plan"


# ============================================================================
# Retrieval
# ============================================================================


def test_retrieval_flags_insufficient_evidence(base_state, settings, papers):
    node = m.make_retrieval({"crossref": m.FixtureSource("crossref", papers[:1])}, settings)
    result = node(base_state)
    assert result["validation"].failure_mode is m.FailureMode.INSUFFICIENT_EVIDENCE


def test_retrieval_survives_a_dead_source(base_state, settings, papers):
    """A 429 or outage on one source must not end the run."""
    node = m.make_retrieval(
        {"crossref": m.FixtureSource("crossref", [], fail_with=m.SourceUnavailable("HTTP 429")),
         "openalex": m.FixtureSource("openalex", papers)},
        settings,
    )
    result = node(base_state)
    assert len(result["retrieved"]) == 3


# ============================================================================
# Processing
# ============================================================================


def test_processing_drops_unparseable_paper_without_failing_batch(base_state, papers):
    base_state["retrieved"] = papers
    good = json.dumps({"fields": [
        {"name": "dataset", "value": "SQuAD", "evidence_span": "SQuAD dataset"}]})
    node = m.make_processing(m.ScriptedLLM({"Paper A": "not json at all", "Abstract": good}))
    result = node(base_state)
    assert len(result["extractions"]) == 2


# ============================================================================
# Validator - the integrity-critical agent, checked from both directions:
# a true claim must pass, a fabricated one must not.
# ============================================================================


def test_validator_passes_traceable_claims(base_state, papers, traceable_extraction):
    base_state["retrieved"] = papers
    base_state["extractions"] = traceable_extraction
    node = m.make_validator(m.ScriptedLLM({"Decide": json.dumps({"relevant": True})}))
    assert node(base_state)["validation"].passed


def test_validator_rejects_fabricated_span(base_state, papers):
    """The core integrity check: an invented quotation must not pass."""
    base_state["retrieved"] = papers
    base_state["extractions"] = [m.Extraction(
        source_id="10.1/a",
        fields=[m.ExtractedField(name="reported_accuracy", value="99.9%",
                                 evidence_span="achieves 99.9% on all benchmarks")])]
    report = m.make_validator(None)(base_state)["validation"]
    assert not report.passed
    assert report.failure_mode is m.FailureMode.ACCURACY


def test_validator_flags_irrelevant_results(base_state, papers, traceable_extraction):
    base_state["retrieved"] = papers
    base_state["extractions"] = traceable_extraction
    node = m.make_validator(m.ScriptedLLM(
        {"Decide": json.dumps({"relevant": False, "note": "off topic"})}))
    assert node(base_state)["validation"].failure_mode is m.FailureMode.RELEVANCE


# ============================================================================
# Human gate and storage
# ============================================================================


def test_storage_withholds_export_without_approval(base_state, settings, tmp_path):
    settings = m.Settings(**{**settings.__dict__, "output_dir": str(tmp_path)})
    base_state["approved"] = False
    result = m.make_storage(settings)(base_state)
    assert result["transitions"][0].decision == "withheld"
    assert not list(tmp_path.iterdir())


def test_escalated_export_is_flagged_in_the_artefact(base_state, settings, tmp_path,
                                                      traceable_extraction):
    settings = m.Settings(**{**settings.__dict__, "output_dir": str(tmp_path)})
    base_state.update(approved=True, verified=traceable_extraction, escalated=True,
                      escalation_reason="accuracy retry budget exhausted")
    m.make_storage(settings)(base_state)
    text = (tmp_path / "test.md").read_text()
    assert "Flagged as unverified" in text


# ============================================================================
# Functional - the whole graph, stubbed only at its edges (model, sources).
# These exercise the wiring that unit tests of individual agents cannot
# reach: conditional edges, state reducers, loop re-entry.
# ============================================================================


def test_happy_path_exports_artefact(papers, auto_settings):
    state = run_graph(llm_for_run(papers), {"crossref": m.FixtureSource("crossref", papers)},
                      auto_settings)
    assert state["export_path"].endswith(".md")
    assert not state["escalated"]
    assert len(state["verified"]) == 3


def test_accuracy_loop_fires_then_escalates(papers, auto_settings):
    """The 'adaptive loop fired' evidence used in the presentation comes
    from exactly this run - it is not a hypothetical capture."""
    state = run_graph(llm_for_run(papers, traceable=False),
                      {"crossref": m.FixtureSource("crossref", papers)}, auto_settings)
    retries = [t for t in state["transitions"]
               if t.node == "orchestrator" and t.detail.startswith("accuracy retry")]
    assert len(retries) == auto_settings.max_accuracy_retries
    assert state["escalated"]
    assert state["export_path"].endswith(".md")  # still terminates with a flagged artefact


def test_run_always_terminates_under_worst_case_failures(papers, auto_settings):
    """If this test ever needs its recursion limit raised, the bounded-
    recovery claim in the design has been broken."""
    state = run_graph(llm_for_run(papers, traceable=False, relevant=False),
                      {"crossref": m.FixtureSource("crossref", papers)}, auto_settings)
    assert state["escalated"]
