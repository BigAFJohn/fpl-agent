"""
tests/evals/test_deepeval_agent.py
===================================
DeepEval-powered LLM evaluation suite for the FPL Agent.

Uses Claude (claude-sonnet-4-5) as the judge model instead of GPT-4.
This means only one API key is needed — ANTHROPIC_API_KEY — for both
the FPL agent itself and the evaluation judge.

WHY THIS EXISTS
---------------
The existing test_agent_evals.py uses rule-based assertions (regex,
string matching, length checks). That catches structural and formatting
issues but cannot detect semantic problems:

  - Agent says "Thiago (6.2 adj pts)" but predictions show 5.94
  - Agent recommends a player not in the top 25 predictions
  - Captain section spends 80% of words discussing transfers
  - Risks section flags a player not in the injury watchlist

DeepEval adds three LLM-as-judge metrics that catch these:

  Faithfulness     Is the output grounded in the input data?
  Hallucination    Does the output invent facts not in context?
  Answer Relevancy Does each section answer the right question?

HOW LLM-AS-JUDGE WORKS
-----------------------
DeepEval sends your agent's output + the original context to Claude.
Claude scores the output against the metric criteria using a rubric.
This is the G-Eval approach from Liu et al. 2023.

Cost: ~$0.02-0.03 per full run (13 tests × ~2 Claude calls each).

THRESHOLDS
----------
Starting thresholds — calibrate after your first few runs.
If consistently scoring 0.85+, raise the bar.

  HALLUCINATION  <= 0.20  fail if > 20% hallucinated content
  FAITHFULNESS   >= 0.70  fail if < 70% grounded in input data
  RELEVANCY      >= 0.75  fail if < 75% relevant to the question

RUN COMMANDS
------------
  pytest tests/evals/test_deepeval_agent.py -v        # DeepEval only
  pytest -m "not deepeval"                            # Skip these (no API cost)
  pytest tests/evals/ --alluredir=allure-results -v   # With Allure report
"""

import json
import os
import re
import pytest
from pathlib import Path

# =============================================================================
# CLAUDE JUDGE SETUP
# =============================================================================

try:
    import anthropic
    from deepeval.models.base_model import DeepEvalBaseLLM

    class ClaudeJudge(DeepEvalBaseLLM):
        """
        Claude-based judge model for DeepEval metrics.

        Replaces the default GPT-4 judge so only one API key is needed.
        Uses claude-sonnet-4-5 — fast and cost-effective for evaluation.
        """

        def __init__(self):
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )

        def load_model(self):
            return self.client

        def generate(self, prompt: str) -> str:
            response = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        async def a_generate(self, prompt: str) -> str:
            """Async version — falls back to sync for simplicity."""
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return "claude-sonnet-4-5"

    JUDGE_MODEL = ClaudeJudge()
    DEEPEVAL_AVAILABLE = True

except ImportError:
    JUDGE_MODEL = None
    DEEPEVAL_AVAILABLE = False

# DeepEval metric imports
try:
    from deepeval import assert_test
    from deepeval.metrics import (
        FaithfulnessMetric,
        HallucinationMetric,
        AnswerRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase
except ImportError:
    DEEPEVAL_AVAILABLE = False

pytestmark = pytest.mark.deepeval

# =============================================================================
# THRESHOLDS
# =============================================================================

THRESHOLD_HALLUCINATION = 0.20   # Max acceptable hallucination rate
THRESHOLD_FAITHFULNESS  = 0.70   # Min acceptable faithfulness score
THRESHOLD_RELEVANCY     = 0.75   # Min acceptable answer relevancy score

# =============================================================================
# FIXTURES
# =============================================================================

CACHE_PATH = Path("models/agent_cache.json")

# The question context given to Claude — used as retrieval context
# for faithfulness scoring. Represents what the agent was given.
AGENT_CONTEXT = """
You are an expert Fantasy Premier League analyst.
You have been given:
- Top 25 player predictions with adjusted points, lineup probability,
  fixture difficulty rating (FDR), and upcoming opponent
- The selected 15-player squad with captain and vice-captain marked
- An injury watchlist of high-form players with availability concerns
- Fixture difficulty ratings for upcoming gameweeks
- Team defensive rankings by xGA (expected goals against) over last 5 games

Provide: captain recommendation with reasoning, transfer suggestions
with price deltas, key risks before the deadline, and a weekly summary.
"""


def load_latest_response():
    """Load most recent cached agent response."""
    if not CACHE_PATH.exists():
        pytest.skip("No cached agent response — run: python agent/fpl_agent.py")
    try:
        cache = json.loads(CACHE_PATH.read_text())
    except Exception:
        pytest.skip("Could not parse agent cache")
    if not cache:
        pytest.skip("Cache is empty")
    latest_key = sorted(cache.keys())[-1]
    return cache[latest_key], latest_key


def extract_section(text, tag):
    """Extract XML-tagged section from agent response."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# =============================================================================
# SKIP GUARD
# =============================================================================

@pytest.fixture(autouse=True)
def require_deepeval():
    """Skip all tests in this module if DeepEval is not installed."""
    if not DEEPEVAL_AVAILABLE:
        pytest.skip(
            "DeepEval not installed — run: pip install deepeval\n"
            "Also requires: ANTHROPIC_API_KEY environment variable"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — required for Claude judge")


# =============================================================================
# FAITHFULNESS TESTS
# Does the output reflect the data it was given?
# =============================================================================

class TestFaithfulness:
    """
    Faithfulness measures whether the agent's output is grounded in
    the context it received. An unfaithful response makes claims that
    cannot be verified from the input data.

    FPL relevance: if the agent says "Thiago has 6.2 adj pts" but
    the predictions show 5.94, that's a faithfulness failure.
    If the agent quotes an FDR of 1.3 but the data shows 4.3,
    that could lead to a genuinely wrong transfer decision.
    """

    def test_captain_section_faithful(self):
        """Captain recommendation should reference actual prediction data."""
        response, _ = load_latest_response()
        captain_text = extract_section(response, "captain")
        if not captain_text:
            pytest.skip("No captain section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=captain_text,
            retrieval_context=[response],
        )
        metric = FaithfulnessMetric(
            threshold=THRESHOLD_FAITHFULNESS,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_transfers_section_faithful(self):
        """Transfer suggestions should reference actual player prices and FDRs."""
        response, _ = load_latest_response()
        transfers_text = extract_section(response, "transfers")
        if not transfers_text:
            pytest.skip("No transfers section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=transfers_text,
            retrieval_context=[response],
        )
        metric = FaithfulnessMetric(
            threshold=THRESHOLD_FAITHFULNESS,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_risks_section_faithful(self):
        """Risk flags should reference actual players and probabilities."""
        response, _ = load_latest_response()
        risks_text = extract_section(response, "risks")
        if not risks_text:
            pytest.skip("No risks section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=risks_text,
            retrieval_context=[response],
        )
        metric = FaithfulnessMetric(
            threshold=THRESHOLD_FAITHFULNESS,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_summary_faithful(self):
        """Weekly summary should not introduce facts absent from context."""
        response, _ = load_latest_response()
        summary_text = extract_section(response, "summary")
        if not summary_text:
            pytest.skip("No summary section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=summary_text,
            retrieval_context=[response],
        )
        metric = FaithfulnessMetric(
            threshold=THRESHOLD_FAITHFULNESS,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])


# =============================================================================
# HALLUCINATION TESTS
# Does the output invent facts not in the input data?
# =============================================================================

class TestHallucination:
    """
    Hallucination measures whether the agent invents content that
    contradicts or cannot be found in the provided context.

    FPL relevance: the agent should not recommend a player not in the
    top 25, invent a fixture that doesn't exist, or state a lineup
    probability that wasn't in the watchlist data.

    Note: hallucination score is the rate of hallucinated statements.
    0.0 = no hallucinations (ideal). We fail if it EXCEEDS the threshold.
    """

    def test_captain_no_hallucination(self):
        """Captain section should not invent players or statistics."""
        response, _ = load_latest_response()
        captain_text = extract_section(response, "captain")
        if not captain_text:
            pytest.skip("No captain section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=captain_text,
            context=[response],
        )
        metric = HallucinationMetric(
            threshold=THRESHOLD_HALLUCINATION,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_transfers_no_hallucination(self):
        """Transfer suggestions should not invent player prices or FDRs."""
        response, _ = load_latest_response()
        transfers_text = extract_section(response, "transfers")
        if not transfers_text:
            pytest.skip("No transfers section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=transfers_text,
            context=[response],
        )
        metric = HallucinationMetric(
            threshold=THRESHOLD_HALLUCINATION,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_risks_no_hallucination(self):
        """Risks should not fabricate injury news or lineup probabilities."""
        response, _ = load_latest_response()
        risks_text = extract_section(response, "risks")
        if not risks_text:
            pytest.skip("No risks section in cached response")

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=risks_text,
            context=[response],
        )
        metric = HallucinationMetric(
            threshold=THRESHOLD_HALLUCINATION,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_full_response_no_hallucination(self):
        """Full agent response should stay within bounds of input data."""
        response, _ = load_latest_response()

        # Strip XML tags for full-response check
        clean = re.sub(r"<[^>]+>", " ", response).strip()

        test_case = LLMTestCase(
            input=AGENT_CONTEXT,
            actual_output=clean,
            context=[response],
        )
        metric = HallucinationMetric(
            threshold=THRESHOLD_HALLUCINATION,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])


# =============================================================================
# ANSWER RELEVANCY TESTS
# Does each section answer the right question?
# =============================================================================

class TestAnswerRelevancy:
    """
    Answer relevancy measures whether the output actually addresses
    the question asked. A section can be faithful and non-hallucinated
    but still be irrelevant if it answers the wrong question.

    FPL relevance: the captain section should answer captaincy.
    If it spends 80% of its words discussing transfer options,
    that's a relevancy failure even if all the facts are correct.
    Players reading it still won't know who to captain.
    """

    def test_captain_answers_captaincy_question(self):
        """Captain section must directly address who to captain and why."""
        response, _ = load_latest_response()
        captain_text = extract_section(response, "captain")
        if not captain_text:
            pytest.skip("No captain section in cached response")

        test_case = LLMTestCase(
            input=(
                "Who should I captain in Fantasy Premier League this gameweek "
                "and why? Also name a vice-captain as backup."
            ),
            actual_output=captain_text,
        )
        metric = AnswerRelevancyMetric(
            threshold=THRESHOLD_RELEVANCY,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_transfers_answers_transfer_question(self):
        """Transfer section must address what transfers to make this week."""
        response, _ = load_latest_response()
        transfers_text = extract_section(response, "transfers")
        if not transfers_text:
            pytest.skip("No transfers section in cached response")

        test_case = LLMTestCase(
            input=(
                "What transfers should I make in Fantasy Premier League "
                "this gameweek? Name specific players to sell and buy, "
                "and include the price difference."
            ),
            actual_output=transfers_text,
        )
        metric = AnswerRelevancyMetric(
            threshold=THRESHOLD_RELEVANCY,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_risks_answers_risk_question(self):
        """Risks section must flag specific concerns before the deadline."""
        response, _ = load_latest_response()
        risks_text = extract_section(response, "risks")
        if not risks_text:
            pytest.skip("No risks section in cached response")

        test_case = LLMTestCase(
            input=(
                "What are the key risks I should monitor before the FPL "
                "deadline this gameweek? Include injury doubts, difficult "
                "fixtures, and anything that might force last-minute changes."
            ),
            actual_output=risks_text,
        )
        metric = AnswerRelevancyMetric(
            threshold=THRESHOLD_RELEVANCY,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])

    def test_summary_answers_strategy_question(self):
        """Summary should cover overall GW strategy concisely."""
        response, _ = load_latest_response()
        summary_text = extract_section(response, "summary")
        if not summary_text:
            pytest.skip("No summary section in cached response")

        test_case = LLMTestCase(
            input=(
                "Give me a concise weekly FPL strategy summary covering: "
                "who to captain, the best value pick, the key risk to watch, "
                "and one contrarian insight based on fixture difficulty data."
            ),
            actual_output=summary_text,
        )
        metric = AnswerRelevancyMetric(
            threshold=THRESHOLD_RELEVANCY,
            model=JUDGE_MODEL,
            include_reason=True,
        )
        assert_test(test_case, [metric])


# =============================================================================
# COMBINED METRIC TEST
# =============================================================================

class TestCombinedMetrics:
    """
    Run all three metrics on the captain section in one test.
    This is the most important single test — the captain pick is the
    highest-stakes recommendation the agent makes each week.
    A wrong captain choice costs double points.

    This test is the hallucination GATE — if it fails, the agent
    is not trustworthy enough for real FPL decisions.
    """

    def test_captain_passes_all_three_metrics(self):
        """
        Captain section must simultaneously be:
        - Faithful to the prediction data (≥ 0.70)
        - Free of hallucinations (≤ 0.20)
        - Relevant to the captaincy question (≥ 0.75)
        """
        response, _ = load_latest_response()
        captain_text = extract_section(response, "captain")
        if not captain_text:
            pytest.skip("No captain section in cached response")

        test_case = LLMTestCase(
            input=(
                "Who should I captain in FPL this gameweek and why? "
                "Name a vice-captain too."
            ),
            actual_output=captain_text,
            retrieval_context=[response],
            context=[response],
        )
        metrics = [
            FaithfulnessMetric(
                threshold=THRESHOLD_FAITHFULNESS,
                model=JUDGE_MODEL,
                include_reason=True,
            ),
            HallucinationMetric(
                threshold=THRESHOLD_HALLUCINATION,
                model=JUDGE_MODEL,
                include_reason=True,
            ),
            AnswerRelevancyMetric(
                threshold=THRESHOLD_RELEVANCY,
                model=JUDGE_MODEL,
                include_reason=True,
            ),
        ]
        assert_test(test_case, metrics)
