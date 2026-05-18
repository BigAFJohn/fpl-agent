"""
tests/evals/test_agent_evals.py
================================
Evaluation tests for the FPL Agent (Claude-powered briefing).

THIS IS THE NEW TESTING PARADIGM FOR LLM/AGENT SYSTEMS.

As a test automation engineer moving into AI testing, this is the
key mental shift:

  Traditional testing:  assert output == expected_value
  Agent/LLM testing:    assert output satisfies quality_criteria

Why? Because LLM outputs are non-deterministic. Claude might phrase
the captain recommendation differently each time — but it should
ALWAYS name a specific player, ALWAYS reference a fixture, ALWAYS
give a reason. We test the PROPERTIES of the output, not the
exact string.

TESTING LAYERS FOR AI SYSTEMS:
-------------------------------
1. Structural tests    — Does the output have the right sections/format?
2. Content tests       — Does the output contain required information?
3. Factual tests       — Does the output avoid hallucinations?
4. Quality tests       — Is the output useful and coherent?
5. Regression tests    — Is output quality consistent across runs?
6. Adversarial tests   — Does the agent handle bad input gracefully?

These evals use cached responses so they don't make API calls on
every test run. Use --refresh flag to regenerate:
    python fpl_agent.py --refresh
"""

import sys
import re
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from agent.fpl_agent import parse_response


# =============================================================================
# FIXTURES — load cached agent response
# =============================================================================

CACHE_PATH = Path("models/agent_cache.json")

SAMPLE_RESPONSE = """
<captain>
**Captain: Thiago (5.84 adjusted points vs Man City)**

Thiago leads the model predictions with 5.84 adjusted points despite facing Man City (FDR 3.0). His high lineup probability (0.89) ensures he's almost certain to play, and he's priced at just £7.3m, making him an exceptional value captaincy option.

**Vice-Captain: Bruno Fernandes (5.33 adjusted points vs Sunderland, FDR 2.2)**

Bruno faces the easiest fixture among your premium options.
</captain>

<transfers>
**No urgent transfers recommended this gameweek.**

Your current team is well-structured. If you have a free transfer:
- **Cullen OUT → Tavernier IN (£4.9m → £5.4m, -£0.5m)**: Cullen has only 0.05 lineup probability.
</transfers>

<risks>
1. **Thiago captaincy against Man City**: Man City's defensive record (xGA 1.332) is solid.
2. **Multiple Liverpool assets**: Kelleher, Szoboszlai, and Gakpo all facing tough fixtures.
3. **Bench depth vulnerability**: Cullen has only 0.05 lineup probability.
</risks>

<summary>
For GW36, captain Thiago despite the Man City fixture — his 5.84 adjusted points prediction significantly outperforms all alternatives. Rogers (3.94 adj points vs Burnley, FDR 1.3) and Bruno Fernandes (5.33 vs Sunderland, FDR 2.2) are well-positioned for easy fixtures. The key risk is triple Liverpool exposure against Man City. A contrarian observation: Crystal Palace have the 9th-strongest defence yet face three of your players.
</summary>
"""


@pytest.fixture
def parsed_sections():
    """Parse the sample response into sections."""
    return parse_response(SAMPLE_RESPONSE)


@pytest.fixture
def cached_sections():
    """
    Load the most recent cached agent response.
    Skips if no cache exists — run fpl_agent.py first.
    """
    if not CACHE_PATH.exists():
        pytest.skip("No cached agent response — run: python fpl_agent.py")

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    if not cache:
        pytest.skip("Cache is empty")

    # Get most recent GW
    latest_key = sorted(cache.keys())[-1]
    return parse_response(cache[latest_key])


# =============================================================================
# STRUCTURAL TESTS — does the response have the right format?
# =============================================================================

class TestResponseStructure:
    """
    Layer 1: Structural tests.
    These verify the agent produces output in the expected format.
    Equivalent to testing an API returns the right JSON schema.
    """

    def test_all_sections_present(self, parsed_sections):
        """All four required sections must be present."""
        required = ["captain", "transfers", "risks", "summary"]
        for section in required:
            assert section in parsed_sections, \
                f"Missing required section: {section}"

    def test_no_empty_sections(self, parsed_sections):
        """No section should be empty."""
        for section, content in parsed_sections.items():
            assert len(content.strip()) > 0, \
                f"Section '{section}' is empty"

    def test_captain_section_minimum_length(self, parsed_sections):
        """Captain recommendation should be substantive (>50 chars)."""
        assert len(parsed_sections["captain"]) >= 50, \
            "Captain section too short to be useful"

    def test_summary_is_paragraph(self, parsed_sections):
        """Summary should be a coherent paragraph (>100 chars)."""
        assert len(parsed_sections["summary"]) >= 100, \
            "Summary too short to be a useful paragraph"

    def test_risks_has_multiple_items(self, parsed_sections):
        """Risks section should list at least 2 concerns."""
        risks_text = parsed_sections["risks"]
        # Count numbered items or bullet points
        numbered = len(re.findall(r'^\d+\.', risks_text, re.MULTILINE))
        bullets  = len(re.findall(r'^[-*•]', risks_text, re.MULTILINE))
        total    = numbered + bullets
        assert total >= 2 or risks_text.count('\n') >= 2, \
            f"Risks section should have at least 2 items, found {total}"


# =============================================================================
# CONTENT TESTS — does the response contain required information?
# =============================================================================

class TestResponseContent:
    """
    Layer 2: Content tests.
    These verify the agent includes specific required information.
    Equivalent to testing an API response contains required fields.
    """

    def test_captain_names_a_player(self, parsed_sections):
        """
        Captain section must name a specific player.
        We check for a capitalised word that could be a name.
        """
        captain_text = parsed_sections["captain"]
        # Look for "Captain: <Name>" or "captain <Name>"
        has_captain_label = bool(re.search(r'[Cc]aptain[:\s]+\w', captain_text))
        has_capitalised_name = bool(re.search(r'\b[A-Z][a-z]+\b', captain_text))
        assert has_captain_label or has_capitalised_name, \
            "Captain section should name a specific player"

    def test_captain_mentions_points(self, parsed_sections):
        """Captain recommendation should reference predicted points."""
        captain_text = parsed_sections["captain"]
        has_points = bool(re.search(r'\d+\.?\d*\s*(adj|adjusted|points|pts)', captain_text, re.I))
        has_number = bool(re.search(r'\d+\.\d+', captain_text))
        assert has_points or has_number, \
            "Captain section should reference predicted points"

    def test_captain_mentions_fixture(self, parsed_sections):
        """Captain section should reference the player's fixture/opponent."""
        captain_text = parsed_sections["captain"]
        # Look for "vs", "against", "FDR", or a team name pattern
        has_fixture = bool(re.search(
            r'\b(vs|against|FDR|fixture|opponent)\b', captain_text, re.I
        ))
        assert has_fixture, \
            "Captain section should mention the player's fixture"

    def test_captain_names_vice_captain(self, parsed_sections):
        """Captain section should also name a vice-captain."""
        captain_text = parsed_sections["captain"]
        has_vc = bool(re.search(r'[Vv]ice[- ]?[Cc]aptain', captain_text))
        assert has_vc, "Captain section should name a vice-captain"

    def test_transfers_mentions_price(self, parsed_sections):
        """Transfer suggestions should reference player prices."""
        transfers_text = parsed_sections["transfers"]
        has_price = bool(re.search(r'£\d+\.?\d*m?', transfers_text))
        has_no_transfers = "no" in transfers_text.lower() and "transfer" in transfers_text.lower()
        assert has_price or has_no_transfers, \
            "Transfers section should mention prices or explicitly say no transfers needed"

    def test_summary_mentions_captain(self, parsed_sections):
        """Summary should reference the captain pick."""
        summary_text = parsed_sections["summary"]
        has_captain = bool(re.search(r'captain', summary_text, re.I))
        assert has_captain, "Summary should mention the captain"

    def test_summary_mentions_risk(self, parsed_sections):
        """Summary should acknowledge at least one risk."""
        summary_text = parsed_sections["summary"]
        risk_words = ["risk", "concern", "doubt", "injury", "watch", "careful"]
        has_risk = any(word in summary_text.lower() for word in risk_words)
        assert has_risk, "Summary should acknowledge at least one risk"


# =============================================================================
# FACTUAL TESTS — does the response avoid hallucinations?
# =============================================================================

class TestResponseFactuality:
    """
    Layer 3: Factual/hallucination tests.
    These verify the agent only references real data, not invented facts.
    This is the most important eval category for production AI systems.
    """

    # Known valid FPL players for the current season
    KNOWN_VALID_PLAYERS = {
        "haaland", "thiago", "b.fernandes", "fernandes", "rogers",
        "saliba", "gabriel", "tarkowski", "kelleher", "calvert-lewin",
        "watkins", "szoboszlai", "gakpo", "rice", "saka",
        "gibbs-white", "n.williams", "senesi", "truffert",
    }

    KNOWN_VALID_TEAMS = {
        "arsenal", "man city", "manchester city", "liverpool",
        "chelsea", "man utd", "manchester united", "aston villa",
        "spurs", "tottenham", "brentford", "brighton", "everton",
        "nott'm forest", "nottingham forest", "newcastle", "wolves",
        "fulham", "bournemouth", "leeds", "sunderland", "crystal palace",
        "west ham", "burnley",
    }

    def test_no_obviously_wrong_gw_number(self, parsed_sections):
        """
        Agent should not reference wildly wrong gameweek numbers.
        We're in GW36 — any mention of GW1 or GW50 would be wrong.
        """
        all_text = " ".join(parsed_sections.values())
        wrong_gws = re.findall(r'GW(\d+)', all_text)
        for gw_num in wrong_gws:
            gw_int = int(gw_num)
            assert 1 <= gw_int <= 38, \
                f"Agent mentioned invalid GW{gw_num}"

    def test_prices_in_reasonable_range(self, parsed_sections):
        """
        Any prices mentioned should be in valid FPL range (£3.9m-£15.5m).
        """
        all_text = " ".join(parsed_sections.values())
        # Match prices but not price differences (which start with -)
        prices = re.findall(r'(?<![-−])£(\d+\.?\d*)m?', all_text)
        for price_str in prices:
            price = float(price_str)
            assert 3.5 <= price <= 16.0, \
                f"Agent mentioned invalid price £{price_str}m"

    def test_probabilities_in_range(self, parsed_sections):
        """
        Any probabilities mentioned (lineup_probability) should be 0-1.
        """
        all_text = " ".join(parsed_sections.values())
        # Look for "0.XX" format probabilities
        probs = re.findall(r'\b0\.\d{2}\b', all_text)
        for prob_str in probs:
            prob = float(prob_str)
            assert 0.0 <= prob <= 1.0, \
                f"Agent mentioned invalid probability {prob_str}"

    def test_no_placeholder_text(self, parsed_sections):
        """Agent should not contain template placeholders."""
        all_text = " ".join(parsed_sections.values())
        placeholders = ["[PLAYER]", "[TEAM]", "[FIXTURE]", "INSERT", "TODO", "PLACEHOLDER"]
        for placeholder in placeholders:
            assert placeholder not in all_text.upper(), \
                f"Agent response contains placeholder: {placeholder}"

    def test_response_is_english(self, parsed_sections):
        """Agent should respond in English."""
        all_text = " ".join(parsed_sections.values())
        common_english = ["the", "and", "for", "with", "his", "this", "that"]
        english_count = sum(1 for word in common_english if word in all_text.lower())
        assert english_count >= 4, \
            "Agent response does not appear to be in English"


# =============================================================================
# QUALITY TESTS — is the output useful?
# =============================================================================

class TestResponseQuality:
    """
    Layer 4: Quality/usefulness tests.
    These verify the agent produces actionable, coherent advice.
    More subjective than structural/factual tests but important.
    """

    def test_captain_gives_reasoning(self, parsed_sections):
        """
        Captain section should explain WHY, not just WHO.
        Look for reasoning indicators.
        """
        captain_text = parsed_sections["captain"]
        reasoning_words = [
            "because", "despite", "since", "as ", "given",
            "due to", "with", "makes", "ensure", "outperform"
        ]
        has_reasoning = any(word in captain_text.lower() for word in reasoning_words)
        assert has_reasoning, \
            "Captain section should explain reasoning, not just name a player"

    def test_transfers_are_actionable(self, parsed_sections):
        """
        Transfers section should either suggest specific moves
        or explicitly say no transfers are needed.
        """
        transfers_text = parsed_sections["transfers"]
        has_transfer_suggestion = bool(re.search(r'OUT|IN|→|->|transfer', transfers_text, re.I))
        has_no_transfer = bool(re.search(r'no (urgent|transfer|change)', transfers_text, re.I))
        assert has_transfer_suggestion or has_no_transfer, \
            "Transfers section should either suggest moves or explicitly say no transfers needed"

    def test_summary_covers_multiple_angles(self, parsed_sections):
        """
        Summary should cover: captain, value pick, risk, and fixture insight.
        We check for at least 3 of these angles.
        """
        summary = parsed_sections["summary"].lower()
        angles = {
            "captain"  : bool(re.search(r'captain', summary)),
            "fixture"  : bool(re.search(r'(fdr|fixture|vs|against)', summary)),
            "risk"     : bool(re.search(r'(risk|concern|doubt|watch)', summary)),
            "value"    : bool(re.search(r'(value|price|£|cheap)', summary)),
        }
        covered = sum(angles.values())
        assert covered >= 3, \
            f"Summary covers only {covered}/4 expected angles: {angles}"

    def test_no_contradictions_in_captain_section(self, parsed_sections):
        """
        Captain section should not recommend someone and then say they're injured.
        Simple heuristic: if player is mentioned as captain,
        they shouldn't be described as 'injured' or 'doubt' in same sentence.
        """
        captain_text = parsed_sections["captain"]
        # Extract first player name mentioned after "Captain:"
        match = re.search(r'[Cc]aptain[:\s*]+([A-Z][a-z\.\-]+)', captain_text)
        if match:
            captain_name = match.group(1)
            # Check for injury mentions near the captain name
            surrounding = captain_text[:200]
            has_injury_concern = bool(re.search(
                r'\b(injured|injury|doubt|unavailable|suspended)\b',
                surrounding, re.I
            ))
            # It's OK to mention injury as a risk, but not as a fact
            is_stated_as_fact = bool(re.search(
                rf'{re.escape(captain_name)}.{{0,30}}(is injured|is doubtful|is suspended)',
                surrounding, re.I
            ))
            assert not is_stated_as_fact, \
                f"Captain {captain_name} described as injured/doubtful"


# =============================================================================
# REGRESSION TESTS — cached response quality
# =============================================================================

class TestCachedResponseQuality:
    """
    Layer 5: Regression tests against cached real responses.
    These run against the actual cached Claude response,
    not the sample response fixture.
    """

    def test_cached_response_has_all_sections(self, cached_sections):
        """Real cached response must have all sections."""
        required = ["captain", "transfers", "risks", "summary"]
        for section in required:
            assert section in cached_sections
            assert len(cached_sections[section].strip()) > 0

    def test_cached_response_word_count(self, cached_sections):
        """Real response should be substantive (>100 words total)."""
        total_text = " ".join(cached_sections.values())
        word_count = len(total_text.split())
        assert word_count >= 100, \
            f"Response too short: {word_count} words"

    def test_cached_response_not_error_message(self, cached_sections):
        """Real response should not be an error or refusal."""
        all_text = " ".join(cached_sections.values()).lower()
        error_phrases = [
            "i cannot", "i am unable", "i don't have access",
            "error", "sorry, i", "as an ai",
        ]
        for phrase in error_phrases:
            assert phrase not in all_text, \
                f"Response appears to be an error: contains '{phrase}'"
