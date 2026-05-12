"""
fpl_agent.py  —  Phase 5: LLM Agent Layer
==========================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why an LLM layer on top of XGBoost
   The XGBoost model produces numbers. Claude produces reasoning.
   Numbers tell you Thiago scores 5.84 predicted points.
   Reasoning tells you "Thiago is your best captain option —
   he leads the league in xGI, Brentford have scored in 8 straight
   games, and Man City's defence has been leaking lately."
   FPL decisions require context, trade-offs, and judgment that
   a prediction table can't provide alone.

2. What data we pass to Claude
   We construct a structured prompt containing:
   - Top 20 predictions with adjusted points, lineup prob, fixture
   - Selected team from the LP optimiser
   - Injury watchlist (high form players with availability concerns)
   - Fixture difficulty table for upcoming gameweeks
   - Current gameweek number
   Claude uses all of this to produce actionable recommendations.

3. Prompt engineering for FPL advice
   We use a structured prompt with explicit sections so Claude
   produces consistent, parseable output:
   - CAPTAIN: one recommendation with reasoning
   - TRANSFERS: up to 3 suggestions (in/out pairs with reasoning)
   - RISKS: injury/suspension concerns to monitor
   - SUMMARY: one-paragraph weekly overview
   We ask for XML tags so we can parse each section independently
   and display them formatted.

4. The agent runs AFTER the model
   The agent doesn't replace the XGBoost model — it interprets it.
   The model produces ranked predictions. The agent explains why
   the top predictions make sense and flags where human judgment
   should override the model (e.g. a player the model rates highly
   but who has a history of blanking in big games).

5. Cost management
   Each weekly run makes one API call with ~2000 tokens of context
   and ~500 tokens of output. At Claude's API pricing this is
   negligible (~$0.01 per run). We cache the response locally
   so re-running the report doesn't make additional API calls.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Missing dependency. Run: pip install anthropic")


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH    = "db/fpl.db"
CACHE_PATH = Path("models/agent_cache.json")


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


# =============================================================================
# DATA BUILDER
# =============================================================================

def build_agent_context(engine):
    """
    Assembles all relevant data into a structured context dict
    that gets passed to Claude. Keeps the prompt concise by
    limiting to the most relevant rows.
    """

    # Current gameweek
    gw_df = pd.read_sql(
        text("SELECT id FROM gameweeks WHERE is_current = TRUE LIMIT 1"),
        engine
    )
    current_gw = int(gw_df["id"].iloc[0]) if not gw_df.empty else 36

    # Top 25 predictions
    top_preds = pd.read_sql(text("""
        SELECT web_name, position, team_name, price,
               predicted_points, adjusted_points,
               lineup_probability, fixture_fdr, opponent_name,
               fpl_status
        FROM predictions
        ORDER BY adjusted_points DESC
        LIMIT 25
    """), engine)

    # Selected team
    selected = pd.read_sql(text("""
        SELECT web_name, position, team_name, price,
               adjusted_points, lineup_probability,
               is_starting, is_captain, is_vice_captain,
               bench_order, opponent_name, fixture_fdr
        FROM selected_team
        ORDER BY is_starting DESC, position, adjusted_points DESC
    """), engine)

    # Injury watchlist
    watchlist = pd.read_sql(text("""
        SELECT p.web_name, lp.team_name, p.status, p.news,
               p.chance_of_playing_this_round,
               pf.avg_points_5gw,
               lp.lineup_probability
        FROM players p
        JOIN player_features pf ON p.id = pf.player_id
          AND pf.season = '2025-26'
          AND pf.gameweek = (SELECT MAX(gameweek) FROM player_features WHERE season = '2025-26')
        JOIN lineup_probability lp ON p.id = lp.player_id
        WHERE lp.lineup_probability < 0.75
          AND pf.avg_points_5gw > 3.5
          AND p.status NOT IN ('u', 'n')
        ORDER BY pf.avg_points_5gw DESC
        LIMIT 10
    """), engine)

    # Upcoming fixtures
    fixtures = pd.read_sql(text("""
        SELECT home_team, away_team, gameweek,
               home_attack_fdr, away_attack_fdr
        FROM fixture_difficulty
        WHERE finished = FALSE
          AND gameweek IS NOT NULL
        ORDER BY gameweek, home_attack_fdr
        LIMIT 20
    """), engine)

    # Team defensive rankings
    defences = pd.read_sql(text("""
        SELECT DISTINCT ON (team_name)
               team_name,
               ROUND(xga_rolling_5::numeric, 3) AS xga_5gw,
               ROUND(defence_strength::numeric, 3) AS strength
        FROM team_defence_ratings
        WHERE season = '2025-26'
          AND xga_rolling_5 IS NOT NULL
        ORDER BY team_name, gameweek DESC
    """), engine)
    defences = defences.sort_values("xga_5gw")

    return {
        "gameweek"   : current_gw,
        "predictions": top_preds.to_dict(orient="records"),
        "selected"   : selected.to_dict(orient="records"),
        "watchlist"  : watchlist.to_dict(orient="records"),
        "fixtures"   : fixtures.to_dict(orient="records"),
        "defences"   : defences.to_dict(orient="records"),
    }


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_prompt(context):
    """
    Constructs the structured prompt for Claude.
    Uses XML tags to help Claude structure its response
    in a way we can parse.
    """
    gw = context["gameweek"]

    # Format predictions table
    preds_lines = []
    for r in context["predictions"][:20]:
        preds_lines.append(
            f"  {r['web_name']:<20} {r['position']:<4} £{float(r['price'] or 0):.1f} "
            f"adj={float(r['adjusted_points'] or 0):.2f} "
            f"linp={float(r['lineup_probability'] or 0):.2f} "
            f"fdr={float(r['fixture_fdr'] or 0):.1f} "
            f"vs {r['opponent_name'] or ''}"
        )
    preds_text = "\n".join(preds_lines)

    # Format selected team
    starters = [r for r in context["selected"] if r["is_starting"]]
    bench    = [r for r in context["selected"] if not r["is_starting"]]

    team_lines = ["  STARTING XI:"]
    for r in starters:
        cap = " (C)" if r["is_captain"] else " (V)" if r["is_vice_captain"] else ""
        team_lines.append(
            f"    {r['web_name']}{cap:<5} {r['position']} "
            f"£{float(r['price'] or 0):.1f} adj={float(r['adjusted_points'] or 0):.2f} "
            f"vs {r['opponent_name'] or ''} FDR={float(r['fixture_fdr'] or 0):.1f}"
        )
    team_lines.append("  BENCH:")
    for r in bench:
        team_lines.append(
            f"    {r['web_name']:<20} {r['position']} "
            f"£{float(r['price'] or 0):.1f} linp={float(r['lineup_probability'] or 0):.2f}"
        )
    team_text = "\n".join(team_lines)

    # Format watchlist
    watch_lines = []
    for r in context["watchlist"]:
        watch_lines.append(
            f"  {r['web_name']:<20} {r['team_name']:<18} "
            f"status={r['status']} linp={float(r['lineup_probability'] or 0):.2f} "
            f"form={float(r['avg_points_5gw'] or 0):.1f} news: {r['news'] or ''}"
        )
    watch_text = "\n".join(watch_lines) if watch_lines else "  None"

    # Format top defensive teams (hardest to score against)
    def_lines = []
    for r in context["defences"][:10]:
        def_lines.append(
            f"  {r['team_name']:<25} xGA/5gw={float(r['xga_5gw'] or 0):.3f}"
        )
    def_text = "\n".join(def_lines)

    prompt = f"""You are an expert Fantasy Premier League analyst. Analyse the data below and provide actionable recommendations for GW{gw}.

Be direct, specific, and concise. Back every recommendation with data from the tables provided.

---
TOP 20 MODEL PREDICTIONS (adjusted points = predicted × lineup probability):
{preds_text}

CURRENT SELECTED TEAM:
{team_text}

INJURY/AVAILABILITY WATCHLIST:
{watch_text}

STRONGEST DEFENCES LAST 5 GAMES (lower xGA = harder to score against):
{def_text}
---

Provide your analysis in the following structure. Use the XML tags exactly as shown:

<captain>
Recommend ONE captain for GW{gw}. Name the player, their adjusted points prediction, their fixture, and explain in 2-3 sentences why they are the best captain option. Also name your vice-captain.
</captain>

<transfers>
Suggest up to 3 transfer moves (player OUT → player IN). For each transfer:
- State the player to sell and why (form, fixture, injury risk)
- State the player to buy and why (form, fixture, value)
- Note the price difference
Keep suggestions realistic — only recommend transfers that improve the team.
If no transfers are clearly beneficial, say so.
</transfers>

<risks>
List 2-3 key risks to monitor before the GW{gw} deadline:
- Injury doubts that could force last-minute changes
- Fixture concerns (tough matchups the model may have underestimated)
- Any other factors affecting the selected team
</risks>

<summary>
Write a single paragraph (4-6 sentences) summarising the key FPL strategy for GW{gw}. 
Include: best captain, top value pick, key risk, and one contrarian observation based on the fixture difficulty data.
</summary>"""

    return prompt


# =============================================================================
# CLAUDE API CALL
# =============================================================================

def call_claude(prompt, gameweek, use_cache=True):
    """
    Calls Claude API with the FPL analysis prompt.
    Caches response locally to avoid repeat API calls on re-runs.
    """
    cache_key = f"gw{gameweek}"

    # Check cache first
    if use_cache and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        if cache_key in cache:
            print("  ✓ Using cached agent response")
            return cache[cache_key]

    print("  Calling Claude API...")
    client = Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text

    # Cache the response
    CACHE_PATH.parent.mkdir(exist_ok=True)
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    cache[cache_key] = result
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

    print("  ✓ Claude response received and cached")
    return result


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_response(response_text):
    """
    Extracts sections from Claude's XML-tagged response.
    Returns dict with captain, transfers, risks, summary keys.
    """
    import re

    sections = {}
    tags = ["captain", "transfers", "risks", "summary"]

    for tag in tags:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response_text, re.DOTALL)
        sections[tag] = match.group(1).strip() if match else ""

    return sections


# =============================================================================
# DISPLAY
# =============================================================================

def display_briefing(sections, gameweek):
    """Prints the formatted weekly briefing."""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print(f"║  FPL AGENT BRIEFING — GW{gameweek}{'':>41}║")
    print(f"║  Powered by Claude{'':>49}║")
    print("╚" + "═" * 68 + "╝")

    if sections.get("captain"):
        print("\n  🎖  CAPTAIN RECOMMENDATION")
        print("  " + "─" * 50)
        for line in sections["captain"].strip().split("\n"):
            print(f"  {line}")

    if sections.get("transfers"):
        print("\n  🔄  TRANSFER SUGGESTIONS")
        print("  " + "─" * 50)
        for line in sections["transfers"].strip().split("\n"):
            print(f"  {line}")

    if sections.get("risks"):
        print("\n  ⚠   KEY RISKS BEFORE DEADLINE")
        print("  " + "─" * 50)
        for line in sections["risks"].strip().split("\n"):
            print(f"  {line}")

    if sections.get("summary"):
        print("\n  📋  WEEKLY SUMMARY")
        print("  " + "─" * 50)
        for line in sections["summary"].strip().split("\n"):
            print(f"  {line}")

    print("\n" + "═" * 70 + "\n")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_fpl_agent(use_cache=True, force_refresh=False):
    """
    Full agent pipeline:
    1. Load prediction data from DB
    2. Build structured prompt
    3. Call Claude API (or use cache)
    4. Parse and display briefing
    """
    start = datetime.now()

    print("=" * 60)
    print(f"FPL Agent  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/3] Building context...")
    context = build_agent_context(engine)
    print(f"  ✓ GW{context['gameweek']} context built")
    print(f"  ✓ {len(context['predictions'])} predictions loaded")
    print(f"  ✓ {len(context['watchlist'])} players on watchlist")

    print("\n[2/3] Building prompt...")
    prompt = build_prompt(context)
    print(f"  ✓ Prompt built ({len(prompt)} chars)")

    print("\n[3/3] Getting Claude analysis...")
    if force_refresh:
        use_cache = False
    response = call_claude(prompt, context["gameweek"], use_cache=use_cache)

    print("\n[4/4] Parsing and displaying briefing...")
    sections = parse_response(response)
    display_briefing(sections, context["gameweek"])

    duration = (datetime.now() - start).total_seconds()
    print(f"  Duration: {duration:.0f}s")

    return sections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FPL Agent — Claude-powered weekly briefing")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh — ignore cache and call Claude API again")
    args = parser.parse_args()

    run_fpl_agent(force_refresh=args.refresh)
