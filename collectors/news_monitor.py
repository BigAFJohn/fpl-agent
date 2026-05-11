"""
news_monitor.py  —  Phase 1, Component 3: News & Lineup Signal Monitor
=======================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why news signals matter more than stats for lineup prediction
   No stat tells you a player is injured — only news does. A player with
   0.8xG per game is worthless if he doesn't start. This component closes
   the gap between "statistically likely to score" and "will actually play".
   News signals are the single biggest edge over pure stats-based FPL tools.

2. RSS feeds — what they are and why we use them
   RSS (Really Simple Syndication) is a standard format that news sites use
   to publish their latest articles as a structured XML feed. We subscribe
   to football news RSS feeds the same way a podcast app subscribes to shows.
   The feedparser library reads any RSS feed and returns clean Python objects.
   No scraping, no HTML parsing — the sites give us the data for free.

3. FPL built-in news — the underused signal
   Our player database already has two fields we haven't used yet:
   - 'news': free-text injury note (e.g. "Knee injury - 75% chance of playing")
   - 'chance_of_playing_this_round': 0-100 numeric
   - 'status': a/d/i/u (available/doubtful/injured/unavailable)
   These update several times per week. We read them directly from our DB.

4. How we use the Claude API to parse unstructured text
   Raw news articles contain sentences like "Salah is a major doubt after
   limping off in training". A human reads this and knows: Salah, Liverpool,
   doubtful, this week. A language model can do the same thing at scale.
   We send the raw text to Claude and ask it to return structured JSON.
   This is called "information extraction" — turning prose into data.

5. The lineup_signals table — what we're building towards
   Every parsed signal gets stored as a row with:
   - player name, club, status, confidence score (0-1), gameweek, source
   In Phase 3 (Feature Engineering) we aggregate these signals per player
   per gameweek into a single lineup_probability score that feeds the model.

6. Confidence scoring — how we weight different sources
   Not all signals are equal. We assign base confidence weights:
   - FPL official status: 0.95 (direct from the club via FPL)
   - RSS injury report:   0.75 (journalist reporting)
   - RSS general news:    0.50 (may be speculative)
   Claude then adjusts these based on the language in the article
   (e.g. "confirmed out" → higher confidence, "reportedly" → lower).

7. APScheduler — automated monitoring
   We use APScheduler to run the monitor on a schedule:
   - Nightly at midnight: full RSS scan + FPL news refresh
   - Every 15 minutes on matchday eve: rapid injury scan
   This runs in the background while you work — you don't touch it.
"""

import feedparser
import json
import time
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from anthropic import Anthropic


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = "db/fpl.db"

# RSS feeds — free, no auth, update multiple times per day
RSS_FEEDS = {
    "BBC Sport Football"    : "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "Sky Sports Football"   : "https://www.skysports.com/rss/12040",
    "The Guardian Football" : "https://www.theguardian.com/football/rss",
}

# Premier League clubs — used to filter articles to EPL-relevant ones only
EPL_CLUBS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds", "Liverpool", "Manchester City", "Manchester United",
    "Newcastle", "Nottingham Forest", "Sunderland", "Tottenham",
    "West Ham", "Wolverhampton", "Wolves",
]

# Keywords that flag an article as injury/availability related.
# We only send articles containing these to Claude — keeps API costs low.
INJURY_KEYWORDS = [
    "injury", "injured", "doubt", "doubtful", "miss", "missing", "ruled out",
    "fitness", "hamstring", "knee", "ankle", "muscle", "strain", "suspension",
    "suspended", "ban", "banned", "return", "recover", "training", "available",
    "unavailable", "start", "bench", "squad", "lineup", "selection",
]

# Claude model for news parsing
CLAUDE_MODEL = "claude-sonnet-4-6"


# =============================================================================
# DATABASE
# =============================================================================

def get_database_engine():
    """Returns SQLAlchemy engine connected to the local SQLite database."""
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_signals_table(engine):
    """
    Creates the lineup_signals table if it doesn't already exist.
    This table is the output of this entire component — every parsed
    news signal lands here as a structured row. Clears signals older
    than 24hrs before each run to avoid unbounded growth.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lineup_signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at    TEXT,
                gameweek        INTEGER,
                player_name     TEXT,
                club            TEXT,
                status          TEXT,
                confidence      REAL,
                signal_type     TEXT,
                source          TEXT,
                headline        TEXT,
                raw_text        TEXT,
                claude_response TEXT
            )
        """))
        conn.commit()

    # Clear signals older than 24hrs to avoid unbounded growth
    with engine.connect() as conn:
        conn.execute(text(
            "DELETE FROM lineup_signals WHERE collected_at < datetime('now', '-24 hours')"
        ))
        conn.commit()

    print("lineup_signals table ready")


# =============================================================================
# FPL OFFICIAL NEWS
# =============================================================================

def collect_fpl_official_news(engine):
    """
    Reads injury news directly from the players table we already have.
    Joins with the teams table to get proper club names instead of IDs.

    The FPL API updates these fields several times per week:
    - status: a=available, d=doubtful, i=injured, u=unavailable
    - news: free-text injury note
    - chance_of_playing_this_round: 0-100

    We convert these into lineup_signals rows using a simple status mapping
    rather than calling Claude — the FPL data is already structured.
    This is the highest-confidence signal source we have.
    """
    print("\n[1/3] FPL official news...")

    status_map = {
        "a": ("available",   0.95),
        "d": ("doubt",       0.50),
        "i": ("injured",     0.05),
        "u": ("unavailable", 0.02),
    }

    with engine.connect() as conn:
        # Get current GW number
        gw_result = conn.execute(text(
            "SELECT id FROM gameweeks WHERE is_current = 1 LIMIT 1"
        ))
        gw_row = gw_result.fetchone()
        current_gw = gw_row[0] if gw_row else 0

        # Join with teams table so we get proper club names, not numeric IDs
        result = conn.execute(text("""
            SELECT p.web_name,
                   t.name,
                   p.status,
                   p.news,
                   p.chance_of_playing_this_round
            FROM players p
            LEFT JOIN teams t ON p.team = t.id
            WHERE p.status != 'a' OR p.news != ''
            ORDER BY t.name, p.web_name
        """))
        players = result.fetchall()

    print(f"  Found {len(players)} players with news/non-available status")

    signals = []
    now = datetime.now(timezone.utc).isoformat()

    for web_name, team_name, status, news, chance in players:
        status_label, base_confidence = status_map.get(status, ("unknown", 0.5))

        # Use chance_of_playing as confidence if available — more precise
        if chance is not None:
            confidence = float(chance) / 100.0
        else:
            confidence = base_confidence

        signals.append({
            "collected_at"   : now,
            "gameweek"       : current_gw,
            "player_name"    : web_name,
            "club"           : team_name or "Unknown",
            "status"         : status_label,
            "confidence"     : confidence,
            "signal_type"    : "fpl_official",
            "source"         : "FPL API",
            "headline"       : news[:200] if news else f"Status: {status}",
            "raw_text"       : news,
            "claude_response": None,
        })

    if signals:
        df = pd.DataFrame(signals)
        df.to_sql("lineup_signals", engine, if_exists="append", index=False)
        print(f"  ✓ {len(signals)} FPL official signals saved")

    return len(signals), current_gw


# =============================================================================
# RSS FEED COLLECTOR
# =============================================================================

def fetch_rss_articles(max_articles_per_feed=20):
    """
    Fetches recent articles from all configured RSS feeds.
    Returns only articles that mention an EPL club AND contain
    injury/availability keywords — avoids sending irrelevant content to Claude.
    """
    print("\n[2/3] RSS feeds...")
    all_articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            articles_checked = 0
            articles_kept = 0

            for entry in feed.entries[:max_articles_per_feed]:
                articles_checked += 1
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                text    = f"{title} {summary}".lower()

                # Filter 1: must mention an EPL club
                if not any(club.lower() in text for club in EPL_CLUBS):
                    continue

                # Filter 2: must contain injury/availability keywords
                if not any(kw in text for kw in INJURY_KEYWORDS):
                    continue

                all_articles.append({
                    "source"   : source_name,
                    "title"    : title,
                    "summary"  : summary[:500],
                    "link"     : entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
                articles_kept += 1

            print(f"  {source_name}: {articles_kept}/{articles_checked} relevant articles")
            time.sleep(1)

        except Exception as e:
            print(f"  ✗ Failed to fetch {source_name}: {e}")

    print(f"  Total relevant articles: {len(all_articles)}")
    return all_articles


# =============================================================================
# CLAUDE API PARSER
# =============================================================================

def parse_articles_with_claude(articles, engine, current_gw):
    """
    Sends relevant news articles to Claude and extracts structured
    lineup signals from the text.

    WHY CLAUDE AND NOT RULES-BASED PARSING?
    Rules like "if 'ruled out' in text → injured" fail on nuance:
    - "Not ruled out for the weekend" → actually available
    - "Doubt but trained fully today" → likely available
    - "Expected to be assessed" → genuinely unknown
    Claude understands context, negation, and hedging language that
    a keyword matcher would get wrong.

    Articles are batched in groups of 5 to reduce API calls.
    """
    if not articles:
        print("\n  No articles to parse")
        return 0

    print(f"\n[3/3] Parsing {len(articles)} articles with Claude...")

    client = Anthropic()
    signals_saved = 0
    now = datetime.now(timezone.utc).isoformat()

    batch_size = 5
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]

        articles_text = ""
        for j, article in enumerate(batch, 1):
            articles_text += f"""
Article {j} (Source: {article['source']}):
Headline: {article['title']}
Text: {article['summary']}
---"""

        prompt = f"""You are an expert Fantasy Premier League analyst extracting player availability signals from news articles.

Analyze these football news articles and extract any information about Premier League player availability, injuries, or lineup hints.

{articles_text}

For each player mentioned with availability information, return a JSON array. Each object must have exactly these fields:
- "player_name": full name as commonly known (e.g. "Mohamed Salah", "Erling Haaland")
- "club": Premier League club name (e.g. "Liverpool", "Manchester City")
- "status": one of exactly: "available", "doubt", "injured", "unavailable"
- "confidence": float 0.0-1.0 (how confident are you this signal is accurate?)
- "detail": one sentence summary of the situation
- "source_article": article number (1-{len(batch)})

Confidence guide:
- 0.9+: confirmed by club, official statement, or manager quote
- 0.7-0.9: credible journalist reporting specific details
- 0.5-0.7: general reporting, no specific source cited
- 0.3-0.5: speculative or vague language ("could", "might", "reportedly")
- <0.3: very uncertain or contradictory information

Return ONLY a valid JSON array. No preamble, no explanation, no markdown.
If no clear availability signals exist, return an empty array: []"""

        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            try:
                parsed_signals = json.loads(response_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    parsed_signals = json.loads(json_match.group())
                else:
                    print(f"  ✗ Could not parse Claude response for batch {i//batch_size + 1}")
                    continue

            rows = []
            for signal in parsed_signals:
                article_idx = signal.get("source_article", 1) - 1
                article = batch[min(article_idx, len(batch)-1)]

                rows.append({
                    "collected_at"   : now,
                    "gameweek"       : current_gw,
                    "player_name"    : signal.get("player_name", ""),
                    "club"           : signal.get("club", ""),
                    "status"         : signal.get("status", "unknown"),
                    "confidence"     : float(signal.get("confidence", 0.5)),
                    "signal_type"    : "rss_injury",
                    "source"         : article["source"],
                    "headline"       : article["title"][:200],
                    "raw_text"       : article["summary"][:500],
                    "claude_response": response_text,
                })

            if rows:
                df = pd.DataFrame(rows)
                df.to_sql("lineup_signals", engine, if_exists="append", index=False)
                signals_saved += len(rows)
                print(f"  Batch {i//batch_size + 1}: {len(rows)} signals extracted")

            time.sleep(0.5)

        except Exception as e:
            print(f"  ✗ Claude API error on batch {i//batch_size + 1}: {e}")

    print(f"  ✓ {signals_saved} signals saved from RSS articles")
    return signals_saved


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_signals(engine):
    """
    Prints a summary of signals collected broken down by type and status.
    Notable signals show players flagged as doubt/injured with confidence bars.
    """
    print("\n  Signal summary:")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT signal_type,
                   status,
                   COUNT(*)                   as count,
                   ROUND(AVG(confidence), 2)  as avg_confidence
            FROM lineup_signals
            WHERE collected_at >= datetime('now', '-1 hour')
            GROUP BY signal_type, status
            ORDER BY signal_type, status
        """))
        rows = result.fetchall()

    print(f"  {'Type':<16} {'Status':<14} {'Count':>6} {'Avg Conf':>10}")
    print(f"  {'-'*50}")
    for row in rows:
        print(f"  {row[0]:<16} {row[1]:<14} {row[2]:>6} {row[3]:>10}")

    print("\n  Notable signals (doubt/injured/unavailable):")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT player_name, club, status, confidence, headline
            FROM lineup_signals
            WHERE status IN ('doubt', 'injured', 'unavailable')
              AND signal_type = 'fpl_official'
            ORDER BY confidence DESC
            LIMIT 15
        """))
        notable = result.fetchall()

    for row in notable:
        bar = "█" * int(row[3] * 10) + "░" * (10 - int(row[3] * 10))
        print(f"  {row[0]:<22} {row[1]:<22} {row[2]:<12} {bar} {row[3]:.2f}")
        print(f"    → {row[4][:80]}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_news_monitor():
    """
    Runs a full news monitoring cycle:
    1. FPL official player news from our database
    2. Relevant articles from RSS feeds
    3. Claude parses RSS articles into structured signals
    4. Summary verification
    """
    print("=" * 60)
    print(f"News Monitor  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_database_engine()
    setup_signals_table(engine)

    fpl_count, current_gw = collect_fpl_official_news(engine)
    articles = fetch_rss_articles(max_articles_per_feed=20)
    rss_count = parse_articles_with_claude(articles, engine, current_gw)

    verify_signals(engine)

    print(f"\n{'='*60}")
    print(f"  FPL official signals : {fpl_count}")
    print(f"  RSS signals parsed   : {rss_count}")
    print(f"  Total this run       : {fpl_count + rss_count}")
    print(f"{'='*60}")


# =============================================================================
# SCHEDULER — for automated background monitoring
# =============================================================================

def start_scheduler():
    """
    Runs the news monitor automatically on a schedule using APScheduler.
    Call this instead of run_news_monitor() for continuous operation.

    Schedule:
    - Every night at midnight: full scan
    - Friday/Saturday 6am: pre-matchday scan
    - Saturday/Sunday every 15 mins 8am-2pm: rapid matchday scan

    Use run_news_monitor() manually for now — switch to this in Phase 7
    when we set up the background service.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(run_news_monitor, "cron", hour=0, minute=0)
    scheduler.add_job(run_news_monitor, "cron", day_of_week="fri,sat", hour=6)
    scheduler.add_job(
        run_news_monitor, "cron",
        day_of_week="sat,sun",
        hour="8-14", minute="*/15"
    )
    print("Scheduler started. Press Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    run_news_monitor()
