"""
dashboard.py  —  Phase 7b: Streamlit Dashboard (v2)
=====================================================
Run: streamlit run dashboard.py
"""

import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="FPL Agent",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

:root {
    --bg:       #060d1f;
    --surface:  #0d1a2d;
    --card:     #0f2035;
    --border:   #1e3a5f;
    --accent:   #00d4ff;
    --gold:     #ffd700;
    --green:    #00e676;
    --amber:    #ffab40;
    --red:      #ff5252;
    --violet:   #b388ff;
    --text:     #ffffff;
    --text2:    #90caf9;
    --muted:    #546e8a;
    --pitch:    #2d5a27;
    --pitch2:   #316128;
    --line:     rgba(255,255,255,0.15);
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1500px; }

/* ── Metrics ── */
.kpi {
    background: linear-gradient(135deg, var(--card) 0%, #0d2040 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.kpi-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.kpi-sub { font-size: 0.72rem; color: var(--text2); margin-top: 0.2rem; }

/* ── Section header ── */
.sh {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid var(--border);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    padding: 0.6rem 1.4rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Prediction row ── */
.pred-row {
    display: grid;
    grid-template-columns: 2.5rem 3rem 1fr 5rem 4rem 4rem 4rem 4rem;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 0.75rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.3rem;
    transition: border-color 0.15s;
}
.pred-row:hover { border-color: var(--accent); }

.rank { font-family:'Barlow Condensed',sans-serif; font-size:0.75rem; color:var(--muted); text-align:center; }
.pos-pill {
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.65rem; font-weight:700;
    padding:0.15rem 0.3rem; border-radius:3px;
    text-align:center; letter-spacing:0.05em;
}
.p-GK  { background:#4a1d9622; color:#b388ff; border:1px solid #4a1d9655; }
.p-DEF { background:#00600022; color:#00e676; border:1px solid #00600055; }
.p-MID { background:#0d47a122; color:#64b5f6; border:1px solid #0d47a155; }
.p-FWD { background:#e65c0022; color:#ffab40; border:1px solid #e65c0055; }

.pname { font-size:0.9rem; font-weight:600; color:#ffffff; }
.pteam { font-size:0.72rem; color:var(--text2); }
.pnum  { font-family:'Barlow Condensed',sans-serif; font-size:0.88rem; font-weight:700; text-align:right; }

.fdr-badge {
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.72rem; font-weight:700;
    padding:0.15rem 0.4rem; border-radius:3px;
    text-align:center; display:inline-block;
}
.fd1{background:#00600033;color:#00e676;border:1px solid #00600066;}
.fd2{background:#00600022;color:#80cbc4;border:1px solid #00600044;}
.fd3{background:#f9a82522;color:#ffcc02;border:1px solid #f9a82544;}
.fd4{background:#e53c3c22;color:#ff7043;border:1px solid #e53c3c44;}
.fd5{background:#e53c3c33;color:#ff5252;border:1px solid #e53c3c66;}

.status-pill {
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.62rem; font-weight:700; letter-spacing:0.05em;
    padding:0.1rem 0.35rem; border-radius:3px; text-transform:uppercase;
}
.s-a{background:#00600022;color:#00e676;border:1px solid #00600044;}
.s-d{background:#f9a82522;color:#ffab40;border:1px solid #f9a82544;}
.s-i{background:#e53c3c22;color:#ff5252;border:1px solid #e53c3c44;}
.s-s{background:#e53c3c22;color:#ff5252;border:1px solid #e53c3c44;}

/* ── Watchlist card ── */
.watch-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.4rem;
}

/* ── Briefing ── */
.brief-block {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.brief-icon-title {
    font-family:'Barlow Condensed',sans-serif;
    font-size:0.85rem; font-weight:700; letter-spacing:0.1em;
    color:var(--accent); text-transform:uppercase;
    margin-bottom:0.75rem;
}
.brief-body { font-size:0.87rem; line-height:1.7; color:#cde4ff; }

/* Pitch */
.pitch-wrap { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=3)
    db = Path("db/fpl.db")
    return create_engine(f"sqlite:///{db}") if db.exists() else None


@st.cache_data(ttl=300)
def q(sql):
    eng = get_engine()
    if eng is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(text(sql), eng)
    except Exception:
        return pd.DataFrame()


def load_gw():
    eng = get_engine()
    if eng is None:
        return "?"
    try:
        r = pd.read_sql(text("SELECT id FROM gameweeks WHERE is_current=TRUE LIMIT 1"), eng)
        return int(r["id"].iloc[0]) if not r.empty else "?"
    except Exception:
        return "?"


def load_agent():
    p = Path("models/agent_cache.json")
    if not p.exists():
        return None
    try:
        cache = json.loads(p.read_text())
        return cache[sorted(cache.keys())[-1]] if cache else None
    except Exception:
        return None


def load_backtest():
    p = Path("models/backtest_results.json")
    return json.loads(p.read_text()) if p.exists() else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def fdr_badge(v):
    if v is None:
        return ""
    c = min(5, max(1, round(float(v))))
    return f'<span class="fdr-badge fd{c}">{float(v):.1f}</span>'


def pos_pill(p):
    return f'<span class="pos-pill p-{p}">{p}</span>'


def status_pill(s, chance=None):
    s = str(s or "a")
    label = s.upper()
    if chance is not None:
        label += f" {int(chance)}%"
    return f'<span class="status-pill s-{s}">{label}</span>'


# ── Pitch View ────────────────────────────────────────────────────────────────

def pitch_view(selected_df):
    """Renders an SVG football pitch with players positioned by formation."""
    starters = selected_df[selected_df["is_starting"]].copy()
    bench    = selected_df[~selected_df["is_starting"]].sort_values("bench_order")

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters = starters.sort_values("position", key=lambda x: x.map(pos_order))

    # Group by position
    groups = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        groups[pos] = starters[starters["position"] == pos].reset_index(drop=True)

    # Pitch dimensions
    W, H = 500, 680
    # Y positions (% of pitch height, from top = GK end)
    y_pos = {"GK": 0.10, "DEF": 0.28, "MID": 0.50, "FWD": 0.72}

    player_nodes = []
    for pos, grp in groups.items():
        n = len(grp)
        for i, (_, p) in enumerate(grp.iterrows()):
            x_pct = (i + 1) / (n + 1)
            y_pct = y_pos[pos]
            is_cap = bool(p.get("is_captain", False))
            is_vc  = bool(p.get("is_vice_captain", False))
            pts    = float(p.get("adjusted_points", 0) or 0)
            prob   = float(p.get("lineup_probability", 0.8) or 0.8)
            player_nodes.append({
                "name"  : str(p["web_name"]),
                "team"  : str(p["team_name"]),
                "pos"   : pos,
                "x"     : x_pct * W,
                "y"     : y_pct * H,
                "cap"   : is_cap,
                "vc"    : is_vc,
                "pts"   : pts,
                "prob"  : prob,
                "price" : float(p.get("price", 0) or 0),
            })

    # Colour map
    pos_colors = {"GK": "#b388ff", "DEF": "#00e676", "MID": "#64b5f6", "FWD": "#ffab40"}

    # Build SVG
    svg_parts = [f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;display:block;border-radius:10px">
  <defs>
    <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="#2d5a27"/>
      <stop offset="50%"  stop-color="#316128"/>
      <stop offset="100%" stop-color="#2a5424"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>

  <!-- Pitch background with stripes -->
  <rect width="{W}" height="{H}" fill="url(#pg)" rx="10"/>
  {"".join(f'<rect x="0" y="{i*H/10}" width="{W}" height="{H/20}" fill="rgba(0,0,0,0.06)"/>' for i in range(0,10,2))}

  <!-- Pitch markings -->
  <!-- Outer border -->
  <rect x="20" y="20" width="{W-40}" height="{H-40}"
        fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5" rx="4"/>
  <!-- Centre circle -->
  <circle cx="{W/2}" cy="{H/2}" r="55"
          fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>
  <!-- Centre spot -->
  <circle cx="{W/2}" cy="{H/2}" r="3" fill="rgba(255,255,255,0.35)"/>
  <!-- Halfway line -->
  <line x1="20" y1="{H/2}" x2="{W-20}" y2="{H/2}"
        stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>
  <!-- Top penalty box -->
  <rect x="{W*0.22}" y="20" width="{W*0.56}" height="{H*0.17}"
        fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>
  <!-- Top 6-yard box -->
  <rect x="{W*0.36}" y="20" width="{W*0.28}" height="{H*0.07}"
        fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
  <!-- Bottom penalty box -->
  <rect x="{W*0.22}" y="{H-20-H*0.17}" width="{W*0.56}" height="{H*0.17}"
        fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>
  <!-- Bottom 6-yard box -->
  <rect x="{W*0.36}" y="{H-20-H*0.07}" width="{W*0.28}" height="{H*0.07}"
        fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>
  <!-- Penalty spots -->
  <circle cx="{W/2}" cy="{H*0.14}" r="2.5" fill="rgba(255,255,255,0.35)"/>
  <circle cx="{W/2}" cy="{H*0.86}" r="2.5" fill="rgba(255,255,255,0.35)"/>
"""]

    # Player nodes
    for p in player_nodes:
        x, y   = p["x"], p["y"]
        col    = pos_colors[p["pos"]]
        name   = p["name"][:13]
        pts    = p["pts"]
        is_cap = p["cap"]
        is_vc  = p["vc"]

        # Glow ring for captain
        if is_cap:
            svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="26" fill="none" stroke="#ffd700" stroke-width="2.5" opacity="0.8"/>')
        elif is_vc:
            svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="26" fill="none" stroke="#b388ff" stroke-width="1.5" opacity="0.7"/>')

        # Avatar circle
        svg_parts.append(f"""
  <circle cx="{x:.1f}" cy="{y:.1f}" r="22"
          fill="{col}22" stroke="{col}" stroke-width="2"/>
  <text x="{x:.1f}" y="{y-6:.1f}" text-anchor="middle" dominant-baseline="middle"
        fill="{col}" font-family="Barlow Condensed,sans-serif"
        font-size="9" font-weight="700">{p["pos"]}</text>
  <text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" dominant-baseline="middle"
        fill="white" font-family="Barlow Condensed,sans-serif"
        font-size="10.5" font-weight="800">{pts:.1f}</text>
""")
        # Captain/VC badge
        if is_cap:
            svg_parts.append(f"""
  <circle cx="{x+18:.1f}" cy="{y-18:.1f}" r="9" fill="#ffd700"/>
  <text x="{x+18:.1f}" y="{y-17:.1f}" text-anchor="middle" dominant-baseline="middle"
        fill="#000" font-family="Barlow Condensed,sans-serif" font-size="9" font-weight="800">C</text>
""")
        elif is_vc:
            svg_parts.append(f"""
  <circle cx="{x+18:.1f}" cy="{y-18:.1f}" r="9" fill="#b388ff"/>
  <text x="{x+18:.1f}" y="{y-17:.1f}" text-anchor="middle" dominant-baseline="middle"
        fill="#000" font-family="Barlow Condensed,sans-serif" font-size="9" font-weight="800">V</text>
""")

        # Name label below circle
        svg_parts.append(f"""
  <rect x="{x-36:.1f}" y="{y+26:.1f}" width="72" height="15" rx="3"
        fill="rgba(0,0,0,0.65)"/>
  <text x="{x:.1f}" y="{y+34:.1f}" text-anchor="middle" dominant-baseline="middle"
        fill="white" font-family="Barlow Condensed,sans-serif"
        font-size="10" font-weight="600">{name}</text>
""")

    svg_parts.append("</svg>")
    return "".join(svg_parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    gw = load_gw()

    preds    = q("SELECT web_name,position,team_name,price,predicted_points,adjusted_points,lineup_probability,fixture_fdr,opponent_name,fpl_status FROM predictions ORDER BY adjusted_points DESC")
    selected = q("SELECT web_name,position,team_name,price,predicted_points,adjusted_points,lineup_probability,is_starting,is_captain,is_vice_captain,bench_order,opponent_name,fixture_fdr,gameweek FROM selected_team ORDER BY is_starting DESC,bench_order NULLS FIRST")
    fixtures = q("SELECT home_team,away_team,gameweek,home_attack_fdr,away_attack_fdr,fpl_fdr_home,fpl_fdr_away,finished FROM fixture_difficulty WHERE finished=FALSE AND gameweek IS NOT NULL ORDER BY gameweek,home_attack_fdr LIMIT 20")
    backtest = load_backtest()
    agent    = load_agent()

    try:
        watchlist = q("""
            SELECT p.web_name, lp.team_name, p.status, p.news,
                   p.chance_of_playing_this_round,
                   pf.avg_points_5gw, lp.lineup_probability
            FROM players p
            JOIN player_features pf ON p.id=pf.player_id
              AND pf.season='2025-26'
              AND pf.gameweek=(SELECT MAX(gameweek) FROM player_features WHERE season='2025-26')
            JOIN lineup_probability lp ON p.id=lp.player_id
            WHERE lp.lineup_probability<0.75
              AND pf.avg_points_5gw>3.5
              AND p.status NOT IN ('u','n')
            ORDER BY pf.avg_points_5gw DESC LIMIT 12
        """)
    except Exception:
        watchlist = pd.DataFrame()

    # ── Header ──────────────────────────────────────────────────────────────
    h1, h2 = st.columns([5, 1])
    with h1:
        st.markdown(f"""
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;
                    font-weight:800;letter-spacing:-0.01em;color:#fff">
            ⚽ FPL <span style="color:var(--accent)">AGENT</span>
        </div>
        <div style="font-size:0.78rem;color:var(--muted);margin-top:0.1rem">
            GW{gw} &nbsp;·&nbsp; {datetime.now().strftime('%a %d %b %Y %H:%M')}
        </div>""", unsafe_allow_html=True)
    with h2:
        if st.button("⟳  Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── KPIs ────────────────────────────────────────────────────────────────
    if not preds.empty and not selected.empty:
        starters_df = selected[selected["is_starting"]]
        cap_df      = selected[selected["is_captain"]]
        start_pts   = float(starters_df["adjusted_points"].sum())
        cap_bonus   = float(cap_df["adjusted_points"].sum()) if not cap_df.empty else 0
        total_pred  = start_pts + cap_bonus
        total_cost  = float(selected["price"].sum())
        top_name    = preds.iloc[0]["web_name"]
        top_pts     = float(preds.iloc[0]["adjusted_points"])
        cap_name    = cap_df.iloc[0]["web_name"] if not cap_df.empty else "—"

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        for col, lbl, val, sub in [
            (c1, "Gameweek",    f"GW{gw}",           "current"),
            (c2, "Predicted",   f"{total_pred:.1f}",  "pts incl. captain"),
            (c3, "Squad Cost",  f"£{total_cost:.1f}m",f"£{100-total_cost:.1f}m left"),
            (c4, "Captain",     cap_name,              f"+{cap_bonus:.1f} pts bonus"),
            (c5, "Top Model",   top_name,              f"{top_pts:.2f} adj pts"),
            (c6, "MAE",         "0.934",               "pts per player"),
        ]:
            col.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5 = st.tabs(["⚽  TEAM","📊  PREDICTIONS","🗓  FIXTURES","🏥  WATCHLIST","🤖  BRIEFING"])

    # ── TEAM TAB ────────────────────────────────────────────────────────────
    with t1:
        if selected.empty:
            st.info("No team selected. Run `python scheduler.py predict`")
        else:
            starters = selected[selected["is_starting"]]
            bench    = selected[~selected["is_starting"]].sort_values("bench_order")
            total_cost = float(selected["price"].sum())

            # Formation string
            pos_counts = starters["position"].value_counts()
            formation  = f"{pos_counts.get('DEF',0)}-{pos_counts.get('MID',0)}-{pos_counts.get('FWD',0)}"

            pc, sc = st.columns([5, 4])

            with pc:
                st.markdown(f'<div class="sh">Formation {formation}</div>', unsafe_allow_html=True)
                svg = pitch_view(selected)
                components.html(f"""
                <!DOCTYPE html>
                <html>
                <head>
                <style>
                  body {{ margin:0; padding:0; background:#060d1f; }}
                  svg {{ width:100%; height:auto; display:block; border-radius:10px; }}
                </style>
                </head>
                <body>{svg}</body>
                </html>
                """, height=900, scrolling=False)

            with sc:
                st.markdown('<div class="sh">Starting XI</div>', unsafe_allow_html=True)

                pos_order = {"GK":0,"DEF":1,"MID":2,"FWD":3}
                for _, r in starters.sort_values("position", key=lambda x: x.map(pos_order)).iterrows():
                    pos   = str(r["position"])
                    name  = str(r["web_name"])
                    team  = str(r["team_name"])
                    pts   = float(r["adjusted_points"] or 0)
                    price = float(r["price"] or 0)
                    prob  = float(r["lineup_probability"] or 0.8)
                    fdr   = r.get("fixture_fdr")
                    opp   = str(r.get("opponent_name","") or "")[:10]
                    cap   = bool(r["is_captain"])
                    vc    = bool(r["is_vice_captain"])

                    cap_html = ""
                    border   = "var(--border)"
                    if cap:
                        cap_html = '<span style="color:#ffd700;font-family:\'Barlow Condensed\',sans-serif;font-size:0.75rem;font-weight:800;margin-left:4px">(C)</span>'
                        border   = "#ffd700"
                    elif vc:
                        cap_html = '<span style="color:#b388ff;font-family:\'Barlow Condensed\',sans-serif;font-size:0.75rem;font-weight:800;margin-left:4px">(V)</span>'
                        border   = "#b388ff"

                    prob_color = "#00e676" if prob >= 0.8 else "#ffab40" if prob >= 0.5 else "#ff5252"
                    prob_pct   = int(prob * 100)

                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:2.8rem 1fr auto;
                                align-items:center;gap:0.5rem;
                                padding:0.5rem 0.75rem;
                                background:var(--card);
                                border:1px solid {border};
                                border-radius:5px;margin-bottom:0.3rem">
                        {pos_pill(pos)}
                        <div>
                            <div style="font-size:0.88rem;font-weight:600;color:#fff;line-height:1.2">
                                {name}{cap_html}
                            </div>
                            <div style="font-size:0.7rem;color:var(--text2)">{team}</div>
                            <div style="background:var(--border);border-radius:2px;height:3px;
                                        width:100%;margin-top:4px;overflow:hidden">
                                <div style="width:{prob_pct}%;height:100%;background:{prob_color};border-radius:2px"></div>
                            </div>
                        </div>
                        <div style="text-align:right">
                            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
                                        font-weight:800;color:var(--accent)">{pts:.2f}</div>
                            <div style="font-size:0.68rem;color:var(--muted)">£{price:.1f}m</div>
                            <div style="margin-top:2px">{fdr_badge(fdr)}</div>
                            <div style="font-size:0.65rem;color:var(--muted)">{opp}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="sh" style="margin-top:1rem">Bench</div>', unsafe_allow_html=True)
                for _, r in bench.iterrows():
                    pos   = str(r["position"])
                    name  = str(r["web_name"])
                    team  = str(r["team_name"])
                    pts   = float(r["adjusted_points"] or 0)
                    price = float(r["price"] or 0)
                    prob  = float(r["lineup_probability"] or 0.5)
                    order = int(r["bench_order"] or 1)

                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:1.5rem 2.8rem 1fr auto;
                                align-items:center;gap:0.5rem;
                                padding:0.4rem 0.75rem;
                                background:rgba(13,26,45,0.5);
                                border:1px solid var(--border);
                                border-radius:5px;margin-bottom:0.25rem;opacity:0.8">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;
                                    color:var(--muted);text-align:center">{order}</div>
                        {pos_pill(pos)}
                        <div>
                            <div style="font-size:0.85rem;font-weight:500;color:#e0e0e0">{name}</div>
                            <div style="font-size:0.68rem;color:var(--muted)">{team}</div>
                        </div>
                        <div style="text-align:right">
                            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.9rem;
                                        font-weight:700;color:var(--text2)">{pts:.2f}</div>
                            <div style="font-size:0.65rem;color:var(--muted)">£{price:.1f}m</div>
                            <div style="font-size:0.65rem;color:var(--muted)">{int(prob*100)}%</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:0.75rem;padding:0.6rem 0.75rem;
                            background:var(--card);border:1px solid var(--border);
                            border-radius:5px;display:flex;justify-content:space-between;
                            font-family:'Barlow Condensed',sans-serif;font-size:0.8rem">
                    <span style="color:var(--muted)">TOTAL COST</span>
                    <span style="color:var(--accent);font-weight:700">£{total_cost:.1f}m
                        <span style="color:var(--muted);font-weight:400"> / £{100-total_cost:.1f}m remaining</span>
                    </span>
                </div>""", unsafe_allow_html=True)

    # ── PREDICTIONS TAB ──────────────────────────────────────────────────────
    with t2:
        if preds.empty:
            st.info("No predictions. Run `python scheduler.py predict`")
        else:
            st.markdown('<div class="sh">Top Predictions</div>', unsafe_allow_html=True)
            pos_filter = st.selectbox("Position", ["All","GK","DEF","MID","FWD"], label_visibility="collapsed")
            filtered = preds if pos_filter == "All" else preds[preds["position"] == pos_filter]

            # Header row
            st.markdown("""
            <div style="display:grid;grid-template-columns:2.5rem 3rem 1fr 5rem 4rem 4rem 4rem 4rem;
                        gap:0.5rem;padding:0.3rem 0.75rem;
                        font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;
                        text-transform:uppercase;letter-spacing:0.1em;color:var(--muted)">
                <div>#</div><div>Pos</div><div>Player</div>
                <div style="text-align:right">Adj Pts</div>
                <div style="text-align:right">Pred</div>
                <div style="text-align:right">LinP</div>
                <div style="text-align:right">FDR</div>
                <div style="text-align:right">Status</div>
            </div>""", unsafe_allow_html=True)

            for i, (_, r) in enumerate(filtered.head(30).iterrows(), 1):
                pos    = str(r.get("position","MID"))
                name   = str(r.get("web_name",""))
                team   = str(r.get("team_name",""))
                adj    = float(r.get("adjusted_points",0) or 0)
                pred   = float(r.get("predicted_points",0) or 0)
                prob   = float(r.get("lineup_probability",0.8) or 0.8)
                fdr    = r.get("fixture_fdr")
                opp    = str(r.get("opponent_name","") or "")[:10]
                status = str(r.get("fpl_status","a") or "a")

                prob_color = "#00e676" if prob>=0.8 else "#ffab40" if prob>=0.5 else "#ff5252"
                rank_color = "#ffd700" if i<=3 else "#fff" if i<=10 else "var(--muted)"

                st.markdown(f"""
                <div class="pred-row">
                    <div class="rank" style="color:{rank_color}">{i}</div>
                    {pos_pill(pos)}
                    <div>
                        <div class="pname">{name}</div>
                        <div class="pteam">{team} · {opp}</div>
                    </div>
                    <div class="pnum" style="color:var(--accent);font-size:1rem">{adj:.2f}</div>
                    <div class="pnum" style="color:var(--text2)">{pred:.2f}</div>
                    <div class="pnum" style="color:{prob_color}">{int(prob*100)}%</div>
                    <div style="text-align:right">{fdr_badge(fdr)}</div>
                    <div style="text-align:right">{status_pill(status)}</div>
                </div>""", unsafe_allow_html=True)

    # ── FIXTURES TAB ────────────────────────────────────────────────────────
    with t3:
        if fixtures.empty:
            st.info("No upcoming fixtures.")
        else:
            st.markdown('<div class="sh">Upcoming Fixtures — Custom FDR vs FPL Rating</div>', unsafe_allow_html=True)
            for _, r in fixtures.iterrows():
                h = float(r["home_attack_fdr"] or 3)
                a = float(r["away_attack_fdr"] or 3)
                gw_num = int(r["gameweek"] or 0)

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:1fr 3.5rem 1fr 5rem;
                            align-items:center;gap:0.75rem;
                            padding:0.6rem 1rem;
                            background:var(--card);border:1px solid var(--border);
                            border-radius:5px;margin-bottom:0.3rem">
                    <div style="text-align:right;display:flex;align-items:center;
                                justify-content:flex-end;gap:0.5rem">
                        <span style="font-size:0.9rem;font-weight:600;color:#fff">{r['home_team']}</span>
                        {fdr_badge(h)}
                    </div>
                    <div style="text-align:center;font-family:'Barlow Condensed',sans-serif;
                                font-size:0.65rem;color:var(--muted);letter-spacing:0.1em">
                        GW{gw_num}<br><span style="font-size:0.7rem;color:var(--border)">vs</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:0.5rem">
                        {fdr_badge(a)}
                        <span style="font-size:0.9rem;font-weight:600;color:#fff">{r['away_team']}</span>
                    </div>
                    <div style="text-align:right;font-family:'Barlow Condensed',sans-serif;
                                font-size:0.7rem;color:var(--muted)">
                        FPL {int(r['fpl_fdr_home'] or 0)}&nbsp;/&nbsp;{int(r['fpl_fdr_away'] or 0)}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:0.7rem;color:var(--muted);margin-top:0.5rem">
            Chip rating = our custom xGA-based FDR &nbsp;|&nbsp; FPL x/y = FPL built-in home/away rating
            </div>""", unsafe_allow_html=True)

    # ── WATCHLIST TAB ────────────────────────────────────────────────────────
    with t4:
        if watchlist.empty:
            st.info("No injury concerns.")
        else:
            st.markdown('<div class="sh">Injury & Availability Watchlist</div>', unsafe_allow_html=True)
            for _, r in watchlist.iterrows():
                s      = str(r.get("status","a") or "a")
                prob   = float(r.get("lineup_probability",0.5) or 0.5)
                form   = float(r.get("avg_points_5gw",0) or 0)
                news   = str(r.get("news","") or "")[:120]
                chance = r.get("chance_of_playing_this_round")
                pc     = f"{int(chance)}%" if chance else ""
                col    = "#ffab40" if s=="d" else "#ff5252" if s in ("i","s") else "#00e676"

                st.markdown(f"""
                <div class="watch-card" style="border-left-color:{col}">
                    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.35rem">
                        <span style="font-size:0.95rem;font-weight:600;color:#fff">{r['web_name']}</span>
                        <span style="font-size:0.75rem;color:var(--text2)">{r['team_name']}</span>
                        {status_pill(s, chance)}
                        <span style="margin-left:auto;font-family:'Barlow Condensed',sans-serif;
                                     font-size:0.8rem;font-weight:700;color:{col}">{int(prob*100)}% start</span>
                        <span style="font-family:'Barlow Condensed',sans-serif;font-size:0.8rem;
                                     font-weight:700;color:var(--accent)">{form:.1f} form</span>
                    </div>
                    <div style="font-size:0.78rem;color:var(--text2)">{news}</div>
                </div>""", unsafe_allow_html=True)

    # ── BRIEFING TAB ─────────────────────────────────────────────────────────
    with t5:
        col_brief, col_back = st.columns([3, 2])

        with col_brief:
            if agent is None:
                st.info("No briefing. Run `python fpl_agent.py`")
            else:
                icons = {"captain":"🎖","transfers":"🔄","risks":"⚠","summary":"📋"}
                titles = {"captain":"Captain Pick","transfers":"Transfer Advice",
                          "risks":"Key Risks","summary":"Weekly Summary"}

                for key in ["captain","transfers","risks","summary"]:
                    m = re.search(rf"<{key}>(.*?)</{key}>", agent, re.DOTALL)
                    if m:
                        content = m.group(1).strip().replace("**","").replace("*","")
                        st.markdown(f"""
                        <div class="brief-block">
                            <div class="brief-icon-title">{icons[key]} &nbsp;{titles[key]}</div>
                            <div class="brief-body">{content}</div>
                        </div>""", unsafe_allow_html=True)

                if st.button("🔄 Regenerate (calls Claude API)"):
                    from fpl_agent import run_fpl_agent
                    with st.spinner("Calling Claude..."):
                        run_fpl_agent(force_refresh=True)
                    st.cache_data.clear()
                    st.rerun()

        with col_back:
            if backtest:
                st.markdown('<div class="sh">Backtest Performance</div>', unsafe_allow_html=True)
                s = backtest.get("summary", {})
                b1,b2 = st.columns(2)
                for col, lbl, val in [
                    (b1,"Avg pts/GW",f"{s.get('avg_pts_gw',0):.1f}"),
                    (b2,"Captain %", f"{s.get('captain_rate',0):.0f}%"),
                ]:
                    col.markdown(f"""
                    <div class="kpi">
                        <div class="kpi-label">{lbl}</div>
                        <div class="kpi-value" style="font-size:1.6rem">{val}</div>
                    </div>""", unsafe_allow_html=True)

                try:
                    import plotly.graph_objects as go
                    gw_data = backtest.get("gameweeks",[])
                    if gw_data:
                        gws = [r["gameweek"] for r in gw_data]
                        pts = [r["actual_points"] for r in gw_data]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=gws, y=pts,
                            marker_color=["#00e676" if p>=55 else "#00d4ff" if p>=45 else "#ff5252" for p in pts],
                            name="Actual pts",
                        ))
                        fig.add_shape(type="line",x0=gws[0],x1=gws[-1],y0=50,y1=50,
                                      line=dict(color="#546e8a",dash="dot",width=1))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#546e8a",size=10),
                            margin=dict(l=0,r=0,t=10,b=0),
                            height=250,showlegend=False,
                            xaxis=dict(gridcolor="#1e3a5f",title="GW",tickfont=dict(size=9)),
                            yaxis=dict(gridcolor="#1e3a5f",title="Pts",tickfont=dict(size=9)),
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                except ImportError:
                    pass

                st.markdown(f"""
                <div style="font-size:0.75rem;color:var(--muted);padding:0.5rem 0;line-height:1.6">
                    <div>Best GW: <span style="color:var(--accent)">GW{s.get('best_gw','?')} ({s.get('best_pts',0):.0f} pts)</span></div>
                    <div>Worst GW: <span style="color:var(--red)">GW{s.get('worst_gw','?')} ({s.get('worst_pts',0):.0f} pts)</span></div>
                    <div>Avg manager benchmark: <span style="color:var(--text2)">50 pts/GW</span></div>
                    <div style="margin-top:0.4rem;color:var(--text2)">{s.get('tier','—')}</div>
                </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
