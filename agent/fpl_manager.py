"""
agent/fpl_manager.py  —  Phase 8e: FPL Auto-Submission with Approval Gate
=========================================================================

USAGE
-----
  # Dry run — shows diff, no submission (safe default)
  python agent/fpl_manager.py

  # Show diff and ask for approval before submitting
  python agent/fpl_manager.py --submit

  # Skip confirmation (use with caution)
  python agent/fpl_manager.py --submit --auto-approve

ENVIRONMENT VARIABLES REQUIRED
-------------------------------
  FPL_EMAIL     your FPL account email
  FPL_PASSWORD  your FPL account password
  FPL_TEAM_ID   your FPL team ID
               (from URL: fantasy.premierleague.com/entry/[TEAM_ID]/history)
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text

FPL_BASE_URL  = "https://fantasy.premierleague.com/api"
FPL_LOGIN_URL = "https://users.premierleague.com/accounts/login/"
APPROVAL_FILE = Path("models/pending_submission.json")
DB_PATH       = "db/fpl.db"
REQUEST_DELAY = 1.0


def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


class FPLClient:
    """Lightweight FPL API client with session-based authentication."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://fantasy.premierleague.com",
            "Referer": "https://fantasy.premierleague.com/",
        })
        self.player_map = {}

    def authenticate(self, email, password):
        # Check for direct access token (bypasses login DNS issue)
        access_token = os.environ.get("FPL_ACCESS_TOKEN")
        if access_token:
            self.session.headers.update({
                "Authorization": f"Bearer {access_token}"
            })
            self.session.cookies.set(
                "access_token", access_token,
                domain=".premierleague.com"
            )
            print("  ✓ Authenticated via access token")
            return True

    def get_bootstrap(self):
        """Loads FPL bootstrap data. Returns dict with current_gw."""
        time.sleep(REQUEST_DELAY)
        try:
            r = self.session.get(f"{FPL_BASE_URL}/bootstrap-static/", timeout=15)
            if r.status_code != 200:
                return None
            data = r.json()
            for p in data.get("elements", []):
                self.player_map[p["id"]] = {
                    "web_name"    : p["web_name"],
                    "element_type": p["element_type"],
                    "now_cost"    : p["now_cost"] / 10,
                }
            current_gw = next(
                (e["id"] for e in data.get("events", []) if e.get("is_current")),
                next(
                    (e["id"] for e in data.get("events", []) if e.get("is_next")),
                    1
                )
            )
            print(f"  ✓ Bootstrap loaded — {len(self.player_map)} players, GW{current_gw}")
            return {"current_gw": current_gw}
        except Exception as e:
            print(f"  ✗ Bootstrap error: {e}")
            return None

    def get_my_team(self, team_id):
        """Fetches current team picks."""
        time.sleep(REQUEST_DELAY)
        try:
            r = self.session.get(f"{FPL_BASE_URL}/my-team/{team_id}/", timeout=10)
            if r.status_code != 200:
                print(f"  ✗ Failed to get team (status {r.status_code})")
                return None
            picks = r.json().get("picks", [])
            print(f"  ✓ Current team loaded — {len(picks)} players")
            return picks
        except Exception as e:
            print(f"  ✗ Get team error: {e}")
            return None

    def submit_lineup(self, team_id, picks):
        """Submits lineup picks to FPL."""
        time.sleep(REQUEST_DELAY)
        try:
            csrftoken = self.session.cookies.get("csrftoken", "")
            r = self.session.post(
                f"{FPL_BASE_URL}/my-team/{team_id}/",
                json={"picks": picks},
                headers={"X-CSRFToken": csrftoken},
                timeout=10,
            )
            if r.status_code == 200:
                print("  ✓ Lineup submitted successfully")
                return True
            print(f"  ✗ Submission failed (status {r.status_code})")
            try:
                print(f"    {r.json()}")
            except Exception:
                print(f"    {r.text[:200]}")
            return False
        except Exception as e:
            print(f"  ✗ Submission error: {e}")
            return False


def load_model_team(engine):
    """Loads the model's recommended team from selected_team table."""
    return pd.read_sql(text("""
        SELECT player_id, web_name, position, team_name, price,
               adjusted_points, is_starting, is_captain,
               is_vice_captain, bench_order
        FROM selected_team
        ORDER BY is_starting DESC, is_captain DESC,
                 adjusted_points DESC, bench_order
    """), engine)


def compute_diff(current_picks, model_team, player_map):
    """Computes the difference between current and model team."""
    current_elements = {p["element"] for p in current_picks}
    model_elements   = set(model_team["player_id"].astype(int).tolist())

    out_elements = current_elements - model_elements
    in_elements  = model_elements  - current_elements

    curr_cap = next((p for p in current_picks if p.get("is_captain")), None)
    curr_cap_name = player_map.get(
        curr_cap["element"], {}
    ).get("web_name", "?") if curr_cap else "?"

    model_cap_row  = model_team[model_team["is_captain"] == True]
    model_cap_name = str(model_cap_row["web_name"].iloc[0]) \
        if not model_cap_row.empty else "?"

    return {
        "transfers_out"  : [
            player_map.get(e, {}).get("web_name", f"ID:{e}")
            for e in out_elements
        ],
        "transfers_in"   : [
            str(r["web_name"])
            for _, r in model_team.iterrows()
            if int(r["player_id"]) in in_elements
        ],
        "n_transfers"    : len(out_elements),
        "captain_change" : curr_cap_name != model_cap_name,
        "current_captain": curr_cap_name,
        "model_captain"  : model_cap_name,
    }


def display_proposed_changes(diff, model_team, gw):
    """Displays proposed changes clearly before asking for approval."""
    print(f"\n{'='*62}")
    print(f"  PROPOSED CHANGES — GW{gw}")
    print(f"{'='*62}")

    if diff["n_transfers"] == 0:
        print("\n  ✓ No transfers needed — same 15 players")
    else:
        print(f"\n  TRANSFERS ({diff['n_transfers']}):")
        for name in diff["transfers_out"]:
            print(f"    OUT ← {name}")
        for name in diff["transfers_in"]:
            print(f"    IN  → {name}")

    if diff["captain_change"]:
        print(f"\n  CAPTAIN CHANGE:")
        print(f"    Current : {diff['current_captain']}")
        print(f"    Model   : {diff['model_captain']}")
    else:
        print(f"\n  ✓ Captain unchanged: {diff['model_captain']}")

    print(f"\n  FULL LINEUP:")
    starters = model_team[model_team["is_starting"] == True].sort_values(
        "adjusted_points", ascending=False
    )
    for _, r in starters.iterrows():
        cap = " ← CAPTAIN" if r["is_captain"] else \
              " ← VICE"   if r["is_vice_captain"] else ""
        new = " ← NEW" if str(r["web_name"]) in diff["transfers_in"] else ""
        print(
            f"    {str(r['web_name']):<22} {str(r['position']):<4} "
            f"£{float(r['price']):>4.1f}  adj={float(r['adjusted_points']):.2f}"
            f"{cap}{new}"
        )

    bench = model_team[model_team["is_starting"] == False].sort_values("bench_order")
    print(f"\n  BENCH:")
    for _, r in bench.iterrows():
        print(
            f"    {int(r['bench_order'] or 0)}. {str(r['web_name']):<22} "
            f"{str(r['position']):<4} £{float(r['price']):>4.1f}"
        )
    print(f"{'='*62}")


def request_approval(dry_run=False, auto_approve=False):
    """Asks for human approval. Returns True if approved."""
    if dry_run:
        print("\n  DRY RUN — not submitting.")
        print("  Run with --submit to actually submit to FPL.")
        return False
    if auto_approve:
        print("\n  AUTO-APPROVE — submitting without confirmation.")
        return True
    print("\n  Submit these changes to FPL?")
    print("  [y] Yes — submit now")
    print("  [n] No  — cancel")
    while True:
        choice = input("\n  Your choice [y/n]: ").strip().lower()
        if choice == "y":
            return True
        elif choice == "n":
            print("  Cancelled.")
            return False
        print("  Please enter y or n")


def save_pending(diff, model_team, gw):
    """Saves pending submission to disk."""
    APPROVAL_FILE.parent.mkdir(exist_ok=True)
    APPROVAL_FILE.write_text(json.dumps({
        "gameweek"  : gw,
        "created_at": datetime.now().isoformat(),
        "status"    : "pending",
        "diff"      : diff,
        "model_team": model_team.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"  ✓ Pending submission saved to {APPROVAL_FILE}")


def build_picks(model_team):
    """Converts model team to FPL API picks format (positions 1-15)."""
    picks   = []
    pos_num = 1

    starters = model_team[model_team["is_starting"] == True]
    gk       = starters[starters["position"] == "GK"]
    outfield = starters[starters["position"] != "GK"].sort_values(
        "adjusted_points", ascending=False
    )
    for _, r in pd.concat([gk, outfield]).iterrows():
        picks.append({
            "element"        : int(r["player_id"]),
            "position"       : pos_num,
            "is_captain"     : bool(r["is_captain"]),
            "is_vice_captain": bool(r["is_vice_captain"]),
        })
        pos_num += 1

    bench     = model_team[model_team["is_starting"] == False]
    bench_gk  = bench[bench["position"] == "GK"]
    bench_out = bench[bench["position"] != "GK"].sort_values("bench_order")
    for _, r in pd.concat([bench_out, bench_gk]).iterrows():
        picks.append({
            "element"        : int(r["player_id"]),
            "position"       : pos_num,
            "is_captain"     : False,
            "is_vice_captain": False,
        })
        pos_num += 1

    return picks


def run_fpl_manager(dry_run=True, auto_approve=False):
    """Full FPL submission pipeline with approval gate."""
    start = datetime.now()

    print("=" * 62)
    print(f"FPL Manager  |  {start.strftime('%Y-%m-%d %H:%M')}")
    if dry_run:
        print("  MODE: DRY RUN (use --submit to actually submit)")
    print("=" * 62)

    email   = os.environ.get("FPL_EMAIL")
    password= os.environ.get("FPL_PASSWORD")
    team_id = os.environ.get("FPL_TEAM_ID")

    if not all([email, password, team_id]):
        print("\n  ✗ Missing credentials. Set:")
        print("    export FPL_EMAIL='your@email.com'")
        print("    export FPL_PASSWORD='yourpassword'")
        print("    export FPL_TEAM_ID='123456'")
        print("\n  Find team ID: fantasy.premierleague.com/entry/[ID]/history")
        return False

    team_id = int(team_id)

    print("\n[1/5] Loading model recommendations...")
    engine     = get_engine()
    model_team = load_model_team(engine)
    if model_team.empty:
        print("  ✗ No team found — run: python scheduler.py predict")
        return False
    gw = pd.read_sql(text("SELECT MAX(gameweek) FROM selected_team"), engine).iloc[0, 0]
    print(f"  ✓ Model team for GW{gw} — {len(model_team)} players")

    print("\n[2/5] Connecting to FPL...")
    client = FPLClient()
    if not client.authenticate(email, password):
        return False

    print("\n[3/5] Loading FPL data...")
    bootstrap = client.get_bootstrap()
    if not bootstrap:
        return False
    current_picks = client.get_my_team(team_id)
    if not current_picks:
        return False

    print("\n[4/5] Computing changes...")
    diff = compute_diff(current_picks, model_team, client.player_map)
    print(f"  Transfers needed : {diff['n_transfers']}")
    cap_str = f"Yes — {diff['model_captain']}" if diff["captain_change"] else "No"
    print(f"  Captain change   : {cap_str}")

    display_proposed_changes(diff, model_team, gw)
    save_pending(diff, model_team, gw)

    approved = request_approval(dry_run=dry_run, auto_approve=auto_approve)
    if not approved:
        return False

    print("\n[5/5] Submitting to FPL...")
    if diff["n_transfers"] > 0:
        print(f"  ⚠ Make these transfers manually on fantasy.premierleague.com first:")
        print(f"    OUT: {', '.join(diff['transfers_out'])}")
        print(f"    IN : {', '.join(diff['transfers_in'])}")
        print()

    picks   = build_picks(model_team)
    success = client.submit_lineup(team_id, picks)

    if success:
        if APPROVAL_FILE.exists():
            pending = json.loads(APPROVAL_FILE.read_text())
            pending["status"]       = "submitted"
            pending["submitted_at"] = datetime.now().isoformat()
            APPROVAL_FILE.write_text(json.dumps(pending, indent=2, default=str))
        print(f"\n  ✓ GW{gw} lineup submitted — Captain: {diff['model_captain']}")

    print(f"\n  Duration: {(datetime.now() - start).total_seconds():.0f}s")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPL Auto-Submission Agent")
    parser.add_argument("--submit",       action="store_true",
                        help="Actually submit (default is dry-run)")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Skip human confirmation")
    args = parser.parse_args()
    run_fpl_manager(dry_run=not args.submit, auto_approve=args.auto_approve)
