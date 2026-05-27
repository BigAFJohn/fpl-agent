"""
prediction/captain_model.py  —  Phase 8d: Captain Classifier v2
================================================================

WHAT THIS MODEL DOES
--------------------
Predicts which player is most likely to be the top scorer in a
given gameweek. Used to select the captain in team_selector.py.

WHY A SEPARATE CAPTAIN MODEL?
------------------------------
The main XGBoost model in predict_points.py predicts AVERAGE points.
Captaincy requires predicting CEILING — who is likely to score BIG
this specific week, not who averages the most.

A player who scores [2, 2, 2, 2, 15] has avg=4.6.
The main model sees a mediocre pick. We see ceiling=15.

FEATURE DESIGN PHILOSOPHY
--------------------------
Based on feature discrimination analysis against historical top scorers:

  STRONGEST signals (top scorers vs non-top, ratio):
    avg_points_10gw      3.48x  — sustained form is predictive
    xgi_season           3.72x  — underlying attacking quality
    selected_by_percent  4.40x  — popular players score big more often

  MODERATE signals:
    max_points_5gw       1.47x  — actual ceiling in recent history
    last_3gw_top10_rate  1.92x  — hot streak signal
    price                ~2x    — premium players captain more reliably

  NOISE signals (removed):
    differential_score  15.67x  — dominated by fringe low-ownership
                                   players who occasionally explode,
                                   causes model to recommend non-starters

OWNERSHIP TIERS (new in this version)
--------------------------------------
Instead of a compound differential_score that creates infinite values
for zero-ownership players, we use categorical ownership tiers:

  is_high_ownership    selected_by_percent > 30%  — template captain
  is_medium_ownership  10-30%                      — semi-popular pick
  is_low_ownership     < 10%                       — differential

This lets XGBoost learn the non-linear relationship between ownership
and captaincy value without one signal dominating.

ACCURACY CONTEXT
----------------
Top-1 accuracy: picking the exact top scorer is very hard.
Even the best human FPL managers achieve ~25-30% top-1.
A random pick from 15 starters = 6.7%.
Our target: 10%+ top-1, 25%+ top-3.

The classifier is most useful for RANKING candidates —
distinguishing B.Fernandes from Calvert-Lewin for captaincy
is more valuable than predicting the exact top scorer.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise ImportError("Run: pip install xgboost scikit-learn")


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH            = "db/fpl.db"
CAPTAIN_MODEL_PATH = Path("models/xgb_captain_classifier.pkl")
TRAIN_SEASONS      = ["2022-23", "2023-24", "2024-25"]

CAPTAIN_FEATURES = [
    # === CEILING FEATURES — actual max scores from history ===
    "max_points_5gw",          # Max score in last 5 GWs (real ceiling)
    "max_points_3gw",          # Max score in last 3 GWs (more recent)
    "last_3gw_top10_rate",     # Fraction of last 3 GWs in top 10 scorers

    # === OWNERSHIP TIERS — replaces compound differential_score ===
    "is_high_ownership",       # > 30% selected — template captain candidate
    "is_medium_ownership",     # 10-30% selected — popular pick
    "is_low_ownership",        # < 10% selected — differential
    "selected_by_percent",     # Raw ownership for fine-grained signal

    # === FORM — strongest historical signal ===
    "avg_points_5gw",
    "avg_points_10gw",
    "avg_points_season",

    # === UNDERLYING QUALITY ===
    "xgi_season",
    "xgi_per90_season",
    "avg_goals_5gw",
    "avg_assists_5gw",
    "avg_ict_5gw",

    # === FIXTURE ===
    "fixture_fdr",
    "opponent_xga_5",
    "opp_xga_rank",
    "was_home",

    # === AVAILABILITY ===
    "lineup_probability",

    # === POSITION / PRICE ===
    "is_attacking",
    "is_premium",
    "price",

    # === POSITION ONE-HOT ===
    "pos_GK",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


# =============================================================================
# CEILING FEATURES FROM PLAYER_HISTORY
# =============================================================================

def compute_ceiling_features(engine):
    """
    Computes real ceiling features from player_history.

    Returns DataFrame with:
      max_points_5gw      — actual max score in last 5 GWs
      max_points_3gw      — actual max score in last 3 GWs
      last_3gw_top10_rate — fraction of last 3 GWs where player
                            ranked in global top 10 scorers

    Uses shift(1) — only past gameweeks, no data leakage.
    Only uses player_history (current season) joined to model_features
    for season context.
    """
    print("  Computing ceiling features from player_history...")

    ph = pd.read_sql("""
        SELECT ph.player_id,
               ph.round     AS gameweek,
               ph.total_points,
               mf.season
        FROM player_history ph
        JOIN (
            SELECT DISTINCT player_id, gameweek, season
            FROM model_features
        ) mf ON ph.player_id = mf.player_id
             AND ph.round     = mf.gameweek
        ORDER BY ph.player_id, mf.season, ph.round
    """, engine)

    if ph.empty:
        print("  ⚠ No player_history data found")
        return pd.DataFrame()

    print(f"  ✓ {len(ph):,} player-gameweek rows loaded")

    ph = ph.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)

    # Global rank per GW for top10 signal
    ph["gw_rank"] = ph.groupby(["season", "gameweek"])["total_points"].rank(
        ascending=False, method="min"
    )
    ph["is_top10"] = (ph["gw_rank"] <= 10).astype(float)

    # Rolling per player — shift(1) prevents leakage
    rows = []
    for (pid, season), grp in ph.groupby(["player_id", "season"]):
        grp  = grp.sort_values("gameweek").reset_index(drop=True)
        pts  = grp["total_points"]
        t10  = grp["is_top10"]

        for i in range(len(grp)):
            past5 = pts.iloc[max(0, i-5):i]
            past3 = pts.iloc[max(0, i-3):i]
            t10_3 = t10.iloc[max(0, i-3):i]
            rows.append({
                "player_id"          : pid,
                "season"             : season,
                "gameweek"           : int(grp.loc[i, "gameweek"]),
                "max_points_5gw"     : float(past5.max()) if len(past5) > 0 else 0.0,
                "max_points_3gw"     : float(past3.max()) if len(past3) > 0 else 0.0,
                "last_3gw_top10_rate": float(t10_3.mean()) if len(t10_3) > 0 else 0.0,
            })

    ceiling_df = pd.DataFrame(rows)
    print(f"  ✓ Ceiling features computed for {len(ceiling_df):,} rows")
    return ceiling_df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def load_training_data(engine):
    """Loads model_features for all seasons with actual_points."""
    print("  Loading captain training data...")
    df = pd.read_sql("""
        SELECT mf.*
        FROM model_features mf
        WHERE mf.actual_points IS NOT NULL
        ORDER BY mf.season, mf.gameweek, mf.player_id
    """, engine)
    print(f"  ✓ {len(df):,} rows loaded from model_features")
    return df


def engineer_captain_features(df, ceiling_df=None):
    """
    Adds all captain-specific features to the dataframe.
    Safe to call on both training data and prediction data.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # Merge real ceiling features
    # -------------------------------------------------------------------------
    if ceiling_df is not None and not ceiling_df.empty:
        df = df.merge(
            ceiling_df[["player_id", "season", "gameweek",
                        "max_points_5gw", "max_points_3gw",
                        "last_3gw_top10_rate"]],
            on=["player_id", "season", "gameweek"],
            how="left"
        )
        fallback_n = df["max_points_5gw"].isna().sum()
        if fallback_n > 0:
            avg = pd.to_numeric(df["avg_points_5gw"], errors="coerce").fillna(0)
            df["max_points_5gw"]      = df["max_points_5gw"].fillna(avg * 1.2)
            df["max_points_3gw"]      = df["max_points_3gw"].fillna(avg * 1.2)
            df["last_3gw_top10_rate"] = df["last_3gw_top10_rate"].fillna(0)
        print(f"  ✓ Ceiling features merged ({fallback_n:,} rows used fallback)")
    else:
        avg = pd.to_numeric(df.get("avg_points_5gw", 0), errors="coerce").fillna(0)
        df["max_points_5gw"]      = avg * 1.2
        df["max_points_3gw"]      = avg * 1.2
        df["last_3gw_top10_rate"] = 0.0

    # -------------------------------------------------------------------------
    # Ownership tiers — categorical, avoids compound ratio problems
    # -------------------------------------------------------------------------
    own = pd.to_numeric(
        df.get("selected_by_percent", pd.Series(5.0, index=df.index)),
        errors="coerce"
    ).fillna(5.0)

    df["selected_by_percent"] = own
    df["is_high_ownership"]   = (own > 30).astype(float)
    df["is_medium_ownership"] = ((own >= 10) & (own <= 30)).astype(float)
    df["is_low_ownership"]    = (own < 10).astype(float)

    # -------------------------------------------------------------------------
    # Position features
    # -------------------------------------------------------------------------
    for pos in ["GK", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(float)

    df["is_attacking"] = df["position"].isin(["FWD", "MID"]).astype(float)
    df["is_premium"]   = (
        pd.to_numeric(df["price"], errors="coerce") > 9.0
    ).astype(float)

    # -------------------------------------------------------------------------
    # Home advantage
    # -------------------------------------------------------------------------
    if "was_home" in df.columns:
        df["was_home"] = pd.to_numeric(df["was_home"], errors="coerce").fillna(0.5)
    else:
        df["was_home"] = 0.5

    # -------------------------------------------------------------------------
    # Fixture quality rank
    # -------------------------------------------------------------------------
    opp_xga = pd.to_numeric(
        df.get("opponent_xga_5", pd.Series(1.5, index=df.index)),
        errors="coerce"
    )
    df["opp_xga_rank"] = opp_xga.rank(pct=True).fillna(0.5)

    # -------------------------------------------------------------------------
    # Fill all missing values with column medians
    # -------------------------------------------------------------------------
    for col in CAPTAIN_FEATURES:
        if col in df.columns:
            median = df[col].median()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                median if pd.notna(median) else 0.0
            )
        else:
            df[col] = 0.0

    return df


def build_captain_labels(df):
    """
    Binary label: 1 if player was the top scorer in their GW, else 0.
    Ties: all tied top scorers get label=1.
    Players with 0 actual_points: label=0.
    """
    df    = df.copy()
    pts   = pd.to_numeric(df["actual_points"], errors="coerce").fillna(0)
    df["_pts"] = pts

    gw_max = df.groupby(["season", "gameweek"])["_pts"].transform("max")
    df["captain_label"] = ((pts == gw_max) & (pts > 0)).astype(int)

    n_labels  = df["captain_label"].sum()
    n_gws     = df.groupby(["season", "gameweek"]).ngroups
    pos_rate  = df["captain_label"].mean()
    print(f"  ✓ Captain labels: {n_labels:,} top scorers across {n_gws:,} GWs")
    print(f"    Positive rate  : {pos_rate:.2%}")
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_captain_classifier(df):
    """
    Trains XGBoost binary classifier.
    Class imbalance handled via scale_pos_weight.
    """
    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[~df["season"].isin(TRAIN_SEASONS)].copy()

    X_train = train_df[CAPTAIN_FEATURES].values
    y_train = train_df["captain_label"].values
    X_val   = val_df[CAPTAIN_FEATURES].values
    y_val   = val_df["captain_label"].values

    pos          = max(int(y_train.sum()), 1)
    neg          = len(y_train) - pos
    spw          = neg / pos

    print(f"  Train: {len(train_df):,} rows, {pos:,} positives")
    print(f"  Val  : {len(val_df):,} rows, {int(y_val.sum()):,} positives")
    print(f"  scale_pos_weight: {spw:.1f}")

    model = xgb.XGBClassifier(
        n_estimators       = 500,
        max_depth          = 4,
        learning_rate      = 0.03,
        subsample          = 0.8,
        colsample_bytree   = 0.7,
        min_child_weight   = 10,
        scale_pos_weight   = spw,
        random_state       = 42,
        n_jobs             = -1,
        eval_metric        = "auc",
        early_stopping_rounds = 40,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  ✓ Training complete — best round: {model.best_iteration}")
    return model, val_df, X_val, y_val


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_captain_model(model, val_df, X_val, y_val):
    """
    Evaluates captain classifier with top-1 and top-3 accuracy.

    Top-1: did our #1 ranked player score the most points that GW?
    Top-3: was our #1 ranked player in the top 3 scorers that GW?

    Top-3 is the FPL-relevant metric — a player who scores 15
    when the top scorer scored 16 is still a great captain.
    """
    print("\n  Evaluating captain classifier v2...")

    probs  = model.predict_proba(X_val)[:, 1]
    val_df = val_df.copy()
    val_df["captain_score"] = probs
    val_df["_pts"] = pd.to_numeric(
        val_df["actual_points"], errors="coerce"
    ).fillna(0)

    try:
        auc = roc_auc_score(y_val, probs)
        print(f"  AUC-ROC          : {auc:.3f}")
    except Exception:
        print("  AUC-ROC          : N/A")

    top1_correct = top3_correct = total = 0

    for (season, gw), gw_df in val_df.groupby(["season", "gameweek"]):
        if len(gw_df) < 50 or gw_df["captain_label"].sum() == 0:
            continue

        best_idx = gw_df["captain_score"].idxmax()
        best_pts = float(gw_df.loc[best_idx, "_pts"])
        gw_max   = gw_df["_pts"].max()
        top3_min = gw_df["_pts"].nlargest(3).min()

        top1_correct += int(best_pts == gw_max and best_pts > 0)
        top3_correct += int(best_pts >= top3_min and best_pts > 0)
        total        += 1

    top1 = top1_correct / total if total > 0 else 0
    top3 = top3_correct / total if total > 0 else 0

    print(f"  Top-1 accuracy   : {top1:.1%} ({top1_correct}/{total} GWs)")
    print(f"  Top-3 accuracy   : {top3:.1%} ({top3_correct}/{total} GWs)")
    print(f"  Baseline top-1   : ~6.7%  (random 1 of 15)")
    print(f"  Baseline top-3   : ~20.0% (random 3 of 15)")

    # Feature importance
    print(f"\n  Top 10 captain features:")
    imp = pd.Series(
        model.feature_importances_, index=CAPTAIN_FEATURES
    ).sort_values(ascending=False)
    bar_max = imp.iloc[0]
    for feat, val in imp.head(10).items():
        bar = "█" * int(val / bar_max * 40)
        print(f"    {feat:<35} {val:.4f} {bar}")

    return top1, top3


# =============================================================================
# SCORING — apply to predictions table
# =============================================================================

def score_predictions(model, engine):
    """
    Applies captain classifier to current predictions table.
    Loads features from model_features (has all required columns).
    Updates predictions table with captain_score and captain_rank.
    """
    print("\n  Scoring predictions with captain classifier v2...")

    preds = pd.read_sql("""
        SELECT mf.*,
               p.web_name,
               p.price          AS price_override,
               p.adjusted_points
        FROM model_features mf
        JOIN predictions p ON mf.player_id = p.player_id
        WHERE mf.season = '2025-26'
          AND mf.gameweek = (
              SELECT MAX(gameweek)
              FROM model_features
              WHERE season = '2025-26'
          )
    """, engine)

    if preds.empty:
        print("  ✗ No prediction data found")
        return

    # Use price from predictions table (more current)
    if "price_override" in preds.columns:
        preds["price"] = preds["price_override"]

    ceiling_df = compute_ceiling_features(engine)
    preds      = engineer_captain_features(preds, ceiling_df)

    scores = model.predict_proba(preds[CAPTAIN_FEATURES].values)[:, 1]
    preds["captain_score"] = scores.round(4)
    preds["captain_rank"]  = preds["captain_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Add columns if needed — separate transactions to avoid abort
    for col, coltype in [("captain_score", "NUMERIC(5,4)"),
                         ("captain_rank",  "INTEGER")]:
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE predictions ADD COLUMN {col} {coltype}"
                ))
        except Exception:
            pass  # Column already exists

    # Update scores
    with engine.begin() as conn:
        for _, row in preds.iterrows():
            conn.execute(text("""
                UPDATE predictions
                SET captain_score = :cs, captain_rank = :cr
                WHERE player_id = :pid
            """), {
                "cs" : float(row["captain_score"]),
                "cr" : int(row["captain_rank"]),
                "pid": int(row["player_id"]),
            })

    # Display top 10
    show = [c for c in ["web_name", "position", "adjusted_points",
                         "max_points_5gw", "selected_by_percent",
                         "captain_score", "captain_rank",
                         "fixture_fdr", "opponent_name"]
            if c in preds.columns]
    top10 = preds.nsmallest(10, "captain_rank")[show].copy()

    print(f"\n  Top 10 captain recommendations (v2):")
    print(f"  {'#':<3} {'Player':<22} {'Pos':<4} {'AdjPts':>7} "
          f"{'MaxPts5':>8} {'Own%':>5} {'CapScore':>9} {'FDR':>4} Opponent")
    print(f"  {'-'*80}")
    for _, r in top10.iterrows():
        wn = r.get("web_name", "")
        name = str(wn) if isinstance(wn, str) else (
            str(wn.iloc[0]) if hasattr(wn, "iloc") else str(wn)
        )
        print(
            f"  {int(r['captain_rank']):<3} {name:<22} "
            f"{str(r.get('position','')):<4} "
            f"{float(r.get('adjusted_points', 0) or 0):>7.2f} "
            f"{float(r.get('max_points_5gw', 0) or 0):>8.1f} "
            f"{float(r.get('selected_by_percent', 0) or 0):>5.1f}% "
            f"{float(r['captain_score'] or 0):>9.4f} "
            f"{float(r.get('fixture_fdr', 0) or 0):>4.1f} "
            f"{str(r.get('opponent_name', '') or '')}"
        )

    print(f"\n  ✓ captain_score v2 added to {len(preds):,} predictions")
    return preds


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_captain_model(model, top1, top3, features):
    CAPTAIN_MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(CAPTAIN_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model"      : model,
            "features"   : features,
            "top1_acc"   : top1,
            "top3_acc"   : top3,
            "version"    : "v2",
            "trained_at" : datetime.now().isoformat(),
        }, f)
    print(f"  ✓ Captain model v2 saved to {CAPTAIN_MODEL_PATH}")


def load_captain_model():
    if not CAPTAIN_MODEL_PATH.exists():
        return None, None
    with open(CAPTAIN_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("top1_acc", data.get("accuracy"))


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_captain_model():
    """
    Full pipeline:
    1. Load model_features from DB
    2. Compute real ceiling features from player_history
    3. Engineer all captain features
    4. Build labels — who was top scorer each GW?
    5. Train XGBoost classifier
    6. Evaluate top-1 and top-3 accuracy
    7. Score current predictions table
    8. Save model
    """
    start  = datetime.now()
    engine = get_engine()

    print("=" * 60)
    print(f"Captain Classifier v2  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    print("\n[1/6] Loading training data...")
    df = load_training_data(engine)

    print("\n[2/6] Computing ceiling features...")
    ceiling_df = compute_ceiling_features(engine)

    print("\n[3/6] Engineering captain features...")
    df = engineer_captain_features(df, ceiling_df)

    print("\n[4/6] Building captain labels...")
    df = build_captain_labels(df)

    print("\n[5/6] Training...")
    model, val_df, X_val, y_val = train_captain_classifier(df)

    print("\n[6/6] Evaluating...")
    top1, top3 = evaluate_captain_model(model, val_df, X_val, y_val)

    print("\n[+] Scoring current predictions...")
    score_predictions(model, engine)

    save_captain_model(model, top1, top3, CAPTAIN_FEATURES)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  Top-1 accuracy   : {top1:.1%}")
    print(f"  Top-3 accuracy   : {top3:.1%}")
    print(f"  Model saved      : {CAPTAIN_MODEL_PATH}")
    print(f"  Duration         : {duration:.0f}s")
    print(f"{'='*60}")

    return model, top1


if __name__ == "__main__":
    run_captain_model()
