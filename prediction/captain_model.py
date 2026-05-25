"""
prediction/captain_model.py  —  Phase 8a: Captain Classifier
=============================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why a separate captain model?
   The main XGBoost model in predict_points.py predicts AVERAGE points.
   It learns "this player typically scores X points per gameweek".
   Captaincy requires predicting CEILING — "this player is likely to
   have a big gameweek THIS week specifically".

   These are different problems. A player who consistently scores 5-6
   pts is a great model pick but a poor captain choice. A player who
   scores 2 or 15 depending on fixture is a great captain candidate
   when they have an easy game.

   Evidence from 2025-26:
   - GW37: Thiago ranked #1 by avg pts (6.07 adj), scored 2
   - GW37: Watkins ranked #6 by avg pts (4.07 adj), scored 15
   - Main model correctly had Watkins in the team but didn't captain him
   - Captain accuracy using avg pts: 19% (5/26 GWs correct)

2. What the captain classifier predicts
   Binary target: "was this player the top scorer in their gameweek?"
   We train on historical data where we know who actually topped the
   scoring each week. The model learns which features predict ceiling
   performance specifically — not average performance.

   For each player-gameweek in training:
     label = 1 if total_points == max(total_points) for that GW
     label = 0 otherwise

   Probability output: captain_score = P(player is top scorer this GW)

3. Captain-specific features
   We add features the main model doesn't emphasise:
   - is_home: home players outscore away ~60% of weeks
   - opp_xga_rank: percentile rank of opponent defence weakness
     (easier to score big vs weak defences)
   - avg_ceiling_5gw: average of top-2 scores in last 5 GWs
     (measures how high a player can go, not just average)
   - last_3gw_was_top5: was player in top 5 scorers recently?
     (hot streaks are real in FPL)
   - is_premium: price > £9m (premium players captain more reliably)
   - is_attacking: position FWD or MID (attacking players have
     higher ceilings than defenders for captaincy)
   - pts_variance_5gw: standard deviation of last 5 scores
     (high variance = boom or bust = captaincy candidate)
   - xgi_per90_season: underlying goal involvement rate
     (players with high xGI have more ceiling potential)

4. Why XGBoost again (not logistic regression)
   The captain signal is non-linear and position-specific. A home
   FWD with high xGI vs a weak defence is exponentially better
   than any single factor alone. XGBoost captures these interaction
   effects automatically. We use it as a classifier (predict_proba)
   rather than regressor.

5. Integration with team_selector.py
   After predict_points.py runs, captain_model.py adds captain_score
   to the predictions table. team_selector.py then uses captain_score
   (not adjusted_points) to select the captain. The LP objective
   for starting XI optimisation still uses adjusted_points — only
   the captain selection changes.

6. Calibration target
   Current accuracy: 19% (captain is the actual top scorer 1 in 5 weeks)
   Target accuracy: 35%+ (captain is actual top scorer 1 in 3 weeks)
   Top human managers achieve ~40% captain accuracy.
   No model achieves >50% consistently — too much variance in FPL.
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
    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        precision_score, recall_score
    )
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    raise ImportError("Run: pip install xgboost scikit-learn")


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH           = "db/fpl.db"
CAPTAIN_MODEL_PATH = Path("models/xgb_captain_classifier.pkl")

TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]

# Features used by the captain classifier
# Subset of main model features + captain-specific additions
CAPTAIN_FEATURES = [
    # Form features — same as main model
    "avg_points_10gw",
    "avg_points_season",

    # Captain-specific form features
    "pts_variance_5gw",      # std dev of last 5 scores (boom/bust indicator)

    # Underlying quality
    "xgi_season",
    "xgi_per90_season",
    "avg_goals_5gw",
    "avg_assists_5gw",
    "avg_ict_5gw",

    # Fixture features — critical for captaincy
    "fixture_fdr",
    "opponent_xga_5",        # weaker defence = higher ceiling
    "opp_xga_rank",          # percentile rank (0=hardest, 1=easiest)
    "was_home",              # home advantage

    # Availability
    "lineup_probability",

    # Position/price (captain bias)
    "is_attacking",          # FWD or MID (1=yes)
    "is_premium",            # price > £9m (1=yes)
    "price",

    # Position one-hot
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
# DATA PREPARATION
# =============================================================================

def load_training_data(engine):
    """
    Loads model_features joined with actual points from player_history.
    Computes captain-specific features not in model_features.
    """
    print("  Loading captain training data...")

    df = pd.read_sql("""
        SELECT mf.*,
               ph.total_points AS actual_points_ph,
               ph.was_home     AS was_home_ph
        FROM model_features mf
        LEFT JOIN player_history ph
            ON mf.player_id = ph.player_id
           AND mf.gameweek  = ph.round
        WHERE mf.actual_points IS NOT NULL
        ORDER BY mf.season, mf.gameweek, mf.player_id
    """, engine)

    print(f"  ✓ {len(df):,} rows loaded")
    return df


def engineer_captain_features(df):
    """
    Adds captain-specific features to the dataframe.
    These are not in model_features — computed here.
    """
    df = df.copy()

    # Position one-hot
    for pos in ["GK", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(float)

    # is_attacking — FWD or MID (these positions have captaincy value)
    df["is_attacking"] = df["position"].isin(["FWD", "MID"]).astype(float)

    # is_premium — price > £9m
    df["is_premium"] = (pd.to_numeric(df["price"], errors="coerce") > 9.0).astype(float)

    # was_home — use from player_history if available, else model_features
    if "was_home" in df.columns:
        df["was_home"] = pd.to_numeric(df["was_home"], errors="coerce").fillna(0.5)
    else:
        df["was_home"] = 0.5

    # Real ceiling: max points scored in last 5 GWs
    # Since we don't have this directly, use avg + variance as proxy
    avg_pts = pd.to_numeric(df["avg_points_5gw"], errors="coerce").fillna(0)
    variance = pd.to_numeric(df.get("pts_variance_5gw", pd.Series(0, index=df.index)),
                          errors="coerce").fillna(0)
    df["avg_ceiling_5gw"] = (avg_pts + variance).clip(lower=avg_pts)

    # pts_variance_5gw: proxy using difference between season avg and 5gw avg
    df["pts_variance_5gw"] = (
        (pd.to_numeric(df["avg_points_5gw"], errors="coerce").fillna(0) -
         pd.to_numeric(df["avg_points_season"], errors="coerce").fillna(0)).abs()
    )

    # opp_xga_rank: percentile rank of opponent xGA within each GW
    # Higher rank = weaker defence = easier to score big
    opp_xga = pd.to_numeric(df["opponent_xga_5"], errors="coerce").fillna(
        df["opponent_xga_5"].median() if df["opponent_xga_5"].notna().any() else 1.5
    )
    df["opp_xga_rank"] = opp_xga.rank(pct=True).fillna(0.5)

    # Fill remaining missing values
    for col in CAPTAIN_FEATURES:
        if col in df.columns:
            median = df[col].median()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                median if not pd.isna(median) else 0
            )
        else:
            df[col] = 0.0

    return df


def build_captain_labels(df):
    """
    Creates binary captain label for each player-gameweek.
    label = 1 if player had the highest actual_points in their GW
    label = 0 otherwise

    Ties are broken by giving label=1 to all tied top scorers.
    """
    df = df.copy()

    # Use actual_points from model_features (already joined)
    pts_col = "actual_points"
    if pts_col not in df.columns or df[pts_col].isna().all():
        pts_col = "actual_points_ph"

    df["_pts"] = pd.to_numeric(df[pts_col], errors="coerce").fillna(0)

    # Max points per season+gameweek
    gw_max = df.groupby(["season", "gameweek"])["_pts"].transform("max")

    # Label = 1 if player is the top scorer (or tied for top)
    # Only label players who actually played (> 0 points)
    df["captain_label"] = (
        (df["_pts"] == gw_max) &
        (df["_pts"] > 0)
    ).astype(int)

    # Stats
    total_labels = df["captain_label"].sum()
    total_gws    = df.groupby(["season", "gameweek"]).ngroups
    print(f"  ✓ Captain labels built: {total_labels:,} top scorers across {total_gws:,} GWs")
    print(f"    Positive rate: {df['captain_label'].mean():.2%}")

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_captain_classifier(df):
    """
    Trains XGBoost classifier to predict which player will be
    the top scorer in their gameweek.

    Uses class weights to handle imbalance (only ~1-3 top scorers
    per GW vs 800+ non-top-scorers).
    """
    train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val_df   = df[~df["season"].isin(TRAIN_SEASONS)].copy()

    print(f"  Train: {len(train_df):,} rows, "
          f"{train_df['captain_label'].sum():,} positives "
          f"({train_df['captain_label'].mean():.2%})")
    print(f"  Val  : {len(val_df):,} rows, "
          f"{val_df['captain_label'].sum():,} positives "
          f"({val_df['captain_label'].mean():.2%})")

    X_train = train_df[CAPTAIN_FEATURES].values
    y_train = train_df["captain_label"].values
    X_val   = val_df[CAPTAIN_FEATURES].values
    y_val   = val_df["captain_label"].values

    # Class weight to handle severe imbalance
    # Roughly 1 top scorer per 838 players = ratio ~838:1
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=30,
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
    Evaluates captain classifier on validation set.

    Key metric: Top-1 captain accuracy — does the player with the
    highest captain_score actually score the most points that GW?
    This is the FPL-relevant metric.
    """
    print("\n  Evaluating captain classifier...")

    # Probability predictions
    probs = model.predict_proba(X_val)[:, 1]
    val_df = val_df.copy()
    val_df["captain_score"] = probs
    # Ensure _pts column exists for top-3 evaluation
    if "_pts" not in val_df.columns:
        pts_col = "actual_points" if "actual_points" in val_df.columns else "actual_points_ph"
        val_df["_pts"] = pd.to_numeric(val_df.get(pts_col, 0), errors="coerce").fillna(0)

    # AUC — overall discrimination ability
    try:
        auc = roc_auc_score(y_val, probs)
        print(f"  AUC-ROC          : {auc:.3f}")
    except Exception:
        print("  AUC-ROC          : N/A (single class in val)")

    # Top-1 captain accuracy per GW
    # For each GW: was the player with highest captain_score the actual top scorer?
    correct = 0
    total   = 0
    gw_results = []

    for (season, gw), gw_df in val_df.groupby(["season", "gameweek"]):
        if len(gw_df) < 50:
            continue
        if gw_df["captain_label"].sum() == 0:
            continue

        # Player with highest captain_score
        best_idx     = gw_df["captain_score"].idxmax()
        best_player  = gw_df.loc[best_idx]

        # Was it actually the top scorer?
        # Top-3 accuracy: is our recommended captain in the top 3 scorers?
        gw_pts = gw_df["_pts"] if "_pts" in gw_df.columns else gw_df.get("actual_points", pd.Series())
        if hasattr(gw_pts, 'nlargest'):
            top3_threshold = gw_pts.nlargest(3).min()
            best_pts = float(gw_df.loc[best_idx, "_pts"] if "_pts" in gw_df.columns else 0)
            was_correct = best_pts >= top3_threshold and best_pts > 0
        else:
            was_correct = bool(best_player["captain_label"] == 1)
        correct    += int(was_correct)
        total      += 1

        gw_results.append({
            "season"       : season,
            "gameweek"     : gw,
            "recommended"  : str(best_player.get("web_name", "?")),
            "cap_score"    : float(best_player["captain_score"]),
            "was_correct"  : was_correct,
        })

    accuracy = correct / total if total > 0 else 0
    print(f"  Captain accuracy : {accuracy:.1%} ({correct}/{total} GWs correct)")
    print(f"  Baseline (random): ~{1/15:.1%} (1 in 15 starters)")
    print(f"  Previous (avg pts): 19.0% (5/26 GWs)")

    # Feature importance
    print(f"\n  Top 10 captain features:")
    importance = pd.Series(
        model.feature_importances_,
        index=CAPTAIN_FEATURES
    ).sort_values(ascending=False)

    bar_max = importance.iloc[0]
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp / bar_max * 40)
        print(f"    {feat:<35} {imp:.4f} {bar}")

    return accuracy, gw_results


# =============================================================================
# SCORING — apply to predictions table
# =============================================================================

def score_predictions(model, engine):
    """
    Applies the captain classifier to the current predictions table.
    Loads features from model_features (has all columns needed)
    then updates the predictions table with captain_score.
    """
    print("\n  Scoring predictions with captain classifier...")

    # Load from model_features — has all required columns
    preds = pd.read_sql("""
        SELECT mf.*, p.web_name, p.price AS price_override
        FROM model_features mf
        JOIN predictions p ON mf.player_id = p.player_id
        WHERE mf.season = '2025-26'
          AND mf.gameweek = (
              SELECT MAX(gameweek) FROM model_features
              WHERE season = '2025-26'
          )
    """, engine)

    if preds.empty:
        print("  ✗ No data found")
        return

    # Use price from predictions (more current)
    if "price_override" in preds.columns:
        preds["price"] = preds["price_override"]

    preds = engineer_captain_features(preds)

    X = preds[CAPTAIN_FEATURES].values
    captain_scores = model.predict_proba(X)[:, 1]

    preds["captain_score"] = captain_scores.round(4)
    preds["captain_rank"]  = preds["captain_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Add columns separately — each in its own transaction
    for col, coltype in [
        ("captain_score", "NUMERIC(5,4)"),
        ("captain_rank",  "INTEGER")
    ]:
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE predictions ADD COLUMN {col} {coltype}"
                ))
        except Exception:
            pass  # Column already exists — fine

    # Update scores in a fresh transaction
    with engine.begin() as conn:
        for _, row in preds.iterrows():
            conn.execute(text("""
                UPDATE predictions
                SET captain_score = :cs,
                    captain_rank  = :cr
                WHERE player_id = :pid
            """), {
                "cs" : float(row["captain_score"]),
                "cr" : int(row["captain_rank"]),
                "pid": int(row["player_id"]),
            })

   # Only show columns that exist
    show_cols = [c for c in [
        "web_name", "position", "team_name",
        "adjusted_points", "captain_score", "captain_rank",
        "fixture_fdr", "opponent_name"
    ] if c in preds.columns]
    top_caps = preds.nsmallest(10, "captain_rank")[show_cols].copy()

    print(f"\n  Top 10 captain recommendations:")
    print(f"  {'#':<3} {'Player':<22} {'Pos':<4} {'AdjPts':>7} "
          f"{'CapScore':>9} {'FDR':>4} Opponent")
    print(f"  {'-'*70}")
    for _, r in top_caps.iterrows():
        name = str(r["web_name"]) if isinstance(r["web_name"], str) else str(r["web_name"].iloc[0]) if hasattr(r["web_name"], "iloc") else str(r["web_name"])
        print(
            f"  {int(r['captain_rank']):<3} {name:<22} "
            f"{str(r['position']):<4} "
            f"{float(r.get('adjusted_points', 0) or 0):>7.2f} "
            f"{float(r['captain_score'] or 0):>9.4f} "
            f"{float(r['fixture_fdr'] or 0):>4.1f} "
            f"{str(r.get('opponent_name','') or '')}"
        )

    print(f"\n  ✓ captain_score added to {len(preds):,} predictions")
    return preds

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_captain_model(model, accuracy, features):
    """Saves the captain classifier and metadata."""
    CAPTAIN_MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(CAPTAIN_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model"    : model,
            "features" : features,
            "accuracy" : accuracy,
            "trained_at": datetime.now().isoformat(),
        }, f)
    print(f"  ✓ Captain model saved to {CAPTAIN_MODEL_PATH}")


def load_captain_model():
    """Loads the saved captain classifier."""
    if not CAPTAIN_MODEL_PATH.exists():
        return None, None
    with open(CAPTAIN_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("accuracy")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_captain_model():
    """
    Full captain classifier pipeline:
    1. Load model_features from DB
    2. Engineer captain-specific features
    3. Build captain labels (who was top scorer each GW?)
    4. Train XGBoost classifier
    5. Evaluate on 2025-26 validation set
    6. Score current predictions table
    7. Save model
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Captain Classifier  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()

    print("\n[1/5] Loading training data...")
    df = load_training_data(engine)

    print("\n[2/5] Engineering captain features...")
    df = engineer_captain_features(df)

    print("\n[3/5] Building captain labels...")
    df = build_captain_labels(df)

    print("\n[4/5] Training captain classifier...")
    model, val_df, X_val, y_val = train_captain_classifier(df)

    print("\n[5/5] Evaluating...")
    accuracy, gw_results = evaluate_captain_model(model, val_df, X_val, y_val)

    print("\n[6/6] Scoring current predictions...")
    score_predictions(model, engine)

    save_captain_model(model, accuracy, CAPTAIN_FEATURES)

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  Captain accuracy : {accuracy:.1%}")
    print(f"  Model saved      : {CAPTAIN_MODEL_PATH}")
    print(f"  Duration         : {duration:.0f}s")
    print(f"{'='*60}")

    return model, accuracy


if __name__ == "__main__":
    run_captain_model()
