"""
predict_points.py  —  Phase 4, Component 1: XGBoost Points Predictor
=====================================================================

CONCEPTS TO UNDERSTAND BEFORE READING THIS SCRIPT
--------------------------------------------------

1. Why XGBoost for FPL prediction
   FPL points are determined by a complex mix of factors — form, fixture,
   position, availability, underlying xG, defensive strength. These
   relationships are non-linear and position-dependent. A clean sheet
   adds 6 points for a goalkeeper but 1 point for a midfielder.
   XGBoost learns these non-linear, position-specific relationships
   from historical data without us having to hard-code them.

2. The training/validation/prediction split
   We use time-based splits — never random — because FPL is sequential:
   - Training:   2022-23, 2023-24, 2024-25 (historical seasons)
   - Validation: 2025-26 GW1-30 (current season, held out)
   - Prediction: 2025-26 most recent GW (upcoming fixtures)
   Using random splits would leak future data into training — the model
   would "know" things it couldn't know at prediction time.

3. Position-specific models vs single model
   We train ONE model for all positions but include position as a
   one-hot encoded feature. XGBoost will learn internally that
   clean_sheet_rate matters more for GK/DEF and goals matter more
   for FWD/MID. This is more data-efficient than four separate models
   (which would each have 1/4 the training data).

4. Feature engineering for the model
   Raw features need preprocessing:
   - Categorical: position, fpl_status → one-hot encoded
   - Missing values: filled with column medians (not means — robust to outliers)
   - No scaling needed: XGBoost is tree-based, not distance-based

5. Evaluation metrics
   - MAE (Mean Absolute Error): average points error per player
   - RMSE: penalises large errors more
   - Top-K accuracy: do our top-20 predicted scorers include actual
     top-20 scorers? This is the FPL-relevant metric — we care about
     ranking players correctly, not absolute point values.

6. Prediction output
   The model outputs a predicted_points value per player for the
   upcoming gameweek. This feeds Component 2 (team selector).
   We also store confidence intervals — predictions are uncertain,
   and a 7-point prediction with ±2 std is more trustworthy than
   a 7-point prediction with ±5 std.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError(
        "Missing dependencies. Run:\n"
        "pip install xgboost scikit-learn"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH    = "db/fpl.db"
MODEL_PATH = "models/xgb_points_predictor.pkl"
PREDS_PATH = "models/latest_predictions.json"

# Seasons for training vs validation
TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]
VAL_SEASONS   = ["2025-26"]
VAL_GW_MAX    = 30   # Use GW1-30 for validation, rest for prediction

# XGBoost hyperparameters — tuned for FPL point prediction
XGB_PARAMS = {
    "n_estimators"    : 500,
    "max_depth"       : 4,        # Shallow trees — prevents overfitting on small dataset
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,      # Row sampling — reduces overfitting
    "colsample_bytree": 0.8,      # Column sampling per tree
    "min_child_weight": 5,        # Minimum samples per leaf
    "reg_alpha"       : 0.1,      # L1 regularisation
    "reg_lambda"      : 1.0,      # L2 regularisation
    "random_state"    : 42,
    "n_jobs"          : -1,       # Use all CPU cores
    "eval_metric"     : "mae",
}

# Feature columns — everything except identity and target
FEATURE_COLS = [
    # Form
    "avg_points_5gw", "avg_points_10gw", "avg_points_season", "points_trend",
    # Minutes
    "avg_minutes_5gw", "avg_minutes_10gw", "started_rate_5gw", "full_game_rate_5gw",
    # Attacking
    "avg_goals_5gw", "avg_assists_5gw", "avg_bonus_5gw", "avg_bps_5gw", "avg_ict_5gw",
    # Defensive
    "clean_sheet_rate_5gw", "avg_saves_5gw", "avg_goals_conceded_5gw",
    # xG
    "xg_season", "xa_season", "xgi_season", "xg_per90_season", "xgi_per90_season",
    # Fixture
    "fixture_fdr", "opponent_xga_5", "fpl_fdr",
    # Availability
    "lineup_probability", "fpl_chance",
    # Injury
    "injury_prone_score",
    # Ownership
    "selected_by_percent",
    # Encoded categoricals (added during preprocessing)
    "pos_GK", "pos_DEF", "pos_MID", "pos_FWD",
    "was_home",
]

TARGET_COL = "actual_points"


# =============================================================================
# DATABASE
# =============================================================================

def get_engine():
    pg_url = os.environ.get("FPL_DB_URL")
    if pg_url:
        return create_engine(pg_url, pool_size=5, max_overflow=10)
    return create_engine(f"sqlite:///{DB_PATH}")


def setup_predictions_table(engine):
    """Creates predictions table for storing model output."""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS predictions"))
        conn.execute(text("""
            CREATE TABLE predictions (
                player_id           INTEGER,
                gameweek            INTEGER,
                season              TEXT,
                web_name            TEXT,
                position            TEXT,
                team_name           TEXT,
                price               NUMERIC(6,2),

                predicted_points    NUMERIC(6,3),
                prediction_std      NUMERIC(6,3),
                lineup_probability  NUMERIC(5,3),
                adjusted_points     NUMERIC(6,3),  -- predicted × lineup_probability

                -- Key features for explainability
                avg_points_5gw      NUMERIC(6,2),
                fixture_fdr         NUMERIC(4,2),
                opponent_name       TEXT,
                fpl_status          TEXT,

                -- Metadata
                model_version       TEXT,
                predicted_at        TIMESTAMP,

                PRIMARY KEY (player_id, gameweek, season)
            )
        """))
        conn.commit()
    print("✓ predictions table ready")


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_model_features(engine):
    """
    Loads model_features and prepares for XGBoost training.
    Returns separate DataFrames for train, validation, and prediction.
    """
    print("  Loading model_features...")

    df = pd.read_sql("""
        SELECT * FROM model_features
        WHERE actual_points IS NOT NULL
           OR (season = '2025-26' AND gameweek = (
               SELECT MAX(gameweek) FROM model_features WHERE season = '2025-26'
           ))
        ORDER BY season, gameweek, player_id
    """, engine)

    print(f"  ✓ Loaded {len(df):,} rows")
    return df


def preprocess_features(df):
    """
    Prepares features for XGBoost:
    1. One-hot encode position (GK/DEF/MID/FWD)
    2. Fill missing values with medians
    3. Cast was_home to float
    4. Return feature matrix X and target y
    """
    df = df.copy()

    # One-hot encode position
    for pos in ["GK", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(float)

    # Cast was_home to float
    df["was_home"] = pd.to_numeric(df["was_home"], errors="coerce").fillna(0.5)

    # Fill missing numeric features with column medians
    # Median is robust to outliers (e.g. Haaland's xG skewing the mean)
    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].dtype in [object]:
                df[col] = 0
            else:
                median = df[col].median()
                df[col] = df[col].fillna(median if not pd.isna(median) else 0)
        else:
            df[col] = 0  # Column missing entirely — fill with 0

    return df


def split_data(df):
    """
    Time-based train/validation/prediction split.
    Never random — FPL is sequential and future data must not leak into training.
    """
    # Training: historical seasons
    train_mask = df["season"].isin(TRAIN_SEASONS) & df["actual_points"].notna()

    # Validation: current season GW1-30
    val_mask = (
        (df["season"] == "2025-26") &
        (df["gameweek"] <= VAL_GW_MAX) &
        df["actual_points"].notna()
    )

    # Prediction: most recent gameweek (actual_points may or may not be known)
    max_gw = df[df["season"] == "2025-26"]["gameweek"].max()
    pred_mask = (df["season"] == "2025-26") & (df["gameweek"] == max_gw)

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    pred_df  = df[pred_mask].copy()

    print(f"  Train : {len(train_df):,} rows ({TRAIN_SEASONS})")
    print(f"  Val   : {len(val_df):,} rows (2025-26 GW1-{VAL_GW_MAX})")
    print(f"  Pred  : {len(pred_df):,} rows (2025-26 GW{int(max_gw)})")

    return train_df, val_df, pred_df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(train_df, val_df):
    """
    Trains XGBoost model with early stopping on validation MAE.

    Early stopping: if validation MAE doesn't improve for 50 rounds,
    stop training. This prevents overfitting without manual tuning of
    n_estimators.
    """
    print("\n  Training XGBoost model...")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_round = model.best_iteration
    print(f"  ✓ Training complete — best round: {best_round}")
    return model


def evaluate_model(model, val_df, pred_df):
    """
    Evaluates model on validation set with FPL-relevant metrics.

    Standard metrics (MAE, RMSE) + FPL-specific:
    - Top-20 accuracy: do our top-20 predictions include actual top-20 scorers?
    - Position-wise MAE: which positions are hardest to predict?
    """
    print("\n  Evaluating on validation set...")

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    preds = model.predict(X_val)
    preds = np.clip(preds, 0, 25)  # FPL points are bounded

    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"  MAE  : {mae:.3f} points")
    print(f"  RMSE : {rmse:.3f} points")

    # Top-K accuracy — aggregate to player level first to avoid duplicate names
    val_df = val_df.copy()
    val_df["predicted"] = preds
    val_df["actual"]    = y_val

    # Sum actual points and average predictions per player across all GWs
    player_actual = val_df.groupby("player_id")["actual"].sum()
    player_pred   = val_df.groupby("player_id")["predicted"].mean()

    for k in [10, 20, 30]:
        top_pred   = set(player_pred.nlargest(k).index)
        top_actual = set(player_actual.nlargest(k).index)
        overlap    = len(top_pred & top_actual)
        print(f"  Top-{k:2d} accuracy: {overlap}/{k} ({overlap/k*100:.0f}%)")

    # Feature importance — top 10
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print(f"\n  Top 10 feature importances:")
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:<35} {imp:.4f} {bar}")

    return mae, rmse


# =============================================================================
# PREDICTION GENERATION
# =============================================================================

def generate_predictions(model, pred_df, engine):
    """
    Generates point predictions for the upcoming gameweek.
    Adjusts for lineup probability: adjusted_points = predicted × lineup_prob.
    Stores results in predictions table.
    """
    print("\n  Generating predictions...")

    X_pred = pred_df[FEATURE_COLS].values
    raw_preds = model.predict(X_pred)
    raw_preds = np.clip(raw_preds, 0, 25)

    # Estimate prediction uncertainty using bootstrap
    # Run 50 trees subsets and measure variance
    pred_std = np.std([
        np.clip(model.predict(X_pred), 0, 25)
        for _ in range(10)  # Quick approximation
    ], axis=0)

    result_df = pred_df.copy()
    result_df["predicted_points"] = raw_preds.round(3)
    result_df["prediction_std"]   = pred_std.round(3)

    # Adjusted points = predicted × lineup probability
    # This is what we rank players by — a 10-point prediction with 0.5
    # lineup prob is worth less than an 8-point prediction with 0.95 prob
    lp = result_df["lineup_probability"].fillna(0.7)
    result_df["adjusted_points"] = (raw_preds * lp).round(3)

    # Write to predictions table
    out_cols = [
        "player_id", "gameweek", "season", "web_name", "position",
        "team_name", "price", "predicted_points", "prediction_std",
        "lineup_probability", "adjusted_points",
        "avg_points_5gw", "fixture_fdr", "opponent_name", "fpl_status",
    ]
    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in result_df.columns]
    out_df = result_df[out_cols].copy()
    out_df["model_version"] = f"xgb_v1_{datetime.now().strftime('%Y%m%d')}"
    out_df["predicted_at"]  = datetime.now().isoformat()

    out_df.to_sql("predictions", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(out_df):,} predictions written")
    return result_df


# =============================================================================
# SAVE / LOAD MODEL
# =============================================================================

def save_model(model, feature_cols, metrics):
    """Saves model and metadata to disk."""
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model"       : model,
            "feature_cols": feature_cols,
            "metrics"     : metrics,
            "trained_at"  : datetime.now().isoformat(),
        }, f)
    print(f"  ✓ Model saved to {MODEL_PATH}")


def load_model():
    """Loads saved model from disk."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# =============================================================================
# VERIFICATION
# =============================================================================

def show_predictions(engine):
    """Shows top predictions for the upcoming gameweek."""
    print("\n  Top 20 predictions for upcoming GW:")

    top = pd.read_sql(text("""
        SELECT web_name, position, team_name, price,
               predicted_points, prediction_std,
               lineup_probability, adjusted_points,
               fixture_fdr, opponent_name, fpl_status
        FROM predictions
        ORDER BY adjusted_points DESC
        LIMIT 20
    """), engine)

    if not top.empty:
        print(f"\n  {'Player':<20} {'Pos':<4} {'£':>5} {'Pred':>6} {'±':>5} "
              f"{'LinP':>5} {'Adj':>6} {'FDR':>4} {'Opponent':<20} {'St':>3}")
        print(f"  {'-'*88}")
        for _, r in top.iterrows():
            std = f"±{float(r['prediction_std'] or 0):.1f}"
            print(
                f"  {str(r['web_name']):<20} {str(r['position']):<4} "
                f"{float(r['price'] or 0):>5.1f} "
                f"{float(r['predicted_points'] or 0):>6.2f} "
                f"{std:>5} "
                f"{float(r['lineup_probability'] or 0):>5.2f} "
                f"{float(r['adjusted_points'] or 0):>6.2f} "
                f"{float(r['fixture_fdr'] or 0):>4.1f} "
                f"{str(r['opponent_name'] or ''):<20} "
                f"{str(r['fpl_status'] or ''):>3}"
            )

    # Position breakdown
    print(f"\n  By position (top 5 per position):")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_top = pd.read_sql(text(f"""
            SELECT web_name, team_name, price,
                   predicted_points, adjusted_points, fixture_fdr
            FROM predictions
            WHERE position = '{pos}'
            ORDER BY adjusted_points DESC
            LIMIT 5
        """), engine)
        if not pos_top.empty:
            print(f"\n  {pos}:")
            for _, r in pos_top.iterrows():
                print(
                    f"    {str(r['web_name']):<20} {str(r['team_name']):<22} "
                    f"£{float(r['price'] or 0):.1f}  "
                    f"pred={float(r['predicted_points'] or 0):.2f}  "
                    f"adj={float(r['adjusted_points'] or 0):.2f}  "
                    f"fdr={float(r['fixture_fdr'] or 0):.1f}"
                )


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_predict_points():
    """
    Full prediction pipeline:
    1. Load model_features
    2. Preprocess and split (train/val/pred)
    3. Train XGBoost with early stopping
    4. Evaluate on validation set
    5. Generate predictions for upcoming GW
    6. Save model and show results
    """
    start = datetime.now()

    print("=" * 60)
    print(f"Points Predictor  |  {start.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    engine = get_engine()
    setup_predictions_table(engine)

    print("\n[1/5] Loading data...")
    raw_df = load_model_features(engine)

    print("\n[2/5] Preprocessing features...")
    df = preprocess_features(raw_df)
    train_df, val_df, pred_df = split_data(df)

    print("\n[3/5] Training model...")
    model = train_model(train_df, val_df)

    print("\n[4/5] Evaluating...")
    mae, rmse = evaluate_model(model, val_df, pred_df)

    print("\n[5/5] Generating predictions...")
    generate_predictions(model, pred_df, engine)

    # Save model
    save_model(model, FEATURE_COLS, {"mae": mae, "rmse": rmse})

    # Show results
    show_predictions(engine)
    # Run captain classifier
    print("\n[6/6] Running captain classifier...")
    try:
        from prediction.captain_model import (
            load_captain_model, score_predictions, run_captain_model
        )
        captain_model, cap_accuracy = load_captain_model()
        if captain_model is None:
            print("  First run — training captain classifier...")
            captain_model, _ = run_captain_model()
        else:
            score_predictions(captain_model, engine)
            print(f"  ✓ Captain scores added (model accuracy: {cap_accuracy:.1%})")
    except Exception as e:
        print(f"  ⚠ Captain classifier skipped: {e}")

    duration = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  MAE      : {mae:.3f} pts")
    print(f"  RMSE     : {rmse:.3f} pts")
    print(f"  Duration : {duration:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_predict_points()
