"""
Phase 6 — ML Model Trainer with Optuna hyperparameter tuning + MLflow tracking.

Flow per training run:
  1. Load samples from db/ml_samples.jsonl
  2. Optuna Bayesian search over LightGBM (or sklearn) hyperparams
     - N_TRIALS trials, each evaluated via 5-fold stratified CV
     - MedianPruner stops bad trials early
  3. Retrain final model on all data with best params
  4. Log everything to MLflow (sqlite:///db/mlflow.db)
  5. Save model to db/ml_model.pkl  (live predictor uses this)

Run manually: python -m ml.trainer
Auto-triggers: when sample_count() >= MIN_TRAINING_SAMPLES
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from loguru import logger

from ml.features import load_samples, sample_count, FEATURE_NAMES, MIN_TRAINING_SAMPLES

_MODEL_PATH  = Path("./db/ml_model.pkl")
_META_PATH   = Path("./db/ml_meta.json")
_MLFLOW_URI  = "sqlite:///db/mlflow.db"
_EXPERIMENT  = "ARIA_ML"
N_TRIALS     = 30          # Optuna search budget (scales with sample count)
CV_FOLDS     = 5


def _get_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        return None


def _lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


# ── Optuna objectives ─────────────────────────────────────────────────────────

def _lgbm_objective(trial, X: list, y: list) -> float:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": 42,
        "verbose": -1,
    }

    X_arr = np.array(X)
    y_arr = np.array(y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_arr[train_idx], y_arr[train_idx])
        preds = clf.predict(X_arr[val_idx])
        scores.append(accuracy_score(y_arr[val_idx], preds))
        trial.report(np.mean(scores), step=fold)
        if trial.should_prune():
            raise __import__("optuna").exceptions.TrialPruned()

    return float(np.mean(scores))


def _sklearn_objective(trial, X: list, y: list) -> float:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
        "max_depth":     trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
    }

    X_arr = np.array(X)
    y_arr = np.array(y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_arr[train_idx], y_arr[train_idx])
        preds = clf.predict(X_arr[val_idx])
        scores.append(accuracy_score(y_arr[val_idx], preds))
        trial.report(np.mean(scores), step=fold)
        if trial.should_prune():
            raise __import__("optuna").exceptions.TrialPruned()

    return float(np.mean(scores))


# ── Main entry points ─────────────────────────────────────────────────────────

def train() -> bool:
    """
    Tune hyperparams with Optuna, retrain on full data, log to MLflow.
    Returns True if training succeeded.
    """
    n = sample_count()
    if n < MIN_TRAINING_SAMPLES:
        logger.info(f"[ML] Not enough samples to train ({n} < {MIN_TRAINING_SAMPLES})")
        return False

    X, y = load_samples()
    if not X:
        return False

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    logger.info(f"[ML] Training on {len(X)} samples ({n_pos} wins, {n_neg} losses)")

    if n_pos < 15 or n_neg < 15:
        logger.warning(f"[ML] Class imbalance too severe ({n_pos}W/{n_neg}L) — skipping train")
        return False

    use_lgbm = _lgbm_available()
    backend  = "lgbm" if use_lgbm else "sklearn"
    n_trials = min(N_TRIALS, max(10, len(X) // 3))

    # ── Optuna study ──────────────────────────────────────────────────────────
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study   = optuna.create_study(direction="maximize", pruner=pruner)
    objective = _lgbm_objective if use_lgbm else _sklearn_objective

    logger.info(f"[ML] Optuna: {n_trials} trials, backend={backend}")
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_cv_acc = study.best_value
    n_pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    logger.info(f"[ML] Best CV accuracy: {best_cv_acc:.1%} params={best_params} ({n_pruned} pruned)")

    # ── Retrain final model on all data ──────────────────────────────────────
    if use_lgbm:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(**best_params, random_state=42)

    X_arr = np.array(X)
    y_arr = np.array(y)
    clf.fit(X_arr, y_arr)

    preds    = clf.predict(X_arr)
    train_acc = float(sum(p == t for p, t in zip(preds, y_arr)) / len(y_arr))
    logger.info(f"[ML] Final train accuracy: {train_acc:.1%}")

    importances: dict[str, float] = {}
    try:
        imp = clf.feature_importances_
        importances = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in imp]))
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"[ML] Top features: {top5}")
    except Exception:
        pass

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow = _get_mlflow()
    if mlflow:
        try:
            mlflow.set_tracking_uri(_MLFLOW_URI)
            mlflow.set_experiment(_EXPERIMENT)
            with mlflow.start_run():
                mlflow.log_param("backend", backend)
                mlflow.log_param("n_trials", n_trials)
                mlflow.log_param("cv_folds", CV_FOLDS)
                mlflow.log_param("min_training_samples", MIN_TRAINING_SAMPLES)
                for k, v in best_params.items():
                    mlflow.log_param(k, v)

                mlflow.log_metric("cv_accuracy", round(best_cv_acc, 4))
                mlflow.log_metric("train_accuracy", round(train_acc, 4))
                mlflow.log_metric("n_samples", len(X))
                mlflow.log_metric("n_wins", n_pos)
                mlflow.log_metric("n_losses", n_neg)
                mlflow.log_metric("win_rate", round(n_pos / len(y), 4))
                mlflow.log_metric("n_trials_pruned", n_pruned)

                for feat, imp_val in importances.items():
                    mlflow.log_metric(f"imp_{feat}", imp_val)

                try:
                    mlflow.sklearn.log_model(clf, artifact_path="model")
                except Exception:
                    pass

                # Live performance: does the boost actually help?
                try:
                    from ml.performance import tracker as ml_perf
                    ml_perf.log_to_mlflow()
                except Exception:
                    pass

            logger.info(f"[ML] MLflow run logged → {_MLFLOW_URI}")
        except Exception as e:
            logger.warning(f"[ML] MLflow logging failed (non-fatal): {e}")

    # ── Save model + metadata ─────────────────────────────────────────────────
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MODEL_PATH.open("wb") as f:
        pickle.dump({"backend": backend, "model": clf}, f)

    meta = {
        "backend": backend,
        "n_samples": len(X),
        "n_wins": n_pos,
        "n_losses": n_neg,
        "cv_accuracy": round(best_cv_acc, 4),
        "train_accuracy": round(train_acc, 4),
        "best_params": best_params,
        "feature_importances": importances,
    }
    _META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info(f"[ML] Model saved to {_MODEL_PATH}")

    # Telegram: ML model retrained notification
    try:
        from notifications.telegram import alert_ml_retrained
        top_feat = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_name = top_feat[0][0] if top_feat else "—"
        from ml.performance import tracker as ml_perf
        verdict  = ml_perf.get_stats().get("verdict", "—")
        alert_ml_retrained(
            pair="ALL",
            accuracy=best_cv_acc * 100,
            top_feature=top_name,
            n_samples=len(X),
            verdict=verdict,
        )
    except Exception:
        pass

    return True


def maybe_train() -> None:
    """Train only if enough new samples since last train."""
    if sample_count() >= MIN_TRAINING_SAMPLES:
        train()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    result = train()
    print("Training", "succeeded" if result else "failed (not enough samples yet)")
