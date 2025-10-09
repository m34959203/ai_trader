"""CLI for meta-label retraining with Purged Walk-Forward CV and experiment tracking."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("ai_trader.meta_label_cli")


@dataclass
class FoldMetrics:
    fold: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: Optional[float]


class ExperimentLogger:
    """Unified interface for MLflow and Weights & Biases tracking."""

    def __init__(
        self,
        run_name: str,
        mlflow_uri: Optional[str] = None,
        mlflow_experiment: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ) -> None:
        self._mlflow = self._init_mlflow(mlflow_uri, mlflow_experiment, run_name)
        self._wandb_run = self._init_wandb(wandb_project, wandb_entity, run_name)

    def log_params(self, params: Dict[str, object]) -> None:
        if self._mlflow is not None:
            try:
                import mlflow

                mlflow.log_params(params)
            except Exception:  # pragma: no cover - third-party guard
                LOG.exception("Failed to log params to MLflow")
        if self._wandb_run is not None:
            try:
                self._wandb_run.config.update(params, allow_val_change=True)
            except Exception:  # pragma: no cover
                LOG.exception("Failed to log params to W&B")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._mlflow is not None:
            try:
                import mlflow

                mlflow.log_metrics(metrics, step=step)
            except Exception:  # pragma: no cover
                LOG.exception("Failed to log metrics to MLflow")
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:  # pragma: no cover
                LOG.exception("Failed to log metrics to W&B")

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self._mlflow is not None:
            try:
                import mlflow

                mlflow.set_tags(tags)
            except Exception:  # pragma: no cover
                LOG.exception("Failed to set MLflow tags")
        if self._wandb_run is not None:
            try:
                self._wandb_run.config.update(tags, allow_val_change=True)
            except Exception:  # pragma: no cover
                LOG.exception("Failed to set W&B tags")

    def finish(self) -> None:
        if self._mlflow is not None:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:  # pragma: no cover
                LOG.exception("Failed to end MLflow run")
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:  # pragma: no cover
                LOG.exception("Failed to end W&B run")

    @staticmethod
    def _init_mlflow(
        tracking_uri: Optional[str], experiment: Optional[str], run_name: str
    ) -> Optional[object]:  # pragma: no cover - optional dependency setup
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment:
                mlflow.set_experiment(experiment_name=experiment)
            mlflow.start_run(run_name=run_name)
            return mlflow
        except Exception as exc:
            LOG.warning("MLflow logging disabled (%s)", exc)
            return None

    @staticmethod
    def _init_wandb(
        project: Optional[str], entity: Optional[str], run_name: str
    ) -> Optional[object]:  # pragma: no cover - optional dependency setup
        if not project:
            return None
        try:
            import wandb

            run = wandb.init(project=project, entity=entity, name=run_name, reinit=True)
            return run
        except Exception as exc:
            LOG.warning("Weights & Biases logging disabled (%s)", exc)
            return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-label training pipeline")
    parser.add_argument("dataset", type=Path, help="Path to the labelled events dataset (CSV/Parquet)")
    parser.add_argument("--model", default="rf", choices=["rf", "xgb"], help="Base learner type")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of Purged CV folds")
    parser.add_argument("--purge-gap", type=int, default=5, help="Number of samples to purge before each test fold")
    parser.add_argument("--label-column", default="meta_label", help="Name of the column with binary meta-labels")
    parser.add_argument(
        "--time-column",
        default="event_end",
        help="Temporal ordering column used for walk-forward split (defaults to 'event_end')",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        help="Explicit list of feature columns (default: auto-detect numeric columns)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=Path("artifacts/meta_model.pkl"))
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path (defaults to <output>.json)")
    parser.add_argument("--mlflow-uri", help="MLflow tracking URI (optional)")
    parser.add_argument(
        "--mlflow-experiment",
        default="meta-labeling",
        help="MLflow experiment name (default: meta-labeling)",
    )
    parser.add_argument("--wandb-project", help="Weights & Biases project name (optional)")
    parser.add_argument("--wandb-entity", help="Weights & Biases entity/team (optional)")
    parser.add_argument("--run-name", help="Custom run name for experiment tracking")
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{path}' does not exist")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def purged_walk_forward_indices(
    timestamps: Sequence[pd.Timestamp | float | int], n_splits: int, purge_gap: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")

    n_samples = len(timestamps)
    if n_samples < n_splits:
        raise ValueError("Not enough samples for the requested number of folds")

    fold_boundaries = np.linspace(0, n_samples, num=n_splits + 1, dtype=int)
    for fold_idx in range(n_splits):
        test_start = fold_boundaries[fold_idx]
        test_end = fold_boundaries[fold_idx + 1]
        train_end = max(0, test_start - purge_gap)
        train_indices = np.arange(0, train_end, dtype=int)
        test_indices = np.arange(test_start, test_end, dtype=int)
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        yield train_indices, test_indices


def build_model(kind: str, seed: int):
    if kind == "rf":
        try:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced",
            )
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("scikit-learn is required for RandomForestClassifier") from exc

    if kind == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore

            return XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=seed,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method=os.environ.get("XGB_TREE_METHOD", "hist"),
                n_jobs=int(os.environ.get("XGB_N_JOBS", "-1")),
            )
        except Exception:
            LOG.warning("xgboost not available, falling back to GradientBoostingClassifier")
            try:
                from sklearn.ensemble import GradientBoostingClassifier

                return GradientBoostingClassifier(random_state=seed)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("GradientBoostingClassifier unavailable") from exc

    raise ValueError(f"Unsupported model kind: {kind}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def select_features(df: pd.DataFrame, label_col: str, feature_cols: Optional[Sequence[str]]) -> List[str]:
    if feature_cols:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing in dataset: {missing}")
        return list(feature_cols)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        raise ValueError("No numeric columns detected for features; specify --feature-cols")
    return numeric_cols


def extract_probabilities(model, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if isinstance(proba, (list, tuple)):
                proba = np.asarray(proba)
            if proba.ndim == 2:
                return proba[:, 1]
        except Exception:  # pragma: no cover - guard
            LOG.exception("Failed to compute predict_proba; skipping AUC")
            return None
    if hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X)
            return (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
        except Exception:  # pragma: no cover
            LOG.exception("Failed to compute decision_function; skipping AUC")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    try:
        df = load_dataset(args.dataset)
    except Exception as exc:
        LOG.error("Unable to load dataset: %s", exc)
        return 1

    if args.label_column not in df.columns:
        LOG.error("Label column '%s' not found in dataset", args.label_column)
        return 1

    if args.time_column in df.columns:
        df = df.sort_values(by=args.time_column)
    else:
        LOG.warning("Time column '%s' not present; using dataset order", args.time_column)
        df = df.copy()

    df = df.reset_index(drop=True)
    feature_cols = select_features(df, args.label_column, args.feature_cols)
    timestamps = (
        pd.to_datetime(df[args.time_column])
        if args.time_column in df.columns
        else pd.Series(np.arange(len(df)), name="index")
    )

    X = df[feature_cols].fillna(method="ffill").fillna(0.0).to_numpy()
    y = df[args.label_column].astype(int).to_numpy()

    try:
        splits = list(purged_walk_forward_indices(timestamps.tolist(), args.cv_folds, args.purge_gap))
    except Exception as exc:
        LOG.error("Failed to build Purged CV splits: %s", exc)
        return 1

    if not splits:
        LOG.error("No valid splits generated; check --cv-folds and --purge-gap settings")
        return 1

    run_name = args.run_name or f"meta-label-{args.model}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    tracker = ExperimentLogger(
        run_name=run_name,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    tracker.log_params(
        {
            "model": args.model,
            "cv_folds": args.cv_folds,
            "purge_gap": args.purge_gap,
            "label_column": args.label_column,
            "time_column": args.time_column,
            "feature_cols": feature_cols,
            "dataset": str(args.dataset),
        }
    )
    tracker.set_tags({"cli": "meta_label_cli"})

    fold_metrics: List[FoldMetrics] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        model = build_model(args.model, args.seed)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_proba = extract_probabilities(model, X[test_idx])
        metrics = compute_metrics(y[test_idx], y_pred, y_proba)
        fold_metrics.append(
            FoldMetrics(
                fold=fold_idx,
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                roc_auc=None if np.isnan(metrics["roc_auc"]) else metrics["roc_auc"],
            )
        )
        tracker.log_metrics({f"fold{fold_idx}_{k}": v for k, v in metrics.items()}, step=fold_idx)
        LOG.info(
            "Fold %s metrics: accuracy=%.4f f1=%.4f precision=%.4f recall=%.4f roc_auc=%s",
            fold_idx,
            metrics["accuracy"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics["roc_auc"]) else "nan",
        )

    aggregate = {
        "accuracy": float(np.mean([m.accuracy for m in fold_metrics])),
        "f1": float(np.mean([m.f1 for m in fold_metrics])),
        "precision": float(np.mean([m.precision for m in fold_metrics])),
        "recall": float(np.mean([m.recall for m in fold_metrics])),
        "roc_auc": float(np.nanmean([m.roc_auc if m.roc_auc is not None else np.nan for m in fold_metrics])),
    }
    tracker.log_metrics({f"cv_{k}": v for k, v in aggregate.items()}, step=len(fold_metrics) + 1)
    LOG.info(
        "Aggregate CV metrics: accuracy=%.4f f1=%.4f precision=%.4f recall=%.4f roc_auc=%s",
        aggregate["accuracy"],
        aggregate["f1"],
        aggregate["precision"],
        aggregate["recall"],
        f"{aggregate['roc_auc']:.4f}" if not np.isnan(aggregate["roc_auc"]) else "nan",
    )

    # Train final model on full dataset
    final_model = build_model(args.model, args.seed)
    final_model.fit(X, y)

    try:
        import joblib

        args.output.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": final_model,
            "feature_columns": feature_cols,
            "label_column": args.label_column,
            "time_column": args.time_column,
            "created_at": datetime.utcnow().isoformat(),
            "metrics": aggregate,
        }
        joblib.dump(artifact, args.output)
        LOG.info("Saved trained model to %s", args.output)
    except Exception as exc:
        LOG.error("Failed to persist model artifact: %s", exc)
        tracker.finish()
        return 1

    summary_path = args.summary or args.output.with_suffix(".json")
    summary = {
        "dataset": str(args.dataset),
        "model": args.model,
        "feature_columns": feature_cols,
        "cv_folds": args.cv_folds,
        "purge_gap": args.purge_gap,
        "metrics": aggregate,
        "fold_metrics": [asdict(metric) for metric in fold_metrics],
        "artifact_path": str(args.output),
        "generated_at": datetime.utcnow().isoformat(),
    }
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("Saved training summary to %s", summary_path)
    except Exception as exc:
        LOG.warning("Unable to write summary file: %s", exc)

    tracker.finish()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
