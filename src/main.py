"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow
(Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python src/main.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import wandb
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model, make_plots, save_metrics, save_plots
from src.infer import run_inference
from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# W&B helpers — treat W&B as an optional dependency
# ---------------------------------------------------------------------------

def _wandb_is_enabled(cfg: Dict[str, Any]) -> bool:
    wandb_cfg = cfg.get('wandb')
    if not isinstance(wandb_cfg, dict):
        return False
    return bool(wandb_cfg.get('enabled', False))


def _wandb_get_str(cfg: Dict[str, Any], key: str, default: str = '') -> str:
    wandb_cfg = cfg.get('wandb')
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(
    cfg: Dict[str, Any], key: str, default: bool = False
) -> bool:
    wandb_cfg = cfg.get('wandb')
    if not isinstance(wandb_cfg, dict):
        return default
    return bool(wandb_cfg.get(key, default))


def _wandb_get_list(cfg: Dict[str, Any], key: str) -> List[str]:
    wandb_cfg = cfg.get('wandb')
    if not isinstance(wandb_cfg, dict):
        return []
    value = wandb_cfg.get(key, [])
    return list(value) if isinstance(value, list) else []


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text())


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / '.env', override=False)

    cfg = load_config("config.yaml")
    logger.info("[main] Pipeline started")

    # ----------------------------
    # Initialize W&B (optional)
    # ----------------------------
    wandb_run = None
    if _wandb_is_enabled(cfg):
        wandb_project = _wandb_get_str(cfg, 'project')
        if not wandb_project:
            raise ValueError(
                'config.yaml: wandb.project must be non-empty '
                'when wandb.enabled is true'
            )
        wandb_run = wandb.init(
            entity=_wandb_get_str(cfg, 'entity') or None,
            project=wandb_project,
            name=_wandb_get_str(cfg, 'name') or None,
            job_type=_wandb_get_str(cfg, 'job_type', default='training'),
            group=_wandb_get_str(cfg, 'group') or None,
            notes=_wandb_get_str(cfg, 'notes') or None,
            tags=_wandb_get_list(cfg, 'tags') or None,
            config=cfg,
        )
        wandb_run.summary['entrypoint'] = 'python -m src.main'
        logger.info('W&B run initialized | name=%s | project=%s',
                    wandb_run.name, wandb_project)
    else:
        logger.info('W&B disabled — continuing without experiment tracking')

    try:
        # ----------------------------
        # 1. Load raw data
        # ----------------------------
        df_raw = load_raw_data(config_path="config.yaml")
        if wandb_run is not None:
            wandb.log({
                'data/raw_rows': int(df_raw.shape[0]),
                'data/raw_cols': int(df_raw.shape[1]),
            })

        # ----------------------------
        # 2. Clean
        # ----------------------------
        target = cfg["target_column"]
        df_clean = clean_dataframe(df_raw, target_column=target)
        if wandb_run is not None:
            wandb.log({
                'data/clean_rows': int(df_clean.shape[0]),
                'data/clean_cols': int(df_clean.shape[1]),
            })

        # ----------------------------
        # 3. Validate
        # ----------------------------
        feat_cfg = cfg["features"]
        val_cfg = cfg["validation"]
        required_columns = list(dict.fromkeys(
            [target]
            + feat_cfg["quantile_bin"]
            + feat_cfg["numeric_passthrough"]
            + feat_cfg["binary_sum_cols"]
        ))

        validate_dataframe(
            df=df_clean,
            required_columns=required_columns,
            check_missing_values=val_cfg["check_missing_values"],
            target_column=target,
            target_allowed_values=[0, 1],
            numeric_non_negative_cols=val_cfg["numeric_non_negative_cols"],
        )

        # ----------------------------
        # 4. Feature preprocessor
        # ----------------------------
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=feat_cfg["quantile_bin"],
            categorical_onehot_cols=feat_cfg["categorical_onehot"],
            numeric_passthrough_cols=feat_cfg["numeric_passthrough"],
            binary_sum_cols=feat_cfg["binary_sum_cols"],
            n_bins=feat_cfg["n_bins"],
        )

        # ----------------------------
        # 5. Three-way split (train / val / test)
        # ----------------------------
        split_cfg = cfg["split"]
        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
            stratify=y,
        )
        rel_val = split_cfg["val_size"] / (1.0 - split_cfg["test_size"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=rel_val,
            random_state=split_cfg["random_state"],
            stratify=y_temp,
        )
        logger.info(
            "[main] Data split | train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # ----------------------------
        # 6. Train
        # ----------------------------
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            problem_type=cfg["problem_type"],
        )

        # ----------------------------
        # 7. Evaluate on validation split
        # ----------------------------
        metrics = evaluate_model(
            model=model,
            X_eval=X_val,
            y_eval=y_val,
            problem_type=cfg["problem_type"],
        )
        if wandb_run is not None:
            wandb.log({
                f'metrics/val/{k}': float(v)
                for k, v in metrics.items()
            })

        plots = make_plots(model, X_test, y_test)

        # ----------------------------
        # 8. Save artifacts
        # ----------------------------
        paths_cfg = cfg["paths"]
        data_cfg = cfg["data"]

        processed_path = Path(data_cfg["processed_path"])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(processed_path, index=False)

        model_artifact_path = Path(paths_cfg["model_path"])
        model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_artifact_path)

        save_metrics(metrics, paths_cfg["metrics_path"])
        save_plots(plots, paths_cfg["plots_path"])

        # ----------------------------
        # 9. Log model artifact to W&B + promote to 'prod'
        # ----------------------------
        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(
                cfg, 'model_artifact_name', default='model'
            )
            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type='model',
                description=(
                    'Scikit-learn pipeline (preprocessing + estimator)'
                ),
            )
            model_artifact.add_file(str(model_artifact_path))
            logged = wandb.log_artifact(model_artifact)
            logged.wait()
            logger.info('Model artifact logged to W&B')

            if _wandb_get_bool(cfg, 'log_processed_data', default=False):
                data_artifact = wandb.Artifact(
                    name=f'{model_artifact_name}-processed-data',
                    type='dataset',
                    description='Processed training dataset',
                )
                data_artifact.add_file(str(processed_path))
                wandb.log_artifact(data_artifact)

        # ----------------------------
        # 10. Inference + optional predictions artifact
        # ----------------------------
        df_preds = run_inference(
            model=model, X_infer=X_test, include_proba=True
        )
        pred_path = Path(paths_cfg["predictions_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_csv(pred_path, index=True)

        if wandb_run is not None and _wandb_get_bool(
            cfg, 'log_predictions', default=False
        ):
            model_artifact_name = _wandb_get_str(
                cfg, 'model_artifact_name', default='model'
            )
            pred_artifact = wandb.Artifact(
                name=f'{model_artifact_name}-predictions',
                type='predictions',
                description='Inference outputs written by the pipeline',
            )
            pred_artifact.add_file(str(pred_path))
            wandb.log_artifact(pred_artifact)

        logger.info("[main] Pipeline complete")
        logger.info("[main] Model saved:       %s", paths_cfg["model_path"])
        logger.info("[main] Metrics saved:     %s", paths_cfg["metrics_path"])
        logger.info("[main] Plots saved:       %s", paths_cfg["plots_path"])
        logger.info(
            "[main] Predictions saved: %s", paths_cfg["predictions_path"]
        )
        logger.info("[main] Metrics: %s", metrics)

    except Exception:
        logger.exception('Pipeline failed')
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise
    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
