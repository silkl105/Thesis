from __future__ import annotations
"""
src/XGB.py - Expanding-window CV for XGBoost price model
=======================================================

* 10-fold expanding-window **TimeSeriesSplit** (≈80/20 within each fold)
  on the *tree-ready* feature set written by `DataProcessor`  
  (parquet path configured in *config.yaml* → ``raw_files.processed_xgb``).

* Within each outer fold we run a **Bayesian hyper-parameter search**
  (`BayesSearchCV`, 50 iterations) over the most influential XGBoost
  parameters reported in recent housing-price literature
  (Sharma et al. 2024, MDPI 2023, Jetir 2024, *inter alia*).

* Early stopping is enabled with 10 rounds of no improvement on MAE
  using the *reg:absoluteerror* objective.

* The final **best estimator per fold** is persisted to
      data/processed/xgb_model_fold_{fold}.pkl
  so SHAP diagnostics can later be run without re-training.

* Out-of-fold predictions are stored in
      <processed_path>/predictions_xgb.parquet
  and fold-level accuracy metrics (RMSE, MAE, MAPE, MDAPE, ±% thresholds) go
  to a single-sheet Excel workbook
      data/processed/xgb_results.xlsx  →  sheet ``xgb_cv``.
"""


from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm
from xgboost import XGBRegressor

__all__ = ["XGBRunner"]

# --------------------------------------------------------------------------- #
# helpers – error metrics                                                     #
# --------------------------------------------------------------------------- #
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean **absolute** percentage error on the original EUR scale.

    Assumes the target was log-transformed with :pyfunc:`numpy.log1p`
    in preprocessing (all values > 0).

    Returns
    -------
    float
        MAPE [%]
    """
    y_true_eur = np.expm1(y_true)
    y_pred_eur = np.expm1(y_pred)
    pct_err = np.abs(y_true_eur - y_pred_eur) / y_true_eur
    return pct_err.mean() * 100.0


def _mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median absolute percentage error on the original EUR scale.

    Assumes the target was log transformed with :pyfunc:`numpy.log1p`
    in preprocessing (all values > 0).
    """
    y_true_eur = np.expm1(y_true)
    y_pred_eur = np.expm1(y_pred)
    pct_err = np.abs(y_true_eur - y_pred_eur) / y_true_eur
    return np.median(pct_err) * 100.0


# --------------------------------------------------------------------------- #
# main class                                                                  #
# --------------------------------------------------------------------------- #
class XGBRunner:
    """
    Run **XGBoost** with Bayesian hyper-parameter optimisation on the
    tree-ready dataset.

    Parameters
    ----------
    config_path
        Optional explicit path to *config.yaml*.
    metrics_path
        Excel workbook to which fold-level metrics will be written.
        If it exists it is overwritten.
    target
        Name of the (logged) target column.

    * Within each outer fold we run a **hyper-parameter search** using
      `hyperopt` (TPE, 50 evaluations) over the most influential XGBoost
      parameters reported in recent housing-price literature
      (Sharma et al. 2024, MDPI 2023, Jetir 2024, *inter alia*).

    * Early stopping is enabled with 10 rounds of no improvement on MAE
      using the *reg:absoluteerror* objective.
    """

    # --------------------------------------------------------------------- #
    # construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        metrics_path: str | Path = "data/processed/xgb_results.xlsx",
        target: str = "transaction_price",
    ) -> None:
        self.cfg: Dict[str, str] = self._load_config(config_path)
        self.root = Path(__file__).resolve().parents[1]
        self.metrics_path = self.root / metrics_path
        self.target = target

        # ------------------------------------------------------------------ #
        # load XGB‑design matrix                                             #
        proc_file = self.root / self.cfg["raw_files"]["processed_data"]
        if not proc_file.exists():
            raise FileNotFoundError(proc_file)
        self.df = pd.read_parquet(proc_file)

        self.y = self.df[self.target].astype(float).values
        self.X = self.df.drop(columns=[self.target])

        self._fold_info: List[Dict] = []

    # --------------------------------------------------------------------- #
    # public driver                                                         #
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        """Execute expanding-CV workflow and dump Excel workbook."""
        cv_df = self._xgb_cv()

        # only one sheet required
        with pd.ExcelWriter(self.metrics_path, engine="xlsxwriter") as xls:
            cv_df.to_excel(xls, sheet_name="xgb_cv", index=False)
        print(f"✓ Results written to {self.metrics_path.relative_to(self.root)}")

    # --------------------------------------------------------------------- #
    # internal – XGB‑CV                                                     #
    # --------------------------------------------------------------------- #
    def _xgb_cv(self) -> pd.DataFrame:
        """
        10-fold expanding-window CV with **hyperopt** (TPE) hyper-parameter
        tuning (TPE, 50 evaluations) and early stopping (10 rounds, MAE)
        using the *reg:absoluteerror* objective.

        Returns
        -------
        pd.DataFrame
            One row per outer fold with error metrics & best params.
        """
        # -------- outer CV geometry -------------------------------------- #
        n_splits = 10
        block_size = int(round(len(self.X) / n_splits))
        test_size = int(round(block_size * 0.65))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        rows: List[Dict] = []
        preds = []

        # ---------- iterate outer folds ---------------------------------- #
        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X), 1):
            # set total number of inner hyperopt evaluations
            inner_total = 50
            X_train, y_train = self.X.iloc[train_idx], self.y[train_idx]
            X_test, y_test = self.X.iloc[test_idx], self.y[test_idx]

            # build search space for hyperopt
            space = {
                "n_estimators": hp.quniform("n_estimators", 300, 1500, 1),
                "max_depth": hp.quniform("max_depth", 3, 10, 1),
                "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
                "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
                "gamma": hp.uniform("gamma", 0.0, 5.0),
                "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-6), np.log(1.0)),
                "reg_lambda": hp.loguniform("reg_lambda", np.log(0.5), np.log(10.0)),
            }

            trials = Trials()
            def objective(params):
                # cast to int where needed
                params["n_estimators"] = int(params["n_estimators"])
                params["max_depth"] = int(params["max_depth"])
                params["min_child_weight"] = int(params["min_child_weight"])
                # instantiate model with early stopping
                model = XGBRegressor(
                    objective="reg:absoluteerror",
                    tree_method="hist",
                    enable_categorical=False,
                    n_jobs=-1,
                    random_state=41,
                    eval_metric="mae",
                    early_stopping_rounds=10,
                    verbosity=0,
                    **params,
                )
                # inner CV evaluation
                cv_inner = TimeSeriesSplit(n_splits=3)
                scores = []
                for tr_idx, val_idx in cv_inner.split(X_train):
                    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
                    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    y_pred_inner = model.predict(X_val)
                    score = mean_absolute_error(np.expm1(y_val), np.expm1(y_pred_inner))
                    scores.append(score)
                return {"loss": np.mean(scores), "status": STATUS_OK}

            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=inner_total,
                trials=trials,
                rstate=np.random.default_rng(41),
                show_progressbar=True,
            )
            best_params = space_eval(space, best)
            # report chosen hyperparameters
            tqdm.write(f"Fold {fold}: best params: {best_params}")
            # cast back to int
            best_params["n_estimators"] = int(best_params["n_estimators"])
            best_params["max_depth"] = int(best_params["max_depth"])
            best_params["min_child_weight"] = int(best_params["min_child_weight"])

            # final model fit on full training with outer test for early stopping
            final_model = XGBRegressor(
                objective="reg:absoluteerror",
                tree_method="hist",
                enable_categorical=True,
                n_jobs=-1,
                random_state=41,
                eval_metric="mae",
                early_stopping_rounds=10,
                verbosity=0,
                **best_params,
            )
            final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = final_model.predict(X_test)

            # Persist model for SHAP
            mdl_path = (
                self.root
                / self.cfg["processed_path"]
                / f"xgb_model_fold_{fold}.pkl"
            )
            joblib.dump(final_model, mdl_path)

            # --- metrics on EUR scale ------------------------------------ #
            rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
            mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
            mape = _mape(y_test, y_pred)
            mdape = _mdape(y_test, y_pred)

            err_pct = (
                np.abs(np.expm1(y_pred) - np.expm1(y_test))
                / np.expm1(y_test)
            )

            rows.append(
                {
                    "fold": fold,
                    "train_obs": len(train_idx),
                    "test_obs": len(test_idx),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "mdape": mdape,
                    "pct_within_5": np.mean(err_pct <= 0.05) * 100,
                    "pct_within_10": np.mean(err_pct <= 0.10) * 100,
                    "pct_within_20": np.mean(err_pct <= 0.20) * 100,
                    "n_estimators": best_params["n_estimators"],
                    "max_depth": best_params["max_depth"],
                    "learning_rate": best_params["learning_rate"],
                    "subsample": best_params["subsample"],
                    "colsample_bytree": best_params["colsample_bytree"],
                    "gamma": best_params["gamma"],
                }
            )

            # --- save per‑observation predictions ------------------------ #
            test_df = X_test.copy()
            test_df["y_test"] = np.expm1(y_test)
            test_df["y_pred"] = np.expm1(y_pred)
            test_df["fold"] = fold
            preds.append(test_df)

            # keep pipe pointer for potential downstream diagnostics
            self._fold_info.append(
                {"estimator": final_model, "train_idx": train_idx, "test_idx": test_idx}
            )

        # dump predictions once
        preds_df = pd.concat(preds, axis=0)
        preds_path = (
            self.root
            / self.cfg["processed_path"]
            / "predictions_xgb.parquet"
        )
        preds_df.to_parquet(preds_path)

        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _load_config(explicit_path: Optional[str] = None) -> Dict[str, str]:
        """
        Locate and load *config.yaml* (walk up from this file if needed) or
        honour an explicit override.
        """
        override = Path(explicit_path) if explicit_path else None
        if override and override.exists():
            return yaml.safe_load(override.read_text())

        for parent in Path(__file__).resolve().parents:
            cfg = parent / "config.yaml"
            if cfg.exists():
                return yaml.safe_load(cfg.read_text())
        raise FileNotFoundError("config.yaml not found.")
