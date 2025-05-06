from __future__ import annotations

"""src/Ridge.py: forecasting + coefficient diagnostics
=======================================================

* Ridge regression (full X) evaluated with a 10-fold
  expanding-window TimeSeriesSplit (≈80/20 within each fold).
  The best regularisation strength (λ) is chosen inside each fold via
  Bayesian hyperparameter search (BayesSearchCV) with time-aware splits.
  Forecast accuracy metrics are stored to Excel, one row per fold.
* Coefficients and cluster summaries are reported for the Ridge model trained on the last CV fold.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from skopt import BayesSearchCV
from skopt.space import Real
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.preprocess import DataProcessor

__all__ = ["RidgeRunner"]

# helpers – metrics ----------------------------------------------------
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute percentage error on original EUR scale (values > 0).
    Assumes target was log-transformed with np.log1p in preprocessing.
    """
    y_true_eur = np.expm1(y_true)
    y_pred_eur = np.expm1(y_pred)
    pct_err = np.abs(y_true_eur - y_pred_eur) / y_true_eur
    return pct_err.mean() * 100

def _mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median absolute percentage error on original EUR scale (values > 0).
    Assumes target was log-transformed with np.log1p in preprocessing.
    """
    y_true_eur = np.expm1(y_true)
    y_pred_eur = np.expm1(y_pred)
    pct_err = np.abs(y_true_eur - y_pred_eur) / y_true_eur
    return np.median(pct_err) * 100


# main class -----------------------------------------------------------
class RidgeRunner:
    """
    Run Ridge-CV, report last-fold Ridge coefficients, and diagnostics on the processed dataset.

    Parameters
    ----------
    config_path
        Optional explicit path to config.yaml.
    metrics_path
        Excel workbook to which fold-level metrics and OLS
        coefficients will be written.  If it exists it is overwritten.
    """
    def __init__(self, config_path: Optional[str] = None, *,
        metrics_path: str | Path = "data/processed/ridge_results.xlsx",
        save_option: bool = False,
        target: Optional[str] = 'transaction_price',
    ) -> None:
        self.cfg = self._load_config(config_path)
        self.root = Path(__file__).resolve().parents[1]
        self.metrics_path = self.root / metrics_path
        self.save_option = save_option
        self.target = target
        self.dp = DataProcessor()

        # ----------------------------------------------------------------
        # load linear design matrix (already one‑hot / ordinal encoded)
        proc_file = self.root / self.cfg["raw_files"]["processed_data"]
        if not proc_file.exists():
            raise FileNotFoundError(proc_file)
        self.df = pd.read_parquet(proc_file)

        self.y = self.df[self.target].astype(float).values  # ensure numeric log‑target
        self.X = self.df.drop(columns=[self.target]).copy()

        self.cat_pass: List[str] = [c for c in self.X.columns if c not in (*self.dp.center_cols, *self.dp.scale_cols)]
        self.transformer: ColumnTransformer = ColumnTransformer(
            [
                ("center", StandardScaler(with_std=False), self.dp.center_cols),
                ("scale", StandardScaler(), self.dp.scale_cols),
                ("pass", "passthrough", self.cat_pass),
            ],
            remainder="drop",
        )
        self._fold_info = []
        # Store pipeline from the last CV fold for coefficient reporting
        self.last_fold_pipe: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # public driver -----------------------------------------------------
    def run(self) -> None:
        """Execute full workflow and dump Excel workbook.

        Runs Ridge-CV, OLS, and diagnostics, and writes all outputs to Excel.
        """
        cv_df = self._ridge_cv()
        coef_df, cluster_df = self._ridge_last_cv()
        diag_df = self._compute_diagnostics()

        with pd.ExcelWriter(self.metrics_path, engine="xlsxwriter") as xls:
            cv_df.to_excel(xls, sheet_name="ridge_cv", index=False)
            coef_df.to_excel(xls, sheet_name="ridge_coef", index=False)
            cluster_df.to_excel(xls, sheet_name="cluster_coef", index=False)
            diag_df.to_excel(xls, sheet_name="diagnostics", index=False)
        print(f"✓ Results written to {self.metrics_path.relative_to(self.root)}")

    # ------------------------------------------------------------------
    # internal – ridge --------------------------------------------------
    def _ridge_cv(self) -> pd.DataFrame:
        """
        10-fold expanding-window CV with Bayesian search for alpha.

        Performs a Bayesian hyperparameter optimization (BayesSearchCV)
        over a log-uniform alpha space on each fold.
        Saves per-observation out-of-fold predictions and computes
        error metrics for each fold.
        """
        n_splits = 10
        block_size = int(round(len(self.X) / n_splits))
        test_size  = int(round(block_size * 0.65))
        tscv = TimeSeriesSplit(n_splits=n_splits,
                               test_size=test_size)
        rows: List[Dict[str, float]] = []
        preds = []
        self._fold_info = []

        for fold, (train_idx, test_idx) in tqdm(enumerate(tscv.split(self.X), 1), total=n_splits, desc="RidgeCV folds"):
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X.iloc[test_idx]
            y_test = self.y[test_idx]

            # Bayesian search for alpha
            bayes = BayesSearchCV(
                Ridge(),
                {"alpha": Real(1e-4, 1e4, prior="log-uniform")},
                n_iter=15,
                cv=TimeSeriesSplit(n_splits=3),
                scoring="neg_mean_absolute_error",
                random_state=41,
                verbose=0,
                n_jobs=2
            )
            pipe = Pipeline([("prep", self.transformer), ("model", bayes)])
            pipe.fit(X_train, y_train)
            # Save pipeline of the last CV fold for later coefficient extraction
            if fold == n_splits:
                self.last_fold_pipe = pipe

            y_pred = pipe.predict(X_test)

            self._fold_info.append({
                "pipe": pipe,
                "train_idx": train_idx,
                "test_idx": test_idx
            })

            rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
            mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
            mape = _mape(y_test, y_pred)
            mdape = _mdape(y_test, y_pred)
            rows.append(
                {
                    "fold": fold,
                    "train_obs": len(train_idx),
                    "test_obs": len(test_idx),
                    "alpha": pipe.named_steps["model"].best_estimator_.alpha,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "mdape": mdape,
                }
            )
            # Compute percent within error thresholds
            err_pct = np.abs(np.expm1(y_pred) - np.expm1(y_test)) / np.expm1(y_test)
            rows[-1].update({
                "pct_within_5": np.mean(err_pct <= 0.05) * 100,
                "pct_within_10": np.mean(err_pct <= 0.10) * 100,
                "pct_within_20": np.mean(err_pct <= 0.20) * 100,
            })
            # Save predictions for this fold
            X_test_flat = X_test.copy()
            X_test_flat['y_test'] = np.expm1(y_test)
            X_test_flat['y_pred'] = np.expm1(y_pred)
            X_test_flat['fold'] = fold
            preds.append(X_test_flat)

        preds_df = pd.concat(preds, axis=0)
        out_path = self.root / self.cfg["processed_path"] / "predictions_ridge.parquet"
        preds_df.to_parquet(out_path)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _compute_diagnostics(self) -> pd.DataFrame:
        """Compute diagnostics once for the whole dataset:
        - Top 5 VIFs (variance inflation factors)
        - Number of Cook's distance outliers
        - Breusch-Pagan p-value
        - Durbin-Watson statistic
        Returns DataFrame with one row.
        """
        # Transform the full dataset once
        X_t = pd.DataFrame(
            self.transformer.fit_transform(self.X),
            columns=self._out_cols(),
            dtype=float
        )
        # Precise VIFs via per‑variable auxiliary regressions
        vifs = []
        X_arr = X_t.values
        for idx, col in enumerate(X_t.columns):
            try:
                v = variance_inflation_factor(X_arr, idx)
            except Exception:
                v = np.nan
            vifs.append({"feature": col, "vif": float(v)})
        top5_vifs = sorted(vifs, key=lambda d: -d["vif"] if np.isfinite(d["vif"]) else -np.inf)[:5]
        # OLS diagnostics on full sample
        model = sm.OLS(self.y, sm.add_constant(X_t, has_constant="add")).fit()
        infl = model.get_influence()
        cooks = infl.cooks_distance[0]
        cooks_thresh = 4 / len(self.y)
        cooks_count = int((cooks > cooks_thresh).sum())
        # Breusch-Pagan test
        _, lm_pvalue, _, _ = het_breuschpagan(model.resid, sm.add_constant(X_t, has_constant="add"))
        # Durbin-Watson
        dw = durbin_watson(model.resid)
        # Return a single-row DataFrame
        return pd.DataFrame([{
            "top5_vifs": json.dumps(top5_vifs),
            "cooks_count": cooks_count,
            "bp_pvalue": lm_pvalue,
            "dw": dw
        }])

    def _ridge_last_cv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute coefficients and cluster summaries for the Ridge model
        fitted on the last CV fold.

        Returns
        -------
        coef_df : pd.DataFrame
            DataFrame with 'term' and 'coef' columns for the intercept and features.
        cluster_df : pd.DataFrame
            DataFrame summarizing clusters with columns:
            'cluster', 'n_terms', 'l2_norm', 'sum_coef'.
        """
        # Extract the best Ridge estimator from the last fold pipeline
        ridge: Ridge = self.last_fold_pipe.named_steps["model"].best_estimator_
        # Build feature names including intercept
        feature_names = ["const"] + self._out_cols()
        # Get coefficient array
        coef_arr = np.concatenate(([ridge.intercept_], ridge.coef_))
        coef_df = pd.DataFrame({"term": feature_names, "coef": coef_arr})

        # Define clusters (same mapping as OLS)
        clusters: Dict[str, List[str]] = {
            "coordinates": ["lat", "lon"],
            "region": [c for c in coef_df.term if c.startswith("nuts3_")],
            "temporal": ["date_of_listing", "month_sin", "month_cos", "construction_yr"],
            "property_size": ["unit_surface", "gross_volume", "parcel_surface", "rooms_nr"],
            "property_quality": ["quality"] + [c for c in coef_df.term if c.startswith("shed_")],
            "property_type": [c for c in coef_df.term if c.startswith(("property_class_", "property_type_"))],
            "lags": ["pc6_price_2y_prior", "pc6_price_6m_minus_2y",
                     "global_price_6m_prior", "global_price_1m_minus_6m"],
            "economy": ["ciss", "unemployment_rate", "mortgage_rate"],
        }

        # Summarize clusters
        sums = []
        for cluster, terms in clusters.items():
            mask = coef_df.term.isin(terms)
            if mask.any():
                vals = coef_df.loc[mask, "coef"].values
                sums.append({
                    "cluster": cluster,
                    "n_terms": int(mask.sum()),
                    "l2_norm": float(np.linalg.norm(vals)),
                    "sum_coef": float(vals.sum()),
                })
        cluster_df = pd.DataFrame(sums)
        return coef_df, cluster_df

    # ------------------------------------------------------------------
    def _out_cols(self) -> List[str]:
        """Return transformed column order (excluding intercept)."""
        return (
              self.dp.center_cols
            + self.dp.scale_cols
            + self.cat_pass
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(env_path: Optional[str] = None) -> Dict[str, str]:
        override = Path(env_path) if env_path else None
        if override and override.exists():
            return yaml.safe_load(override.read_text())
        for parent in Path(__file__).resolve().parents:
            p = parent / "config.yaml"
            if p.exists():
                return yaml.safe_load(p.read_text())
        raise FileNotFoundError("config.yaml not found.")
