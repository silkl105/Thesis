from __future__ import annotations

"""src/OLS.py: forecasting + coefficient diagnostics
=======================================================

* Region-specific OLS models evaluated with a 10-fold
  expanding-window TimeSeriesSplit (≈80/20 within each fold).
  In each fold, three separate OLS regressions are fitted on the
  training data for three predefined NUTS3 region groups, and
  predictions are made on the corresponding test observations.
  Forecast accuracy metrics are stored to Excel, one row per fold.
* A global full-sample OLS is also fitted for coefficient interpretation.
  We report point estimates + heteroskedasticity-robust HC3 s.e.,
  and map each term to feature clusters as in RidgeRunner.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from src.preprocess import DataProcessor

__all__ = ["OLSRunner"]

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

class OLSRunner:
    """
    Run region-specific OLS-CV + global OLS on the processed dataset.

    Parameters
    ----------
    config_path : Optional[str]
        Optional explicit path to config.yaml.
    metrics_path : str or Path
        Excel workbook to which fold-level metrics and OLS
        coefficients will be written. If it exists it is overwritten.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        metrics_path: str | Path = "data/processed/ols_results.xlsx",
        save_option: bool = False,
        target: Optional[str] = "transaction_price",
    ) -> None:
        self.cfg = self._load_config(config_path)
        self.root = Path(__file__).resolve().parents[1]
        self.metrics_path = self.root / metrics_path
        self.save_option = save_option
        self.target = target
        self.dp = DataProcessor()

        # load linear design matrix
        proc_file = self.root / self.cfg["raw_files"]["processed_lin"]
        if not proc_file.exists():
            raise FileNotFoundError(proc_file)
        self.df = pd.read_parquet(proc_file)

        self.y = self.df[self.target].astype(float).values
        self.X = self.df.drop(columns=[self.target]).copy()

        # build transformer (must match RidgeRunner)
        self.cat_pass: List[str] = [
            c for c in self.X.columns
            if c not in (*self.dp.center_cols, *self.dp.scale_cols)
        ]
        self.transformer: ColumnTransformer = ColumnTransformer(
            [
                ("center", StandardScaler(with_std=False), self.dp.center_cols),
                ("scale", StandardScaler(), self.dp.scale_cols),
                ("pass", "passthrough", self.cat_pass),
            ],
            remainder="drop",
        )

        self._fold_info: List[Dict] = []

    def run(self) -> None:
        """Execute full workflow and dump Excel workbook."""
        cv_df, cv_by_model_df = self._ols_cv()
        coef_df, cluster_df = self._ols_full()
        diag_df = self._compute_diagnostics()

        with pd.ExcelWriter(self.metrics_path, engine="xlsxwriter") as xls:
            cv_df.to_excel(xls, sheet_name="ols_cv", index=False)
            cv_by_model_df.to_excel(xls, sheet_name="ols_cv_by_model", index=False)
            coef_df.to_excel(xls, sheet_name="ols_coef", index=False)
            cluster_df.to_excel(xls, sheet_name="cluster_coef", index=False)
            diag_df.to_excel(xls, sheet_name="diagnostics", index=False)
        print(f"✓ Results written to {self.metrics_path.relative_to(self.root)}")

    def _ols_cv(self):
        """
        10-fold expanding-window CV for region-specific OLS.
        Return two DataFrames: one aggregated across all test folds,
        and one with per-model metrics within each fold.

        Fits three separate OLS regressions per fold on predefined
        NUTS3 groups, then aggregates predictions and computes metrics.
        """
        # define region groups
        region_groups = {
            "model1": ["212", "213", "221", "225", "230"],
            "model2": ["224", "226", "415", "416"],
            "model3": ["327", "32B", "350"],
        }

        n_splits = 10
        block = int(round(len(self.X) / n_splits))
        test_size = int(round(block * 0.65))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        rows: List[Dict[str, float]] = []
        rows_model: List[Dict[str, float]] = []
        preds: List[pd.DataFrame] = []

        for fold, (train_idx, test_idx) in tqdm(enumerate(tscv.split(self.X), 1),
                                                total=n_splits, desc="OLSCV folds"):
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X.iloc[test_idx]
            y_test = self.y[test_idx]
            # Fit transformer once on full training fold to avoid zero-variance on subsets
            fold_transformer = clone(self.transformer)
            fold_transformer.fit(X_train)

            preds_fold = []
            # Identify all region dummy columns for this fold
            region_dummy_cols = [c for c in X_train.columns if c.startswith("nuts3_region_")]

            # fit each region-specific OLS
            for grp_name, codes in region_groups.items():
                cols = [f"nuts3_region_NL{c}" for c in codes]
                mask_train = X_train[cols].sum(axis=1).astype(bool)
                mask_test = X_test[cols].sum(axis=1).astype(bool)
                # For model1, include observations without any region dummy (NL211 dropped as categorical column processed with drop_first=True)
                if grp_name == "model1":
                    missing_train = ~X_train[region_dummy_cols].sum(axis=1).astype(bool)
                    missing_test = ~X_test[region_dummy_cols].sum(axis=1).astype(bool)
                    mask_train = mask_train | missing_train
                    mask_test = mask_test | missing_test

                if not mask_train.any():
                    continue

                # Transform once for the fold
                X_tr_arr = fold_transformer.transform(X_train.loc[mask_train])
                X_te_arr = fold_transformer.transform(X_test.loc[mask_test])

                # Replace any NaNs (can arise from rare categories or missing values)
                X_tr_arr = np.nan_to_num(X_tr_arr, nan=0.0)
                X_te_arr = np.nan_to_num(X_te_arr, nan=0.0)

                # Fit plain OLS via scikit-learn
                ols = LinearRegression(n_jobs=-1)
                ols.fit(X_tr_arr, y_train[mask_train])

                y_pred = ols.predict(X_te_arr)

                df_pred = X_test.loc[mask_test].copy()
                df_pred["y_test"] = np.expm1(y_test[mask_test])
                df_pred["y_pred"] = np.expm1(y_pred)
                df_pred["fold"] = fold
                df_pred["model"] = grp_name
                preds_fold.append(df_pred)

                # per‑model metrics
                y_true_grp = df_pred["y_test"].values
                y_pred_grp = df_pred["y_pred"].values
                rmse_g = np.sqrt(mean_squared_error(y_true_grp, y_pred_grp))
                mae_g = mean_absolute_error(y_true_grp, y_pred_grp)
                mape_g = _mape(np.log1p(y_true_grp), np.log1p(y_pred_grp))
                mdape_g = _mdape(np.log1p(y_true_grp), np.log1p(y_pred_grp))
                err_g = np.abs(y_pred_grp - y_true_grp) / y_true_grp
                rows_model.append({
                    "fold": fold,
                    "model": grp_name,
                    "rmse": rmse_g,
                    "mae": mae_g,
                    "mape": mape_g,
                    "mdape": mdape_g,
                    "pct_within_5": np.mean(err_g <= 0.05) * 100,
                    "pct_within_10": np.mean(err_g <= 0.10) * 100,
                    "pct_within_20": np.mean(err_g <= 0.20) * 100
                })

            # aggregate preds and compute metrics
            fold_preds = pd.concat(preds_fold, axis=0)
            preds.append(fold_preds)

            y_true = fold_preds["y_test"].values
            y_pred = fold_preds["y_pred"].values

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = _mape(np.log1p(y_true), np.log1p(y_pred))
            mdape = _mdape(np.log1p(y_true), np.log1p(y_pred))
            err_pct = np.abs(y_pred - y_true) / y_true

            rows.append({
                "fold": fold,
                "train_obs": len(train_idx),
                "test_obs": len(test_idx),
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "mdape": mdape,
                "pct_within_5": np.mean(err_pct <= 0.05) * 100,
                "pct_within_10": np.mean(err_pct <= 0.10) * 100,
                "pct_within_20": np.mean(err_pct <= 0.20) * 100
            })

        # save predictions
        preds_df = pd.concat(preds, axis=0)
        out_path = self.root / self.cfg["processed_path"] / "predictions_ols.parquet"
        preds_df.to_parquet(out_path)

        cv_df = pd.DataFrame(rows)
        cv_by_model_df = pd.DataFrame(rows_model)
        return cv_df, cv_by_model_df

    def _ols_full(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit OLS once (full sample) and return coefficient tables."""
        X_all = pd.DataFrame(
            self.transformer.fit_transform(self.X),
            columns=self._out_cols(), dtype=float
        )
        X_all = sm.add_constant(X_all, has_constant="add")
        model = sm.OLS(self.y, X_all).fit(cov_type="HC3")

        params = model.params
        se = model.bse
        pvalues = model.pvalues
        ci = model.conf_int()
        coef_df = pd.DataFrame({
            "term": params.index,
            "coef": params.values,
            "se_robust": se.values,
            "p_value": pvalues.values,
            "[0.025": ci[0].values,
            "0.975]": ci[1].values,
        })

        # cluster mapping same as in RidgeRunner
        clusters: Dict[str, List[str]] = {
            "coordinates": ["lat", "lon"],
            "region": [c for c in coef_df.term if c.startswith("nuts3_")],
            "temporal": ["date_of_listing", "month_sin", "month_cos", "construction_yr"],
            "property_size": ["unit_surface", "gross_volume", "parcel_surface", "rooms_nr"],
            "property_quality": ["quality"] + [c for c in coef_df.term if c.startswith("shed_")],
            "property_type": [c for c in coef_df.term if c.startswith(("property_class_", "property_type_"))],
            "lags": ["pc6_price_2y_prior", "pc6_price_6m_minus_2y", "global_price_6m_prior", "global_price_1m_minus_6m"],
            "economy": ["ciss", "unemployment_rate", "mortgage_rate"],
        }

        sums = []
        for clust, terms in clusters.items():
            mask = coef_df.term.isin(terms)
            if mask.any():
                coefs = coef_df.loc[mask, "coef"].values
                sums.append({
                    "cluster": clust,
                    "n_terms": mask.sum(),
                    "l2_norm": float(np.linalg.norm(coefs)),
                    "sum_coef": float(coefs.sum()),
                })
        cluster_df = pd.DataFrame(sums)
        return coef_df, cluster_df

    def _compute_diagnostics(self) -> pd.DataFrame:
        """Compute the same diagnostics on full sample as RidgeRunner."""
        # Transform the full feature matrix and ensure it is NaN‑free
        X_arr = self.transformer.fit_transform(self.X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        X_t = pd.DataFrame(X_arr, columns=self._out_cols(), dtype=float)

        vifs = []
        for idx, col in enumerate(X_t.columns):
            try:
                v = sm.stats.outliers_influence.variance_inflation_factor(X_arr, idx)
            except Exception:
                v = np.nan
            vifs.append({"feature": col, "vif": float(v)})
        top5_vifs = sorted(vifs, key=lambda d: -d["vif"] if np.isfinite(d["vif"]) else -np.inf)[:5]

        model = sm.OLS(self.y, sm.add_constant(X_t, has_constant="add")).fit()
        infl = model.get_influence()
        cooks = infl.cooks_distance[0]
        cooks_thresh = 4 / len(self.y)
        cooks_count = int((cooks > cooks_thresh).sum())
        _, lm_p, _, _ = sm.stats.diagnostic.het_breuschpagan(model.resid, sm.add_constant(X_t, has_constant="add"))
        dw = sm.stats.stattools.durbin_watson(model.resid)

        return pd.DataFrame([{
            "top5_vifs": json.dumps(top5_vifs),
            "cooks_count": cooks_count,
            "bp_pvalue": lm_p,
            "dw": dw
        }])

    def _out_cols(self) -> List[str]:
        """Return transformed column order (excluding intercept)."""
        return self.dp.center_cols + self.dp.scale_cols + self.cat_pass

    @staticmethod
    def _load_config(env_path: Optional[str] = None) -> Dict[str, str]:
        override = Path(env_path) if env_path else None
        if override and override.exists():
            return yaml.safe_load(override.read_text())
        for parent in Path(__file__).resolve().parents:
            cfg = parent / "config.yaml"
            if cfg.exists():
                return yaml.safe_load(cfg.read_text())
        raise FileNotFoundError("config.yaml not found.")