from __future__ import annotations

"""
src/SHAP.py – SHAP diagnostics for the XGBoost price model
==========================================================

This module provides :class:`XGBSHAPAnalyzer`, a lightweight helper that
loads the **fold‑10** XGBoost model trained by :class:`~src.XGB.XGBRunner`,
computes SHAP values on the *tree‑ready* feature matrix, and writes tidy
Excel summaries of

* per‑feature importance (mean │SHAP│),
* aggregated cluster contributions (same taxonomy as OLS/Ridge runners).

The heavy lifting is delegated to *shap.TreeExplainer* which is fast on
tree models, so even a fairly large dataset is handled comfortably.
"""

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import yaml

__all__ = ["SHAPAnalyzer"]


class SHAPAnalyzer:
    """
    Run SHAP analysis for the XGBoost house-price model.

    Parameters
    ----------
    config_path
        Optional explicit path to *config.yaml*.
    model_fold
        Outer CV fold whose best estimator should be analysed.  Defaults to
        the 10th fold which, by construction, is the model trained on the
        **largest** training window.
    sample_size
        If not *None*, a stratified random sample (without replacement) of
        at most this many rows is used for SHAP calculation to keep run-time
        bounded.  Set to *None* to use the full dataset.
    output_path
        Excel workbook to which the summaries will be written (two sheets:
        ``feature_importance`` and ``cluster_importance``).  Existing files
        are overwritten.
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        model_fold: int = 10,
        sample_size: Optional[int] = 5_000,
        output_path: str | Path = "data/processed/shap_results.xlsx",
        target: str = "transaction_price",
    ) -> None:

        # ---------- locate project paths -------------------------------- #
        self.cfg: Dict[str, str] = self._load_config(config_path)
        self.root = Path(__file__).resolve().parents[1]

        # ---------- data ------------------------------------------------- #
        xgb_file = self.root / self.cfg["raw_files"]["processed_data"]
        if not xgb_file.exists():
            raise FileNotFoundError(xgb_file)
        self.df = pd.read_parquet(xgb_file)

        if target not in self.df.columns:
            raise KeyError(f"Target column '{target}' absent from {xgb_file}")
        self.X = self.df.drop(columns=[target])
        self.y = self.df[target].astype(float).values  # noqa: not used yet

        # ---------- model ------------------------------------------------ #
        self.model_path = (
            self.root
            / self.cfg["processed_path"]
            / f"xgb_model_fold_{model_fold}.pkl"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Saved model for fold {model_fold} not found: {self.model_path}"
            )
        self.model = joblib.load(self.model_path)

        # ---------- options --------------------------------------------- #
        self.sample_size = sample_size
        self.output_path = self.root / output_path

        # Pre‑compute indices so we can reuse the same subset for multiple
        # plots / summaries if the user calls the public helpers repeatedly.
        self._sample_idx = (
            np.random.default_rng(41).choice(
                len(self.X),
                size=min(sample_size, len(self.X)),
                replace=False,
            )
            if sample_size is not None and sample_size < len(self.X)
            else slice(None)
        )

        self._explainer: Optional[shap.Explainer] = None
        self._shap_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # figure helpers                                                     #
    # ------------------------------------------------------------------ #
    def _ensure_outdir(self) -> Path:
        """
        Ensure the *processed* directory exists and return the absolute
        path.  Figures are written directly next to the parquet/model
        artefacts so everything lives under ``data/processed``.
        """
        outdir = self.root / self.cfg["processed_path"]
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _save_fig(self, fig: "plt.Figure", name: str) -> None:  # noqa: quotes for ForwardRef
        """Save *fig* to *data/processed* (PNG, 150 dpi) and close."""
        outdir = self._ensure_outdir()
        fig.savefig(outdir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # SHAP plots                                                         #
    # ------------------------------------------------------------------ #
    def _plot_summary(self, shap_vals: np.ndarray) -> None:
        """
        Beeswarm summary of all SHAP values (global explanation).

        The plot is saved as ``shap_summary.png``.
        """
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            self.X.iloc[self._sample_idx],
            show=False,
            plot_type="dot",
        )
        self._save_fig(fig, "shap_summary.png")

    def _plot_bar(self, shap_vals: np.ndarray) -> None:
        """
        Simple bar chart of mean │SHAP│ values for the 20 most important
        features.  Saved as ``shap_bar_top20.png``.
        """
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_vals,
            self.X.iloc[self._sample_idx],
            show=False,
            plot_type="bar",
            max_display=20,
        )
        self._save_fig(fig, "shap_bar_top20.png")

    def _plot_dependence(self, shap_vals: np.ndarray, top_n: int = 4) -> None:
        """
        SHAP *dependence* plots for the ``top_n`` most important
        predictors with automatic interaction lookup.  Files are written
        as ``shap_dependence_<feature>.png``.
        """
        # rank features by mean |shap|
        mean_abs = np.abs(shap_vals).mean(axis=0)
        ranked = np.argsort(mean_abs)[::-1][:top_n]
        X_s = self.X.iloc[self._sample_idx]

        for idx in ranked:
            feat = self.X.columns[idx]
            fig = plt.figure(figsize=(8, 5))
            # automatically colour by the feature with strongest interaction
            shap.dependence_plot(
                idx,
                shap_vals,
                X_s,
                interaction_index="auto",
                show=False,
            )
            fname = f"shap_dependence_{feat}.png".replace("[", "").replace("]", "")
            self._save_fig(fig, fname)

    # ------------------------------------------------------------------ #
    # public driver                                                      #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Compute SHAP values, generate summaries, and write the Excel file.

        *Mean absolute SHAP values* are used for feature importances
        throughout, because they are both easy to interpret and robust for
        ranking.
        """
        shap_vals = self._compute_shap_values()
        imp_df = self._feature_importance(shap_vals)
        cluster_df = self._cluster_importance(imp_df)

        with pd.ExcelWriter(self.output_path, engine="xlsxwriter") as xls:
            imp_df.to_excel(xls, sheet_name="feature_importance", index=False)
            cluster_df.to_excel(xls, sheet_name="cluster_importance", index=False)
        print(f"✓ SHAP summaries written to {self.output_path.relative_to(self.root)}")
        self._generate_plots()
        print("✓ SHAP figures saved to data/processed/")

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _compute_shap_values(self) -> np.ndarray:
        """
        Compute (and memoise) SHAP values for *self.X[self._sample_idx]*,
        using interventional perturbation to handle multicollinearity accurately.
        The DataFrame is converted to a numeric numpy array before passing to the explainer,
        and feature names are supplied for consistent mapping in plots.

        Returns
        -------
        np.ndarray
            Matrix of shap values with shape *(n_samples, n_features)*.
        """
        if self._shap_values is None:
            if self._explainer is None:
                background_df = self.X.iloc[self._sample_idx]
                background_num = self._prepare_numeric(background_df)
                self._explainer = shap.TreeExplainer(
                    self.model,
                    data=background_num.values,
                    feature_perturbation="interventional",
                    feature_names=list(background_num.columns)
                )
            X_s = self.X.iloc[self._sample_idx]
            X_s_num = self._prepare_numeric(X_s)
            self._shap_values = self._explainer.shap_values(X_s_num.values)
        return self._shap_values

    def _prepare_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all feature columns to pure numeric floats:
        - category columns → integer codes
        - pandas Int64 columns → floats
        - other numeric types → floats
        """
        df_num = df.copy()
        for col in df_num.columns:
            if pd.api.types.is_categorical_dtype(df_num[col].dtype):
                df_num[col] = df_num[col].cat.codes
            elif pd.api.types.is_integer_dtype(df_num[col].dtype):
                df_num[col] = df_num[col].astype(float)
            else:
                df_num[col] = df_num[col].astype(float)
        return df_num

    # ---------- tidy summaries ---------------------------------------- #
    def _feature_importance(self, shap_vals: np.ndarray) -> pd.DataFrame:
        """
        Return a tidy DataFrame of per-feature mean │SHAP│ values.

        The DataFrame is sorted descending by importance.

        Parameters
        ----------
        shap_vals
            SHAP values produced by :pymeth:`_compute_shap_values`.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``mean_abs_shap``.
        """
        mean_abs = np.abs(shap_vals).mean(axis=0)
        imp_df = (
            pd.DataFrame(
                {"feature": self.X.columns, "mean_abs_shap": mean_abs}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return imp_df

    # ------------------------------------------------------------------ #
    def _cluster_importance(self, imp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate feature importances into curated **clusters**.

        The taxonomy mirrors the one used by *OLSRunner* / *RidgeRunner*
        so cross-model comparisons are straightforward.

        Parameters
        ----------
        imp_df
            Output of :pymeth:`_feature_importance`.

        Returns
        -------
        pd.DataFrame
            Columns: ``cluster``, ``n_terms``, ``sum_mean_abs_shap``.
        """
        clusters: Dict[str, List[str]] = {
            "coordinates": ["lat", "lon"],
            "region": [c for c in imp_df.feature if c.startswith("nuts3_")],
            "temporal": ["date_of_listing", "month_sin", "month_cos", "construction_yr"],
            "property_size": [
                "unit_surface",
                "gross_volume",
                "parcel_surface",
                "rooms_nr",
            ],
            "property_quality": ["quality"] + [c for c in imp_df.feature if c.startswith("shed_")],
            "property_type": [
                c
                for c in imp_df.feature
                if c.startswith(("property_class_", "property_type_"))
            ],
            "lags": [
                "pc6_price_2y_prior",
                "pc6_price_6m_minus_2y",
                "global_price_6m_prior",
                "global_price_1m_minus_6m",
            ],
            "economy": ["ciss", "unemployment_rate", "mortgage_rate"],
        }

        rows = []
        for cluster, terms in clusters.items():
            mask = imp_df.feature.isin(terms)
            if mask.any():
                vals = imp_df.loc[mask, "mean_abs_shap"].values
                rows.append(
                    {
                        "cluster": cluster,
                        "n_terms": int(mask.sum()),
                        "sum_mean_abs_shap": float(vals.sum()),
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values("sum_mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    # static helpers                                                     #
    # ------------------------------------------------------------------ #
    def _generate_plots(self) -> None:
        """
        Produce and persist a standard set of SHAP visualisations:

        * Beeswarm summary of all features.
        * Bar chart of top‑20 mean │SHAP│ values.
        * Dependence plots for the four most influential features.
        """
        shap_vals = self._compute_shap_values()
        self._plot_summary(shap_vals)
        self._plot_bar(shap_vals)
        self._plot_dependence(shap_vals)
    
    @staticmethod
    def _load_config(explicit_path: Optional[str] = None) -> Dict[str, str]:
        """
        Locate and parse *config.yaml*.

        The search order is

        1. **explicit_path** if provided,
        2. walk up the directory tree from *this file*.
        """
        override = Path(explicit_path) if explicit_path else None
        if override and override.exists():
            return yaml.safe_load(override.read_text())

        for parent in Path(__file__).resolve().parents:
            cfg = parent / "config.yaml"
            if cfg.exists():
                return yaml.safe_load(cfg.read_text())
        raise FileNotFoundError("config.yaml not found.")