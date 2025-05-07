from __future__ import annotations

"""
src/SHAP.py - SHAP diagnostics for the XGBoost price model (global, local & categorical plots)
=============================================================================================

This module provides :class:`XGBSHAPAnalyzer`, a lightweight helper that
loads the **fold-10** XGBoost model trained by :class:`~src.XGB.XGBRunner`,
computes SHAP values on the *tree-ready* feature matrix, and writes tidy
Excel summaries of

* per-feature importance (mean │SHAP│),
* aggregated cluster contributions (same taxonomy as OLS/Ridge runners).

Optionally, one-hot dummies can be **collapsed** back to their parent categorical feature for global importance metrics and summary plots (`collapse_categories=True`).

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
import matplotlib
import yaml
from collections import defaultdict
try:
    from shap.plots import colors as shap_colors  # type: ignore
except ImportError:  # pragma: no cover – legacy SHAP
    shap_colors = None  # pytype: disable=invalid-assignment

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
    collapse_categories
        If True, one-hot encoded dummy features are collapsed back to their
        parent categorical feature for global importance metrics and summary plots.
    output_path
        Excel workbook to which the summaries will be written (two sheets:
        ``feature_importance`` and ``cluster_importance``).  Existing files
        are overwritten.

    Produces:
    * per-feature importance (mean │SHAP│),
    * aggregated cluster contributions (same taxonomy as OLS/Ridge runners),
    * summary beeswarm,
    * bar chart,
    * dependence plots,
    * categorical dependence plots,
    * **heat-map**,
    * local waterfall explanations.
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
        collapse_categories: bool = True,
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
        self.target = target

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
        # prefixes that mark dummy columns originating from one original categorical field
        self._cat_prefixes: List[str] = [
            "property_class",
            "property_type",
            "shed",
            "nuts3_region",
            "monument",
        ]
        self.collapse_categories = collapse_categories
        self.output_path = self.root / output_path

        # Directory for all SHAP figures (configurable via config.yaml → shap_path)
        self.shap_dir: Path = self.root / self.cfg.get("shap_path", "data/processed/SHAP")
        self.shap_dir.mkdir(parents=True, exist_ok=True)

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
    def _save_fig(self, fig=None, name: str = "") -> None:
        """
        Save a matplotlib figure or axes to disk, ensuring tight layout.
        Accepts a matplotlib.figure.Figure, matplotlib.axes.Axes, or None (uses plt.gcf()).
        """
        # Determine the figure object to save
        if fig is None:
            fig = plt.gcf()
        elif isinstance(fig, matplotlib.axes.Axes):
            fig = fig.figure
        # If not a Figure now, fallback to plt.gcf()
        if not isinstance(fig, matplotlib.figure.Figure):
            fig = plt.gcf()
        # Call tight_layout before saving
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(self.shap_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # SHAP plots                                                         #
    # ------------------------------------------------------------------ #
    def _plot_summary(self, shap_vals: np.ndarray) -> None:
        """
        Beeswarm summary of all SHAP values (global explanation).
        While we collapsed the one-hot dummies for the bar plot,
        here we keep them separate to show the distribution of SHAP.

        The plot is saved as ``shap_summary.png``.
        """
        plt.close("all")
        X_viz = self.X.iloc[self._sample_idx].copy()
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            X_viz,
            show=False,
            plot_type="dot",
        )
        self._save_fig(fig, "shap_summary.png")

    def _plot_bar(self, shap_vals: np.ndarray) -> None:
        """
        Simple bar chart of mean │SHAP│ values for the 20 most important
        features.  If `collapse_categories` is True, one-hot dummies are
        collapsed back to their parent categorical feature by summing
        their SHAP contributions.  Saved as ``shap_bar_top20.png``.
        """
        plt.close("all")
        # prepare data for plotting
        if self.collapse_categories:
            # collapse one-hot dummies by summing SHAP values
            agg_vals, agg_names = self._aggregate_shap(
                shap_vals, list(self.X.columns)
            )
            # build a dummy DataFrame for feature names (values not used in bar plot)
            plot_X = pd.DataFrame(
                np.zeros_like(agg_vals),
                columns=agg_names
            )
            vals_to_plot = agg_vals
        else:
            plot_X = self.X.iloc[self._sample_idx].copy()
            plot_X.columns = [
                self._base_feat_name(c, self._cat_prefixes) for c in plot_X.columns
            ]
            vals_to_plot = shap_vals

        # create bar summary plot
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(
            vals_to_plot,
            plot_X,
            show=False,
            plot_type="bar",
            max_display=20,
        )
        self._save_fig(fig, "shap_bar_top20.png")

    def _plot_dependence_pairs(
        self,
        shap_vals: np.ndarray,
        pairs: List[tuple[str, str]],
    ) -> None:
        """
        SHAP dependence plots for specific (feature, hue) pairs.
        Each plot is saved as ``dependence_<feature>.png``.
        """
        plt.close("all")
        # subset to the sampled rows and post-1900 if applicable
        X_s = self.X.iloc[self._sample_idx]
        mask = self._post1900_mask(X_s)
        X_s, shap_vals_sub = X_s.loc[mask], shap_vals[mask]
        for x_feat, hue_feat in pairs:
            idx_x = self.X.columns.get_loc(x_feat)
            shap.dependence_plot(
                idx_x,
                shap_vals_sub,
                X_s,
                interaction_index=hue_feat,
                show=False,
            )
            fig = plt.gcf()
            self._save_fig(fig, f"dependence_{x_feat}.png")
            
    def _plot_heatmap(self, shap_vals: np.ndarray, top_n: int = 10) -> None:
        """Draw a SHAP heat‑map for the *top_n* most influential features.

        The heat‑map orders instances by their summed SHAP contribution and
        features by their mean |SHAP| magnitude – a compact overview that
        helps spot heterogeneity across observations.
        """
        plt.close("all")
        # --- choose *top_n* features -------------------------------------------------
        top_idx = np.argsort(np.abs(shap_vals).mean(0))[-top_n:]
        exp = shap.Explanation(
            values=shap_vals[:, top_idx],
            base_values=np.repeat(self._explainer.expected_value, shap_vals.shape[0]),
            data=self.X.iloc[self._sample_idx, top_idx],
            feature_names=[self.X.columns[i] for i in top_idx],
        )
        order = np.argsort(exp.values.sum(1))
        shap.plots.heatmap(
            exp,
            instance_order=order,
            show=False,
        )
        self._save_fig(plt.gcf(), f"shap_heatmap_top{top_n}.png")

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

    # ------------------------------------------------------------------ #
    # categorical aggregation helpers                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _base_feat_name(col: str, prefixes: List[str]) -> str:
        """
        Collapse a one‑hot dummy column back to its *base* categorical
        feature name.  A column is considered a dummy if it *starts with*
        one of *prefixes* followed by an underscore.
        """
        for p in prefixes:
            if col.startswith(f"{p}_"):
                return p
        return col

    def _aggregate_shap(
        self, shap_vals: np.ndarray, feature_names: List[str]
    ) -> tuple[np.ndarray, List[str]]:
        """
        Sum SHAP contributions of dummies that belong to the same
        high-level categorical feature.

        Returns
        -------
        agg_vals : (n_samples, n_collapsed_features) ndarray
        agg_names : list[str]
        """
        groups: defaultdict[str, List[int]] = defaultdict(list)
        for idx, name in enumerate(feature_names):
            groups[self._base_feat_name(name, self._cat_prefixes)].append(idx)

        agg_names: List[str] = list(groups.keys())
        agg_vals = np.zeros((shap_vals.shape[0], len(agg_names)))
        for j, idxs in enumerate(groups.values()):
            agg_vals[:, j] = shap_vals[:, idxs].sum(axis=1)

        return agg_vals, agg_names

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
        if self.collapse_categories:
            shap_vals, feature_list = self._aggregate_shap(
                shap_vals, list(self.X.columns)
            )
        else:
            feature_list = list(self.X.columns)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        imp_df = (
            pd.DataFrame(
                {"feature": feature_list, "mean_abs_shap": mean_abs}
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
    # masking helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _post1900_mask(X: pd.DataFrame) -> np.ndarray:
        """Rows with ``construction_yr`` ≥ 1900 (or all rows if column missing)."""
        if "construction_yr" in X.columns:
            return X["construction_yr"].fillna(0).ge(1900).values
        return np.ones(len(X), dtype=bool)

    # ------------------------------------------------------------------ #
    # static helpers                                                     #
    # ------------------------------------------------------------------ #
    def _generate_plots(self) -> None:
        """
        Produce and persist a standard set of SHAP visualisations:

        * Beeswarm summary of all features.
        * Bar chart of top-20 mean │SHAP│ values.
        * Heat-map of top features across all samples.
        * Additional domain-driven dependence plots for specific feature pairs.
        * Categorical dependence plots and local waterfall explanations.
        """
        shap_vals = self._compute_shap_values()
        self._plot_summary(shap_vals)
        self._plot_bar(shap_vals)
        self._plot_heatmap(shap_vals)
        custom_pairs = [
            ("gross_volume", "parcel_surface"),
            ("pc6_price_2y_prior", "global_price_6m_prior"),
            ("lon", "lat"),
            ("construction_yr", "quality"),
            ("unit_surface", "rooms_nr"),
            ("global_price_6m_prior", "date_of_listing"),
            ("parcel_surface", "property_type_Tussenwoning")
        ]
        self._plot_dependence_pairs(shap_vals, custom_pairs)

        # categorical dependence plots
        self._plot_categorical_dependence(shap_vals, "nuts3_region_", hue="construction_yr")
        self._plot_categorical_dependence(shap_vals, "property_class_", hue="parcel_surface")
        self._plot_categorical_dependence(shap_vals, "shed_", hue="quality")

        # local explanations
        self._plot_local_waterfall(shap_vals)

    def _plot_categorical_dependence(
        self,
        shap_vals: np.ndarray,
        prefix: str,
        hue: str = "unit_surface",
    ) -> None:
        """
        Aggregate one-hot SHAP contributions for *prefix* and plot them
        versus the category, coloured by *hue*.

        The figure is saved as ``<prefix>_dependence.png``.
        """
        plt.close("all")
        X_s = self.X.iloc[self._sample_idx]
        mask = self._post1900_mask(X_s)
        X_s, shap_vals = X_s.loc[mask], shap_vals[mask]

        cols = [c for c in self.X.columns if c.startswith(prefix)]
        if not cols:  # nothing to plot
            return
        idxs = [self.X.columns.get_loc(c) for c in cols]
        cat_shap = shap_vals[:, idxs].sum(axis=1)
        categories = (
            X_s[cols]
            .idxmax(axis=1)
            .str.replace(prefix, "")
            .str.lstrip("_")
        )
        hue_vals = X_s[hue] if hue in X_s else None

        df = pd.DataFrame(
            {"category": categories, "shap": cat_shap, "hue": hue_vals}
        )
        cat_to_x = {c: i for i, c in enumerate(sorted(df.category.unique()))}

        fig, ax = plt.subplots(figsize=(12, 6))
        # use SHAP's native red↔blue diverging palette when available
        cmap = shap_colors.red_blue if shap_colors is not None else "coolwarm"
        scatter = ax.scatter(
            df.category.map(cat_to_x),
            df.shap,
            c=df.hue,
            cmap=cmap,
            s=7,
            alpha=1.0,
        )
        ax.set_xticks(list(cat_to_x.values()))
        ax.set_xticklabels(cat_to_x.keys(), rotation=45, ha="right")
        ax.set_xlabel(prefix.rstrip("_"))
        ax.set_ylabel(f"SHAP value for {self.target}")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(hue)
        cbar.ax.yaxis.set_tick_params(labelsize=8)
        self._save_fig(fig, f"{prefix.rstrip('_')}_dependence.png")

    def _plot_local_waterfall(
        self,
        shap_vals: np.ndarray,
        indices: Optional[List[int]] | None = None,
    ) -> None:
        """
        Plot local SHAP waterfall explanations for selected sample indices.
        If no indices are provided, defaults to the 100th and 1000th samples,
        as well as the 10th highest and 10th lowest target values in the sample.

        Parameters
        ----------
        shap_vals : numpy.ndarray
            Matrix returned by :meth:`_compute_shap_values`, shaped
            (n_samples, n_features).
        indices : list[int] or None, optional
            Row indices within the sampled subset to visualize. If None,
            defaults as described above.
        """
        if indices is None:
            n_samples = shap_vals.shape[0]
            default_indices: list[int] = []
            if n_samples >= 100:
                default_indices.append(99)
            if n_samples >= 1000:
                default_indices.append(999)
            # Determine sample target values
            if isinstance(self._sample_idx, slice):
                y_sample = self.y
            else:
                y_sample = self.y[self._sample_idx]
            if y_sample.shape[0] >= 10:
                sorted_idx = np.argsort(y_sample)
                default_indices.append(int(sorted_idx[9]))
                default_indices.append(int(sorted_idx[-10]))
            indices = sorted(set(default_indices))
        max_row = shap_vals.shape[0] - 1
        for local_idx in indices:
            if local_idx > max_row:
                continue  # skip out‑of‑range requests silently
            exp = shap.Explanation(
                values=shap_vals[local_idx],
                base_values=self._explainer.expected_value,
                data=self.X.iloc[self._sample_idx].iloc[local_idx],
                feature_names=self.X.columns,
            )
            shap.plots.waterfall(exp, show=False)
            fig = plt.gcf()
            self._save_fig(fig, f"local_waterfall_idx{local_idx}.png")
    
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