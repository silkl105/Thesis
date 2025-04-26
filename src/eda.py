from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import branca
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml
from matplotlib.ticker import StrMethodFormatter


class EDA:
    """
    Quick-look EDA on pre-processed housing data.

    Parameters
    ----------
    config_path
        Optional path to config.yaml.  If omitted, walk up from this file
        until one is found, or honour the ``CONFIG_PATH`` environment variable.
    save_option
        If True, every figure is written to ``<figs_path>`` in the repo.
    """

    # ------------------------------------------------------------------
    # construction
    def __init__(self, config_path: Optional[str] = None, *, save_option: bool = False) -> None:
        self.cfg = self._load_config(config_path)
        self.root = Path(__file__).resolve().parents[1]
        self.fig_dir = self.root / self.cfg["figs_path"]
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.save_option = save_option

        processed = self.root / self.cfg["raw_files"]["processed_transactions"]
        if not processed.is_file():
            raise FileNotFoundError(f"Processed data missing: {processed}")
        self.df = pd.read_parquet(processed)

        sns.set_style("whitegrid")

    # ------------------------------------------------------------------
    # public quick looks
    def descriptive_stats(self) -> pd.DataFrame:
        """Return ``describe`` summary for key numeric columns."""
        cols = [
            "unit_surface",
            "gross_volume",
            "rooms_nr",
            "initial_list_price",
            "transaction_price",
            "duration",
        ]
        return self.df[cols].describe(
            percentiles=[0.01, 0.05, 0.95, 0.99]
        ).round(2)

    def missing_values(self) -> pd.Series:
        """Count NaNs per column (non-zero only)."""
        mv = self.df.isna().sum()
        return mv[mv > 0]

    # ---------- simple bar charts -------------------------------------
    def transactions_by_year(self) -> None:
        """Bar chart of annual deal counts (annotated in k)."""
        yearly = self.df["date_of_transaction"].dt.year.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly.plot(kind="bar", ax=ax)
        ax.set(
            title="Number of Transactions by Year",
            xlabel="Year",
            ylabel="Count",
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: format(int(x), ","))
        )
        ax.set_xticklabels(yearly.index, rotation=0)
        for i, v in yearly.items():
            ax.text(i - yearly.index.min(), v, f"{v/1_000:.0f}k", ha="center", va="bottom")
        self._save_fig(fig, "transactions_by_year.png")

    # ---------- distributions -----------------------------------------
    def histograms(self) -> None:
        """Histograms with optional clipping annotation on extremes."""
        def _axis_fmt(ax):  # local helper
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        cfg = [
            ("gross_volume", "Gross Volume (m³)", (0, 1_000), "right"),
            ("transaction_price", "Transaction Price (EUR)", None, None),
            ("unit_surface", "Unit Surface (m²)", (0, 400), "right"),
            ("construction_yr", "Year of Construction", (1875, 2020), "left"),
        ]

        for col, label, xlim, side in cfg:
            s = self.df[col].dropna()
            if s.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))

            kws = {"bins": 50}
            if xlim:
                kws["binrange"] = xlim
            sns.histplot(s, ax=ax, **kws)
            _axis_fmt(ax)

            pct = 0.0
            if xlim and side:
                lo, hi = xlim
                pct = (
                    (s > hi).sum() if side == "right" else (s < lo).sum()
                ) / len(s) * 100

            suffix = f"\n{'> ' if side=='right' else '< '}limit: {pct:.2f}%" if side else ""
            ax.set(
                title=f"Distribution of {label}{suffix}",
                xlabel=label,
                ylabel="Count",
            )
            self._save_fig(fig, f"hist_{col}.png")

    # ---------- normality checks --------------------------------------
    def qqplots(self) -> None:
        """Side-by-side Q–Q plots (log vs. original scale)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        sm.graphics.qqplot(self.df["transaction_price"], line="45", fit=True, ax=ax1)
        ax1.set_title("Q–Q Plot (log scale)")
        sm.graphics.qqplot(np.expm1(self.df["transaction_price"]), line="45", fit=True, ax=ax2)
        ax2.set_title("Q–Q Plot (original scale)")
        self._save_fig(fig, "qqplots.png")

    # ---------- price trends ------------------------------------------
    def avg_price_by_year(self) -> None:
        """Scatter + OLS trend of average price per calendar year."""
        yr_avg = (
            self.df.groupby(self.df["date_of_transaction"].dt.year)["transaction_price"]
            .apply(lambda x: np.expm1(x).mean())
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.regplot(
            x=yr_avg.index, y=yr_avg.values, scatter_kws={"s": 50}, line_kws={"color": "red"}, ax=ax
        )
        ax.set(
            title="Average Transaction Price by Year",
            xlabel="Year",
            ylabel="Price (EUR)",
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(int(x), ",")))
        ax.set_xticks(yr_avg.index[::2])
        ax.grid(alpha=0.3)
        self._save_fig(fig, "avg_price_by_year.png")

    def avg_price_by_quarter(self) -> None:
        """Scatter + trend for average price per *year-quarter*."""
        q_avg = (
            self.df.groupby("date_of_transaction_yearquarter")["transaction_price"]
            .apply(lambda x: np.expm1(x).mean())
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.regplot(
            x=np.arange(len(q_avg)),
            y=q_avg.values,
            scatter_kws={"s": 50},
            line_kws={"color": "red"},
            ax=ax,
        )
        ax.set(
            title="Average Transaction Price by Quarter",
            xlabel="Year / Quarter",
            ylabel="Price (EUR)",
        )
        ticks = list(range(0, len(q_avg), 8))
        ax.set_xticks(ticks)
        ax.set_xticklabels(q_avg.index[::8])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(int(x), ",")))
        ax.grid(alpha=0.3)
        self._save_fig(fig, "avg_price_by_quarter.png")

    def avg_price_by_month(self) -> None:
        """Seasonality plot of average price by calendar month."""
        m_avg = (
            self.df.groupby(self.df["date_of_transaction"].dt.month)["transaction_price"]
            .apply(lambda x: np.expm1(x).mean())
        )
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig, ax = plt.subplots(figsize=(12, 6))
        m_avg.plot(kind="bar", ax=ax)
        ax.set(
            title="Average Transaction Price by Calendar Month",
            xlabel="Month",
            ylabel="Price (EUR)",
        )
        ax.set_xticklabels(labels, rotation=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(int(x), ",")))
        for i, v in enumerate(m_avg):
            ax.text(i, v, f"{v/1_000:.0f}k", ha="center", va="bottom")
        self._save_fig(fig, "avg_price_by_month.png")

    # ---------- categorical health-check ------------------------------
    def categorical_rollup_analysis(self, threshold: float = 0.01) -> pd.DataFrame:
        """
        For key categoricals, show total levels and how many fall below
        the relative-frequency threshold.
        """
        cats = [
            "use_type",
            "addition",
            "property_class",
            "property_type",
            "qual_inside",
            "qual_outside",
            "shed",
            "monument",
        ]
        rows = []
        for col in cats:
            vc = self.df[col].value_counts(normalize=True)
            rows.append(
                {
                    "Column": col,
                    "Total": vc.size,
                    f"< {threshold:.0%}": (vc < threshold).sum(),
                }
            )
        return pd.DataFrame(rows)

    # ---------- variance‑inflation diagnostics ------------------------
    def vif_table(self, df_tmp: pd.DataFrame = None , cols: Optional[list[str]] = None, thresh: float = 10.0) -> pd.DataFrame:
        """
        Variance Inflation Factors for a list of predictors.

        The method makes temporary conversions for period / datetime
        columns so they become numeric only within the VIF calculation.
        """
        if df_tmp is None:
            df_tmp = self.df
        if not isinstance(df_tmp, pd.DataFrame):
            raise TypeError("df_tmp must be a pandas DataFrame")
        else:
            df_tmp = df_tmp
        if cols is None:
            cols = [
                "construction_yr", "unit_surface", "gross_volume",
                "parcel_surface", "rooms_nr", "date_of_listing",
                "date_of_listing_month", "date_of_listing_year",
                "date_of_listing_yearquarter",
                "lon", "lat",
                "pc6_price_2y_prior", "pc6_price_6m_prior",
                "global_price_1m_prior", "global_price_6m_prior"
            ]

        tmp = df_tmp[cols].copy()
        # Drop any rows with missing values so correlations are computed on the same sample
        tmp = tmp.dropna()

        if "date_of_listing_yearquarter" in tmp.columns and pd.api.types.is_period_dtype(
            tmp["date_of_listing_yearquarter"]
        ):
            tmp["date_of_listing_yearquarter"] = (
                tmp["date_of_listing_yearquarter"]
                .apply(lambda p: p.ordinal if hasattr(p, "ordinal") else np.nan)
                .astype("float64")
            )
        if "date_of_listing" in tmp.columns and pd.api.types.is_datetime64_any_dtype(
            tmp["date_of_listing"]
        ):
            tmp["date_of_listing"] = tmp["date_of_listing"].map(
                lambda d: d.toordinal() if pd.notna(d) else np.nan
            )

        non_num = tmp.select_dtypes(exclude=[np.number]).columns
        if len(non_num):
            raise TypeError(f"Non-numeric columns present after coercion: {list(non_num)}")

        mat = tmp.astype("float64").values
        vif_vals = [
            variance_inflation_factor(mat, i) for i in range(mat.shape[1])
        ]
        out = (
            pd.Series(vif_vals, index=tmp.columns, name="vif")
            .to_frame()
            .assign(high=lambda d: d["vif"] > thresh)
            .sort_values("vif", ascending=False)
        )
        return out

    # ---------- correlation matrix ------------------------------------
    def correlation_matrix(self) -> None:
        """Lower-triangle heatmap of Pearson correlations."""
        corr = self.df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", ax=ax)
        ax.set_title("Correlation Matrix (lower triangle)")
        self._save_fig(fig, "correlation_matrix.png")

    # ---------- violin + scatter --------------------------------------
    def violin_plots(self) -> None:
        """Violin plots for *property_type* and *qual_inside* vs. price."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.violinplot(data=self.df, x="property_type", y="transaction_price", ax=ax1)
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_title("Price by Property Type")

        sns.violinplot(data=self.df, x="qual_inside", y="transaction_price", ax=ax2)
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_title("Price by Interior Quality")

        plt.tight_layout()
        self._save_fig(fig, "violin_plots.png")

    def scatter_unit_surface(self) -> None:
        """LOWESS-smoothed scatter of surface vs. price by interior quality."""
        qual = ["slecht", "matig", "redelijk", "goed", "uitstekend"]
        fig = sns.lmplot(
            x="unit_surface",
            y="transaction_price",
            hue="qual_inside",
            data=self.df.query("unit_surface <= 750 and qual_inside in @qual"),
            lowess=True,
            hue_order=qual,
            height=6,
            aspect=1.5,
            legend=False,
        )
        plt.legend(loc="lower right")
        plt.title("Unit Surface vs. Transaction Price (LOWESS)")
        self._save_fig(fig.fig, "scatter_unit_surface.png")  # lmplot returns FacetGrid

    def rolling_mean_price(self) -> None:
        """12-month rolling mean of *log-prices* (displayed on EUR scale)."""
        monthly = (
            self.df.set_index("date_of_transaction")
            .resample("M")["transaction_price"]
            .mean()
        )
        roll = np.expm1(monthly.rolling(12).mean())
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(roll.index, roll.values, linewidth=2)
        ax.set(
            title="12-Month Rolling Mean of Transaction Prices",
            xlabel="Year",
            ylabel="Price (EUR)",
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(int(x), ",")))
        ax.grid(alpha=0.3)
        self._save_fig(fig, "rolling_mean_price.png")

    # ---------- interactive maps --------------------------------------
    def make_maps(self, map_type: str = "pc6") -> folium.Map:
        """
        Interactive folium map:

        * map_type='pc6'. coloured PC6 centroids, Gelderland outline.
        * map_type='nuts3': choropleth of NUTS-3 mean prices.
        """
        if map_type not in {"pc6", "nuts3"}:
            raise ValueError("map_type must be 'pc6' or 'nuts3'")

        m = folium.Map()
        pc6_df = (
            self.df.groupby("pc6", as_index=False)
            .agg(lat=("lat", "first"), lon=("lon", "first"), price=("transaction_price", "mean"))
        )
        pc6_df["price"] = np.expm1(pc6_df["price"])

        # bounding box
        m.fit_bounds(
            [
                [pc6_df.lat.min() - 0.1, pc6_df.lon.min() - 0.1],
                [pc6_df.lat.max() + 0.1, pc6_df.lon.max() + 0.1],
            ]
        )

        if map_type == "pc6":
            prov = gpd.read_file(self.root / self.cfg["raw_files"]["provinces_map"])
            folium.GeoJson(
                prov,
                style_function=lambda f: {
                    "fillColor": "#add8e6"
                    if f["properties"]["statnaam"] == "Gelderland"
                    else "#ffffff",
                    "color": "black",
                    "weight": 2,
                    "fillOpacity": 0.5,
                },
            ).add_to(m)

            cmap = branca.colormap.LinearColormap(["green", "red"], vmin=200_000, vmax=800_000)
            cmap.add_to(m)
            for _, row in pc6_df.iterrows():
                folium.CircleMarker(
                    location=(row.lat, row.lon),
                    radius=1,
                    color=cmap(row.price),
                    fill=True,
                    fill_opacity=0.5,
                ).add_to(m)
            self._maybe_save_map(m, "map_pc6.html")
            return m

        # ---------- NUTS-3 --------------------------------------------
        nuts3_geo = gpd.read_file(self.root / self.cfg["raw_files"]["nuts3_map"])
        price_df = (
            self.df.groupby("nuts3_region")["transaction_price"]
            .mean()
            .apply(np.expm1)
            .rename("price")
            .reset_index()
        )
        nuts3 = nuts3_geo.merge(price_df, left_on="nuts3_code", right_on="nuts3_region", how="left")
        cmap = branca.colormap.LinearColormap(
            ["green", "red"],
            vmin=nuts3.price.min(),
            vmax=nuts3.price.max(),
        )
        cmap.add_to(m)
        folium.GeoJson(
            nuts3,
            style_function=lambda feat: {
                "fillColor": cmap(feat["properties"].get("price"))
                if feat["properties"].get("price") is not None
                else "#ffffff",
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["statnaam", "nuts3_code", "price"],
                aliases=["Region:", "NUTS-3:", "Mean price (€):"],
                localize=True,
            ),
        ).add_to(m)
        self._maybe_save_map(m, "map_nuts3.html")
        return m

    # ------------------------------------------------------------------
    # helpers
    @staticmethod
    def _load_config(env_path: Optional[str] = None) -> dict:
        """Load YAML config, walking up from this file if needed."""
        override = os.environ.get("CONFIG_PATH", env_path)
        if override and Path(override).exists():
            return yaml.safe_load(Path(override).read_text())
        for parent in Path(__file__).resolve().parents:
            p = parent / "config.yaml"
            if p.exists():
                return yaml.safe_load(p.read_text())
        raise FileNotFoundError("config.yaml not found.")

    def _save_fig(self, fig: plt.Figure, name: str) -> None:
        """Save fig if ``save_option`` is true, then show."""
        if self.save_option:
            fig.savefig(self.fig_dir / name, dpi=150, bbox_inches="tight")
        plt.show()

    def _maybe_save_map(self, m: folium.Map, name: str) -> None:
        if self.save_option:
            m.save(self.fig_dir / name)