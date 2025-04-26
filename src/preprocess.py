from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyreadr
import yaml


class DataProcessor:
    """
    End-to-end pipeline for processing raw housing dataset.

    Stages
    ------
    1. Raw ingest: read the RData dump.
    2. Clean: dtype fixes, parcel imputation, postcode prep.
    3. Feature-engineering: geographic joins + rolling-price lags.
    4. Final prep: collapse sparse categories, write two modelling
       datasets (XGBoost and OLS).

    Notes
    -----
    * Category collapses are based on similarity in mean logged prices,
      not on counts alone.
    * Rolling means use left-closed windows so the listing day itself
      never leaks into the look-back features.
    """

    # construction ----------------------------------------------------
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        save_option: bool = False,
    ) -> None:
        self.config: Dict[str, Any] = self._load_config(config_path)
        root = Path(__file__).resolve().parents[1]

        self.raw_path: Path = root / self.config["raw_path"]
        self.proc_path: Path = root / self.config["processed_path"]
        self.rdata_file: Path = root / self.config["raw_files"]["transactions"]
        self.shp_file: Path = root / self.config["raw_files"]["shapefile"]

        self.proc_path.mkdir(parents=True, exist_ok=True)
        self.save_option = save_option

    # public driver ---------------------------------------------------
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the full pipeline and optionally store the raw-feature parquet.

        Returns
        -------
        df_raw, df_xgb, df_lin
            * df_raw: after feature engineering but before category collapse.
            * df_xgb: ready for tree models.
            * df_lin: one-hot / ordinal-encoded for OLS.
        """
        df = self.load_transactions()
        df = self.clean(df)
        df_feat = self.feature_engineering(df)
        df_xgb, df_lin = self.clean_2(df_feat.copy())

        if self.save_option:
            self.save_processed(df_feat, "processed_data.parquet")
            self.save_processed(df_xgb, "processed_data_xgb.parquet")
            self.save_processed(df_lin, "processed_data_lin.parquet")
        else:
            return df_feat, df_xgb, df_lin

    # stage 1 – ingest ---------------------------------------------------
    def load_transactions(self) -> pd.DataFrame:
        """Read the **.rdata** bundle produced by NVM’s export script."""
        res = pyreadr.read_r(self.rdata_file)
        return next(iter(res.values()))

    # stage 2 – cleaning -------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
        """
        Fix dtypes, impute small gaps, create postcode cuts, and
        log-transform price columns.
        """
        # ----- integers -------------------------------------------------
        int_cols = [
            "unit_id",
            "property_id",
            "housenumber",
            "gross_volume",
            "parcel_surface",
            "rooms_nr",
            "initial_list_price",
            "last_list_price",
            "transaction_price",
            "id",
            "duration",
        ]
        self._coerce_ints(df, int_cols)

        # R’s integer64 artefacts
        df["construction_yr"] = self._repair_integer64(df["construction_yr"])
        df["unit_surface"] = self._repair_integer64(df["unit_surface"])

        # ----- categoricals --------------------------------------------
        cats = [
            "use_type",
            "property_class",
            "property_type",
            "qual_inside",
            "qual_outside",
            "shed",
            "monument",
            "place",
            "province",
            "addition",
            "street",
            "pc6",
        ]
        df[cats] = df[cats].astype("category")

        # Construction period bucket
        df["construction_per"] = (
            df["construction_per"]
            .fillna(df["construction_yr"].apply(self._construction_period))
            .fillna("Unknown")
            .astype("category")
        )

        # ----- dates ---------------------------------------------------
        df["date_of_listing"] = pd.to_datetime(df["date_of_listing"], errors="coerce")
        df["date_of_transaction"] = pd.to_datetime(df["date_of_transaction"], errors="coerce")

        # ----- parcel imputation ---------------------------------------
        global_med = round(df["parcel_surface"].median())
        by_pc6 = (
            df.groupby("pc6", observed=True)["parcel_surface"]
            .transform(lambda x: round(x.median()) if x.notna().any() else global_med)
        )
        df["parcel_surface"] = df["parcel_surface"].fillna(by_pc6)

        # ----- postcode granularity ------------------------------------
        df["pc4"] = df["pc6"].str[:4].astype("category")
        df["pc5"] = df["pc6"].str[:5].astype("category")

        # ----- log prices ----------------------------------------------
        for col in ["transaction_price", "initial_list_price", "last_list_price"]:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[col] = np.log1p(vals.clip(lower=0))

        return df

    # stage 3 – feature engineering -------------------------------------
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        * Merge PC6 centroids / NUTS regions.
        * Create hierarchical rolling-mean price features
          (PC6→PC5→PC4→NUTS3→NUTS2→Global) for 6m and 2y.
        * Add global rolling means (1m and 6m).
        """
        df = self._attach_geography(df)
        df = self._add_temporal_columns(df)
        df = self._compute_local_rollups(df)
        df = self._compute_global_rollups(df)
        df = df.dropna(
            subset=[
                "pc6_price_6m_prior",
                "pc6_price_2y_prior",
                "global_price_1m_prior",
                "global_price_6m_prior",
            ]
        )
        return df

    # stage 4 – final modelling prep -------------------------------------
    def clean_2(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collapse sparse categories (why each collapse is done is noted
        inline), then emit two parquet files:

        * processed_data_xgb.parquet: minimal feature engineering.
        * processed_data_lin.parquet: one-hot & ordinal encoded.
        """
        df = df.copy()
        
        # ---------- category collapses ---------------------------------
        df["use_type"] = pd.Series(
            np.where(df["use_type"] == "woonfunctie", "woonfunctie", "other"),
            dtype="category"
        )

        collapse_map = {
            # Property class: merge small yet price-similar classes
            "property_class": {
                "grachtenpand": "woonboerderij",
                "landhuis": "villa",
            },
            # Quality: merge in-between categories based on mean price ladders
            "qual_inside": {
                "goed tot uitstekend": "uitstekend",
                "slecht tot matig": "slecht",
                "matig tot redelijk": "redelijk",
                "redelijk tot goed": "redelijk",
            },
            "qual_outside": {
                "goed tot uitstekend": "uitstekend",
                "slecht tot matig": "slecht",
                "matig tot redelijk": "redelijk",
                "redelijk tot goed": "redelijk",
                "onbekend [MP]": "slecht",
            },
            # Shed: price driven mainly by material, not siting
            "shed": {
                "aangebouwd hout": "wood",
                "vrijstaand hout": "wood",
                "box": "wood",
                "aangebouwd steen": "stone",
                "vrijstaand steen": "stone",
                "inpandig": "indoor",
                "geen": "none",
            },
        }

        for col, mapper in collapse_map.items():
            df[col] = (
                df[col].astype(str).replace(mapper).astype("category")
            )

        # ---------- drop columns not to be used by models -------------
        drop_cols = [
            "unit_id",
            "property_id",
            "street",
            "housenumber",
            "addition",
            "initial_list_price",
            "last_list_price",
            "date_of_transaction",
            "duration",
            "id",
            "date_of_transaction_yearquarter",
            "date_of_transaction_year",
            "date_of_transaction_month"
        ]
        df_xgb = df.drop(columns=drop_cols).copy()

        # -- drop temporal duplicates --------------------------------------
        drop_temp = ["date_of_listing_year", "date_of_listing_yearquarter", "date_of_listing_month"]
        df_xgb.drop(columns=drop_temp, inplace=True, errors="ignore")

        # -- global lags: keep only 6-month level + diff -------------------
        df_xgb["global_price_1m_minus_6m"] = (
            df_xgb["global_price_1m_prior"] - df_xgb["global_price_6m_prior"]
        )
        df_xgb.drop(columns=["global_price_1m_prior"], inplace=True)

        # -- local lags: keep only 2-year level + diff -------------------
        df_xgb["pc6_price_6m_minus_2y"] = (
            df_xgb["pc6_price_6m_prior"] - df_xgb["pc6_price_2y_prior"]
        )
        df_xgb.drop(columns=["global_price_1m_prior"], inplace=True)

        # ---------- OLS encoding --------------------------------------
        df_lin = df_xgb.copy()

        df_lin = pd.get_dummies(
            df_lin,
            columns=[
                "use_type",
                "property_class",
                "property_type",
                "shed",
                "monument",
                "province",
                "nuts3_region",
            ],
            drop_first=True,
        )

        period_order = [
            "<1906",
            "1906-1930",
            "1931-1944",
            "1945-1959",
            "1960-1970",
            "1971-1980",
            "1981-1990",
            "1991-2000",
            "2001-2010",
            "2011-2020",
            "Unknown",
        ]
        df_lin["construction_per"] = pd.Categorical(
            df_lin["construction_per"], categories=period_order, ordered=True
        ).codes

        qual_levels = ["slecht", "matig", "redelijk", "goed", "uitstekend"]
        for qcol in ["qual_inside", "qual_outside"]:
            df_lin[qcol] = pd.Categorical(
                df_lin[qcol], categories=qual_levels, ordered=True
            ).codes

        df_lin["date_of_listing"] = df_lin["date_of_listing"].apply(lambda d: d.toordinal())

        return df_xgb, df_lin

    # basic I/O helpers -------------------------------------------------
    def save_processed(self, df: pd.DataFrame, filename: str) -> None:
        """Write *df* to the processed folder in parquet (PyArrow) format."""
        df.to_parquet(self.proc_path / filename, engine="pyarrow", index=False)


    # -------------------------------------------------------------------
    #                           INTERNALS                               
    @staticmethod
    def _load_config(explicit: Optional[str]) -> Dict[str, Any]:
        """Find and parse *config.yaml* or honour ``CONFIG_PATH``."""
        path = os.environ.get("CONFIG_PATH", explicit)
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        for parent in Path(__file__).resolve().parents:
            cfg = parent / "config.yaml"
            if cfg.exists():
                with cfg.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError("config.yaml not found.")

    # ----- dtype helpers ------------------------------------------------
    @staticmethod
    def _coerce_ints(df: pd.DataFrame, cols: list[str]) -> None:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].round().astype("Int64")

    @staticmethod
    def _repair_integer64(s: pd.Series) -> pd.Series:
        """
        Decode R “integer64” columns that land in Python as
        tiny-looking ``float64`` values (≈9 × 10⁻³²¹).

        Context
        -------
        *construction_yr* and *unit_surface* were stored in the R
        workspace as `bit64::integer64`, whose raw 64‑bit integer
        payloads are simply re‑interpreted by R as doubles.
        When the data are round‑tripped through **pyreadr** they
        arrive in pandas as ``float64`` with *identical* underlying
        bits.

        All we need to do is view those bits as signed 64‑bit
        integers – **no byte‑swapping or heuristics required** – and
        then cast to pandas’ nullable ``Int64``.
        """
        # View the raw float64 buffer as int64
        arr_f = s.to_numpy(dtype="float64")
        nan_mask = np.isnan(arr_f)
        raw = arr_f.view(np.int64)

        # Build a nullable integer Series, restoring missing values
        fixed = pd.Series(raw, index=s.index, name=s.name, dtype="Int64")
        fixed[nan_mask] = pd.NA
        return fixed

    @staticmethod
    def _construction_period(year: float | int | None) -> str | None:
        """Map numeric year to a decennial bucket."""
        if pd.isna(year):
            return None
        yr = int(year)
        bins = [
            (1906, "<1906"),
            (1930, "1906-1930"),
            (1944, "1931-1944"),
            (1959, "1945-1959"),
            (1970, "1960-1970"),
            (1980, "1971-1980"),
            (1990, "1981-1990"),
            (2000, "1991-2000"),
            (2010, "2001-2010"),
            (9999, "2011-2020"),
        ]
        return next(label for upper, label in bins if yr <= upper)

    # ----- geo / temporal -----------------------------------------------
    def _attach_geography(self, df: pd.DataFrame) -> pd.DataFrame:
        pc = gpd.read_file(self.shp_file, engine="pyogrio")
        pc["pc6"] = pc["POSTCODE"].str.replace(r"\s+", "", regex=True)

        # Read raw region codes as strings
        pc["nuts3_region"] = (
            pc["NUTS3_2024"]
            .astype(str)
            .str.replace(r"[ '\\\"]+", "", regex=True)
            .replace({"NL364": "NL224", "NL421": "NL416"})
        )
        pc["nuts2_region"] = pc["nuts3_region"].str[:4]

        pc = pc.to_crs(epsg=4326)
        pc["lon"] = pc.geometry.x
        pc["lat"] = pc.geometry.y

        df = df.merge(
            pc[["pc6", "nuts3_region", "nuts2_region", "lon", "lat"]],
            on="pc6",
            how="left",
        )

        df["nuts3_region"] = df["nuts3_region"].astype("category")
        df["nuts2_region"] = df["nuts2_region"].astype("category")

        prov_map = {
            "NL13": "Drenthe",
            "NL23": "Flevoland",
            "NL12": "Friesland",
            "NL22": "Gelderland",
            "NL11": "Groningen",
            "NL42": "Limburg",
            "NL41": "North Brabant",
            "NL32": "North Holland",
            "NL21": "Overijssel",
            "NL33": "South Holland",
            "NL35": "Utrecht",
            "NL34": "Zeeland",
        }
        df["province"] = df["nuts2_region"].map(prov_map).astype("category")
        return df

    @staticmethod
    def _add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
        df["date_of_transaction_month"] = df["date_of_transaction"].dt.month
        df["date_of_transaction_yearquarter"] = df["date_of_transaction"].dt.to_period("Q")
        df["date_of_transaction_year"] = df["date_of_transaction"].dt.year

        df["date_of_listing_month"] = df["date_of_listing"].dt.month
        df["date_of_listing_yearquarter"] = df["date_of_listing"].dt.to_period("Q")
        df["date_of_listing_year"] = df["date_of_listing"].dt.year
        # Encode seasonality with the first Fourier harmonic
        theta = 2 * np.pi * df["date_of_listing"].dt.month / 12
        df["month_sin"] = np.sin(theta)
        df["month_cos"] = np.cos(theta)
        
        return df

    # ----- rolling means ------------------------------------------------
    def _compute_local_rollups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hierarchical fallback (≥3 obs) over PC6→PC5→PC4→NUTS3→NUTS2→Global
        for 6-month and 2-year windows.
        """
        df["_row_id"] = np.arange(len(df))

        slim = df[
            [
                "_row_id",
                "transaction_price",
                "pc6",
                "pc5",
                "pc4",
                "nuts3_region",
                "nuts2_region",
                "date_of_transaction",
                "date_of_listing",
            ]
        ]

        con = duckdb.connect(":memory:")
        con.register("tbl", slim)

        def tpl(level: str, field: str, win: str) -> str:
            # Determine alias suffix: e.g. "2 years" -> "2y", "6 months" -> "6m"
            num, unit = win.split()
            suffix = f"{num}{unit[0]}"
            return f"""
                SELECT l._row_id,
                       AVG(t.transaction_price) AS {level}_{suffix},
                       COUNT(*)                AS n
                FROM tbl AS l
                LEFT JOIN tbl AS t
                       ON t.{field} = l.{field}
                      AND t.date_of_transaction
                          BETWEEN l.date_of_listing - INTERVAL '{win}'
                              AND l.date_of_listing - INTERVAL '1 day'
                GROUP BY l._row_id
            """

        sql = f"""
        WITH
            pc6_2y AS ({tpl('pc6', 'pc6', "2 years")}),
            pc5_2y AS ({tpl('pc5', 'pc5', "2 years")}),
            pc4_2y AS ({tpl('pc4', 'pc4', "2 years")}),
            n3_2y  AS ({tpl('n3',  'nuts3_region', "2 years")}),
            n2_2y  AS ({tpl('n2',  'nuts2_region', "2 years")}),

            g_2y AS (
                SELECT l._row_id,
                       AVG(t.transaction_price) AS g_2y,
                       COUNT(*)                AS n
                FROM tbl AS l
                LEFT JOIN tbl AS t
                       ON t.date_of_transaction BETWEEN
                          l.date_of_listing - INTERVAL '2 years'
                          AND l.date_of_listing - INTERVAL '1 day'
                GROUP BY l._row_id
            ),

            pc6_6m AS ({tpl('pc6', 'pc6', "6 months")}),
            pc5_6m AS ({tpl('pc5', 'pc5', "6 months")}),
            pc4_6m AS ({tpl('pc4', 'pc4', "6 months")}),
            n3_6m  AS ({tpl('n3',  'nuts3_region', "6 months")}),
            n2_6m  AS ({tpl('n2',  'nuts2_region', "6 months")}),

            g_6m AS (
                SELECT l._row_id,
                       AVG(t.transaction_price) AS g_6m,
                       COUNT(*)                AS n
                FROM tbl AS l
                LEFT JOIN tbl AS t
                       ON t.date_of_transaction BETWEEN
                          l.date_of_listing - INTERVAL '6 months'
                          AND l.date_of_listing - INTERVAL '1 day'
                GROUP BY l._row_id
            )

        SELECT
            l._row_id,
            CASE
                WHEN pc6_2y.n >= 3 THEN pc6_2y.pc6_2y
                WHEN pc5_2y.n >= 3 THEN pc5_2y.pc5_2y
                WHEN pc4_2y.n >= 3 THEN pc4_2y.pc4_2y
                WHEN n3_2y.n  >= 3 THEN n3_2y.n3_2y
                WHEN n2_2y.n  >= 3 THEN n2_2y.n2_2y
                ELSE g_2y.g_2y
            END AS pc6_price_2y_prior,

            CASE
                WHEN pc6_6m.n >= 3 THEN pc6_6m.pc6_6m
                WHEN pc5_6m.n >= 3 THEN pc5_6m.pc5_6m
                WHEN pc4_6m.n >= 3 THEN pc4_6m.pc4_6m
                WHEN n3_6m.n  >= 3 THEN n3_6m.n3_6m
                WHEN n2_6m.n  >= 3 THEN n2_6m.n2_6m
                ELSE g_6m.g_6m
            END AS pc6_price_6m_prior
        FROM tbl l
        LEFT JOIN pc6_2y USING (_row_id)
        LEFT JOIN pc5_2y USING (_row_id)
        LEFT JOIN pc4_2y USING (_row_id)
        LEFT JOIN n3_2y  USING (_row_id)
        LEFT JOIN n2_2y  USING (_row_id)
        LEFT JOIN g_2y   USING (_row_id)
        LEFT JOIN pc6_6m USING (_row_id)
        LEFT JOIN pc5_6m USING (_row_id)
        LEFT JOIN pc4_6m USING (_row_id)
        LEFT JOIN n3_6m  USING (_row_id)
        LEFT JOIN n2_6m  USING (_row_id)
        LEFT JOIN g_6m   USING (_row_id)
        ORDER BY _row_id
        """

        roll = con.execute(sql).df()
        df["pc6_price_2y_prior"] = roll["pc6_price_2y_prior"]
        df["pc6_price_6m_prior"] = roll["pc6_price_6m_prior"]
        df.drop(columns="_row_id", inplace=True)
        return df

    @staticmethod
    def _compute_global_rollups(df: pd.DataFrame) -> pd.DataFrame:
        daily = (
            df.groupby(df["date_of_transaction"].dt.normalize(), observed=False)
            .agg(sum=("transaction_price", "sum"), cnt=("transaction_price", "count"))
            .sort_index()
        )

        def roll(days: int) -> pd.Series:
            s = daily["sum"].rolling(f"{days}D", closed="left").sum()
            c = daily["cnt"].rolling(f"{days}D", closed="left").sum()
            return s / c

        g1 = roll(30).rename("global_price_1m_prior")
        g6 = roll(180).rename("global_price_6m_prior")

        all_days = pd.date_range(
            df["date_of_transaction"].min().normalize(),
            df["date_of_transaction"].max().normalize(),
            freq="D",
        )
        df["global_price_1m_prior"] = df["date_of_listing"].dt.normalize().map(
            g1.reindex(all_days)
        )
        df["global_price_6m_prior"] = df["date_of_listing"].dt.normalize().map(
            g6.reindex(all_days)
        )

        return df