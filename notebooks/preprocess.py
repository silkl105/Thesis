from __future__ import annotations

"""Pre‑processing utilities used throughout the project.

This module cleans the raw data extract and exposes two public
symbols:

``clean``
    In‑place cleaning function returning the optimised ``DataFrame``.

``CONFIG``
    A dict‑like view on *config.yaml* located at the repository root.

The implementation is self‑contained: all file paths are resolved
relative to the project root so that the code continues to work when the
repo is cloned elsewhere (GitHub, CI, production, …).
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import numpy as np
import pandas as pd
import yaml
import geopandas as gpd

# Configuration helpers
def _load_config() -> Dict[str, Any]:
    """Search upward for *config.yaml* and load it with *PyYAML*."""

    # Allow overriding config path via environment variable
    config_env = os.environ.get("CONFIG_PATH")
    if config_env:
        cfg = Path(config_env)
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"CONFIG_PATH set to '{config_env}', but file does not exist.")

    for parent in Path(__file__).resolve().parents:
        cfg = parent / "config.yaml"
        if cfg.exists():
            with cfg.open("r", encoding="utf‑8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "Could not locate 'config.yaml'. Place it in the repository root "
        "or set the CONFIG_PATH environment variable."
    )

CONFIG: Dict[str, Any] = _load_config()


# Utility helpers
def _repair_integer64(series: pd.Series) -> pd.Series:
    """Repair R's integer64 columns that got coerced to float64.

    The algorithm re‑interprets the underlying bits: when the absolute
    value is <1e‑100 it is the 1e‑321 sub‑normal
    encoding used by bit64.
    """

    vals = series.to_numpy(dtype="float64")
    mask = ~np.isnan(vals)
    # The 1e‑321 sub‑normals end up as absurdly tiny floats
    if (np.abs(vals[mask]) < 1e-100).all():
        ints = vals.view(np.uint64)
        # Wrong byte‑order → byteswap
        if ints[mask].mean() > 10_000:
            ints = vals.byteswap().view(np.uint64)
        out = pd.Series(ints, index=series.index, name=series.name).astype("Int64")
        out[~mask] = pd.NA
        return out
    # Otherwise we assume they were genuine floats, round them
    return series.round().astype("Int64")

def _construction_period(year: Optional[float | int]) -> Optional[str]:
    """Map construction_yr to period buckets."""

    if pd.isna(year):
        return None
    year = int(round(year))
    if year < 1906:
        return "<1906"
    if year <= 1930:
        return "1906-1930"
    if year <= 1944:
        return "1931-1944"
    if year <= 1959:
        return "1945-1959"
    if year <= 1970:
        return "1960-1970"
    if year <= 1980:
        return "1971-1980"
    if year <= 1990:
        return "1981-1990"
    if year <= 2000:
        return "1991-2000"
    if year <= 2010:
        return "2001-2010"
    return "2011-2020"

# Public API
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and optimize the transaction DataFrame for analysis.

    This function performs:
    - Dtype conversion and memory optimization.
    - Repair of integer64 artifacts from R imports.
    - Imputation of `construction_per` based on `construction_yr` buckets.
    - Imputation of `parcel_surface` by PC6-level median with global fallback.
    - Log1p transformation of price columns to reduce skew while preserving zeros.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned and optimized DataFrame ready for feature engineering.
    """
    # 1. dtype optimisation
    int_columns = [
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

    cat_columns = [
        "use_type",
        "property_class",
        "property_type",
        "construction_per",
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

    # Integer‑like columns (except construction_yr/unit_surface → handled later)
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].round().astype("Int64")

    # Repair the integer64 artefacts.
    df["construction_yr"] = _repair_integer64(df["construction_yr"])
    df["unit_surface"] = _repair_integer64(df["unit_surface"])

    # Categorical columns (construction_per temporarily left as‑is)
    for col in cat_columns:
        if col != "construction_per":
            df[col] = df[col].astype("category")

    # Date columns
    df["date_of_listing"] = pd.to_datetime(df["date_of_listing"], errors="coerce")
    df["date_of_transaction"] = pd.to_datetime(df["date_of_transaction"], errors="coerce")

    # 2. Impute construction_per from construction_yr buckets (vectorized)
    df["construction_per"] = df["construction_per"].fillna(df["construction_yr"].apply(_construction_period))
    df["construction_per"] = df["construction_per"].fillna("Unknown").astype("category")

    # 3. Impute parcel_surface – postcode median fallback global median
    overall_parcel_median = round(df["parcel_surface"].median())
    
    # Function to safely calculate group median, falling back to overall median
    def safe_median(x):
        if x.isna().all():  # If group is all NAs
            return overall_parcel_median
        group_median = x.median()  # Calculate group median
        return round(group_median if pd.notna(group_median) else overall_parcel_median)

    pc6_medians = df.groupby("pc6", observed=True)["parcel_surface"].transform(safe_median)
    
    df.loc[df["parcel_surface"].isna(), "parcel_surface"] = pc6_medians[df["parcel_surface"].isna()]

    # 4. Transform monetary values – log1p
    for col in ["transaction_price", "initial_list_price", "last_list_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Convert negatives to NaN with warning
        neg_mask = df[col] < 0
        if neg_mask.any():
            print(f"Warning: Found {neg_mask.sum()} negative values in {col}. Converting to NaN.")
        df[col] = np.where(neg_mask, np.nan, df[col])
        # Apply log1p transformation to reduce right skew while preserving zeros
        df[col] = np.log1p(df[col])

    return df



def feature_engineering(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Engineer geographic, provincial, and listing‑based lagged price features.

    This function:
      1. Loads PC6 shapefile and extract NUTS3/NUTS2 regions and centroids.
      2. Consolidate small NUTS3 regions ('NL364'→'NL224', 'NL421'→'NL416') to reduce sparsity.
      3. Maps NUTS2 regions to Dutch provinces (e.g. NL35 → Utrecht).
      4. Merges geographic and province fields into the DataFrame.
      5. Extracts date_of_transaction and date_of_listing components: year and quarter.
      6. Builds listing_period_q (quarter) and listing_period_m (month) from date_of_listing.
      7. Computes mean transaction_price lagged by 1 month and 1 and 4 quarters at PC6, NUTS3, NUTS2, and global levels, with fallback PC6→NUTS3→NUTS2→global.
      8. Drops any rows missing any of the six new lagged features to ensure a complete dataset for OLS regression.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transactions DataFrame (no missing critical fields).
    filepath : str
        Path to the PC6 shapefile.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with:
          - `province`, `lon`, `lat`, `date_of_*_yearquarter`
          - `pc6_price_m1_prior`, `pc6_price_q1_prior`, `pc6_price_q4_prior`
          - `global_price_m1_prior`, `global_price_q1_prior`, `global_price_q4_prior`
        and without any of the six lag features missing.

    Raises
    ------
    ValueError
        If any PC6 fails to map to a region or centroid.
    """
    # Load spatial reference data
    pc_gdf = gpd.read_file(filepath, engine="pyogrio")
    pc_gdf["pc6"] = pc_gdf["POSTCODE"].str.replace(r"\s+", "", regex=True)
    pc_gdf["nuts3_region"] = (
        pc_gdf["NUTS3_2024"]
        .astype(str)
        .str.replace(r"[ '\\\\\"]+", "", regex=True)
    )
    # Consolidate very small NUTS3 regions into neighbors
    pc_gdf["nuts3_region"] = pc_gdf["nuts3_region"].replace({
        "NL364": "NL224",
        "NL421": "NL416",
    }).astype("category")
    pc_gdf["nuts2_region"] = pc_gdf["nuts3_region"].str[:4].astype("category")
    # Extract centroids in WGS84
    pc_gdf = pc_gdf.to_crs(epsg=4326)
    pc_gdf["lon"] = pc_gdf.geometry.x
    pc_gdf["lat"] = pc_gdf.geometry.y

    centroids = pc_gdf[["pc6", "nuts3_region", "nuts2_region", "lon", "lat"]]
    df = df.merge(centroids, on="pc6", how="left")

    # Map NUTS2 region to province
    province_map = {
        "NL13": "Drenthe",    "NL23": "Flevoland",
        "NL12": "Friesland",  "NL22": "Gelderland",
        "NL11": "Groningen",  "NL42": "Limburg",
        "NL41": "North Brabant", "NL32": "North Holland",
        "NL21": "Overijssel",   "NL33": "South Holland",
        "NL35": "Utrecht",      "NL34": "Zeeland",
    }
    df["province"] = df["nuts2_region"].map(province_map).astype("category")

    # Temporal features
    df["date_of_transaction_year"] = df["date_of_transaction"].dt.year
    df["date_of_listing_year"] = df["date_of_listing"].dt.year
    # Create PeriodDtype quarters for transaction and listing (kept as Period for lag arithmetic)
    df["date_of_transaction_yearquarter"] = df["date_of_transaction"].dt.to_period("Q")
    df["date_of_listing_yearquarter"] = df["date_of_listing"].dt.to_period("Q")
    # Listing periods used for referencing lags
    df["listing_period_q"] = df["date_of_listing_yearquarter"]
    df["listing_period_m"] = df["date_of_listing"].dt.to_period("M")

    # Quarterly means
    pc6_q_pm    = df.groupby(["pc6", "listing_period_q"], observed=True)["transaction_price"].mean()
    nuts3_q_pm  = df.groupby(["nuts3_region", "listing_period_q"], observed=True)["transaction_price"].mean()
    nuts2_q_pm  = df.groupby(["nuts2_region", "listing_period_q"], observed=True)["transaction_price"].mean()
    global_q_pm = df.groupby("listing_period_q")["transaction_price"].mean()

    # Monthly means
    pc6_m_pm    = df.groupby(["pc6", "listing_period_m"], observed=True)["transaction_price"].mean()
    nuts3_m_pm  = df.groupby(["nuts3_region", "listing_period_m"], observed=True)["transaction_price"].mean()
    nuts2_m_pm  = df.groupby(["nuts2_region", "listing_period_m"], observed=True)["transaction_price"].mean()
    global_m_pm = df.groupby("listing_period_m")["transaction_price"].mean()

    # Build lag keys
    df["_key_pc6_q1"]   = list(zip(df["pc6"], df["listing_period_q"] - 1))
    df["_key_pc6_q4"]   = list(zip(df["pc6"], df["listing_period_q"] - 4))
    df["_key_nuts3_q1"] = list(zip(df["nuts3_region"], df["listing_period_q"] - 1))
    df["_key_nuts3_q4"] = list(zip(df["nuts3_region"], df["listing_period_q"] - 4))
    df["_key_nuts2_q1"] = list(zip(df["nuts2_region"], df["listing_period_q"] - 1))
    df["_key_nuts2_q4"] = list(zip(df["nuts2_region"], df["listing_period_q"] - 4))
    df["_key_pc6_m1"]   = list(zip(df["pc6"], df["listing_period_m"] - 1))
    df["_key_nuts3_m1"] = list(zip(df["nuts3_region"], df["listing_period_m"] - 1))
    df["_key_nuts2_m1"] = list(zip(df["nuts2_region"], df["listing_period_m"] - 1))

    # Lagged features with PC6→NUTS3→NUTS2→global fallback
    df["pc6_price_m1_prior"] = (
        df["_key_pc6_m1"].map(pc6_m_pm)
          .fillna(df["_key_nuts3_m1"].map(nuts3_m_pm))
          .fillna(df["_key_nuts2_m1"].map(nuts2_m_pm))
          .fillna((df["listing_period_m"] - 1).map(global_m_pm))
    )
    df["pc6_price_q1_prior"] = (
        df["_key_pc6_q1"].map(pc6_q_pm)
          .fillna(df["_key_nuts3_q1"].map(nuts3_q_pm))
          .fillna(df["_key_nuts2_q1"].map(nuts2_q_pm))
          .fillna((df["listing_period_q"] - 1).map(global_q_pm))
    )
    df["pc6_price_q4_prior"] = (
        df["_key_pc6_q4"].map(pc6_q_pm)
          .fillna(df["_key_nuts3_q4"].map(nuts3_q_pm))
          .fillna(df["_key_nuts2_q4"].map(nuts2_q_pm))
          .fillna((df["listing_period_q"] - 4).map(global_q_pm))
    )

    df["global_price_m1_prior"] = (df["listing_period_m"] - 1).map(global_m_pm)
    df["global_price_q1_prior"] = (df["listing_period_q"] - 1).map(global_q_pm)
    df["global_price_q4_prior"] = (df["listing_period_q"] - 4).map(global_q_pm)

    # Clean up helper keys and listing period fields
    df.drop(columns=[c for c in df.columns if c.startswith("_key_")], inplace=True)
    df.drop(columns=["listing_period_q", "listing_period_m"], inplace=True)

    # Discard any rows missing the six lag features
    df = df.dropna(subset=[
        "pc6_price_m1_prior", "pc6_price_q1_prior", "pc6_price_q4_prior",
        "global_price_m1_prior", "global_price_q1_prior", "global_price_q4_prior"
    ])

    # Optionally cast yearquarter fields to categorical for downstream
    df["date_of_transaction_yearquarter"] = df["date_of_transaction_yearquarter"].astype("category")
    df["date_of_listing_yearquarter"] = df["date_of_listing_yearquarter"].astype("category")

    return df

__all__ = ["CONFIG", "clean", "feature_engineering"]