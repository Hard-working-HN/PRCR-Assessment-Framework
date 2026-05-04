import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point


warnings.filterwarnings('ignore', category=UserWarning)


CSV_PATH = 'Total_Data_with_QSstar_merged.csv'
SHP_PATH = 'County2023.shp'
OUTPUT_DIR = './step1_outputs'

SHP_CODE_COL = 'code'
MAIN_CODE_COL = 'Code'
YEAR_COL = 'Year'
CII_COL = 'S_star'
TRAJ_COL = 'trajectory_class'
STATE_COL = 'cii_state'
FRONTIER_COL = 'Frontier_OOF'
GAP_COL = 'Gap_OOF'

TRAJECTORY_MAP = {
    0: 'trajectory_0',
    1: 'trajectory_1',
    2: 'trajectory_2',
    3: 'trajectory_3'
}

STATE_MAP = {
    0: 'state_0',
    1: 'state_1',
    2: 'state_2',
    3: 'state_3'
}

PROJECTED_CRS = 'EPSG:6933'

YEAR_MIN = 2014
YEAR_MAX = 2022


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_code(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    try:
        s_num = int(float(s))
        return str(s_num).zfill(6)
    except Exception:
        pass

    digits = ''.join((ch for ch in s if ch.isdigit()))

    if digits == '':
        return np.nan

    return digits.zfill(6)


def minmax_scale(series: pd.Series, reverse: bool = False) -> pd.Series:
    s = series.astype(float).copy()
    s = s.replace([np.inf, -np.inf], np.nan)

    valid = s.dropna()

    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    s_min = valid.min()
    s_max = valid.max()

    if np.isclose(s_max, s_min):
        out = pd.Series(0.5, index=series.index, dtype=float)
    else:
        out = (s - s_min) / (s_max - s_min)

    if reverse:
        out = 1 - out

    return out


def calc_trend_slope(years: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(years) & np.isfinite(values)

    x = years[mask]
    y = values[mask]

    if len(x) < 2:
        return np.nan

    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def calc_improve_ratio(values: np.ndarray) -> float:
    vals = pd.Series(values).dropna().values

    if len(vals) < 2:
        return np.nan

    diff = np.diff(vals)

    return float((diff > 0).mean())


def build_stability_metrics(
    df_region: pd.DataFrame,
    year_col: str,
    cii_col: str
) -> pd.Series:
    tmp = df_region.sort_values(year_col).copy()

    years = tmp[year_col].to_numpy(dtype=float)
    vals = tmp[cii_col].to_numpy(dtype=float)

    cii_mean = np.nanmean(vals)
    cii_std = np.nanstd(vals, ddof=0)
    cii_range = np.nanmax(vals) - np.nanmin(vals) if np.isfinite(vals).any() else np.nan

    if np.isfinite(cii_mean) and (not np.isclose(cii_mean, 0)):
        cii_cv = cii_std / abs(cii_mean)
    else:
        cii_cv = np.nan

    slope = calc_trend_slope(years, vals)
    improve_ratio = calc_improve_ratio(vals)

    return pd.Series({
        'cii_mean': cii_mean,
        'cii_std': cii_std,
        'cii_cv': cii_cv,
        'cii_range': cii_range,
        'trend_slope': slope,
        'improve_ratio': improve_ratio
    })


def add_class_labels(
    df: pd.DataFrame,
    traj_col: str,
    state_col: str
) -> pd.DataFrame:
    out = df.copy()

    if traj_col in out.columns:
        out['trajectory_label'] = out[traj_col].map(TRAJECTORY_MAP)

    if state_col in out.columns:
        out['cii_state_label'] = out[state_col].map(STATE_MAP)

    return out


def build_adjacency_edges(
    gdf_proj: gpd.GeoDataFrame,
    code_col: str
) -> pd.DataFrame:
    gdf_proj = gdf_proj[[code_col, 'geometry']].copy().reset_index(drop=True)

    sindex = gdf_proj.sindex

    edges = []

    geoms = gdf_proj.geometry.values
    codes = gdf_proj[code_col].values

    for i, geom in enumerate(geoms):
        candidate_idx = list(sindex.intersection(geom.bounds))

        for j in candidate_idx:
            if j <= i:
                continue

            geom_j = geoms[j]

            if geom.touches(geom_j):
                dist_km = geom.centroid.distance(geom_j.centroid) / 1000.0
                edges.append((codes[i], codes[j], dist_km))

    edge_df = pd.DataFrame(
        edges,
        columns=[
            'Code_1',
            'Code_2',
            'centroid_distance_km'
        ]
    )

    return edge_df


ensure_dir(OUTPUT_DIR)


df = pd.read_csv(CSV_PATH)


required_cols = [
    MAIN_CODE_COL,
    YEAR_COL,
    CII_COL,
    TRAJ_COL,
    STATE_COL,
    FRONTIER_COL,
    GAP_COL
]

missing = [
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(f'The main table is missing required columns: {missing}')


df[MAIN_CODE_COL] = df[MAIN_CODE_COL].apply(normalize_code)
df = df[df[MAIN_CODE_COL].notna()].copy()

df = df[
    (df[YEAR_COL] >= YEAR_MIN)
    & (df[YEAR_COL] <= YEAR_MAX)
].copy()

df = df.sort_values(
    [MAIN_CODE_COL, YEAR_COL]
).reset_index(drop=True)

df = add_class_labels(
    df,
    TRAJ_COL,
    STATE_COL
)


rename_map = {
    MAIN_CODE_COL: 'Code',
    YEAR_COL: 'Year',
    CII_COL: 'CII_norm',
    FRONTIER_COL: 'Frontier',
    GAP_COL: 'Gap'
}

df = df.rename(columns=rename_map)


gdf = gpd.read_file(SHP_PATH)


if SHP_CODE_COL not in gdf.columns:
    raise ValueError(f'The code column was not found in the SHP file: {SHP_CODE_COL}')


gdf[SHP_CODE_COL] = gdf[SHP_CODE_COL].apply(normalize_code)
gdf = gdf[gdf[SHP_CODE_COL].notna()].copy()

main_codes = set(df['Code'].unique())

gdf = gdf[gdf[SHP_CODE_COL].isin(main_codes)].copy()

if gdf[SHP_CODE_COL].duplicated().any():
    gdf = gdf.dissolve(
        by=SHP_CODE_COL,
        as_index=False
    )


if gdf.crs is None:
    raise ValueError(
        'The SHP file has no coordinate reference system information. '
        'Please assign the correct CRS to the SHP file first.'
    )


gdf_proj = gdf.to_crs(PROJECTED_CRS).copy()

gdf_proj['area_km2'] = gdf_proj.geometry.area / 1000000.0
gdf_proj['centroid_proj'] = gdf_proj.geometry.centroid
gdf_proj['repr_point_proj'] = gdf_proj.geometry.representative_point()

centroid_gs = gpd.GeoSeries(
    gdf_proj['centroid_proj'],
    crs=PROJECTED_CRS
).to_crs(gdf.crs)

repr_gs = gpd.GeoSeries(
    gdf_proj['repr_point_proj'],
    crs=PROJECTED_CRS
).to_crs(gdf.crs)

gdf_proj['centroid_lon'] = centroid_gs.x
gdf_proj['centroid_lat'] = centroid_gs.y
gdf_proj['repr_lon'] = repr_gs.x
gdf_proj['repr_lat'] = repr_gs.y


region_static = gdf_proj[
    [
        SHP_CODE_COL,
        'area_km2',
        'centroid_lon',
        'centroid_lat',
        'repr_lon',
        'repr_lat'
    ]
].copy()

region_static = region_static.rename(
    columns={
        SHP_CODE_COL: 'Code'
    }
)

region_static = region_static.sort_values('Code').reset_index(drop=True)


stability_df = (
    df
    .groupby('Code', group_keys=False)
    .apply(
        lambda x: build_stability_metrics(
            x,
            'Year',
            'CII_norm'
        )
    )
    .reset_index()
)


latest_year = int(df['Year'].max())

df_latest = df[df['Year'] == latest_year].copy()

df_latest = (
    df_latest
    .sort_values(['Code', 'Year'])
    .drop_duplicates(
        subset=['Code'],
        keep='first'
    )
)


region_profile = df_latest.merge(
    stability_df,
    on='Code',
    how='left'
)

region_profile = region_profile.merge(
    region_static,
    on='Code',
    how='left'
)


region_profile['level_score'] = minmax_scale(
    region_profile['CII_norm'],
    reverse=False
)

region_profile['frontier_closeness_score'] = minmax_scale(
    region_profile['Gap'],
    reverse=True
)

region_profile['low_volatility_score'] = minmax_scale(
    region_profile['cii_cv'],
    reverse=True
)

region_profile['trend_score'] = minmax_scale(
    region_profile['trend_slope'],
    reverse=False
)

region_profile['improve_ratio_score'] = minmax_scale(
    region_profile['improve_ratio'],
    reverse=False
)


region_profile['stability_score'] = (
    0.4 * region_profile['low_volatility_score'].fillna(0)
    + 0.3 * region_profile['trend_score'].fillna(0)
    + 0.3 * region_profile['improve_ratio_score'].fillna(0)
)


region_profile['source_score_base'] = (
    0.4 * region_profile['level_score'].fillna(0)
    + 0.3 * region_profile['frontier_closeness_score'].fillna(0)
    + 0.3 * region_profile['stability_score'].fillna(0)
)


region_profile['gap_score'] = minmax_scale(
    region_profile['Gap'],
    reverse=False
)

region_profile['low_level_need_score'] = minmax_scale(
    region_profile['CII_norm'],
    reverse=True
)


region_profile['demand_score_base'] = (
    0.6 * region_profile['gap_score'].fillna(0)
    + 0.4 * region_profile['low_level_need_score'].fillna(0)
)


node_year_prepared = (
    df
    .merge(
        stability_df,
        on='Code',
        how='left'
    )
    .merge(
        region_static,
        on='Code',
        how='left'
    )
)


node_year_prepared = node_year_prepared.merge(
    region_profile[
        [
            'Code',
            'source_score_base',
            'demand_score_base',
            'stability_score'
        ]
    ],
    on='Code',
    how='left'
)


adj_edges = build_adjacency_edges(
    gdf_proj=gdf_proj.rename(
        columns={
            SHP_CODE_COL: 'Code'
        }
    ),
    code_col='Code'
)


adj_edges_rev = adj_edges.rename(
    columns={
        'Code_1': 'Code_2',
        'Code_2': 'Code_1'
    }
)


adj_edges_bidirectional = pd.concat(
    [
        adj_edges,
        adj_edges_rev
    ],
    ignore_index=True
)


adj_edges_bidirectional = (
    adj_edges_bidirectional
    .sort_values(['Code_1', 'Code_2'])
    .reset_index(drop=True)
)


node_year_out = os.path.join(
    OUTPUT_DIR,
    'node_year_prepared.csv'
)

region_static_out = os.path.join(
    OUTPUT_DIR,
    'region_static_from_shp.csv'
)

region_profile_out = os.path.join(
    OUTPUT_DIR,
    'region_profile_step1.csv'
)

adj_edges_out = os.path.join(
    OUTPUT_DIR,
    'adjacency_edges_step1.csv'
)


node_year_prepared.to_csv(
    node_year_out,
    index=False,
    encoding='utf-8-sig'
)

region_static.to_csv(
    region_static_out,
    index=False,
    encoding='utf-8-sig'
)

region_profile.to_csv(
    region_profile_out,
    index=False,
    encoding='utf-8-sig'
)

adj_edges_bidirectional.to_csv(
    adj_edges_out,
    index=False,
    encoding='utf-8-sig'
)