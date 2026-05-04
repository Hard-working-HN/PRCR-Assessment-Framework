import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

from libpysal.weights import Queen
from esda import Moran, Moran_Local


warnings.filterwarnings('ignore', category=UserWarning)


shp_path = 'County2023.shp'
csv_path = 'Total_Data_with_QSstar.csv'

shp_code_col = 'code'
csv_code_col = 'Code'
year_col = 'Year'
value_col = 'S_star'

alpha = 0.05
permutations = 999


def clean_code(series):
    return (
        series
        .astype(str)
        .str.replace('\\.0$', '', regex=True)
        .str.strip()
        .str.zfill(6)
    )


county_gdf = gpd.read_file(shp_path)
qs_df = pd.read_csv(csv_path)


county_gdf[shp_code_col] = clean_code(county_gdf[shp_code_col])
qs_df[csv_code_col] = clean_code(qs_df[csv_code_col])


qs_df[year_col] = pd.to_numeric(
    qs_df[year_col],
    errors='coerce'
)

qs_df[value_col] = pd.to_numeric(
    qs_df[value_col],
    errors='coerce'
)


qs_df = qs_df[
    [
        csv_code_col,
        year_col,
        value_col
    ]
].copy()


qs_df = qs_df.dropna(
    subset=[
        value_col
    ]
).copy()


qs_df = qs_df[
    qs_df[year_col] != 2013
].copy()


county_gdf = county_gdf[
    county_gdf.geometry.notna()
    & ~county_gdf.geometry.is_empty
].copy()


years = sorted(
    qs_df[year_col]
    .dropna()
    .unique()
    .astype(int)
    .tolist()
)


global_results = []
local_results_all = []


for year in years:
    year_df = qs_df[
        qs_df[year_col] == year
    ].copy()

    gdf_year = county_gdf[
        [
            shp_code_col,
            'geometry'
        ]
    ].merge(
        year_df,
        left_on=shp_code_col,
        right_on=csv_code_col,
        how='inner'
    )

    if gdf_year.empty:
        continue

    local_full = gdf_year[
        [
            shp_code_col,
            value_col
        ]
    ].copy()

    local_full[year_col] = year
    local_full['local_moran_I'] = np.nan
    local_full['local_moran_p'] = np.nan
    local_full['local_moran_z'] = np.nan
    local_full['quadrant'] = np.nan
    local_full['cluster_type'] = np.nan
    local_full['status'] = 'Not computed'

    valid = gdf_year.copy()

    if valid[value_col].nunique() <= 1:
        global_results.append({
            'Year': year,
            'N_used': len(valid),
            'Global_Moran_I': np.nan,
            'Expected_I': np.nan,
            'Z_score': np.nan,
            'P_value': np.nan,
            'Status': 'Constant values, skipped'
        })

        local_full['status'] = 'Constant values, skipped'
        local_results_all.append(local_full)

        continue

    w = Queen.from_dataframe(
        valid,
        use_index=True
    )

    isolated_idx = [
        idx for idx in valid.index
        if len(w.neighbors[idx]) == 0
    ]

    isolated_codes = valid.loc[
        isolated_idx,
        shp_code_col
    ].tolist()

    if len(isolated_codes) > 0:
        pass

    if isolated_idx:
        valid = valid.drop(
            index=isolated_idx
        ).copy()

    if len(valid) < 2 or valid[value_col].nunique() <= 1:
        global_results.append({
            'Year': year,
            'N_used': len(valid),
            'Global_Moran_I': np.nan,
            'Expected_I': np.nan,
            'Z_score': np.nan,
            'P_value': np.nan,
            'Status': 'Insufficient valid observations after removing isolates'
        })

        local_full.loc[
            local_full[shp_code_col].isin(isolated_codes),
            'status'
        ] = 'No neighbors'

        local_full.loc[
            ~local_full[shp_code_col].isin(isolated_codes),
            'status'
        ] = 'Insufficient valid observations after removing isolates'

        local_results_all.append(local_full)

        continue

    w = Queen.from_dataframe(
        valid,
        use_index=True
    )

    w.transform = 'r'

    moran_global = Moran(
        valid[value_col],
        w,
        permutations=permutations
    )

    global_results.append({
        'Year': year,
        'N_used': len(valid),
        'Global_Moran_I': moran_global.I,
        'Expected_I': moran_global.EI,
        'Z_score': moran_global.z_sim,
        'P_value': moran_global.p_sim,
        'Status': 'Computed'
    })

    moran_local = Moran_Local(
        valid[value_col],
        w,
        permutations=permutations
    )

    local_calc = valid[
        [
            shp_code_col,
            value_col
        ]
    ].copy()

    local_calc[year_col] = year
    local_calc['local_moran_I'] = moran_local.Is
    local_calc['local_moran_p'] = moran_local.p_sim
    local_calc['local_moran_z'] = moran_local.z_sim
    local_calc['quadrant'] = moran_local.q

    quadrant_map = {
        1: 'High-High',
        2: 'Low-High',
        3: 'Low-Low',
        4: 'High-Low'
    }

    local_calc['cluster_type'] = np.where(
        local_calc['local_moran_p'] < alpha,
        local_calc['quadrant'].map(quadrant_map),
        'Not significant'
    )

    local_calc['status'] = 'Computed'

    local_full = local_full.merge(
        local_calc[
            [
                shp_code_col,
                'local_moran_I',
                'local_moran_p',
                'local_moran_z',
                'quadrant',
                'cluster_type',
                'status'
            ]
        ],
        on=shp_code_col,
        how='left',
        suffixes=('', '_new')
    )

    for col in [
        'local_moran_I',
        'local_moran_p',
        'local_moran_z',
        'quadrant',
        'cluster_type',
        'status'
    ]:
        local_full[col] = local_full[f'{col}_new'].combine_first(
            local_full[col]
        )

        local_full.drop(
            columns=[
                f'{col}_new'
            ],
            inplace=True
        )

    if isolated_codes:
        local_full.loc[
            local_full[shp_code_col].isin(isolated_codes),
            'status'
        ] = 'No neighbors'

    local_results_all.append(local_full)


global_results_df = pd.DataFrame(global_results)

local_results_df = (
    pd.concat(
        local_results_all,
        ignore_index=True
    )
    if local_results_all
    else pd.DataFrame()
)


global_csv = 'QSstar_global_moran_results.csv'
local_csv = 'QSstar_local_moran_results.csv'


global_results_df.to_csv(
    global_csv,
    index=False,
    encoding='utf-8-sig'
)

local_results_df.to_csv(
    local_csv,
    index=False,
    encoding='utf-8-sig'
)