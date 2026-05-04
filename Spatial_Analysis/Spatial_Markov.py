import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

from libpysal.weights import Queen, lag_spatial


warnings.filterwarnings('ignore')


INPUT_CSV = 'Total_Data_with_QSstar.csv'
INPUT_SHP = 'County2023.shp'
OUT_DIR = 'Spatial_Markov_outputs'

os.makedirs(OUT_DIR, exist_ok=True)


CSV_CODE_COL = 'Code'
YEAR_COL = 'Year'
VALUE_COL = 'S_star'

SHP_CODE_COL = 'code'

WEAK_UPPER = 0.2
MEDIUM_UPPER = 0.5
DROP_YEAR = 2013


DUPLICATE_FILE = os.path.join(
    OUT_DIR,
    'QSstar_duplicate_code_year_rows.csv'
)

UNMATCHED_FILE = os.path.join(
    OUT_DIR,
    'QSstar_unmatched_code_year_rows.csv'
)

ISOLATES_FILE = os.path.join(
    OUT_DIR,
    'QSstar_spatial_markov_isolates.csv'
)

PANEL_FILE = os.path.join(
    OUT_DIR,
    'QSstar_spatial_markov_panel.csv'
)

LOCAL_COUNTS_FILE = os.path.join(
    OUT_DIR,
    'QSstar_local_state_counts_by_year.csv'
)

NEIGHBOR_COUNTS_FILE = os.path.join(
    OUT_DIR,
    'QSstar_neighbor_state_counts_by_year.csv'
)

TRANSITION_PAIRS_FILE = os.path.join(
    OUT_DIR,
    'QSstar_spatial_markov_transition_pairs.csv'
)

SUMMARY_FILE = os.path.join(
    OUT_DIR,
    'QSstar_spatial_markov_summary.csv'
)

COUNTS_LONG_FILE = os.path.join(
    OUT_DIR,
    'QSstar_spatial_markov_transition_counts_long.csv'
)

PROBS_LONG_FILE = os.path.join(
    OUT_DIR,
    'QSstar_spatial_markov_transition_probs_long.csv'
)


STATE_ORDER = [
    1,
    2,
    3,
    4
]


STATE_NAME_MAP = {
    1: 'State 1: Non-synergistic / imbalanced',
    2: 'State 2: Weak synergy',
    3: 'State 3: Moderate synergy',
    4: 'State 4: High synergy'
}


NEIGHBOR_STATE_NAME_MAP = {
    1: 'Neighbor State 1: Non-synergistic / imbalanced',
    2: 'Neighbor State 2: Weak synergy',
    3: 'Neighbor State 3: Moderate synergy',
    4: 'Neighbor State 4: High synergy'
}


def clean_code(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.replace('\\.0$', '', regex=True)
        .str.strip()
        .str.zfill(6)
    )


def assign_state(
    x: float,
    weak_upper: float = 0.2,
    medium_upper: float = 0.5
) -> int:
    if pd.isna(x):
        return np.nan

    if x <= 0:
        return 1
    elif x <= weak_upper:
        return 2
    elif x <= medium_upper:
        return 3
    else:
        return 4


def validate_inputs():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f'Input CSV file not found: {INPUT_CSV}'
        )

    if not os.path.exists(INPUT_SHP):
        raise FileNotFoundError(
            f'Input SHP file not found: {INPUT_SHP}'
        )


def build_transition_matrix(sub_df: pd.DataFrame):
    counts_num = pd.crosstab(
        sub_df['local_state_t'],
        sub_df['local_state_t1']
    ).reindex(
        index=STATE_ORDER,
        columns=STATE_ORDER,
        fill_value=0
    )

    row_sums = counts_num.sum(axis=1).replace(
        0,
        np.nan
    )

    probs_num = counts_num.div(
        row_sums,
        axis=0
    )

    counts_show = counts_num.copy()
    probs_show = probs_num.copy()

    counts_show.index = [
        STATE_NAME_MAP[i]
        for i in counts_show.index
    ]

    counts_show.columns = [
        STATE_NAME_MAP[i]
        for i in counts_show.columns
    ]

    probs_show.index = [
        STATE_NAME_MAP[i]
        for i in probs_show.index
    ]

    probs_show.columns = [
        STATE_NAME_MAP[i]
        for i in probs_show.columns
    ]

    return counts_num, probs_num, counts_show, probs_show


validate_inputs()


df = pd.read_csv(
    INPUT_CSV,
    encoding='utf-8-sig'
)

gdf = gpd.read_file(INPUT_SHP)


required_csv_cols = [
    CSV_CODE_COL,
    YEAR_COL,
    VALUE_COL
]

missing_csv_cols = [
    c for c in required_csv_cols
    if c not in df.columns
]

if missing_csv_cols:
    raise ValueError(
        f'CSV is missing required columns: {missing_csv_cols}'
    )


if SHP_CODE_COL not in gdf.columns:
    raise ValueError(
        f'SHP is missing the administrative code field: {SHP_CODE_COL}'
    )


df[CSV_CODE_COL] = clean_code(df[CSV_CODE_COL])
gdf[SHP_CODE_COL] = clean_code(gdf[SHP_CODE_COL])

df[YEAR_COL] = pd.to_numeric(
    df[YEAR_COL],
    errors='coerce'
)

df[VALUE_COL] = pd.to_numeric(
    df[VALUE_COL],
    errors='coerce'
)


df = df[
    [
        CSV_CODE_COL,
        YEAR_COL,
        VALUE_COL
    ]
].copy()


gdf = gdf[
    [
        SHP_CODE_COL,
        'geometry'
    ]
].copy()


gdf = gdf[
    gdf.geometry.notna()
    & ~gdf.geometry.is_empty
].copy()


shp_dups = gdf[
    gdf.duplicated(
        subset=[
            SHP_CODE_COL
        ],
        keep=False
    )
].copy()


if not shp_dups.empty:
    raise ValueError(
        f'Duplicate administrative codes exist in SHP field {SHP_CODE_COL}. '
        f'Please handle them before rerunning the script.'
    )


df = df.dropna(
    subset=[
        CSV_CODE_COL,
        YEAR_COL,
        VALUE_COL
    ]
).copy()


df = df[
    df[YEAR_COL] != DROP_YEAR
].copy()


df[YEAR_COL] = df[YEAR_COL].astype(int)


dups = df[
    df.duplicated(
        subset=[
            CSV_CODE_COL,
            YEAR_COL
        ],
        keep=False
    )
].copy()


if not dups.empty:
    dups.to_csv(
        DUPLICATE_FILE,
        index=False,
        encoding='utf-8-sig'
    )

    raise ValueError(
        f'Duplicate Code-Year combinations were found and exported to: {DUPLICATE_FILE}\n'
        f'Please handle duplicate values before rerunning the script.'
    )


unmatched = df[
    ~df[CSV_CODE_COL].isin(gdf[SHP_CODE_COL])
].copy()


if not unmatched.empty:
    unmatched.to_csv(
        UNMATCHED_FILE,
        index=False,
        encoding='utf-8-sig'
    )


df = df[
    df[CSV_CODE_COL].isin(gdf[SHP_CODE_COL])
].copy()


df = df.sort_values(
    [
        CSV_CODE_COL,
        YEAR_COL
    ]
).reset_index(drop=True)


years = sorted(
    df[YEAR_COL]
    .unique()
    .tolist()
)


if len(years) == 0:
    raise ValueError(
        'After removing 2013 and invalid records, no usable yearly data remain.'
    )


panel_list = []
isolates_list = []


for year in years:
    year_df = df[
        df[YEAR_COL] == year
    ].copy()

    gdf_year = gdf.merge(
        year_df,
        left_on=SHP_CODE_COL,
        right_on=CSV_CODE_COL,
        how='inner'
    ).copy()

    if gdf_year.empty:
        continue

    gdf_year = gdf_year.reset_index(drop=True)

    w0 = Queen.from_dataframe(
        gdf_year,
        use_index=True
    )

    isolated_idx = [
        i for i in gdf_year.index
        if len(w0.neighbors.get(i, [])) == 0
    ]

    if isolated_idx:
        iso_codes = gdf_year.loc[
            isolated_idx,
            CSV_CODE_COL
        ].tolist()

        for code in iso_codes:
            isolates_list.append({
                'Code': code,
                'Year': year,
                'Reason': 'No neighbors'
            })

    valid = gdf_year.drop(
        index=isolated_idx
    ).copy().reset_index(drop=True)

    if len(valid) < 2:
        continue

    w = Queen.from_dataframe(
        valid,
        use_index=True
    )

    w.transform = 'r'

    y = valid[VALUE_COL].astype(float).values

    ws = lag_spatial(
        w,
        y
    )

    valid['WS_star'] = ws

    valid['n_neighbors'] = [
        len(w.neighbors[i])
        for i in valid.index
    ]

    valid['local_state'] = valid[VALUE_COL].apply(
        lambda x: assign_state(
            x,
            weak_upper=WEAK_UPPER,
            medium_upper=MEDIUM_UPPER
        )
    )

    valid['local_state_name'] = valid['local_state'].map(
        STATE_NAME_MAP
    )

    valid['neighbor_state'] = valid['WS_star'].apply(
        lambda x: assign_state(
            x,
            weak_upper=WEAK_UPPER,
            medium_upper=MEDIUM_UPPER
        )
    )

    valid['neighbor_state_name'] = valid['neighbor_state'].map(
        NEIGHBOR_STATE_NAME_MAP
    )

    panel_list.append(
        valid[
            [
                CSV_CODE_COL,
                YEAR_COL,
                VALUE_COL,
                'WS_star',
                'n_neighbors',
                'local_state',
                'local_state_name',
                'neighbor_state',
                'neighbor_state_name'
            ]
        ].copy()
    )


if len(panel_list) == 0:
    raise ValueError(
        'After removing isolated regions, all years have no valid samples. '
        'Spatial Markov analysis cannot continue.'
    )


panel = pd.concat(
    panel_list,
    ignore_index=True
)


panel = panel.sort_values(
    [
        CSV_CODE_COL,
        YEAR_COL
    ]
).reset_index(drop=True)


isolates_df = pd.DataFrame(isolates_list)


if isolates_df.empty:
    isolates_df = pd.DataFrame(
        columns=[
            'Code',
            'Year',
            'Reason'
        ]
    )


isolates_df.to_csv(
    ISOLATES_FILE,
    index=False,
    encoding='utf-8-sig'
)


panel.to_csv(
    PANEL_FILE,
    index=False,
    encoding='utf-8-sig'
)


local_counts = pd.crosstab(
    panel[YEAR_COL],
    panel['local_state']
).reindex(
    columns=STATE_ORDER,
    fill_value=0
)


local_counts.columns = [
    STATE_NAME_MAP[i]
    for i in local_counts.columns
]


local_counts.index.name = 'Year'


local_counts.to_csv(
    LOCAL_COUNTS_FILE,
    encoding='utf-8-sig'
)


neighbor_counts = pd.crosstab(
    panel[YEAR_COL],
    panel['neighbor_state']
).reindex(
    columns=STATE_ORDER,
    fill_value=0
)


neighbor_counts.columns = [
    NEIGHBOR_STATE_NAME_MAP[i]
    for i in neighbor_counts.columns
]


neighbor_counts.index.name = 'Year'


neighbor_counts.to_csv(
    NEIGHBOR_COUNTS_FILE,
    encoding='utf-8-sig'
)


panel['Year_next'] = panel.groupby(CSV_CODE_COL)[YEAR_COL].shift(-1)
panel['S_star_next'] = panel.groupby(CSV_CODE_COL)[VALUE_COL].shift(-1)
panel['WS_star_next'] = panel.groupby(CSV_CODE_COL)['WS_star'].shift(-1)
panel['local_state_next'] = panel.groupby(CSV_CODE_COL)['local_state'].shift(-1)
panel['local_state_name_next'] = panel.groupby(CSV_CODE_COL)['local_state_name'].shift(-1)
panel['neighbor_state_next'] = panel.groupby(CSV_CODE_COL)['neighbor_state'].shift(-1)
panel['neighbor_state_name_next'] = panel.groupby(CSV_CODE_COL)['neighbor_state_name'].shift(-1)


transitions = panel[
    panel['Year_next'] == panel[YEAR_COL] + 1
].copy()


if transitions.empty:
    raise ValueError(
        'No valid adjacent-year transition samples were formed for t -> t+1.'
    )


transitions = transitions.rename(
    columns={
        YEAR_COL: 'Year_t',
        VALUE_COL: 'S_star_t',
        'WS_star': 'WS_star_t',
        'local_state': 'local_state_t',
        'local_state_name': 'local_state_name_t',
        'neighbor_state': 'neighbor_state_t',
        'neighbor_state_name': 'neighbor_state_name_t'
    }
)


transitions['Year_t1'] = transitions['Year_next'].astype(int)
transitions['local_state_t1'] = transitions['local_state_next'].astype(int)
transitions['neighbor_state_t1'] = transitions['neighbor_state_next'].astype(int)


transitions = transitions[
    [
        CSV_CODE_COL,
        'Year_t',
        'Year_t1',
        'S_star_t',
        'S_star_next',
        'WS_star_t',
        'WS_star_next',
        'local_state_t',
        'local_state_name_t',
        'local_state_t1',
        'local_state_name_next',
        'neighbor_state_t',
        'neighbor_state_name_t',
        'neighbor_state_t1',
        'neighbor_state_name_next'
    ]
].copy()


transitions = transitions.rename(
    columns={
        'S_star_next': 'S_star_t1',
        'WS_star_next': 'WS_star_t1',
        'local_state_name_next': 'local_state_name_t1',
        'neighbor_state_name_next': 'neighbor_state_name_t1'
    }
)


transitions.to_csv(
    TRANSITION_PAIRS_FILE,
    index=False,
    encoding='utf-8-sig'
)


summary_rows = []
counts_long_rows = []
probs_long_rows = []


for ns in STATE_ORDER:
    sub = transitions[
        transitions['neighbor_state_t'] == ns
    ].copy()

    counts_num, probs_num, counts_show, probs_show = build_transition_matrix(
        sub
    )

    counts_path = os.path.join(
        OUT_DIR,
        f'QSstar_spatial_markov_transition_counts_neighbor_state{ns}.csv'
    )

    probs_path = os.path.join(
        OUT_DIR,
        f'QSstar_spatial_markov_transition_probs_neighbor_state{ns}.csv'
    )

    counts_show.to_csv(
        counts_path,
        encoding='utf-8-sig'
    )

    probs_show.to_csv(
        probs_path,
        encoding='utf-8-sig'
    )

    for i in STATE_ORDER:
        for j in STATE_ORDER:
            counts_long_rows.append({
                'neighbor_state_t': ns,
                'neighbor_state_name_t': NEIGHBOR_STATE_NAME_MAP[ns],
                'from_state': i,
                'from_state_name': STATE_NAME_MAP[i],
                'to_state': j,
                'to_state_name': STATE_NAME_MAP[j],
                'count': counts_num.loc[i, j]
            })

            probs_long_rows.append({
                'neighbor_state_t': ns,
                'neighbor_state_name_t': NEIGHBOR_STATE_NAME_MAP[ns],
                'from_state': i,
                'from_state_name': STATE_NAME_MAP[i],
                'to_state': j,
                'to_state_name': STATE_NAME_MAP[j],
                'probability': probs_num.loc[i, j]
            })

    total_n_under_neighbor = len(sub)

    for s in STATE_ORDER:
        n_from = counts_num.loc[s].sum()

        stay_prob = probs_num.loc[s, s]

        up_prob = probs_num.loc[
            s,
            [
                j for j in STATE_ORDER
                if j > s
            ]
        ].sum()

        down_prob = probs_num.loc[
            s,
            [
                j for j in STATE_ORDER
                if j < s
            ]
        ].sum()

        summary_rows.append({
            'neighbor_state_t': ns,
            'neighbor_state_name_t': NEIGHBOR_STATE_NAME_MAP[ns],
            'N_total_transitions_under_this_neighbor_state': total_n_under_neighbor,
            'from_state': s,
            'from_state_name': STATE_NAME_MAP[s],
            'N_from_state': n_from,
            'stay_probability': stay_prob,
            'upward_probability': up_prob,
            'downward_probability': down_prob
        })


summary_df = pd.DataFrame(summary_rows)
counts_long_df = pd.DataFrame(counts_long_rows)
probs_long_df = pd.DataFrame(probs_long_rows)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding='utf-8-sig'
)


counts_long_df.to_csv(
    COUNTS_LONG_FILE,
    index=False,
    encoding='utf-8-sig'
)


probs_long_df.to_csv(
    PROBS_LONG_FILE,
    index=False,
    encoding='utf-8-sig'
)


for ns in STATE_ORDER:
    pass