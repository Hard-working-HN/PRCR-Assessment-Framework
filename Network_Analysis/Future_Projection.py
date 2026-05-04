import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings('ignore')


STEP1_DIR = './step1_outputs'
OUTPUT_DIR = './step4_outputs'

NODE_YEAR_PATH = os.path.join(STEP1_DIR, 'node_year_prepared.csv')
ADJ_PATH = os.path.join(STEP1_DIR, 'adjacency_edges_step1.csv')
STATIC_PATH = os.path.join(STEP1_DIR, 'region_static_from_shp.csv')

TRAIN_YEARS = list(range(2014, 2023))
FUTURE_YEARS = list(range(2023, 2031))

TOP_K = 8
K_NEAREST = 30
MIN_DISTANCE_KM = 1.0
PRED_PROB_THRESHOLD = 0.6
TREND_DAMPING = 0.7

ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
DELTA = 0.5

MATCH_W_ECON = 0.2
MATCH_W_LAND = 0.2
MATCH_W_ACTIVITY = 0.2
MATCH_W_POLICY = 0.2
MATCH_W_COMPLEMENT = 0.2

USE_ADJ_BONUS = True
ADJ_BONUS = 1.05

RANDOM_STATE = 42
SAVE_YEARLY_PAIR_FEATURES = False

ECON_VARS = [
    'GDP',
    'POP',
    'PIR',
    'PIU',
    'TSS',
    'GEF',
    'GRF'
]

LAND_VARS = [
    'Cropland',
    'Urban',
    'Bare_areas',
    'Forest',
    'Grassland',
    'Wetland',
    'Water_bodies'
]

ACTIVITY_VARS = [
    'Light',
    'Road',
    'NIA',
    'GIA',
    'VAP',
    'VAF'
]

POLICY_VARS = [
    'LCCP',
    'ECERFP',
    'ZWC',
    'TNP'
]

TRAJ_CANDIDATES = [
    'trajectory_class',
    'traj_class'
]

STATE_CANDIDATES = [
    'cii_state'
]

CII_CANDIDATES = [
    'CII_norm'
]

GAP_CANDIDATES = [
    'Gap'
]

STABILITY_CANDIDATES = [
    'stability_score'
]


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def pick_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(f'Candidate columns not found: {candidates}')

    return None


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


def haversine_array(lat1, lon1, lat2, lon2):
    R = 6371.0088

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def safe_fill_mid(df, cols):
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).fillna(0.5)

    return out


def topk_flag_by_source(df, weight_col, k):
    tmp = df[['source', 'target', weight_col]].copy()

    tmp = tmp.sort_values(
        ['source', weight_col],
        ascending=[True, False]
    )

    tmp['rank_in_source'] = (
        tmp
        .groupby('source')[weight_col]
        .rank(method='first', ascending=False)
    )

    tmp['strong_edge_flag'] = (
        tmp['rank_in_source'] <= k
    ).astype(int)

    return tmp[
        [
            'source',
            'target',
            'rank_in_source',
            'strong_edge_flag'
        ]
    ]


def build_candidate_pairs(static_df, adj_df, k_nearest=30):
    nodes = static_df[
        [
            'Code',
            'centroid_lon',
            'centroid_lat'
        ]
    ].drop_duplicates().copy()

    nodes = nodes.sort_values('Code').reset_index(drop=True)

    coords_rad = np.radians(
        nodes[
            [
                'centroid_lat',
                'centroid_lon'
            ]
        ].to_numpy()
    )

    nn = NearestNeighbors(
        n_neighbors=min(k_nearest + 1, len(nodes)),
        metric='haversine'
    )

    nn.fit(coords_rad)

    distances, indices = nn.kneighbors(coords_rad)

    nearest_records = []
    codes = nodes['Code'].tolist()

    for i, src in enumerate(codes):
        for _, j in zip(distances[i, 1:], indices[i, 1:]):
            tgt = codes[j]
            nearest_records.append((src, tgt))

    nearest_df = pd.DataFrame(
        nearest_records,
        columns=[
            'source',
            'target'
        ]
    ).drop_duplicates()

    adj_use = adj_df[
        [
            'Code_1',
            'Code_2'
        ]
    ].copy()

    adj_use = adj_use.rename(
        columns={
            'Code_1': 'source',
            'Code_2': 'target'
        }
    )

    adj_use['source'] = adj_use['source'].astype(str).str.zfill(6)
    adj_use['target'] = adj_use['target'].astype(str).str.zfill(6)

    adj_use = adj_use[
        adj_use['source'] != adj_use['target']
    ].drop_duplicates()

    adj_rev = adj_use.rename(
        columns={
            'source': 'target',
            'target': 'source'
        }
    )

    adj_use = pd.concat(
        [
            adj_use,
            adj_rev
        ],
        ignore_index=True
    ).drop_duplicates()

    pairs = pd.concat(
        [
            nearest_df,
            adj_use
        ],
        ignore_index=True
    ).drop_duplicates()

    pairs = pairs[
        pairs['source'] != pairs['target']
    ].copy()

    lon_map = dict(
        zip(
            nodes['Code'],
            nodes['centroid_lon']
        )
    )

    lat_map = dict(
        zip(
            nodes['Code'],
            nodes['centroid_lat']
        )
    )

    pairs['source_lon'] = pairs['source'].map(lon_map)
    pairs['source_lat'] = pairs['source'].map(lat_map)
    pairs['target_lon'] = pairs['target'].map(lon_map)
    pairs['target_lat'] = pairs['target'].map(lat_map)

    lat1 = np.radians(pairs['source_lat'].to_numpy())
    lon1 = np.radians(pairs['source_lon'].to_numpy())
    lat2 = np.radians(pairs['target_lat'].to_numpy())
    lon2 = np.radians(pairs['target_lon'].to_numpy())

    pairs['distance_km'] = haversine_array(
        lat1,
        lon1,
        lat2,
        lon2
    )

    pairs['distance_km'] = pairs['distance_km'].clip(
        lower=MIN_DISTANCE_KM
    )

    adj_set = set(
        zip(
            adj_use['source'],
            adj_use['target']
        )
    )

    pairs['is_adjacent'] = pairs.apply(
        lambda r: 1 if (r['source'], r['target']) in adj_set else 0,
        axis=1
    )

    pairs = pairs[
        [
            'source',
            'target',
            'distance_km',
            'is_adjacent'
        ]
    ].copy()

    return pairs


def prepare_year_node_features(
    node_df_year,
    cii_col,
    gap_col,
    stability_col,
    econ_cols,
    land_cols,
    activity_cols,
    policy_cols,
    traj_col=None,
    state_col=None
):
    df = node_df_year.copy()

    for c in econ_cols + land_cols + activity_cols + policy_cols:
        if c in df.columns:
            df[c] = minmax_scale(df[c], reverse=False)

    df['level_score'] = minmax_scale(
        df[cii_col],
        reverse=False
    )

    df['frontier_closeness_score'] = minmax_scale(
        df[gap_col],
        reverse=True
    )

    if stability_col in df.columns:
        df['stability_score_dyn'] = (
            df[stability_col]
            .astype(float)
            .fillna(df[stability_col].median())
        )

        df['stability_score_dyn'] = minmax_scale(
            df['stability_score_dyn'],
            reverse=False
        )
    else:
        df['stability_score_dyn'] = 0.5

    df['source_score_dyn'] = (
        0.4 * df['level_score'].fillna(0)
        + 0.3 * df['frontier_closeness_score'].fillna(0)
        + 0.3 * df['stability_score_dyn'].fillna(0)
    )

    df['gap_score_dyn'] = minmax_scale(
        df[gap_col],
        reverse=False
    )

    df['low_level_need_score_dyn'] = minmax_scale(
        df[cii_col],
        reverse=True
    )

    df['demand_score_dyn'] = (
        0.6 * df['gap_score_dyn'].fillna(0)
        + 0.4 * df['low_level_need_score_dyn'].fillna(0)
    )

    keep_cols = [
        'Code',
        'Year',
        cii_col,
        gap_col,
        'source_score_dyn',
        'demand_score_dyn'
    ]

    keep_cols += econ_cols + land_cols + activity_cols + policy_cols

    if traj_col is not None and traj_col in df.columns:
        keep_cols.append(traj_col)

    if state_col is not None and state_col in df.columns:
        keep_cols.append(state_col)

    if (
        stability_col is not None
        and stability_col in df.columns
        and stability_col not in keep_cols
    ):
        keep_cols.append(stability_col)

    return df[keep_cols].copy()


def compute_similarity_from_merged(
    merged_df,
    cols,
    prefix_s='s_',
    prefix_t='t_'
):
    if len(cols) == 0:
        return np.full(len(merged_df), np.nan)

    s_cols = [
        prefix_s + c
        for c in cols
    ]

    t_cols = [
        prefix_t + c
        for c in cols
    ]

    temp = merged_df[s_cols + t_cols].copy()
    temp = safe_fill_mid(temp, s_cols + t_cols)

    xs = temp[s_cols].to_numpy(dtype=float)
    xt = temp[t_cols].to_numpy(dtype=float)

    dist = np.sqrt(
        np.sum(
            (xs - xt) ** 2,
            axis=1
        )
    )

    sim = 1.0 / (1.0 + dist)

    return sim


def build_pair_features_for_year(
    year,
    year_node_df,
    candidate_pairs,
    cii_col,
    gap_col,
    econ_cols,
    land_cols,
    activity_cols,
    policy_cols,
    traj_col=None,
    state_col=None,
    current_edge_override=None
):
    src_df = year_node_df.copy().add_prefix('s_')
    tgt_df = year_node_df.copy().add_prefix('t_')

    pair = candidate_pairs.copy()

    pair = pair.merge(
        src_df,
        left_on='source',
        right_on='s_Code',
        how='left'
    )

    pair = pair.merge(
        tgt_df,
        left_on='target',
        right_on='t_Code',
        how='left'
    )

    pair['econ_sim'] = compute_similarity_from_merged(
        pair,
        econ_cols
    )

    pair['land_sim'] = compute_similarity_from_merged(
        pair,
        land_cols
    )

    pair['activity_sim'] = compute_similarity_from_merged(
        pair,
        activity_cols
    )

    pair['policy_sim'] = compute_similarity_from_merged(
        pair,
        policy_cols
    )

    pair['source_score'] = pair['s_source_score_dyn']
    pair['demand_score'] = pair['t_demand_score_dyn']

    pair['source_cii'] = pair[f's_{cii_col}']
    pair['target_cii'] = pair[f't_{cii_col}']

    pair['source_gap'] = pair[f's_{gap_col}']
    pair['target_gap'] = pair[f't_{gap_col}']

    pair['complementarity'] = (
        pair['source_score']
        * pair['demand_score']
    )

    if (
        traj_col is not None
        and f's_{traj_col}' in pair.columns
        and f't_{traj_col}' in pair.columns
    ):
        pair['source_traj'] = pair[f's_{traj_col}']
        pair['target_traj'] = pair[f't_{traj_col}']

        pair['same_traj'] = (
            pair['source_traj'] == pair['target_traj']
        ).astype(int)
    else:
        pair['source_traj'] = np.nan
        pair['target_traj'] = np.nan
        pair['same_traj'] = np.nan

    if (
        state_col is not None
        and f's_{state_col}' in pair.columns
        and f't_{state_col}' in pair.columns
    ):
        pair['source_state'] = pair[f's_{state_col}']
        pair['target_state'] = pair[f't_{state_col}']

        pair['same_state'] = (
            pair['source_state'] == pair['target_state']
        ).astype(int)
    else:
        pair['source_state'] = np.nan
        pair['target_state'] = np.nan
        pair['same_state'] = np.nan

    temp_parts = []
    temp_weights = []

    if not pair['econ_sim'].isna().all():
        temp_parts.append(
            pair['econ_sim'].fillna(pair['econ_sim'].median())
        )
        temp_weights.append(MATCH_W_ECON)

    if not pair['land_sim'].isna().all():
        temp_parts.append(
            pair['land_sim'].fillna(pair['land_sim'].median())
        )
        temp_weights.append(MATCH_W_LAND)

    if not pair['activity_sim'].isna().all():
        temp_parts.append(
            pair['activity_sim'].fillna(pair['activity_sim'].median())
        )
        temp_weights.append(MATCH_W_ACTIVITY)

    if not pair['policy_sim'].isna().all():
        temp_parts.append(
            pair['policy_sim'].fillna(pair['policy_sim'].median())
        )
        temp_weights.append(MATCH_W_POLICY)

    if not pair['complementarity'].isna().all():
        temp_parts.append(
            pair['complementarity'].fillna(pair['complementarity'].median())
        )
        temp_weights.append(MATCH_W_COMPLEMENT)

    if len(temp_parts) == 0:
        pair['match_score'] = np.nan
    else:
        w = np.array(
            temp_weights,
            dtype=float
        )

        w = w / w.sum()

        vals = np.zeros(
            len(pair),
            dtype=float
        )

        for i, part in enumerate(temp_parts):
            vals += w[i] * part.to_numpy(dtype=float)

        pair['match_score'] = vals

    pair['edge_weight_base'] = (
        pair['source_score'] ** ALPHA
        * pair['demand_score'] ** BETA
        * pair['match_score'] ** GAMMA
        / pair['distance_km'] ** DELTA
    )

    if USE_ADJ_BONUS:
        pair.loc[
            pair['is_adjacent'] == 1,
            'edge_weight_base'
        ] *= ADJ_BONUS

    pair['edge_weight_base'] = (
        pair['edge_weight_base']
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    pair['edge_weight_base_norm'] = minmax_scale(
        pair['edge_weight_base'],
        reverse=False
    )

    current_flag = topk_flag_by_source(
        pair,
        'edge_weight_base',
        TOP_K
    )

    pair = pair.merge(
        current_flag,
        on=[
            'source',
            'target'
        ],
        how='left'
    )

    pair['current_strong_edge'] = (
        pair['strong_edge_flag']
        .fillna(0)
        .astype(int)
    )

    pair = pair.drop(
        columns=[
            'strong_edge_flag'
        ]
    )

    if current_edge_override is not None:
        ov = current_edge_override.copy()

        ov = ov.rename(
            columns={
                'pred_current_strong_edge': 'current_strong_edge_override',
                'pred_current_edge_weight_norm': 'current_edge_weight_norm_override'
            }
        )

        pair = pair.merge(
            ov[
                [
                    'source',
                    'target',
                    'current_strong_edge_override',
                    'current_edge_weight_norm_override'
                ]
            ],
            on=[
                'source',
                'target'
            ],
            how='left'
        )

        pair['current_strong_edge'] = (
            pair['current_strong_edge_override']
            .fillna(pair['current_strong_edge'])
            .astype(int)
        )

        pair['edge_weight_base_norm'] = (
            pair['current_edge_weight_norm_override']
            .fillna(pair['edge_weight_base_norm'])
        )

        pair = pair.drop(
            columns=[
                'current_strong_edge_override',
                'current_edge_weight_norm_override'
            ]
        )

    pair['Year'] = year

    keep_cols = [
        'Year',
        'source',
        'target',
        'distance_km',
        'is_adjacent',
        'source_score',
        'demand_score',
        'source_cii',
        'target_cii',
        'source_gap',
        'target_gap',
        'econ_sim',
        'land_sim',
        'activity_sim',
        'policy_sim',
        'complementarity',
        'match_score',
        'edge_weight_base',
        'edge_weight_base_norm',
        'rank_in_source',
        'current_strong_edge',
        'source_traj',
        'target_traj',
        'same_traj',
        'source_state',
        'target_state',
        'same_state'
    ]

    return pair[keep_cols].copy()


def build_node_roles_from_edge_table(
    edge_df,
    weight_col,
    year,
    all_nodes=None
):
    G = nx.DiGraph()

    if all_nodes is not None:
        G.add_nodes_from(all_nodes)
    else:
        nodes = pd.unique(
            pd.concat(
                [
                    edge_df['source'],
                    edge_df['target']
                ],
                ignore_index=True
            )
        )

        G.add_nodes_from(nodes)

    for _, row in edge_df.iterrows():
        G.add_edge(
            row['source'],
            row['target'],
            weight=float(row[weight_col])
        )

    out_degree_w = dict(
        G.out_degree(weight='weight')
    )

    in_degree_w = dict(
        G.in_degree(weight='weight')
    )

    G_for_centrality = nx.DiGraph()
    G_for_centrality.add_nodes_from(G.nodes())

    for u, v, d in G.edges(data=True):
        w = d['weight']
        dist = 1.0 / max(w, 1e-12)

        G_for_centrality.add_edge(
            u,
            v,
            weight=w,
            distance=dist
        )

    betweenness = nx.betweenness_centrality(
        G_for_centrality,
        weight='distance',
        normalized=True
    )

    closeness = nx.closeness_centrality(
        G_for_centrality,
        distance='distance'
    )

    out = pd.DataFrame({
        'Year': year,
        'Code': list(G.nodes()),
        'weighted_out_degree': [
            out_degree_w.get(n, 0.0)
            for n in G.nodes()
        ],
        'weighted_in_degree': [
            in_degree_w.get(n, 0.0)
            for n in G.nodes()
        ],
        'betweenness': [
            betweenness.get(n, 0.0)
            for n in G.nodes()
        ],
        'closeness': [
            closeness.get(n, 0.0)
            for n in G.nodes()
        ]
    })

    out['out_degree_score'] = minmax_scale(
        out['weighted_out_degree']
    )

    out['in_degree_score'] = minmax_scale(
        out['weighted_in_degree']
    )

    out['betweenness_score'] = minmax_scale(
        out['betweenness']
    )

    out['closeness_score'] = minmax_scale(
        out['closeness']
    )

    out_q90 = out['weighted_out_degree'].quantile(0.9)
    in_q90 = out['weighted_in_degree'].quantile(0.9)
    bet_q95 = out['betweenness'].quantile(0.95)

    out['is_source_core'] = (
        out['weighted_out_degree'] >= out_q90
    ).astype(int)

    out['is_receiver_core'] = (
        out['weighted_in_degree'] >= in_q90
    ).astype(int)

    out['is_bridge_core'] = (
        out['betweenness'] >= bet_q95
    ).astype(int)

    return out


def summarize_network(
    edge_df,
    weight_col,
    year,
    all_nodes=None
):
    G = nx.DiGraph()

    if all_nodes is not None:
        G.add_nodes_from(all_nodes)
    else:
        nodes = pd.unique(
            pd.concat(
                [
                    edge_df['source'],
                    edge_df['target']
                ],
                ignore_index=True
            )
        )

        G.add_nodes_from(nodes)

    for _, row in edge_df.iterrows():
        G.add_edge(
            row['source'],
            row['target'],
            weight=float(row[weight_col])
        )

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    weak_components = (
        nx.number_weakly_connected_components(G)
        if num_nodes > 0
        else np.nan
    )

    avg_out_degree = (
        np.mean(
            [
                d
                for _, d in G.out_degree()
            ]
        )
        if num_nodes > 0
        else np.nan
    )

    avg_in_degree = (
        np.mean(
            [
                d
                for _, d in G.in_degree()
            ]
        )
        if num_nodes > 0
        else np.nan
    )

    avg_clustering = (
        nx.average_clustering(G.to_undirected())
        if num_nodes > 0
        else np.nan
    )

    return pd.DataFrame([
        {
            'Year': year,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'weak_components': weak_components,
            'avg_out_degree': avg_out_degree,
            'avg_in_degree': avg_in_degree,
            'avg_clustering': avg_clustering
        }
    ])


def calc_region_feature_trend(
    df,
    code_col,
    year_col,
    feature_cols
):
    trend_dict = {}

    for code, sub in df.groupby(code_col):
        sub = sub.sort_values(year_col)

        trend_dict[code] = {}

        x = sub[year_col].to_numpy(dtype=float)

        for feat in feature_cols:
            if feat not in sub.columns:
                continue

            y = pd.to_numeric(
                sub[feat],
                errors='coerce'
            ).to_numpy(dtype=float)

            mask = np.isfinite(x) & np.isfinite(y)

            if mask.sum() >= 3:
                try:
                    slope = np.polyfit(
                        x[mask],
                        y[mask],
                        1
                    )[0]
                except Exception:
                    slope = 0.0
            else:
                slope = 0.0

            trend_dict[code][feat] = float(slope)

    return trend_dict


def build_future_year_node_df(
    base_df_2022,
    trend_dict,
    future_year,
    base_year,
    feature_cols,
    damping=0.7,
    lower_quantile=0.01,
    upper_quantile=0.99,
    global_reference_df=None
):
    df = base_df_2022.copy()

    step = future_year - base_year

    lower_bounds = {}
    upper_bounds = {}

    if global_reference_df is not None:
        for feat in feature_cols:
            if feat in global_reference_df.columns:
                vals = pd.to_numeric(
                    global_reference_df[feat],
                    errors='coerce'
                )

                lower_bounds[feat] = vals.quantile(lower_quantile)
                upper_bounds[feat] = vals.quantile(upper_quantile)

    for feat in feature_cols:
        if feat not in df.columns:
            continue

        new_vals = []

        for _, row in df.iterrows():
            code = row['Code']
            last_val = row[feat]

            if pd.isna(last_val):
                new_vals.append(np.nan)
                continue

            slope = trend_dict.get(code, {}).get(feat, 0.0)

            pred_val = float(last_val) + damping * step * slope

            if feat in lower_bounds and pd.notna(lower_bounds[feat]):
                pred_val = max(pred_val, lower_bounds[feat])

            if feat in upper_bounds and pd.notna(upper_bounds[feat]):
                pred_val = min(pred_val, upper_bounds[feat])

            new_vals.append(pred_val)

        df[feat] = new_vals

    df['Year'] = future_year

    return df


ensure_dir(OUTPUT_DIR)


node_year = pd.read_csv(NODE_YEAR_PATH)
adj = pd.read_csv(ADJ_PATH)
region_static = pd.read_csv(STATIC_PATH)

node_year['Code'] = node_year['Code'].astype(str).str.zfill(6)
region_static['Code'] = region_static['Code'].astype(str).str.zfill(6)

adj['Code_1'] = adj['Code_1'].astype(str).str.zfill(6)
adj['Code_2'] = adj['Code_2'].astype(str).str.zfill(6)

cii_col = pick_col(
    node_year,
    CII_CANDIDATES,
    required=True
)

gap_col = pick_col(
    node_year,
    GAP_CANDIDATES,
    required=True
)

stability_col = pick_col(
    node_year,
    STABILITY_CANDIDATES,
    required=True
)

traj_col = pick_col(
    node_year,
    TRAJ_CANDIDATES,
    required=False
)

state_col = pick_col(
    node_year,
    STATE_CANDIDATES,
    required=False
)

econ_cols = [
    c for c in ECON_VARS
    if c in node_year.columns
]

land_cols = [
    c for c in LAND_VARS
    if c in node_year.columns
]

activity_cols = [
    c for c in ACTIVITY_VARS
    if c in node_year.columns
]

policy_cols = [
    c for c in POLICY_VARS
    if c in node_year.columns
]

if (
    len(econ_cols) == 0
    and len(land_cols) == 0
    and len(activity_cols) == 0
    and len(policy_cols) == 0
):
    raise ValueError(
        'None of the specified ECON_VARS, LAND_VARS, ACTIVITY_VARS, '
        'or POLICY_VARS exist in node_year_prepared.csv. '
        'Please check the column names.'
    )


node_year = node_year[
    node_year['Year'].isin(TRAIN_YEARS)
].copy()


year_code_sets = []

for y in TRAIN_YEARS:
    year_code_sets.append(
        set(
            node_year.loc[
                node_year['Year'] == y,
                'Code'
            ].unique()
        )
    )


common_codes = sorted(
    list(
        set.intersection(*year_code_sets)
    )
)


node_year = node_year[
    node_year['Code'].isin(common_codes)
].copy()


region_static = region_static[
    region_static['Code'].isin(common_codes)
].copy()


candidate_pairs = build_candidate_pairs(
    region_static,
    adj,
    k_nearest=K_NEAREST
)


candidate_pairs = candidate_pairs[
    candidate_pairs['source'].isin(common_codes)
    & candidate_pairs['target'].isin(common_codes)
].copy()


year_nodes = {}
pair_features_by_year = {}


for y in TRAIN_YEARS:
    df_y = node_year[
        node_year['Year'] == y
    ].copy()

    df_y = prepare_year_node_features(
        df_y,
        cii_col,
        gap_col,
        stability_col,
        econ_cols,
        land_cols,
        activity_cols,
        policy_cols,
        traj_col=traj_col,
        state_col=state_col
    )

    year_nodes[y] = df_y

    pair_y = build_pair_features_for_year(
        y,
        df_y,
        candidate_pairs,
        cii_col=cii_col,
        gap_col=gap_col,
        econ_cols=econ_cols,
        land_cols=land_cols,
        activity_cols=activity_cols,
        policy_cols=policy_cols,
        traj_col=traj_col,
        state_col=state_col,
        current_edge_override=None
    )

    pair_features_by_year[y] = pair_y

    if SAVE_YEARLY_PAIR_FEATURES:
        pair_y.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f'pair_features_{y}.csv'
            ),
            index=False,
            encoding='utf-8-sig'
        )


transition_records = []


for i in range(len(TRAIN_YEARS) - 1):
    t = TRAIN_YEARS[i]
    t1 = TRAIN_YEARS[i + 1]

    df_t = pair_features_by_year[t].copy()

    df_t1 = pair_features_by_year[t1][
        [
            'source',
            'target',
            'current_strong_edge'
        ]
    ].copy()

    df_t1 = df_t1.rename(
        columns={
            'current_strong_edge': 'next_strong_edge'
        }
    )

    trans = df_t.merge(
        df_t1,
        on=[
            'source',
            'target'
        ],
        how='left'
    )

    trans['next_strong_edge'] = (
        trans['next_strong_edge']
        .fillna(0)
        .astype(int)
    )

    trans['transition_from'] = t
    trans['transition_to'] = t1

    transition_records.append(trans)


transition_df = pd.concat(
    transition_records,
    ignore_index=True
)


transition_out = os.path.join(
    OUTPUT_DIR,
    'transition_train_dataset.csv'
)


transition_df.to_csv(
    transition_out,
    index=False,
    encoding='utf-8-sig'
)


train_df = transition_df[
    transition_df['transition_to'] < 2022
].copy()


valid_df = transition_df[
    transition_df['transition_to'] == 2022
].copy()


feature_cols = [
    'distance_km',
    'is_adjacent',
    'source_score',
    'demand_score',
    'source_cii',
    'target_cii',
    'source_gap',
    'target_gap',
    'econ_sim',
    'land_sim',
    'activity_sim',
    'policy_sim',
    'complementarity',
    'match_score',
    'edge_weight_base',
    'edge_weight_base_norm',
    'current_strong_edge',
    'source_traj',
    'target_traj',
    'same_traj',
    'source_state',
    'target_state',
    'same_state'
]


feature_cols = [
    c for c in feature_cols
    if c in transition_df.columns
]


X_train = train_df[feature_cols].copy().fillna(-1)
y_train = train_df['next_strong_edge'].astype(int)

X_valid = valid_df[feature_cols].copy().fillna(-1)
y_valid = valid_df['next_strong_edge'].astype(int)


model_name = None
model = None


try:
    from xgboost import XGBClassifier

    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-06)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    model_name = 'XGBoost'

except Exception:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    model_name = 'RandomForest'


valid_pred_prob = model.predict_proba(X_valid)[:, 1]
valid_pred_label = (
    valid_pred_prob >= 0.5
).astype(int)

val_auc = roc_auc_score(
    y_valid,
    valid_pred_prob
)

val_ap = average_precision_score(
    y_valid,
    valid_pred_prob
)

val_acc = accuracy_score(
    y_valid,
    valid_pred_label
)

val_f1 = f1_score(
    y_valid,
    valid_pred_label
)


valid_result = valid_df[
    [
        'source',
        'target',
        'transition_from',
        'transition_to',
        'next_strong_edge'
    ]
].copy()


valid_result['pred_prob'] = valid_pred_prob
valid_result['pred_label_05'] = valid_pred_label


valid_out = os.path.join(
    OUTPUT_DIR,
    'validation_pred_2022.csv'
)


valid_result.to_csv(
    valid_out,
    index=False,
    encoding='utf-8-sig'
)


X_all = transition_df[feature_cols].copy().fillna(-1)
y_all = transition_df['next_strong_edge'].astype(int)


try:
    if model_name == 'XGBoost':
        from xgboost import XGBClassifier

        pos_rate = y_all.mean()
        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-06)

        final_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )

    else:
        final_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

except Exception:
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


final_model.fit(
    X_all,
    y_all
)


if hasattr(final_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values(
        'importance',
        ascending=False
    )

    feat_imp_out = os.path.join(
        OUTPUT_DIR,
        'feature_importance_link_prediction.csv'
    )

    feat_imp.to_csv(
        feat_imp_out,
        index=False,
        encoding='utf-8-sig'
    )

else:
    feat_imp = None
    feat_imp_out = None


future_network_summary_list = []
future_node_roles_list = []


trend_feature_cols = [
    cii_col,
    gap_col
] + econ_cols + land_cols + activity_cols + policy_cols


trend_feature_cols = [
    c for c in trend_feature_cols
    if c in node_year.columns
]


trend_source_df = node_year[
    node_year['Year'].isin(TRAIN_YEARS)
].copy()


trend_dict = calc_region_feature_trend(
    trend_source_df,
    code_col='Code',
    year_col='Year',
    feature_cols=trend_feature_cols
)


base_2022_raw = node_year[
    node_year['Year'] == 2022
].copy()


base_2022_raw = base_2022_raw[
    base_2022_raw['Code'].isin(common_codes)
].copy()


current_override_df = None


for future_year in FUTURE_YEARS:
    future_raw_df = build_future_year_node_df(
        base_df_2022=base_2022_raw,
        trend_dict=trend_dict,
        future_year=future_year,
        base_year=2022,
        feature_cols=trend_feature_cols,
        damping=TREND_DAMPING,
        lower_quantile=0.01,
        upper_quantile=0.99,
        global_reference_df=trend_source_df
    )

    if traj_col is not None and traj_col in base_2022_raw.columns:
        future_raw_df[traj_col] = base_2022_raw[traj_col].values

    if state_col is not None and state_col in base_2022_raw.columns:
        future_raw_df[state_col] = base_2022_raw[state_col].values

    if stability_col is not None and stability_col in base_2022_raw.columns:
        future_raw_df[stability_col] = base_2022_raw[stability_col].values

    future_raw_out = os.path.join(
        OUTPUT_DIR,
        f'future_node_features_{future_year}.csv'
    )

    future_raw_df.to_csv(
        future_raw_out,
        index=False,
        encoding='utf-8-sig'
    )

    future_node_df = prepare_year_node_features(
        future_raw_df,
        cii_col=cii_col,
        gap_col=gap_col,
        stability_col=stability_col,
        econ_cols=econ_cols,
        land_cols=land_cols,
        activity_cols=activity_cols,
        policy_cols=policy_cols,
        traj_col=traj_col,
        state_col=state_col
    )

    current_pair = build_pair_features_for_year(
        future_year,
        future_node_df,
        candidate_pairs,
        cii_col=cii_col,
        gap_col=gap_col,
        econ_cols=econ_cols,
        land_cols=land_cols,
        activity_cols=activity_cols,
        policy_cols=policy_cols,
        traj_col=traj_col,
        state_col=state_col,
        current_edge_override=current_override_df
    )

    X_future = current_pair[feature_cols].copy().fillna(-1)

    pred_prob = final_model.predict_proba(X_future)[:, 1]

    pred_df = current_pair[
        [
            'source',
            'target',
            'distance_km',
            'is_adjacent',
            'source_score',
            'demand_score',
            'match_score',
            'edge_weight_base',
            'current_strong_edge'
        ]
    ].copy()

    pred_df['pred_prob'] = pred_prob

    pred_df['pred_prob_norm'] = minmax_scale(
        pred_df['pred_prob'],
        reverse=False
    )

    pred_df['pred_keep_flag'] = (
        pred_df['pred_prob'] >= PRED_PROB_THRESHOLD
    ).astype(int)

    pred_df = pred_df.sort_values(
        [
            'source',
            'pred_prob'
        ],
        ascending=[
            True,
            False
        ]
    )

    pred_df['pred_rank_in_source'] = (
        pred_df
        .groupby('source')['pred_prob']
        .rank(method='first', ascending=False)
    )

    pred_df['pred_strong_edge'] = (
        (pred_df['pred_keep_flag'] == 1)
        & (pred_df['pred_rank_in_source'] <= TOP_K)
    ).astype(int)

    pred_topk = pred_df[
        pred_df['pred_strong_edge'] == 1
    ].copy()

    pred_topk['Year'] = future_year

    pred_out = os.path.join(
        OUTPUT_DIR,
        f'predicted_edges_{future_year}_topk.csv'
    )

    pred_topk.to_csv(
        pred_out,
        index=False,
        encoding='utf-8-sig'
    )

    node_roles_future = build_node_roles_from_edge_table(
        pred_topk,
        'pred_prob',
        future_year,
        all_nodes=common_codes
    )

    future_node_roles_list.append(node_roles_future)

    net_sum = summarize_network(
        pred_topk,
        'pred_prob',
        future_year,
        all_nodes=common_codes
    )

    future_network_summary_list.append(net_sum)

    current_override_df = pred_df[
        [
            'source',
            'target',
            'pred_strong_edge',
            'pred_prob_norm'
        ]
    ].copy()

    current_override_df = current_override_df.rename(
        columns={
            'pred_strong_edge': 'pred_current_strong_edge',
            'pred_prob_norm': 'pred_current_edge_weight_norm'
        }
    )


future_network_summary = pd.concat(
    future_network_summary_list,
    ignore_index=True
)


future_node_roles = pd.concat(
    future_node_roles_list,
    ignore_index=True
)


future_summary_out = os.path.join(
    OUTPUT_DIR,
    'future_network_summary_2023_2030.csv'
)


future_roles_out = os.path.join(
    OUTPUT_DIR,
    'future_node_roles_2023_2030.csv'
)


future_network_summary.to_csv(
    future_summary_out,
    index=False,
    encoding='utf-8-sig'
)


future_node_roles.to_csv(
    future_roles_out,
    index=False,
    encoding='utf-8-sig'
)


if feat_imp_out is not None:
    pass


for y in FUTURE_YEARS:
    pass


if feat_imp is not None:
    pass