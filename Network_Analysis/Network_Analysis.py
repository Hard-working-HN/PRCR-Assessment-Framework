import os
import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx


warnings.filterwarnings('ignore')


INPUT_DIR = './step1_outputs'
OUTPUT_DIR = './step2_outputs'

REGION_PROFILE_PATH = os.path.join(INPUT_DIR, 'region_profile_step1.csv')
ADJ_EDGES_PATH = os.path.join(INPUT_DIR, 'adjacency_edges_step1.csv')

BASE_YEAR = 2022
TOP_K = 8
REMOVE_SELF_LOOP = True

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
MIN_DISTANCE_KM = 1.0

ECON_VARS = [
    'GDP',
    'POP',
    'PIR',
    'PIU',
    'TSS',
    'GRF',
    'GEF'
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


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def minmax_scale(series: pd.Series, reverse: bool = False) -> pd.Series:
    s = series.astype(float).copy()
    s = s.replace([np.inf, -np.inf], np.nan)

    valid = s.dropna()

    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    s_min = valid.min()
    s_max = valid.max()

    if np.isclose(s_min, s_max):
        out = pd.Series(0.5, index=series.index, dtype=float)
    else:
        out = (s - s_min) / (s_max - s_min)

    if reverse:
        out = 1 - out

    return out


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def vector_similarity(row_i, row_j, cols):
    vals_i = []
    vals_j = []

    for c in cols:
        vi = row_i.get(c, np.nan)
        vj = row_j.get(c, np.nan)

        if pd.notna(vi) and pd.notna(vj):
            vals_i.append(float(vi))
            vals_j.append(float(vj))

    if len(vals_i) == 0:
        return np.nan

    xi = np.array(vals_i, dtype=float)
    xj = np.array(vals_j, dtype=float)

    dist = np.linalg.norm(xi - xj)
    sim = 1.0 / (1.0 + dist)

    return float(sim)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088

    lon1, lat1, lon2, lat2 = map(
        math.radians,
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return R * c


def calc_pair_distance_km(row_i, row_j):
    return haversine_km(
        row_i['centroid_lon'],
        row_i['centroid_lat'],
        row_j['centroid_lon'],
        row_j['centroid_lat']
    )


def topk_by_source(edge_df: pd.DataFrame, k: int) -> pd.DataFrame:
    edge_df = edge_df.sort_values(
        ['source', 'edge_weight'],
        ascending=[True, False]
    ).copy()

    out = (
        edge_df
        .groupby('source', as_index=False, group_keys=False)
        .head(k)
        .copy()
    )

    out = out.reset_index(drop=True)

    return out


def safe_mean(values):
    vals = pd.Series(values).dropna()

    if len(vals) == 0:
        return np.nan

    return float(vals.mean())


ensure_dir(OUTPUT_DIR)


region = pd.read_csv(REGION_PROFILE_PATH)
adj = pd.read_csv(ADJ_EDGES_PATH)


required_region_cols = [
    'Code',
    'CII_norm',
    'Gap',
    'source_score_base',
    'demand_score_base',
    'centroid_lon',
    'centroid_lat'
]

missing_region = [
    c for c in required_region_cols
    if c not in region.columns
]

if missing_region:
    raise ValueError(
        f'region_profile_step1.csv is missing required columns: {missing_region}'
    )


required_adj_cols = [
    'Code_1',
    'Code_2',
    'centroid_distance_km'
]

missing_adj = [
    c for c in required_adj_cols
    if c not in adj.columns
]

if missing_adj:
    raise ValueError(
        f'adjacency_edges_step1.csv is missing required columns: {missing_adj}'
    )


region['Code'] = region['Code'].astype(str).str.zfill(6)
adj['Code_1'] = adj['Code_1'].astype(str).str.zfill(6)
adj['Code_2'] = adj['Code_2'].astype(str).str.zfill(6)


econ_cols = [
    c for c in ECON_VARS
    if c in region.columns
]

land_cols = [
    c for c in LAND_VARS
    if c in region.columns
]

activity_cols = [
    c for c in ACTIVITY_VARS
    if c in region.columns
]

policy_cols = [
    c for c in POLICY_VARS
    if c in region.columns
]


if (
    len(econ_cols) == 0
    and len(land_cols) == 0
    and len(activity_cols) == 0
    and len(policy_cols) == 0
):
    raise ValueError(
        'None of the specified ECON_VARS, LAND_VARS, ACTIVITY_VARS, '
        'or POLICY_VARS exist in region_profile_step1.csv. '
        'Please check the column names.'
    )


region_std = region.copy()

for col in econ_cols + land_cols + activity_cols + policy_cols:
    region_std[col] = minmax_scale(
        region_std[col],
        reverse=False
    )


adj['is_adjacent'] = 1
adj_pairs = set(
    zip(
        adj['Code_1'],
        adj['Code_2']
    )
)


records = []

region_dict = {
    row['Code']: row
    for _, row in region_std.iterrows()
}

codes = region_std['Code'].tolist()

n_codes = len(codes)
total_pairs = n_codes * n_codes
counter = 0


for source in codes:
    row_s = region_dict[source]

    for target in codes:
        counter += 1

        if REMOVE_SELF_LOOP and source == target:
            continue

        row_t = region_dict[target]

        distance_km = calc_pair_distance_km(
            row_s,
            row_t
        )

        if pd.isna(distance_km) or distance_km < MIN_DISTANCE_KM:
            distance_km = MIN_DISTANCE_KM

        is_adjacent = 1 if (source, target) in adj_pairs else 0

        econ_sim = (
            vector_similarity(row_s, row_t, econ_cols)
            if len(econ_cols) > 0
            else np.nan
        )

        land_sim = (
            vector_similarity(row_s, row_t, land_cols)
            if len(land_cols) > 0
            else np.nan
        )

        activity_sim = (
            vector_similarity(row_s, row_t, activity_cols)
            if len(activity_cols) > 0
            else np.nan
        )

        policy_sim = (
            vector_similarity(row_s, row_t, policy_cols)
            if len(policy_cols) > 0
            else np.nan
        )

        source_score = safe_float(
            row_s['source_score_base'],
            default=np.nan
        )

        demand_score = safe_float(
            row_t['demand_score_base'],
            default=np.nan
        )

        complement = (
            source_score * demand_score
            if pd.notna(source_score) and pd.notna(demand_score)
            else np.nan
        )

        parts = []
        weights = []

        if pd.notna(econ_sim):
            parts.append(econ_sim)
            weights.append(MATCH_W_ECON)

        if pd.notna(land_sim):
            parts.append(land_sim)
            weights.append(MATCH_W_LAND)

        if pd.notna(activity_sim):
            parts.append(activity_sim)
            weights.append(MATCH_W_ACTIVITY)

        if pd.notna(policy_sim):
            parts.append(policy_sim)
            weights.append(MATCH_W_POLICY)

        if pd.notna(complement):
            parts.append(complement)
            weights.append(MATCH_W_COMPLEMENT)

        if len(parts) == 0:
            match_score = np.nan
        else:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()

            match_score = float(
                np.sum(
                    np.array(parts, dtype=float) * weights
                )
            )

        if (
            pd.notna(source_score)
            and pd.notna(demand_score)
            and pd.notna(match_score)
        ):
            edge_weight = (
                source_score ** ALPHA
                * demand_score ** BETA
                * match_score ** GAMMA
                / distance_km ** DELTA
            )

            if USE_ADJ_BONUS and is_adjacent == 1:
                edge_weight *= ADJ_BONUS

        else:
            edge_weight = np.nan

        records.append({
            'year': BASE_YEAR,
            'source': source,
            'target': target,
            'distance_km': distance_km,
            'is_adjacent': is_adjacent,
            'source_score': source_score,
            'demand_score': demand_score,
            'econ_sim': econ_sim,
            'land_sim': land_sim,
            'activity_sim': activity_sim,
            'policy_sim': policy_sim,
            'complementarity': complement,
            'match_score': match_score,
            'edge_weight': edge_weight
        })

        if counter % 500000 == 0:
            pass


pair_df = pd.DataFrame(records)

pair_df = pair_df[
    pair_df['edge_weight'].notna()
].copy()

pair_df = pair_df[
    pair_df['edge_weight'] > 0
].copy()


edges_topk = topk_by_source(
    pair_df,
    TOP_K
)


edges_topk['edge_weight_norm'] = minmax_scale(
    edges_topk['edge_weight'],
    reverse=False
)


G = nx.DiGraph()


for _, row in region.iterrows():
    code = str(row['Code']).zfill(6)
    G.add_node(code)


for _, row in edges_topk.iterrows():
    G.add_edge(
        row['source'],
        row['target'],
        weight=float(row['edge_weight']),
        weight_norm=float(row['edge_weight_norm'])
    )


out_degree_w = dict(
    G.out_degree(weight='weight')
)

in_degree_w = dict(
    G.in_degree(weight='weight')
)


G_for_centrality = nx.DiGraph()


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


node_roles = pd.DataFrame({
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


node_roles['out_degree_score'] = minmax_scale(
    node_roles['weighted_out_degree']
)

node_roles['in_degree_score'] = minmax_scale(
    node_roles['weighted_in_degree']
)

node_roles['betweenness_score'] = minmax_scale(
    node_roles['betweenness']
)

node_roles['closeness_score'] = minmax_scale(
    node_roles['closeness']
)


out_q90 = node_roles['weighted_out_degree'].quantile(0.9)
in_q90 = node_roles['weighted_in_degree'].quantile(0.9)
bet_q95 = node_roles['betweenness'].quantile(0.95)


node_roles['is_source_core'] = (
    node_roles['weighted_out_degree'] >= out_q90
).astype(int)

node_roles['is_receiver_core'] = (
    node_roles['weighted_in_degree'] >= in_q90
).astype(int)

node_roles['is_bridge_core'] = (
    node_roles['betweenness'] >= bet_q95
).astype(int)


num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)


if num_nodes > 0:
    weak_components = nx.number_weakly_connected_components(G)
else:
    weak_components = np.nan


avg_out_degree = safe_mean([
    d for _, d in G.out_degree()
])

avg_in_degree = safe_mean([
    d for _, d in G.in_degree()
])


G_u = G.to_undirected()

avg_clustering = (
    nx.average_clustering(G_u)
    if G_u.number_of_nodes() > 0
    else np.nan
)


network_summary = pd.DataFrame([
    {
        'year': BASE_YEAR,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'weak_components': weak_components,
        'avg_out_degree': avg_out_degree,
        'avg_in_degree': avg_in_degree,
        'avg_clustering': avg_clustering,
        'top_k_per_source': TOP_K,
        'alpha': ALPHA,
        'beta': BETA,
        'gamma': GAMMA,
        'delta': DELTA,
        'adj_bonus': ADJ_BONUS if USE_ADJ_BONUS else 1.0
    }
])


region_with_roles = region.merge(
    node_roles,
    on='Code',
    how='left'
)


pair_out = os.path.join(
    OUTPUT_DIR,
    f'pair_features_{BASE_YEAR}.csv'
)

edges_out = os.path.join(
    OUTPUT_DIR,
    f'network_edges_{BASE_YEAR}_topk.csv'
)

roles_out = os.path.join(
    OUTPUT_DIR,
    f'node_roles_{BASE_YEAR}.csv'
)

summary_out = os.path.join(
    OUTPUT_DIR,
    f'network_summary_{BASE_YEAR}.csv'
)

region_roles_out = os.path.join(
    OUTPUT_DIR,
    f'region_profile_with_roles_{BASE_YEAR}.csv'
)


pair_df.to_csv(
    pair_out,
    index=False,
    encoding='utf-8-sig'
)

edges_topk.to_csv(
    edges_out,
    index=False,
    encoding='utf-8-sig'
)

node_roles.to_csv(
    roles_out,
    index=False,
    encoding='utf-8-sig'
)

network_summary.to_csv(
    summary_out,
    index=False,
    encoding='utf-8-sig'
)

region_with_roles.to_csv(
    region_roles_out,
    index=False,
    encoding='utf-8-sig'
)