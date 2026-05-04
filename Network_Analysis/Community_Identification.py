import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx


warnings.filterwarnings('ignore')


INPUT_DIR = './step2_outputs'
OUTPUT_DIR = './step3_outputs'

BASE_YEAR = 2022

EDGES_PATH = os.path.join(INPUT_DIR, f'network_edges_{BASE_YEAR}_topk.csv')
REGION_PATH = os.path.join(INPUT_DIR, f'region_profile_with_roles_{BASE_YEAR}.csv')

USE_UNDIRECTED_FOR_COMMUNITY = True
TOP_N_CORRIDOR_PER_COMMUNITY_PAIR = 10
MIN_COMMUNITY_SIZE = 5


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


def safe_mean(x):
    x = pd.Series(x).dropna()

    if len(x) == 0:
        return np.nan

    return float(x.mean())


def safe_sum(x):
    x = pd.Series(x).dropna()

    if len(x) == 0:
        return 0.0

    return float(x.sum())


def pick_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(f'Candidate columns not found: {candidates}')

    return None


def classify_community(row):
    source = row['avg_source_core_share']
    receiver = row['avg_receiver_core_share']
    bridge = row['avg_bridge_core_share']
    cii = row['avg_CII_norm']
    gap = row['avg_Gap']

    if row['community_size'] < MIN_COMMUNITY_SIZE:
        return 'small-scale community'

    if source >= 0.1 and cii >= row['global_avg_CII_norm'] and gap <= row['global_avg_Gap']:
        return 'source-output community'

    if receiver >= 0.1 and gap >= row['global_avg_Gap']:
        return 'recipient-catch-up community'

    if bridge >= 0.1:
        return 'bridge-transfer community'

    return 'mixed community'


ensure_dir(OUTPUT_DIR)


edges = pd.read_csv(EDGES_PATH)
region = pd.read_csv(REGION_PATH)


for col in ['source', 'target']:
    edges[col] = edges[col].astype(str).str.zfill(6)

region['Code'] = region['Code'].astype(str).str.zfill(6)


CII_COL = pick_col(region, ['CII_norm'], required=True)
GAP_COL = pick_col(region, ['Gap'], required=True)

SOURCE_CORE_COL = pick_col(region, ['is_source_core'], required=True)
RECEIVER_CORE_COL = pick_col(region, ['is_receiver_core'], required=True)
BRIDGE_CORE_COL = pick_col(region, ['is_bridge_core'], required=True)

OUTDEG_COL = pick_col(region, ['weighted_out_degree'], required=True)
INDEG_COL = pick_col(region, ['weighted_in_degree'], required=True)
BETWEEN_COL = pick_col(region, ['betweenness'], required=True)

TRAJ_COL = pick_col(region, ['trajectory_class', 'traj_class'])
TRAJ_LABEL_COL = pick_col(region, ['trajectory_label'])

STATE_COL = pick_col(region, ['cii_state'])
STATE_LABEL_COL = pick_col(region, ['cii_state_label'])


if USE_UNDIRECTED_FOR_COMMUNITY:
    G_comm = nx.Graph()

    for _, row in edges.iterrows():
        s = row['source']
        t = row['target']
        w = float(row['edge_weight'])

        if G_comm.has_edge(s, t):
            G_comm[s][t]['weight'] += w
        else:
            G_comm.add_edge(s, t, weight=w)

else:
    G_comm = nx.DiGraph()

    for _, row in edges.iterrows():
        G_comm.add_edge(
            row['source'],
            row['target'],
            weight=float(row['edge_weight'])
        )


for code in region['Code'].unique():
    if code not in G_comm:
        G_comm.add_node(code)


community_method = None
community_map = {}


try:
    import community as community_louvain

    partition = community_louvain.best_partition(
        G_comm,
        weight='weight',
        resolution=1.0,
        random_state=42
    )

    community_map = partition
    community_method = 'louvain'

except Exception:
    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(
        G_comm,
        weight='weight'
    )

    for cid, comm in enumerate(communities):
        for node in comm:
            community_map[node] = cid

    community_method = 'greedy_modularity'


community_assignment = pd.DataFrame({
    'Code': list(community_map.keys()),
    'community_id': list(community_map.values())
})


unique_ids = sorted(community_assignment['community_id'].unique())

id_map = {
    old: new
    for new, old in enumerate(unique_ids)
}

community_assignment['community_id'] = community_assignment['community_id'].map(id_map)


region_comm = region.merge(
    community_assignment,
    on='Code',
    how='left'
)


edges_comm = (
    edges
    .merge(
        community_assignment.rename(
            columns={
                'Code': 'source',
                'community_id': 'source_community'
            }
        ),
        on='source',
        how='left'
    )
    .merge(
        community_assignment.rename(
            columns={
                'Code': 'target',
                'community_id': 'target_community'
            }
        ),
        on='target',
        how='left'
    )
)


edges_comm['is_intercommunity'] = (
    edges_comm['source_community'] != edges_comm['target_community']
).astype(int)


community_node_summary = (
    region_comm
    .groupby('community_id')
    .agg(
        community_size=('Code', 'count'),
        avg_CII_norm=(CII_COL, 'mean'),
        avg_Gap=(GAP_COL, 'mean'),
        avg_weighted_out_degree=(OUTDEG_COL, 'mean'),
        avg_weighted_in_degree=(INDEG_COL, 'mean'),
        avg_betweenness=(BETWEEN_COL, 'mean'),
        avg_source_core_share=(SOURCE_CORE_COL, 'mean'),
        avg_receiver_core_share=(RECEIVER_CORE_COL, 'mean'),
        avg_bridge_core_share=(BRIDGE_CORE_COL, 'mean')
    )
    .reset_index()
)


if TRAJ_COL is not None:
    traj_mode = (
        region_comm
        .groupby('community_id')[TRAJ_COL]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
        .rename(columns={TRAJ_COL: 'dominant_trajectory_class'})
    )

    community_node_summary = community_node_summary.merge(
        traj_mode,
        on='community_id',
        how='left'
    )


if TRAJ_LABEL_COL is not None:
    traj_label_mode = (
        region_comm
        .groupby('community_id')[TRAJ_LABEL_COL]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
        .rename(columns={TRAJ_LABEL_COL: 'dominant_trajectory_label'})
    )

    community_node_summary = community_node_summary.merge(
        traj_label_mode,
        on='community_id',
        how='left'
    )


if STATE_COL is not None:
    state_mode = (
        region_comm
        .groupby('community_id')[STATE_COL]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
        .rename(columns={STATE_COL: 'dominant_state_class'})
    )

    community_node_summary = community_node_summary.merge(
        state_mode,
        on='community_id',
        how='left'
    )


if STATE_LABEL_COL is not None:
    state_label_mode = (
        region_comm
        .groupby('community_id')[STATE_LABEL_COL]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
        .rename(columns={STATE_LABEL_COL: 'dominant_state_label'})
    )

    community_node_summary = community_node_summary.merge(
        state_label_mode,
        on='community_id',
        how='left'
    )


community_edge_summary = (
    edges_comm
    .groupby(['source_community', 'target_community'])
    .agg(
        edge_count=('edge_weight', 'count'),
        total_edge_weight=('edge_weight', 'sum'),
        avg_edge_weight=('edge_weight', 'mean')
    )
    .reset_index()
)


internal_summary = (
    community_edge_summary[
        community_edge_summary['source_community'] == community_edge_summary['target_community']
    ]
    .copy()
    .rename(
        columns={
            'source_community': 'community_id',
            'edge_count': 'internal_edge_count',
            'total_edge_weight': 'internal_total_edge_weight',
            'avg_edge_weight': 'internal_avg_edge_weight'
        }
    )[
        [
            'community_id',
            'internal_edge_count',
            'internal_total_edge_weight',
            'internal_avg_edge_weight'
        ]
    ]
)


external_out_summary = (
    community_edge_summary[
        community_edge_summary['source_community'] != community_edge_summary['target_community']
    ]
    .groupby('source_community')
    .agg(
        external_out_edge_count=('edge_count', 'sum'),
        external_out_total_weight=('total_edge_weight', 'sum')
    )
    .reset_index()
    .rename(columns={'source_community': 'community_id'})
)


external_in_summary = (
    community_edge_summary[
        community_edge_summary['source_community'] != community_edge_summary['target_community']
    ]
    .groupby('target_community')
    .agg(
        external_in_edge_count=('edge_count', 'sum'),
        external_in_total_weight=('total_edge_weight', 'sum')
    )
    .reset_index()
    .rename(columns={'target_community': 'community_id'})
)


community_summary = community_node_summary.merge(
    internal_summary,
    on='community_id',
    how='left'
)

community_summary = community_summary.merge(
    external_out_summary,
    on='community_id',
    how='left'
)

community_summary = community_summary.merge(
    external_in_summary,
    on='community_id',
    how='left'
)


for col in [
    'internal_edge_count',
    'internal_total_edge_weight',
    'internal_avg_edge_weight',
    'external_out_edge_count',
    'external_out_total_weight',
    'external_in_edge_count',
    'external_in_total_weight'
]:
    if col in community_summary.columns:
        community_summary[col] = community_summary[col].fillna(0)


community_summary['internal_strength_ratio'] = (
    community_summary['internal_total_edge_weight']
    / (
        community_summary['internal_total_edge_weight']
        + community_summary['external_out_total_weight']
        + community_summary['external_in_total_weight']
        + 1e-12
    )
)


community_summary['global_avg_CII_norm'] = community_summary['avg_CII_norm'].mean()
community_summary['global_avg_Gap'] = community_summary['avg_Gap'].mean()

community_summary['community_type'] = community_summary.apply(
    classify_community,
    axis=1
)


intercommunity_edges = edges_comm[
    edges_comm['is_intercommunity'] == 1
].copy()


intercommunity_edges['community_pair'] = intercommunity_edges.apply(
    lambda r: (
        f"{min(r['source_community'], r['target_community'])}_"
        f"{max(r['source_community'], r['target_community'])}"
    ),
    axis=1
)


intercommunity_corridors = (
    intercommunity_edges
    .sort_values(
        ['community_pair', 'edge_weight'],
        ascending=[True, False]
    )
    .groupby(
        'community_pair',
        as_index=False,
        group_keys=False
    )
    .head(TOP_N_CORRIDOR_PER_COMMUNITY_PAIR)
    .copy()
)


corridor_summary = (
    intercommunity_edges
    .groupby(['community_pair', 'source_community', 'target_community'])
    .agg(
        edge_count=('edge_weight', 'count'),
        total_edge_weight=('edge_weight', 'sum'),
        avg_edge_weight=('edge_weight', 'mean')
    )
    .reset_index()
)


corridor_summary = (
    corridor_summary
    .sort_values(
        ['community_pair', 'total_edge_weight'],
        ascending=[True, False]
    )
    .groupby(
        'community_pair',
        as_index=False,
        group_keys=False
    )
    .head(1)
)


intercommunity_corridors = intercommunity_corridors.merge(
    corridor_summary[
        [
            'community_pair',
            'edge_count',
            'total_edge_weight',
            'avg_edge_weight'
        ]
    ],
    on='community_pair',
    how='left',
    suffixes=('', '_pair')
)


intercommunity_corridors = (
    intercommunity_corridors
    .sort_values(
        ['total_edge_weight', 'edge_weight'],
        ascending=[False, False]
    )
    .reset_index(drop=True)
)


region_comm = region_comm.merge(
    community_summary[
        [
            'community_id',
            'community_type'
        ]
    ],
    on='community_id',
    how='left'
)


assignment_out = os.path.join(
    OUTPUT_DIR,
    f'community_assignment_{BASE_YEAR}.csv'
)

summary_out = os.path.join(
    OUTPUT_DIR,
    f'community_summary_{BASE_YEAR}.csv'
)

region_out = os.path.join(
    OUTPUT_DIR,
    f'region_profile_with_community_{BASE_YEAR}.csv'
)

corridor_out = os.path.join(
    OUTPUT_DIR,
    f'intercommunity_corridors_{BASE_YEAR}.csv'
)

node_summary_out = os.path.join(
    OUTPUT_DIR,
    f'community_node_summary_{BASE_YEAR}.csv'
)


community_assignment.to_csv(
    assignment_out,
    index=False,
    encoding='utf-8-sig'
)

community_summary.to_csv(
    summary_out,
    index=False,
    encoding='utf-8-sig'
)

region_comm.to_csv(
    region_out,
    index=False,
    encoding='utf-8-sig'
)

intercommunity_corridors.to_csv(
    corridor_out,
    index=False,
    encoding='utf-8-sig'
)

community_node_summary.to_csv(
    node_summary_out,
    index=False,
    encoding='utf-8-sig'
)