import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import itertools
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import shap
except ImportError as e:
    raise ImportError('Please install shap first: pip install shap') from e


torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


CSV_PATH = 'Total_Data_with_QSstar_merged.csv'
SHP_PATH = 'County2023.shp'

CSV_CODE_COL = 'Code'
SHP_CODE_COL = 'code'
YEAR_COL = 'Year'
PROVINCE_COL = 'Province_ID'

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

BATCH_SIZE = 1
EPOCHS = 2222
LR = 0.001
WEIGHT_DECAY = 1e-05
PATIENCE = 444
LR_REDUCE_PATIENCE = 111
LR_REDUCE_FACTOR = 0.8

DIST_K = 8
SIM_K = 8

HIDDEN_DIM = 128
GRAPH_HIDDEN_DIM = 64
STATIC_HIDDEN_DIM = 16
ADP_EMB_DIM = 16
DROPOUT = 0.1
LAMBDA_SMOOTH = 0.0001

RESULT_DIR = 'gspatial_randomrowsplit_results'
METRICS_DIR = os.path.join(RESULT_DIR, 'metrics')
PRED_DIR = os.path.join(RESULT_DIR, 'predictions')
SHAP_DIR = os.path.join(RESULT_DIR, 'shap_direct')
ALE_DIR = os.path.join(RESULT_DIR, 'ale')
MODEL_DIR = os.path.join(RESULT_DIR, 'model')
INTERACTION_DIR = os.path.join(RESULT_DIR, 'interaction')

SHAP_BACKGROUND_SIZE = 256
SHAP_PROGRESS_EVERY = 50
SHAP_BATCH_SIZE = 64
SHAP_LOCAL_SMOOTHING = 0.0
SHAP_TEST_NODE_SIZE = None

ALE_N_BINS = 20
ALE_TOP_N_FEATURES = 9
ALE_MIN_GROUP_SIZE = 50
ALE_MAX_SAMPLES = None
ALE_PLOT_MAX_GROUPS = 8

DO_PROVINCE_ALE = False
DO_CII_CLASS_ALE = True

CII_CLASS_ORDER = [
    'non cooperation (CII ≤ 0)',
    'low cooperation (0 < CII ≤ 0.2)',
    'moderate cooperation (0.2 < CII ≤ 0.5)',
    'high cooperation (CII > 0.5)'
]

INTERACTION_SCREEN_TOP_K = 12
INTERACTION_TOP_PAIRS_TO_PLOT = 4
INTERACTION_MAIN_BINS = 8
INTERACTION_MOD_BINS = 4
INTERACTION_MIN_CELL = 5
INTERACTION_MAX_POINTS_PLOT = 2000

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


def clean_code(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s.endswith('.0'):
        s = s[:-2]

    return s


def safe_filename(x):
    s = str(x)

    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, '_')

    return s


def smape(y_true, y_pred, eps=1e-08):
    return (
        np.mean(
            2.0 * np.abs(y_pred - y_true) /
            (np.abs(y_true) + np.abs(y_pred) + eps)
        ) * 100
    )


def wmape(y_true, y_pred, eps=1e-08):
    return (
        np.sum(np.abs(y_true - y_pred)) /
        (np.sum(np.abs(y_true)) + eps) * 100
    )


def mape(y_true, y_pred, eps=1e-08):
    return (
        np.mean(
            np.abs((y_true - y_pred) / (np.abs(y_true) + eps))
        ) * 100
    )


def regression_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE(%)': mape(y_true, y_pred),
        'sMAPE(%)': smape(y_true, y_pred),
        'WMAPE(%)': wmape(y_true, y_pred),
        'n': len(y_true)
    }


def row_normalize(mat, eps=1e-08):
    row_sum = mat.sum(axis=1, keepdims=True)
    return mat / (row_sum + eps)


def symmetric_binary_knn(indices, n_nodes):
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for i in range(n_nodes):
        for j in indices[i]:
            if i != j:
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A


def add_self_loop(A):
    A = A.copy()
    np.fill_diagonal(A, 1.0)
    return A


def detect_numeric_feature_cols(df, exclude_cols):
    feature_cols = []

    for c in df.columns:
        if c in exclude_cols:
            continue

        s = pd.to_numeric(df[c], errors='coerce')

        if s.notna().sum() > 0:
            df[c] = s
            feature_cols.append(c)

    return df, feature_cols


def ensure_dirs():
    dirs = [
        RESULT_DIR,
        METRICS_DIR,
        PRED_DIR,
        SHAP_DIR,
        ALE_DIR,
        MODEL_DIR,
        INTERACTION_DIR,
        os.path.join(ALE_DIR, 'global'),
        os.path.join(ALE_DIR, 'province'),
        os.path.join(ALE_DIR, 'cii_class')
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)


def random_split_indices(
    n,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-08:
        raise ValueError('TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.')

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    n_test = n - n_train - n_valid

    if n_train < 1 or n_valid < 1 or n_test < 1:
        raise ValueError(
            'The sample size is too small to ensure that train, valid, '
            'and test each contain at least one sample.'
        )

    train_idx = perm[:n_train]
    valid_idx = perm[n_train:n_train + n_valid]
    test_idx = perm[n_train + n_valid:]

    return train_idx, valid_idx, test_idx


def aggregate_group_mean_abs_shap(shap_df, feature_names, group_col):
    rows = []

    for g, sub in shap_df.groupby(group_col, dropna=False):
        vals = np.abs(sub[feature_names].values.astype(np.float64))
        mean_abs = vals.mean(axis=0)

        row = {
            group_col: g,
            'n': len(sub)
        }

        for f, v in zip(feature_names, mean_abs):
            row[f] = v

        rows.append(row)

    wide_df = pd.DataFrame(rows)

    if len(wide_df) == 0:
        return wide_df, pd.DataFrame(
            columns=[group_col, 'n', 'feature', 'mean_abs_shap']
        )

    long_df = (
        wide_df
        .melt(
            id_vars=[group_col, 'n'],
            var_name='feature',
            value_name='mean_abs_shap'
        )
        .sort_values(
            [group_col, 'mean_abs_shap'],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )

    return wide_df, long_df


def plot_group_shap_heatmap(
    group_wide_df,
    group_col,
    feature_names,
    save_path,
    top_n_features=10
):
    if len(group_wide_df) == 0:
        return

    global_mean = group_wide_df[feature_names].mean(axis=0).sort_values(ascending=False)
    top_feats = global_mean.index.tolist()[:top_n_features]

    plot_df = group_wide_df[[group_col] + top_feats].copy()
    plot_df = plot_df.sort_values(group_col).reset_index(drop=True)

    mat = plot_df[top_feats].values.astype(float)

    plt.figure(figsize=(1.2 * len(top_feats) + 4, 0.35 * len(plot_df) + 3))
    plt.imshow(mat, aspect='auto')
    plt.colorbar(label='Mean |SHAP|')
    plt.xticks(np.arange(len(top_feats)), top_feats, rotation=45, ha='right')
    plt.yticks(np.arange(len(plot_df)), plot_df[group_col].astype(str).tolist())
    plt.xlabel('Feature')
    plt.ylabel(group_col)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def get_quantile_bin_edges(x_raw, n_bins=20):
    x_raw = np.asarray(x_raw).reshape(-1)
    x_raw = x_raw[~np.isnan(x_raw)]

    if len(x_raw) < max(20, n_bins + 1):
        return None

    edges = np.quantile(x_raw, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) < 3:
        return None

    return edges


def raw_to_scaled_value(raw_value, scaler, feat_idx):
    mu = float(scaler.mean_[feat_idx])
    sd = float(scaler.scale_[feat_idx])

    if abs(sd) < 1e-12:
        return 0.0

    return (float(raw_value) - mu) / sd


def assign_cii_class(y):
    y = float(y)

    if y <= 0:
        return 'non cooperation (CII ≤ 0)'
    elif y <= 0.2:
        return 'low cooperation (0 < CII ≤ 0.2)'
    elif y <= 0.5:
        return 'moderate cooperation (0.2 < CII ≤ 0.5)'
    else:
        return 'high cooperation (CII > 0.5)'


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) < 2:
        return np.nan

    var = np.var(y_true)

    if var < 1e-12:
        return 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    return 1.0 - ss_res / (ss_tot + 1e-12)


def get_quantile_bin_codes(x, n_bins):
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if len(x) < max(20, n_bins + 1):
        return None, None

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)

    if len(edges) < 3:
        return None, None

    codes = np.digitize(x, edges[1:-1], right=True)

    return codes.astype(int), edges


def compute_ordered_shap_interaction_score(
    x_main,
    shap_main,
    x_mod,
    n_main_bins=8,
    n_mod_bins=4,
    min_cell=5
):
    x_main = np.asarray(x_main, dtype=np.float64).reshape(-1)
    shap_main = np.asarray(shap_main, dtype=np.float64).reshape(-1)
    x_mod = np.asarray(x_mod, dtype=np.float64).reshape(-1)

    ok = ~np.isnan(x_main) & ~np.isnan(shap_main) & ~np.isnan(x_mod)

    x_main = x_main[ok]
    shap_main = shap_main[ok]
    x_mod = x_mod[ok]

    if len(x_main) < max(100, n_main_bins * n_mod_bins * 2):
        return np.nan

    main_codes, _ = get_quantile_bin_codes(x_main, n_main_bins)
    mod_codes, _ = get_quantile_bin_codes(x_mod, n_mod_bins)

    if main_codes is None or mod_codes is None:
        return np.nan

    tmp = pd.DataFrame({
        'main_bin': main_codes,
        'mod_bin': mod_codes,
        'shap': shap_main
    })

    base_mean = tmp.groupby('main_bin')['shap'].mean().to_dict()
    tmp['pred_base'] = tmp['main_bin'].map(base_mean)

    cell_stats = (
        tmp.groupby(['main_bin', 'mod_bin'])['shap']
        .agg(['mean', 'size'])
        .reset_index()
    )

    cell_mean_map = {
        (int(r['main_bin']), int(r['mod_bin'])): float(r['mean'])
        for _, r in cell_stats.iterrows()
        if int(r['size']) >= min_cell
    }

    pred_joint = []

    for mb, xb, pb in zip(
        tmp['main_bin'].values,
        tmp['mod_bin'].values,
        tmp['pred_base'].values
    ):
        pred_joint.append(cell_mean_map.get((int(mb), int(xb)), float(pb)))

    pred_joint = np.asarray(pred_joint, dtype=np.float64)

    r2_base = safe_r2(tmp['shap'].values, tmp['pred_base'].values)
    r2_joint = safe_r2(tmp['shap'].values, pred_joint)

    if np.isnan(r2_base) or np.isnan(r2_joint):
        return np.nan

    return max(0.0, float(r2_joint - r2_base))


def screen_top_interactions(
    shap_df,
    feature_value_df,
    importance_df,
    feature_names,
    top_k=12,
    top_pairs=4,
    n_main_bins=8,
    n_mod_bins=4,
    min_cell=5
):
    candidate_features = importance_df['feature'].tolist()[:top_k]
    score_rows = []

    importance_index = importance_df.set_index('feature')

    for f1, f2 in itertools.combinations(candidate_features, 2):
        i = feature_names.index(f1)
        j = feature_names.index(f2)

        x1 = feature_value_df[f1].values.astype(np.float64)
        x2 = feature_value_df[f2].values.astype(np.float64)

        s1 = shap_df[f1].values.astype(np.float64)
        s2 = shap_df[f2].values.astype(np.float64)

        score_12 = compute_ordered_shap_interaction_score(
            x_main=x1,
            shap_main=s1,
            x_mod=x2,
            n_main_bins=n_main_bins,
            n_mod_bins=n_mod_bins,
            min_cell=min_cell
        )

        score_21 = compute_ordered_shap_interaction_score(
            x_main=x2,
            shap_main=s2,
            x_mod=x1,
            n_main_bins=n_main_bins,
            n_mod_bins=n_mod_bins,
            min_cell=min_cell
        )

        if np.isnan(score_12) and np.isnan(score_21):
            continue

        score_sym = np.nanmean([score_12, score_21])

        main_feature = (
            f1
            if importance_index.loc[f1, 'mean_abs_shap'] >= importance_index.loc[f2, 'mean_abs_shap']
            else f2
        )

        mod_feature = f2 if main_feature == f1 else f1

        score_rows.append({
            'feature_1': f1,
            'feature_2': f2,
            'score_f1_given_f2': score_12,
            'score_f2_given_f1': score_21,
            'interaction_score': score_sym,
            'main_feature_for_plot': main_feature,
            'mod_feature_for_plot': mod_feature
        })

    score_df = pd.DataFrame(score_rows)

    if len(score_df) == 0:
        return score_df, score_df

    score_df = score_df.sort_values('interaction_score', ascending=False).reset_index(drop=True)
    top_df = score_df.head(top_pairs).copy()

    return score_df, top_df


def plot_top_interaction_dependence_grid(
    top_pairs_df,
    shap_df,
    feature_value_df,
    save_path,
    max_points=2000,
    random_seed=42
):
    if len(top_pairs_df) == 0:
        return

    n_plot = len(top_pairs_df)
    n_cols = 2
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.RandomState(random_seed)

    for ax, (_, row) in zip(axes, top_pairs_df.iterrows()):
        main_f = row['main_feature_for_plot']
        mod_f = row['mod_feature_for_plot']

        x = feature_value_df[main_f].values.astype(np.float64)
        y = shap_df[main_f].values.astype(np.float64)
        c = feature_value_df[mod_f].values.astype(np.float64)

        ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(c)

        x = x[ok]
        y = y[ok]
        c = c[ok]

        if len(x) > max_points:
            idx = rng.choice(len(x), size=max_points, replace=False)
            x = x[idx]
            y = y[idx]
            c = c[idx]

        sc = ax.scatter(
            x,
            y,
            c=c,
            cmap='viridis',
            s=18,
            alpha=0.75,
            edgecolors='none'
        )

        ax.axhline(0, linestyle='--', linewidth=1.0, color='gray', alpha=0.8)
        ax.set_xlabel(main_f, fontsize=12)
        ax.set_ylabel(f'SHAP value of {main_f}', fontsize=12)
        ax.set_title(f'{main_f} × {mod_f}', fontsize=13)

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(mod_f, fontsize=11)

        ax.tick_params(labelsize=10)

    for ax in axes[n_plot:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def save_checkpoint(
    checkpoint_path,
    best_state,
    best_epoch,
    best_valid_r2,
    feature_cols,
    node_codes,
    all_years,
    x_scaler,
    static_scaler,
    feature_fill_values,
    static_feature_cols
):
    payload = {
        'model_state_dict': best_state,
        'best_epoch': int(best_epoch),
        'best_valid_r2': float(best_valid_r2),
        'feature_cols': list(feature_cols),
        'node_codes': list(node_codes),
        'all_years': list(all_years),
        'x_scaler_mean': np.asarray(x_scaler.mean_, dtype=np.float32),
        'x_scaler_scale': np.asarray(x_scaler.scale_, dtype=np.float32),
        'static_scaler_mean': np.asarray(static_scaler.mean_, dtype=np.float32),
        'static_scaler_scale': np.asarray(static_scaler.scale_, dtype=np.float32),
        'feature_fill_values': feature_fill_values,
        'static_feature_cols': list(static_feature_cols),
        'seed': int(SEED)
    }

    torch.save(payload, checkpoint_path)


ensure_dirs()

csv_df = pd.read_csv(CSV_PATH)

target_col = csv_df.columns[-1]

assert CSV_CODE_COL in csv_df.columns, f'CSV is missing {CSV_CODE_COL}'
assert YEAR_COL in csv_df.columns, f'CSV is missing {YEAR_COL}'
assert PROVINCE_COL in csv_df.columns, f'CSV is missing {PROVINCE_COL}'

csv_df[CSV_CODE_COL] = csv_df[CSV_CODE_COL].apply(clean_code)
csv_df[PROVINCE_COL] = csv_df[PROVINCE_COL].apply(clean_code)
csv_df[YEAR_COL] = pd.to_numeric(csv_df[YEAR_COL], errors='coerce').astype(int)

dup_mask = csv_df.duplicated([CSV_CODE_COL, YEAR_COL])

if dup_mask.any():
    raise ValueError(
        'Duplicate Code-Year combinations exist in the CSV file. '
        'Please remove duplicates first.'
    )

province_nunique = csv_df.groupby(CSV_CODE_COL)[PROVINCE_COL].nunique(dropna=True)

if (province_nunique > 1).any():
    bad_codes = province_nunique[province_nunique > 1].index.tolist()[:20]
    raise ValueError(
        f'Some Code values correspond to multiple Province_ID values. '
        f'The first 20 examples are: {bad_codes}'
    )

code_to_province = (
    csv_df[[CSV_CODE_COL, PROVINCE_COL]]
    .dropna()
    .drop_duplicates(subset=[CSV_CODE_COL])
    .set_index(CSV_CODE_COL)[PROVINCE_COL]
    .to_dict()
)

shp_gdf = gpd.read_file(SHP_PATH)

assert SHP_CODE_COL in shp_gdf.columns, f'SHP is missing {SHP_CODE_COL}'

shp_gdf[SHP_CODE_COL] = shp_gdf[SHP_CODE_COL].apply(clean_code)
shp_gdf = shp_gdf[[SHP_CODE_COL, 'geometry']].copy()
shp_gdf = shp_gdf.dropna(subset=[SHP_CODE_COL]).reset_index(drop=True)

if shp_gdf.crs is None:
    raise ValueError(
        'The SHP file is missing CRS information. '
        'Please assign a projection to the SHP file first.'
    )

if shp_gdf.crs.is_geographic:
    shp_gdf = shp_gdf.to_crs(epsg=3857)

csv_codes = set(csv_df[CSV_CODE_COL].dropna().unique().tolist())
shp_codes = set(shp_gdf[SHP_CODE_COL].dropna().unique().tolist())

common_codes = sorted(csv_codes & shp_codes)

if len(common_codes) == 0:
    raise ValueError(
        'No matching codes were found between the CSV Code field and the SHP code field.'
    )

csv_df = csv_df[csv_df[CSV_CODE_COL].isin(common_codes)].copy()
shp_gdf = shp_gdf[shp_gdf[SHP_CODE_COL].isin(common_codes)].copy()

shp_gdf = shp_gdf.drop_duplicates(subset=[SHP_CODE_COL]).reset_index(drop=True)

node_codes = shp_gdf[SHP_CODE_COL].tolist()

code2idx = {
    c: i
    for i, c in enumerate(node_codes)
}

idx2code = {
    i: c
    for c, i in code2idx.items()
}

n_nodes = len(node_codes)

exclude_cols = [
    CSV_CODE_COL,
    YEAR_COL,
    PROVINCE_COL,
    target_col
]

csv_df, feature_cols = detect_numeric_feature_cols(csv_df, exclude_cols)

all_years = sorted(csv_df[YEAR_COL].unique().tolist())

full_index = (
    pd.MultiIndex
    .from_product(
        [node_codes, all_years],
        names=[CSV_CODE_COL, YEAR_COL]
    )
    .to_frame(index=False)
)

panel_df = full_index.merge(csv_df, on=[CSV_CODE_COL, YEAR_COL], how='left')

panel_df[target_col] = pd.to_numeric(panel_df[target_col], errors='coerce')
panel_df['node_idx'] = panel_df[CSV_CODE_COL].map(code2idx)
panel_df[PROVINCE_COL] = panel_df[CSV_CODE_COL].map(code_to_province)

panel_df = panel_df.sort_values([YEAR_COL, 'node_idx']).reset_index(drop=True)

feature_fill_values = {}

for c in feature_cols:
    val = pd.to_numeric(panel_df[c], errors='coerce').mean()

    if pd.isna(val):
        val = 0.0

    feature_fill_values[c] = float(val)
    panel_df[c] = pd.to_numeric(panel_df[c], errors='coerce').fillna(val)

valid_target_mask = panel_df[target_col].notna().values
n_valid_rows = int(valid_target_mask.sum())

if n_valid_rows < 3:
    raise ValueError(
        'The number of valid target samples is too small to split into '
        'train, valid, and test sets.'
    )

train_row_idx_rel, valid_row_idx_rel, test_row_idx_rel = random_split_indices(
    n_valid_rows,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    seed=SEED
)

valid_global_idx = np.where(valid_target_mask)[0]

train_global_idx = valid_global_idx[train_row_idx_rel]
valid_global_idx2 = valid_global_idx[valid_row_idx_rel]
test_global_idx = valid_global_idx[test_row_idx_rel]

panel_df['split'] = 'unused'
panel_df.loc[train_global_idx, 'split'] = 'train'
panel_df.loc[valid_global_idx2, 'split'] = 'valid'
panel_df.loc[test_global_idx, 'split'] = 'test'

split_counts = panel_df.loc[
    panel_df['split'].isin(['train', 'valid', 'test']),
    'split'
].value_counts()

panel_split_path = os.path.join(PRED_DIR, 'panel_split_assignments.csv')

panel_df[
    [CSV_CODE_COL, YEAR_COL, PROVINCE_COL, 'node_idx', 'split', target_col]
].to_csv(
    panel_split_path,
    index=False,
    encoding='utf-8-sig'
)

X_train_fit = panel_df.loc[
    panel_df['split'] == 'train',
    feature_cols
].values.astype(np.float32)

x_scaler = StandardScaler()
x_scaler.fit(X_train_fit)

year_to_X_scaled = {}
year_to_X_raw = {}
year_to_y = {}
year_to_train_mask = {}
year_to_valid_mask = {}
year_to_test_mask = {}
year_to_province = {}

for y in all_years:
    sub = (
        panel_df[panel_df[YEAR_COL] == y]
        .sort_values('node_idx')
        .reset_index(drop=True)
    )

    X_raw_mat = sub[feature_cols].values.astype(np.float32)
    X_scaled_mat = x_scaler.transform(X_raw_mat).astype(np.float32)
    y_vec = sub[target_col].values.astype(np.float32)

    train_mask = (sub['split'].values == 'train') & ~np.isnan(y_vec)
    valid_mask = (sub['split'].values == 'valid') & ~np.isnan(y_vec)
    test_mask = (sub['split'].values == 'test') & ~np.isnan(y_vec)

    year_to_X_raw[y] = X_raw_mat
    year_to_X_scaled[y] = X_scaled_mat
    year_to_y[y] = y_vec

    year_to_train_mask[y] = train_mask.astype(np.bool_)
    year_to_valid_mask[y] = valid_mask.astype(np.bool_)
    year_to_test_mask[y] = test_mask.astype(np.bool_)

    year_to_province[y] = sub[PROVINCE_COL].astype(str).values

    if np.isnan(y_vec).any():
        miss_n = int(np.isnan(y_vec).sum())

geom_df = shp_gdf.copy()

geom_df['centroid_x'] = geom_df.geometry.centroid.x
geom_df['centroid_y'] = geom_df.geometry.centroid.y
geom_df['area'] = geom_df.geometry.area
geom_df['node_idx'] = geom_df[SHP_CODE_COL].map(code2idx)

geom_df = geom_df.sort_values('node_idx').reset_index(drop=True)

static_feature_cols = [
    'centroid_x',
    'centroid_y',
    'area'
]

static_matrix = geom_df[static_feature_cols].values.astype(np.float32)

static_scaler = StandardScaler()
static_matrix = static_scaler.fit_transform(static_matrix).astype(np.float32)


def build_geo_adjacency(gdf, code_col):
    gdf = gdf[[code_col, 'geometry']].copy().reset_index(drop=True)

    n = len(gdf)

    A = np.zeros((n, n), dtype=np.float32)

    sindex = gdf.sindex

    for i, geom in enumerate(gdf.geometry):
        cand_idx = list(sindex.query(geom, predicate='touches'))

        for j in cand_idx:
            if i != j:
                A[i, j] = 1.0

    A = np.maximum(A, A.T)

    return A


geo_gdf = shp_gdf.copy().reset_index(drop=True)

A_geo_bin = build_geo_adjacency(geo_gdf, SHP_CODE_COL)
A_geo_norm = row_normalize(add_self_loop(A_geo_bin))

coords = np.column_stack([
    geom_df['centroid_x'].values,
    geom_df['centroid_y'].values
]).astype(np.float32)

nbrs_dist = NearestNeighbors(
    n_neighbors=min(DIST_K + 1, n_nodes),
    metric='euclidean'
)

nbrs_dist.fit(coords)

dist_indices = nbrs_dist.kneighbors(coords, return_distance=False)
dist_indices = [idxs[1:] for idxs in dist_indices]

A_dist_bin = symmetric_binary_knn(dist_indices, n_nodes)
A_dist_norm = row_normalize(add_self_loop(A_dist_bin))

sim_basis = static_matrix.copy()

nbrs_sim = NearestNeighbors(
    n_neighbors=min(SIM_K + 1, n_nodes),
    metric='cosine'
)

nbrs_sim.fit(sim_basis)

sim_indices = nbrs_sim.kneighbors(sim_basis, return_distance=False)
sim_indices = [idxs[1:] for idxs in sim_indices]

A_sim_bin = symmetric_binary_knn(sim_indices, n_nodes)
A_sim_norm = row_normalize(add_self_loop(A_sim_bin))


def build_year_graph_samples(mask_dict):
    samples = []

    for y in all_years:
        mask = mask_dict[y]

        if mask.sum() == 0:
            continue

        samples.append({
            'target_year': y,
            'X_scaled': year_to_X_scaled[y].astype(np.float32),
            'X_raw': year_to_X_raw[y].astype(np.float32),
            'y': year_to_y[y].astype(np.float32),
            'province': year_to_province[y].astype(str),
            'mask': mask.astype(np.bool_)
        })

    return samples


train_samples = build_year_graph_samples(year_to_train_mask)
valid_samples = build_year_graph_samples(year_to_valid_mask)
test_samples = build_year_graph_samples(year_to_test_mask)

if len(train_samples) == 0 or len(valid_samples) == 0 or len(test_samples) == 0:
    raise ValueError(
        'The train, validation, or test samples are empty. '
        'Please check the random split logic.'
    )


class YearGraphMaskedDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return (
            torch.tensor(s['X_scaled'], dtype=torch.float32),
            torch.tensor(s['y'], dtype=torch.float32),
            torch.tensor(s['mask'], dtype=torch.bool),
            torch.tensor(int(s['target_year']), dtype=torch.long)
        )


train_ds = YearGraphMaskedDataset(train_samples)
valid_ds = YearGraphMaskedDataset(valid_samples)
test_ds = YearGraphMaskedDataset(test_samples)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


class DiffusionGraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, order=2):
        super().__init__()

        self.order = order
        self.proj = nn.Linear((order + 1) * in_dim, out_dim)

    def forward(self, x, A):
        outs = [x]
        xk = x

        for _ in range(self.order):
            xk = torch.bmm(A, xk)
            outs.append(xk)

        h = torch.cat(outs, dim=-1)

        return self.proj(h)


class AdaptiveGraphGenerator(nn.Module):

    def __init__(self, n_nodes, static_dim, emb_dim):
        super().__init__()

        self.node_emb1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

        self.static_proj1 = nn.Linear(static_dim, emb_dim, bias=False)
        self.static_proj2 = nn.Linear(static_dim, emb_dim, bias=False)

    def forward(self, static_x):
        e1 = self.node_emb1 + self.static_proj1(static_x)
        e2 = self.node_emb2 + self.static_proj2(static_x)

        logits = F.relu(torch.matmul(e1, e2.T))
        A_adp = F.softmax(logits, dim=1)

        return A_adp, e1


class SpatialGraphFusionRegressor(nn.Module):

    def __init__(
        self,
        n_nodes,
        in_dim,
        static_dim,
        A_geo_norm,
        A_dist_norm,
        A_sim_norm,
        A_geo_bin,
        static_matrix,
        hidden_dim=128,
        graph_hidden_dim=64,
        static_hidden_dim=16,
        adp_emb_dim=16,
        dropout=0.1
    ):
        super().__init__()

        self.register_buffer('A_geo_norm', torch.tensor(A_geo_norm, dtype=torch.float32))
        self.register_buffer('A_dist_norm', torch.tensor(A_dist_norm, dtype=torch.float32))
        self.register_buffer('A_sim_norm', torch.tensor(A_sim_norm, dtype=torch.float32))
        self.register_buffer('A_geo_bin', torch.tensor(A_geo_bin, dtype=torch.float32))
        self.register_buffer('static_x', torch.tensor(static_matrix, dtype=torch.float32))

        self.n_nodes = n_nodes

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, static_hidden_dim),
            nn.ReLU()
        )

        self.adp_graph = AdaptiveGraphGenerator(
            n_nodes=n_nodes,
            static_dim=static_dim,
            emb_dim=adp_emb_dim
        )

        self.graph_logits = nn.Parameter(torch.zeros(4))

        self.gconv1 = DiffusionGraphConv(hidden_dim, graph_hidden_dim, order=2)
        self.gconv2 = DiffusionGraphConv(graph_hidden_dim, graph_hidden_dim, order=2)

        self.dropout = nn.Dropout(dropout)

        self.fusion_gate = nn.Sequential(
            nn.Linear(graph_hidden_dim + static_hidden_dim, graph_hidden_dim),
            nn.Sigmoid()
        )

        self.static_to_graph = nn.Linear(static_hidden_dim, graph_hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(graph_hidden_dim + static_hidden_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_dim, 1)
        )

    def build_fused_graph(self, batch_size):
        A_adp, e_ctx = self.adp_graph(self.static_x)

        gate = F.softmax(self.graph_logits, dim=0)

        A_geo = self.A_geo_norm.unsqueeze(0).expand(batch_size, -1, -1)
        A_dist = self.A_dist_norm.unsqueeze(0).expand(batch_size, -1, -1)
        A_sim = self.A_sim_norm.unsqueeze(0).expand(batch_size, -1, -1)
        A_adp_b = A_adp.unsqueeze(0).expand(batch_size, -1, -1)

        A_fused = (
            gate[0] * A_geo +
            gate[1] * A_dist +
            gate[2] * A_sim +
            gate[3] * A_adp_b
        )

        A_fused = A_fused / (A_fused.sum(dim=-1, keepdim=True) + 1e-08)

        return A_fused, A_adp, e_ctx

    def forward(self, x):
        B = x.size(0)

        h = F.relu(self.input_proj(x))

        A_fused, A_adp, e_ctx = self.build_fused_graph(B)

        h_graph = F.relu(self.gconv1(h, A_fused))
        h_graph = self.dropout(h_graph)

        h_graph = F.relu(self.gconv2(h_graph, A_fused))
        h_graph = self.dropout(h_graph)

        static_h = self.static_proj(self.static_x)
        static_h = static_h.unsqueeze(0).expand(B, -1, -1)

        static_proj = self.static_to_graph(static_h)

        gate = self.fusion_gate(torch.cat([h_graph, static_h], dim=-1))

        h_fused = gate * h_graph + (1.0 - gate) * static_proj

        h_out = torch.cat([h_fused, static_h], dim=-1)

        y_pred = self.head(h_out).squeeze(-1)

        D = self.A_geo_bin.sum(dim=1)
        lap_e = D.unsqueeze(1) * e_ctx - torch.matmul(self.A_geo_bin, e_ctx)

        smooth_reg = (e_ctx * lap_e).sum() / (self.A_geo_bin.sum() + 1e-08)

        reg_dict = {
            'smooth_reg': smooth_reg
        }

        return y_pred, reg_dict


model = SpatialGraphFusionRegressor(
    n_nodes=n_nodes,
    in_dim=len(feature_cols),
    static_dim=static_matrix.shape[1],
    A_geo_norm=A_geo_norm,
    A_dist_norm=A_dist_norm,
    A_sim_norm=A_sim_norm,
    A_geo_bin=A_geo_bin,
    static_matrix=static_matrix,
    hidden_dim=HIDDEN_DIM,
    graph_hidden_dim=GRAPH_HIDDEN_DIM,
    static_hidden_dim=STATIC_HIDDEN_DIM,
    adp_emb_dim=ADP_EMB_DIM,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None

    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    preds_all = []
    trues_all = []
    years_all = []

    for batch in loader:
        x, y_true, mask, target_year = batch

        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)
        mask = mask.to(DEVICE).bool()

        with torch.set_grad_enabled(is_train):
            y_pred, reg_dict = model(x)

            pred_sel = y_pred[mask]
            true_sel = y_true[mask]

            if pred_sel.numel() == 0:
                continue

            mse_loss = criterion(pred_sel, true_sel)
            loss = mse_loss + LAMBDA_SMOOTH * reg_dict['smooth_reg']

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        n_batches += 1

        pred_np = pred_sel.detach().cpu().numpy().reshape(-1)
        true_np = true_sel.detach().cpu().numpy().reshape(-1)

        preds_all.append(pred_np)
        trues_all.append(true_np)
        years_all.append(np.repeat(int(target_year.item()), len(true_np)))

    if n_batches == 0:
        raise ValueError(
            'The current loader has no samples available for loss calculation.'
        )

    preds_all = np.concatenate(preds_all)
    trues_all = np.concatenate(trues_all)
    years_all = np.concatenate(years_all)

    avg_loss = total_loss / n_batches
    avg_mse = total_mse / n_batches

    return avg_loss, avg_mse, trues_all, preds_all, years_all


def evaluate_with_state(model, state_dict, train_loader, valid_loader, test_loader):
    model.load_state_dict(state_dict)
    model.eval()

    _, _, y_train_true, y_train_pred, years_train = run_epoch(
        model,
        train_loader,
        optimizer=None
    )

    _, _, y_valid_true, y_valid_pred, years_valid = run_epoch(
        model,
        valid_loader,
        optimizer=None
    )

    _, _, y_test_true, y_test_pred, years_test = run_epoch(
        model,
        test_loader,
        optimizer=None
    )

    train_metrics = regression_metrics(y_train_true, y_train_pred)
    valid_metrics = regression_metrics(y_valid_true, y_valid_pred)
    test_metrics = regression_metrics(y_test_true, y_test_pred)

    return {
        'train': (train_metrics, y_train_true, y_train_pred, years_train),
        'valid': (valid_metrics, y_valid_true, y_valid_pred, years_valid),
        'test': (test_metrics, y_test_true, y_test_pred, years_test)
    }


best_valid_r2 = -np.inf
best_state = None
best_epoch = -1

patience_count = 0
lr_wait_count = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, _, y_train_true, y_train_pred, _ = run_epoch(
        model,
        train_loader,
        optimizer
    )

    valid_loss, _, y_valid_true, y_valid_pred, _ = run_epoch(
        model,
        valid_loader,
        optimizer=None
    )

    train_r2 = r2_score(y_train_true, y_train_pred)
    valid_r2 = r2_score(y_valid_true, y_valid_pred)

    current_lr = optimizer.param_groups[0]['lr']

    if epoch % 10 == 0 or epoch == 1:
        pass

    if valid_r2 > best_valid_r2:
        best_valid_r2 = valid_r2
        best_epoch = epoch
        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }

        patience_count = 0
        lr_wait_count = 0

    else:
        patience_count += 1
        lr_wait_count += 1

        if lr_wait_count >= LR_REDUCE_PATIENCE:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= LR_REDUCE_FACTOR

            new_lr = optimizer.param_groups[0]['lr']
            lr_wait_count = 0

        if patience_count >= PATIENCE:
            break


best_result = evaluate_with_state(
    model,
    best_state,
    train_loader,
    valid_loader,
    test_loader
)

result_df = pd.DataFrame([
    {'split': 'train', **best_result['train'][0]},
    {'split': 'valid', **best_result['valid'][0]},
    {'split': 'test', **best_result['test'][0]}
])

metrics_path = os.path.join(METRICS_DIR, 'train_valid_test_metrics.csv')

result_df.to_csv(
    metrics_path,
    index=False,
    encoding='utf-8-sig'
)

checkpoint_path = os.path.join(MODEL_DIR, 'best_model_checkpoint.pt')

save_checkpoint(
    checkpoint_path=checkpoint_path,
    best_state=best_state,
    best_epoch=best_epoch,
    best_valid_r2=best_valid_r2,
    feature_cols=feature_cols,
    node_codes=node_codes,
    all_years=all_years,
    x_scaler=x_scaler,
    static_scaler=static_scaler,
    feature_fill_values=feature_fill_values,
    static_feature_cols=static_feature_cols
)


def predict_samples_for_split(model, samples):
    rows = []

    model.eval()

    with torch.no_grad():
        for s in samples:
            x = torch.tensor(
                s['X_scaled'],
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            y_pred, _ = model(x)

            y_pred = y_pred.squeeze(0).detach().cpu().numpy()

            y_true = s['y']
            mask = s['mask']
            target_year = int(s['target_year'])
            province_arr = s['province']

            node_indices = np.where(mask)[0]

            for node_idx in node_indices:
                rows.append({
                    'target_year': target_year,
                    'node_idx': int(node_idx),
                    'Code': idx2code[int(node_idx)],
                    PROVINCE_COL: province_arr[node_idx],
                    'y_true': float(y_true[node_idx]),
                    'y_pred': float(y_pred[node_idx])
                })

    return pd.DataFrame(rows)


model.load_state_dict(best_state)

train_pred_df = predict_samples_for_split(model, train_samples)
valid_pred_df = predict_samples_for_split(model, valid_samples)
test_pred_df = predict_samples_for_split(model, test_samples)

train_pred_path = os.path.join(PRED_DIR, 'train_true_pred.csv')
valid_pred_path = os.path.join(PRED_DIR, 'valid_true_pred.csv')
test_pred_path = os.path.join(PRED_DIR, 'test_true_pred.csv')
all_pred_path = os.path.join(PRED_DIR, 'all_splits_true_pred.csv')

train_pred_df.to_csv(
    train_pred_path,
    index=False,
    encoding='utf-8-sig'
)

valid_pred_df.to_csv(
    valid_pred_path,
    index=False,
    encoding='utf-8-sig'
)

test_pred_df.to_csv(
    test_pred_path,
    index=False,
    encoding='utf-8-sig'
)

pd.concat(
    [
        train_pred_df.assign(split='train'),
        valid_pred_df.assign(split='valid'),
        test_pred_df.assign(split='test')
    ],
    axis=0,
    ignore_index=True
).to_csv(
    all_pred_path,
    index=False,
    encoding='utf-8-sig'
)


class SingleNodeSpatialWrapper(nn.Module):

    def __init__(self, base_model, base_x_scaled, node_idx):
        super().__init__()

        self.base_model = base_model
        self.base_model.eval()

        self.node_idx = int(node_idx)

        self.register_buffer(
            'base_x_scaled',
            base_x_scaled.clone()
        )

    def forward(self, x_node_scaled):
        B = x_node_scaled.size(0)

        full_x = self.base_x_scaled.unsqueeze(0).repeat(B, 1, 1).clone()
        full_x[:, self.node_idx, :] = x_node_scaled

        y_pred, _ = self.base_model(full_x)

        out = y_pred[:, self.node_idx].unsqueeze(1)

        return out


def build_local_node_arrays(samples, include_pred=False, pred_df=None):
    rows = []
    x_scaled_list = []
    x_raw_list = []

    pred_lookup = None

    if include_pred:
        pred_lookup = {
            (int(r.target_year), int(r.node_idx)): float(r.y_pred)
            for _, r in pred_df.iterrows()
        }

    for s in samples:
        target_year = int(s['target_year'])
        mask = s['mask']

        node_indices = np.where(mask)[0]

        for node_idx in node_indices:
            x_scaled_list.append(s['X_scaled'][node_idx])
            x_raw_list.append(s['X_raw'][node_idx])

            row = {
                'target_year': target_year,
                'node_idx': int(node_idx),
                'Code': idx2code[int(node_idx)],
                PROVINCE_COL: s['province'][node_idx],
                'y_true': float(s['y'][node_idx])
            }

            if include_pred:
                row['y_pred'] = pred_lookup[target_year, int(node_idx)]

            rows.append(row)

    meta_df = pd.DataFrame(rows)
    x_scaled_arr = np.stack(x_scaled_list, axis=0).astype(np.float32)
    x_raw_arr = np.stack(x_raw_list, axis=0).astype(np.float32)

    return meta_df, x_scaled_arr, x_raw_arr


train_meta_df, train_x_scaled_local, train_x_raw_local = build_local_node_arrays(
    train_samples,
    include_pred=False
)

test_meta_df, test_x_scaled_local, test_x_raw_local = build_local_node_arrays(
    test_samples,
    include_pred=True,
    pred_df=test_pred_df
)

if SHAP_TEST_NODE_SIZE is not None and len(test_meta_df) > SHAP_TEST_NODE_SIZE:
    rng = np.random.RandomState(SEED)

    sampled_idx = rng.choice(
        len(test_meta_df),
        size=SHAP_TEST_NODE_SIZE,
        replace=False
    )

    test_meta_df = test_meta_df.iloc[sampled_idx].reset_index(drop=True)
    test_x_scaled_local = test_x_scaled_local[sampled_idx]
    test_x_raw_local = test_x_raw_local[sampled_idx]

bg_size = min(SHAP_BACKGROUND_SIZE, len(train_meta_df))

bg_idx = np.random.choice(
    len(train_meta_df),
    size=bg_size,
    replace=False
)

bg_x = torch.tensor(
    train_x_scaled_local[bg_idx],
    dtype=torch.float32,
    device=DEVICE
)

base_context_map = {}

for y in all_years:
    base_context_map[int(y)] = {
        'base_x_scaled': torch.tensor(
            year_to_X_scaled[y],
            dtype=torch.float32,
            device=DEVICE
        )
    }

model.load_state_dict(best_state)
model.eval()

raw_feature_names = feature_cols.copy()

shap_rows = []
feature_value_rows = []

importance_accum = np.zeros(len(raw_feature_names), dtype=np.float64)

total_nodes = len(test_meta_df)

for idx, row in test_meta_df.iterrows():
    target_year = int(row['target_year'])
    node_idx = int(row['node_idx'])

    ctx = base_context_map[target_year]

    wrapper = SingleNodeSpatialWrapper(
        base_model=model,
        base_x_scaled=ctx['base_x_scaled'],
        node_idx=node_idx
    ).to(DEVICE)

    wrapper.eval()

    explainer = shap.GradientExplainer(
        wrapper,
        bg_x,
        batch_size=SHAP_BATCH_SIZE,
        local_smoothing=SHAP_LOCAL_SMOOTHING
    )

    x_eval = torch.tensor(
        test_x_scaled_local[idx:idx + 1],
        dtype=torch.float32,
        device=DEVICE
    )

    shap_vals = explainer.shap_values(x_eval)

    sv = np.asarray(shap_vals)[0].reshape(-1)
    x_raw_val = test_x_raw_local[idx].reshape(-1)

    importance_accum += np.abs(sv)

    meta = {
        'target_year': target_year,
        'node_idx': node_idx,
        'Code': row['Code'],
        PROVINCE_COL: row[PROVINCE_COL],
        'y_true': row['y_true'],
        'y_pred': row['y_pred']
    }

    shap_row = meta.copy()
    feat_row = meta.copy()

    for feat_name, sval, fval in zip(raw_feature_names, sv, x_raw_val):
        shap_row[feat_name] = float(sval)
        feat_row[feat_name] = float(fval)

    shap_rows.append(shap_row)
    feature_value_rows.append(feat_row)

    if (idx + 1) % SHAP_PROGRESS_EVERY == 0 or idx + 1 == total_nodes:
        pass

shap_df = pd.DataFrame(shap_rows)
feature_value_df = pd.DataFrame(feature_value_rows)

shap_path = os.path.join(SHAP_DIR, 'spatial_direct_shap_raw.csv')
feature_value_path = os.path.join(SHAP_DIR, 'spatial_direct_shap_feature_values.csv')

shap_df.to_csv(
    shap_path,
    index=False,
    encoding='utf-8-sig'
)

feature_value_df.to_csv(
    feature_value_path,
    index=False,
    encoding='utf-8-sig'
)

importance_df = (
    pd.DataFrame({
        'feature': raw_feature_names,
        'mean_abs_shap': importance_accum / total_nodes
    })
    .sort_values('mean_abs_shap', ascending=False)
    .reset_index(drop=True)
)

importance_path = os.path.join(SHAP_DIR, 'spatial_direct_shap_importance.csv')

importance_df.to_csv(
    importance_path,
    index=False,
    encoding='utf-8-sig'
)

shap_matrix = shap_df[raw_feature_names].values.astype(np.float32)
feature_matrix = feature_value_df[raw_feature_names].values.astype(np.float32)

feature_df = pd.DataFrame(
    feature_matrix,
    columns=raw_feature_names
)

plot_top_n = len(raw_feature_names)

plt.figure(figsize=(8, 6))

shap.summary_plot(
    shap_matrix,
    feature_df,
    feature_names=raw_feature_names,
    show=False,
    max_display=plot_top_n,
    plot_type='dot'
)

plt.tight_layout()

beeswarm_path = os.path.join(SHAP_DIR, 'spatial_direct_shap_beeswarm.png')

plt.savefig(
    beeswarm_path,
    dpi=600,
    bbox_inches='tight'
)

plt.close()

plt.figure(figsize=(8, 6))

shap.summary_plot(
    shap_matrix,
    feature_df,
    feature_names=raw_feature_names,
    show=False,
    max_display=plot_top_n,
    plot_type='bar'
)

plt.tight_layout()

bar_path = os.path.join(SHAP_DIR, 'spatial_direct_shap_bar.png')

plt.savefig(
    bar_path,
    dpi=600,
    bbox_inches='tight'
)

plt.close()

year_shap_wide, year_shap_long = aggregate_group_mean_abs_shap(
    shap_df=shap_df,
    feature_names=raw_feature_names,
    group_col='target_year'
)

year_shap_wide.to_csv(
    os.path.join(SHAP_DIR, 'spatial_direct_shap_by_year_wide.csv'),
    index=False,
    encoding='utf-8-sig'
)

year_shap_long.to_csv(
    os.path.join(SHAP_DIR, 'spatial_direct_shap_by_year_long.csv'),
    index=False,
    encoding='utf-8-sig'
)

plot_group_shap_heatmap(
    group_wide_df=year_shap_wide,
    group_col='target_year',
    feature_names=raw_feature_names,
    save_path=os.path.join(SHAP_DIR, 'spatial_direct_shap_year_heatmap_top10.png'),
    top_n_features=10
)

province_shap_wide, province_shap_long = aggregate_group_mean_abs_shap(
    shap_df=shap_df,
    feature_names=raw_feature_names,
    group_col=PROVINCE_COL
)

province_shap_wide.to_csv(
    os.path.join(SHAP_DIR, 'spatial_direct_shap_by_province_wide.csv'),
    index=False,
    encoding='utf-8-sig'
)

province_shap_long.to_csv(
    os.path.join(SHAP_DIR, 'spatial_direct_shap_by_province_long.csv'),
    index=False,
    encoding='utf-8-sig'
)

plot_group_shap_heatmap(
    group_wide_df=province_shap_wide,
    group_col=PROVINCE_COL,
    feature_names=raw_feature_names,
    save_path=os.path.join(SHAP_DIR, 'spatial_direct_shap_province_heatmap_top10.png'),
    top_n_features=10
)

interaction_score_df, top_interactions_df = screen_top_interactions(
    shap_df=shap_df,
    feature_value_df=feature_value_df,
    importance_df=importance_df,
    feature_names=raw_feature_names,
    top_k=INTERACTION_SCREEN_TOP_K,
    top_pairs=INTERACTION_TOP_PAIRS_TO_PLOT,
    n_main_bins=INTERACTION_MAIN_BINS,
    n_mod_bins=INTERACTION_MOD_BINS,
    min_cell=INTERACTION_MIN_CELL
)

interaction_score_path = os.path.join(
    INTERACTION_DIR,
    'interaction_screen_scores.csv'
)

top_interactions_path = os.path.join(
    INTERACTION_DIR,
    'top_interactions_top4.csv'
)

interaction_score_df.to_csv(
    interaction_score_path,
    index=False,
    encoding='utf-8-sig'
)

top_interactions_df.to_csv(
    top_interactions_path,
    index=False,
    encoding='utf-8-sig'
)

if len(top_interactions_df) > 0:
    interaction_fig_path = os.path.join(
        INTERACTION_DIR,
        'top4_interaction_dependence.png'
    )

    plot_top_interaction_dependence_grid(
        top_pairs_df=top_interactions_df,
        shap_df=shap_df,
        feature_value_df=feature_value_df,
        save_path=interaction_fig_path,
        max_points=INTERACTION_MAX_POINTS_PLOT,
        random_seed=SEED
    )


def compute_1d_ale_for_subset(
    meta_df_sub,
    X_scaled_sub,
    X_raw_sub,
    feature_name,
    feature_idx,
    base_context_map,
    model,
    scaler,
    n_bins=20
):
    x_raw_col = X_raw_sub[:, feature_idx]

    edges = get_quantile_bin_edges(
        x_raw_col,
        n_bins=n_bins
    )

    if edges is None:
        return None

    local_effects = []
    bin_centers = []
    bin_counts = []

    for b in range(len(edges) - 1):
        left = edges[b]
        right = edges[b + 1]

        if b == 0:
            in_bin = (x_raw_col >= left) & (x_raw_col <= right)
        else:
            in_bin = (x_raw_col > left) & (x_raw_col <= right)

        idxs = np.where(in_bin)[0]

        if len(idxs) == 0:
            local_effects.append(0.0)
            bin_centers.append((left + right) / 2.0)
            bin_counts.append(0)
            continue

        diffs = []

        for i in idxs:
            row = meta_df_sub.iloc[i]

            target_year = int(row['target_year'])
            node_idx = int(row['node_idx'])

            wrapper = SingleNodeSpatialWrapper(
                base_model=model,
                base_x_scaled=base_context_map[target_year]['base_x_scaled'],
                node_idx=node_idx
            ).to(DEVICE)

            wrapper.eval()

            x_pair = np.repeat(
                X_scaled_sub[i:i + 1],
                2,
                axis=0
            ).astype(np.float32)

            x_pair[0, feature_idx] = raw_to_scaled_value(
                left,
                scaler,
                feature_idx
            )

            x_pair[1, feature_idx] = raw_to_scaled_value(
                right,
                scaler,
                feature_idx
            )

            x_pair_t = torch.tensor(
                x_pair,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():
                preds = wrapper(x_pair_t).detach().cpu().numpy().reshape(-1)

            diffs.append(float(preds[1] - preds[0]))

        local_eff = float(np.mean(diffs))

        local_effects.append(local_eff)
        bin_centers.append((left + right) / 2.0)
        bin_counts.append(len(idxs))

    local_effects = np.asarray(local_effects, dtype=np.float64)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    bin_counts = np.asarray(bin_counts, dtype=np.float64)

    ale_vals = np.cumsum(local_effects) - 0.5 * local_effects

    weights = bin_counts / max(bin_counts.sum(), 1.0)

    ale_vals_centered = ale_vals - np.sum(ale_vals * weights)

    out_df = pd.DataFrame({
        'feature': feature_name,
        'bin_left': edges[:-1],
        'bin_right': edges[1:],
        'bin_center': bin_centers,
        'local_effect': local_effects,
        'ale': ale_vals_centered,
        'n_bin': bin_counts.astype(int)
    })

    return out_df


def plot_ale_curve(df_ale, title, save_path):
    plt.figure(figsize=(6, 4))

    plt.plot(
        df_ale['bin_center'].values,
        df_ale['ale'].values,
        marker='o'
    )

    plt.xlabel(df_ale['feature'].iloc[0])
    plt.ylabel('ALE')
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()


ale_meta_df = test_meta_df.copy()
ale_X_scaled = test_x_scaled_local.copy()
ale_X_raw = test_x_raw_local.copy()

if ALE_MAX_SAMPLES is not None and len(ale_meta_df) > ALE_MAX_SAMPLES:
    rng = np.random.RandomState(SEED)

    idx_sample = rng.choice(
        len(ale_meta_df),
        size=ALE_MAX_SAMPLES,
        replace=False
    )

    ale_meta_df = ale_meta_df.iloc[idx_sample].reset_index(drop=True)
    ale_X_scaled = ale_X_scaled[idx_sample]
    ale_X_raw = ale_X_raw[idx_sample]

ale_meta_df['CII_Class'] = ale_meta_df['y_true'].apply(assign_cii_class)

if ALE_TOP_N_FEATURES is None:
    ale_feature_names = raw_feature_names.copy()
else:
    ale_feature_names = importance_df['feature'].tolist()[:ALE_TOP_N_FEATURES]

global_ale_all = []

for feat_name in ale_feature_names:
    feat_idx = raw_feature_names.index(feat_name)

    df_ale = compute_1d_ale_for_subset(
        meta_df_sub=ale_meta_df,
        X_scaled_sub=ale_X_scaled,
        X_raw_sub=ale_X_raw,
        feature_name=feat_name,
        feature_idx=feat_idx,
        base_context_map=base_context_map,
        model=model,
        scaler=x_scaler,
        n_bins=ALE_N_BINS
    )

    if df_ale is None:
        continue

    global_ale_all.append(df_ale)

    csv_path = os.path.join(
        ALE_DIR,
        'global',
        f'{safe_filename(feat_name)}_global_ale.csv'
    )

    fig_path = os.path.join(
        ALE_DIR,
        'global',
        f'{safe_filename(feat_name)}_global_ale.png'
    )

    df_ale.to_csv(
        csv_path,
        index=False,
        encoding='utf-8-sig'
    )

    plot_ale_curve(
        df_ale,
        title=f'Global ALE - {feat_name}',
        save_path=fig_path
    )

if len(global_ale_all) > 0:
    global_ale_long = pd.concat(
        global_ale_all,
        axis=0,
        ignore_index=True
    )

    global_ale_long.to_csv(
        os.path.join(ALE_DIR, 'global', 'global_ale_all_features_long.csv'),
        index=False,
        encoding='utf-8-sig'
    )
else:
    global_ale_long = pd.DataFrame()

cii_class_ale_all = []

if DO_CII_CLASS_ALE:
    class_counts = ale_meta_df['CII_Class'].value_counts()

    class_keep = [
        c
        for c in CII_CLASS_ORDER
        if class_counts.get(c, 0) >= ALE_MIN_GROUP_SIZE
    ]

    for feat_name in ale_feature_names:
        feat_idx = raw_feature_names.index(feat_name)

        class_plot_frames = []

        for cii_class in class_keep:
            sel = ale_meta_df['CII_Class'].values == cii_class

            if sel.sum() < ALE_MIN_GROUP_SIZE:
                continue

            sub_meta = ale_meta_df.loc[sel].reset_index(drop=True)
            sub_X_scaled = ale_X_scaled[sel]
            sub_X_raw = ale_X_raw[sel]

            df_ale = compute_1d_ale_for_subset(
                meta_df_sub=sub_meta,
                X_scaled_sub=sub_X_scaled,
                X_raw_sub=sub_X_raw,
                feature_name=feat_name,
                feature_idx=feat_idx,
                base_context_map=base_context_map,
                model=model,
                scaler=x_scaler,
                n_bins=ALE_N_BINS
            )

            if df_ale is None:
                continue

            df_ale['CII_Class'] = cii_class

            cii_class_ale_all.append(df_ale)
            class_plot_frames.append((cii_class, len(sub_meta), df_ale))

        if len(class_plot_frames) > 0:
            plt.figure(figsize=(7, 5))

            for cii_class, n_cls, df_ale_c in class_plot_frames:
                plt.plot(
                    df_ale_c['bin_center'].values,
                    df_ale_c['ale'].values,
                    marker='o',
                    linewidth=1.4,
                    label=f'{cii_class} (n={n_cls})'
                )

            plt.xlabel(feat_name)
            plt.ylabel('ALE')
            plt.title(f'CII-class ALE - {feat_name}')
            plt.legend(fontsize=8)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    ALE_DIR,
                    'cii_class',
                    f'{safe_filename(feat_name)}_cii_class_ale_compare.png'
                ),
                dpi=600,
                bbox_inches='tight'
            )

            plt.close()

if len(cii_class_ale_all) > 0:
    cii_class_ale_long = pd.concat(
        cii_class_ale_all,
        axis=0,
        ignore_index=True
    )

    cii_class_ale_long.to_csv(
        os.path.join(ALE_DIR, 'cii_class', 'cii_class_ale_all_features_long.csv'),
        index=False,
        encoding='utf-8-sig'
    )
else:
    cii_class_ale_long = pd.DataFrame()

province_ale_all = []

if DO_PROVINCE_ALE:
    province_counts = ale_meta_df[PROVINCE_COL].value_counts()
    province_keep = province_counts[province_counts >= ALE_MIN_GROUP_SIZE].index.tolist()

    for feat_name in ale_feature_names:
        feat_idx = raw_feature_names.index(feat_name)

        province_plot_frames = []

        for prov in province_keep:
            sel = ale_meta_df[PROVINCE_COL].astype(str).values == str(prov)

            if sel.sum() < ALE_MIN_GROUP_SIZE:
                continue

            sub_meta = ale_meta_df.loc[sel].reset_index(drop=True)
            sub_X_scaled = ale_X_scaled[sel]
            sub_X_raw = ale_X_raw[sel]

            df_ale = compute_1d_ale_for_subset(
                meta_df_sub=sub_meta,
                X_scaled_sub=sub_X_scaled,
                X_raw_sub=sub_X_raw,
                feature_name=feat_name,
                feature_idx=feat_idx,
                base_context_map=base_context_map,
                model=model,
                scaler=x_scaler,
                n_bins=ALE_N_BINS
            )

            if df_ale is None:
                continue

            df_ale[PROVINCE_COL] = prov

            province_ale_all.append(df_ale)
            province_plot_frames.append((prov, len(sub_meta), df_ale))

        if len(province_plot_frames) > 0:
            province_plot_frames = sorted(
                province_plot_frames,
                key=lambda x: x[1],
                reverse=True
            )[:ALE_PLOT_MAX_GROUPS]

            plt.figure(figsize=(7, 5))

            for prov, n_prov, df_ale_p in province_plot_frames:
                plt.plot(
                    df_ale_p['bin_center'].values,
                    df_ale_p['ale'].values,
                    marker='o',
                    linewidth=1.2,
                    label=f'{prov} (n={n_prov})'
                )

            plt.xlabel(feat_name)
            plt.ylabel('ALE')
            plt.title(f'Province-grouped ALE - {feat_name}')
            plt.legend(fontsize=8)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    ALE_DIR,
                    'province',
                    f'{safe_filename(feat_name)}_province_ale_compare.png'
                ),
                dpi=600,
                bbox_inches='tight'
            )

            plt.close()

if len(province_ale_all) > 0:
    province_ale_long = pd.concat(
        province_ale_all,
        axis=0,
        ignore_index=True
    )

    province_ale_long.to_csv(
        os.path.join(ALE_DIR, 'province', 'province_ale_all_features_long.csv'),
        index=False,
        encoding='utf-8-sig'
    )
else:
    province_ale_long = pd.DataFrame()