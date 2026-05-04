import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans, silhouette_score

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


INPUT_CSV = 'QSstar_wide.csv'
OUT_DIR = 'DTW_trajectory_results_4clusters'

YEAR_START = 2013
YEAR_END = 2022
USE_BASE_YEAR_2013 = True

FINAL_K = 4
RANDOM_SEEDS = list(range(10))
N_JOBS = -1
SIL_SAMPLE_SIZE = None
MAX_ITER = 50
N_INIT_EVAL = 5
N_INIT_FINAL = 10

REORDER_CLUSTER_BY_LAST_YEAR = True

NAME_MAP = {
    1: 'Persistent deteriorating',
    2: 'Low-level stagnant',
    3: 'Steady improvement',
    4: 'Significantly improved'
}


os.makedirs(OUT_DIR, exist_ok=True)


df_wide = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
df_wide.columns = df_wide.columns.map(str)

all_years = list(range(YEAR_START, YEAR_END + 1))

if not USE_BASE_YEAR_2013 and YEAR_START == 2013:
    years = list(range(2014, YEAR_END + 1))
else:
    years = all_years.copy()

year_cols = [str(y) for y in years]

required_cols = ['Code'] + year_cols
missing_cols = [c for c in required_cols if c not in df_wide.columns]

if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')

df_wide = df_wide[required_cols].copy()
df_wide['Code'] = df_wide['Code'].astype(str).str.strip()

for c in year_cols:
    df_wide[c] = pd.to_numeric(df_wide[c], errors='coerce')

if df_wide[year_cols].isna().any().any():
    na_count = int(df_wide[year_cols].isna().sum().sum())
    raise ValueError(
        f'Missing values exist in the time series, with {na_count} missing values in total. '
        f'Please handle them first.'
    )

codes = df_wide['Code'].values

X = df_wide[year_cols].values.astype(float)
X_ts = X[:, :, np.newaxis]

n_regions, n_years = X.shape

desc = pd.Series(X.flatten()).describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
)

desc.to_csv(
    os.path.join(OUT_DIR, 'S_star_distribution_summary.csv'),
    encoding='utf-8-sig'
)


def evaluate_one_model(
    X_ts,
    k,
    seed,
    sample_size=1200,
    n_jobs=-1,
    max_iter=50,
    n_init=2,
    rng_seed=42
):
    model = TimeSeriesKMeans(
        n_clusters=k,
        metric='dtw',
        max_iter=max_iter,
        n_init=n_init,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=1
    )

    labels = model.fit_predict(X_ts)

    counts = np.bincount(labels, minlength=k)

    if sample_size is None:
        sil = silhouette_score(
            X_ts,
            labels,
            metric='dtw',
            n_jobs=n_jobs
        )
    else:
        rng = np.random.default_rng(rng_seed)

        if sample_size >= len(X_ts):
            sil = silhouette_score(
                X_ts,
                labels,
                metric='dtw',
                n_jobs=n_jobs
            )
        else:
            sample_idx = rng.choice(
                len(X_ts),
                size=sample_size,
                replace=False
            )

            sil = silhouette_score(
                X_ts[sample_idx],
                labels[sample_idx],
                metric='dtw',
                n_jobs=n_jobs
            )

    return labels, model, sil, counts


results = []
best_row = None

for seed in RANDOM_SEEDS:
    labels, model, sil, counts = evaluate_one_model(
        X_ts=X_ts,
        k=FINAL_K,
        seed=seed,
        sample_size=SIL_SAMPLE_SIZE,
        n_jobs=N_JOBS,
        max_iter=MAX_ITER,
        n_init=N_INIT_EVAL,
        rng_seed=42 + seed
    )

    row = {
        'k': FINAL_K,
        'seed': int(seed),
        'silhouette_dtw': float(sil),
        'inertia': float(model.inertia_),
        'min_cluster_n': int(counts.min()),
        'max_cluster_n': int(counts.max()),
        'cluster_sizes': str(counts.tolist())
    }

    results.append(row)

    if best_row is None or sil > best_row['silhouette_dtw']:
        best_row = row


eval_df = pd.DataFrame(results).sort_values(
    'silhouette_dtw',
    ascending=False
)

eval_df.to_csv(
    os.path.join(OUT_DIR, 'DTW_seed_evaluation_k4.csv'),
    index=False,
    encoding='utf-8-sig'
)

best_seed = int(best_row['seed'])

pd.DataFrame([best_row]).to_csv(
    os.path.join(OUT_DIR, 'DTW_best_seed_k4.csv'),
    index=False,
    encoding='utf-8-sig'
)


final_model = TimeSeriesKMeans(
    n_clusters=FINAL_K,
    metric='dtw',
    max_iter=MAX_ITER,
    n_init=N_INIT_FINAL,
    n_jobs=N_JOBS,
    random_state=best_seed,
    verbose=1
)

labels_raw = final_model.fit_predict(X_ts)
centers_raw = final_model.cluster_centers_.squeeze(-1)


if REORDER_CLUSTER_BY_LAST_YEAR:
    order = np.argsort(centers_raw[:, -1])
    old_to_new = {
        old_idx: new_idx + 1
        for new_idx, old_idx in enumerate(order)
    }
else:
    old_to_new = {
        old_idx: old_idx + 1
        for old_idx in range(FINAL_K)
    }


labels = np.array(
    [old_to_new[x] for x in labels_raw],
    dtype=int
)

centers = np.zeros_like(centers_raw)

for old_idx, new_idx in old_to_new.items():
    centers[new_idx - 1, :] = centers_raw[old_idx, :]


cluster_df = pd.DataFrame({
    'Code': codes,
    'cluster': labels
})

cluster_df['cluster_name'] = cluster_df['cluster'].map(NAME_MAP)

cluster_df.to_csv(
    os.path.join(OUT_DIR, 'DTW_cluster_labels.csv'),
    index=False,
    encoding='utf-8-sig'
)


center_df = pd.DataFrame(
    centers,
    columns=year_cols
)

center_df.insert(0, 'cluster', np.arange(1, FINAL_K + 1))
center_df['cluster_name'] = center_df['cluster'].map(NAME_MAP)

center_df.to_csv(
    os.path.join(OUT_DIR, 'DTW_cluster_centers.csv'),
    index=False,
    encoding='utf-8-sig'
)


mapping_df = pd.DataFrame({
    'old_cluster_0based': list(old_to_new.keys()),
    'new_cluster_1based': list(old_to_new.values())
}).sort_values('new_cluster_1based')

mapping_df.to_csv(
    os.path.join(OUT_DIR, 'DTW_cluster_reorder_mapping.csv'),
    index=False,
    encoding='utf-8-sig'
)


wide_with_cluster_df = df_wide.merge(
    cluster_df,
    on='Code',
    how='left'
)

wide_with_cluster_df.to_csv(
    os.path.join(OUT_DIR, 'QSstar_wide_with_cluster.csv'),
    index=False,
    encoding='utf-8-sig'
)


final_silhouette_full = silhouette_score(
    X_ts,
    labels_raw,
    metric='dtw',
    n_jobs=N_JOBS
)

pd.DataFrame([{
    'k': FINAL_K,
    'best_seed': best_seed,
    'final_silhouette_full': float(final_silhouette_full)
}]).to_csv(
    os.path.join(OUT_DIR, 'DTW_final_full_silhouette.csv'),
    index=False,
    encoding='utf-8-sig'
)


cluster_stat_rows = []

for c in range(1, FINAL_K + 1):
    sub = wide_with_cluster_df.loc[
        wide_with_cluster_df['cluster'] == c,
        year_cols
    ].astype(float)

    row = {
        'cluster': c,
        'cluster_name': NAME_MAP[c],
        'n': len(sub),
        'share': len(sub) / len(wide_with_cluster_df),
        'mean_start': sub[year_cols[0]].mean(),
        'mean_end': sub[year_cols[-1]].mean(),
        'mean_all_years': sub.values.mean()
    }

    cluster_stat_rows.append(row)

cluster_stat_df = pd.DataFrame(cluster_stat_rows)

cluster_stat_df.to_csv(
    os.path.join(OUT_DIR, 'DTW_cluster_statistics.csv'),
    index=False,
    encoding='utf-8-sig'
)


years_int = [int(y) for y in year_cols]


fig, ax = plt.subplots(figsize=(10, 6))

for c in range(1, FINAL_K + 1):
    sub = wide_with_cluster_df.loc[
        wide_with_cluster_df['cluster'] == c,
        year_cols
    ].astype(float)

    mean_curve = sub.mean(axis=0).values
    q25_curve = sub.quantile(0.25, axis=0).values
    q75_curve = sub.quantile(0.75, axis=0).values

    ax.plot(
        years_int,
        mean_curve,
        linewidth=2.5,
        label=NAME_MAP[c]
    )

    ax.fill_between(
        years_int,
        q25_curve,
        q75_curve,
        alpha=0.18
    )

ax.axhline(0, linestyle='--', linewidth=1)

ax.set_xlabel('', fontsize=14)
ax.set_ylabel('Synergy index', fontsize=14)

ax.tick_params(axis='both', labelsize=12)
ax.legend(frameon=False, fontsize=12)

plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, 'Fig3a_cluster_mean_trajectories.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

for i in range(X.shape[0]):
    ax.plot(
        years_int,
        X[i],
        color='lightgray',
        linewidth=0.5,
        alpha=0.3
    )

for c in range(1, FINAL_K + 1):
    ax.plot(
        years_int,
        centers[c - 1, :],
        linewidth=2.8,
        label=NAME_MAP[c]
    )

ax.axhline(0, linestyle='--', linewidth=1)

ax.set_xlabel('', fontsize=14)
ax.set_ylabel('Synergy index', fontsize=14)

ax.tick_params(axis='both', labelsize=12)
ax.legend(frameon=False, fontsize=12)

plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, 'Fig3b_all_trajectories_with_centers.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


count_df = cluster_df['cluster'].value_counts().sort_index().reset_index()
count_df.columns = ['cluster', 'n']
count_df['share'] = count_df['n'] / count_df['n'].sum()
count_df['cluster_name'] = count_df['cluster'].map(NAME_MAP)


fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    count_df['cluster_name'],
    count_df['n']
)

ax.set_xlabel('')
ax.set_ylabel('Number of regions', fontsize=14)

ax.tick_params(axis='x', labelsize=11, rotation=15)
ax.tick_params(axis='y', labelsize=12)

for i, row in count_df.iterrows():
    ax.text(
        i,
        row['n'],
        f"{row['share']:.1%}",
        ha='center',
        va='bottom',
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, 'Fig3d_cluster_sizes.png'),
    dpi=600,
    bbox_inches='tight'
)

plt.close()


long_rows = []

for i, code in enumerate(codes):
    for j, year in enumerate(year_cols):
        long_rows.append({
            'Code': code,
            'Year': int(year),
            'S_star': X[i, j],
            'cluster': labels[i],
            'cluster_name': NAME_MAP[labels[i]]
        })

cluster_long_df = pd.DataFrame(long_rows)

cluster_long_df.to_csv(
    os.path.join(OUT_DIR, 'QSstar_long_with_cluster.csv'),
    index=False,
    encoding='utf-8-sig'
)