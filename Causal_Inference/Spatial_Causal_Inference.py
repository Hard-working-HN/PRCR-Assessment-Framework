import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone


CSV_PATH = 'Total_Data_with_QSstar_merged.csv'
SHP_PATH = 'County2023.shp'
OUTPUT_DIR = './dynamic_spatial_policy_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_CODE_COL = 'Code'
SHP_CODE_COL = 'code'
YEAR_COL = 'Year'
TARGET_COL = 'S_star'

POLICY_COLS = ['LCCP', 'ECERFP', 'ZWC']
TNP_COL = 'TNP'

CONTROL_COLS = [
    'POP', 'GDP', 'VAP', 'VAF', 'NIA', 'GIA', 'Road', 'TSS',
    'GRF', 'GEF', 'PIU', 'PIR', 'Light', 'Cropland', 'Urban',
    'Bare_areas', 'Forest', 'Grassland', 'Wetland', 'Water_bodies'
]

W_METHOD = 'queen'
KNN_K = 8
SHP_CRS_IF_MISSING = 'EPSG:4326'
ADD_SPATIAL_CONTROL_LAGS = True

LAG_LIST = [1, 2]
N_SPLITS = 5
RANDOM_STATE = 42


def normalize_code(series):
    s = series.astype(str).str.strip()
    s = s.str.replace('\\.0$', '', regex=True)
    return s


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def fill_numeric_by_year_median(df, cols, year_col):
    for c in cols:
        if c not in df.columns:
            continue

        df[c] = pd.to_numeric(df[c], errors='coerce')
        global_median = df[c].median()

        if pd.isna(global_median):
            global_median = 0

        df[c] = (
            df.groupby(year_col)[c]
            .transform(lambda x: x.fillna(x.median()))
            .fillna(global_median)
            .fillna(0)
        )

    return df


def load_and_prepare_shp(shp_path, shp_code_col):
    gdf = gpd.read_file(shp_path)

    if shp_code_col not in gdf.columns:
        raise ValueError(
            f'The column {shp_code_col} was not found in the SHP file.\n'
            f'Current SHP columns are: {list(gdf.columns)}'
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(SHP_CRS_IF_MISSING)

    gdf = gdf[[shp_code_col, 'geometry']].copy()
    gdf['__code__'] = normalize_code(gdf[shp_code_col])

    gdf = gdf[gdf['__code__'].notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gdf['geometry'] = gdf.geometry.buffer(0)

    gdf = gdf.dissolve(by='__code__', as_index=False)

    return gdf[['__code__', 'geometry']].copy()


def build_queen_edges(gdf):
    left = gdf[['__code__', 'geometry']].rename(columns={'__code__': 'code_i'})
    right = gdf[['__code__', 'geometry']].rename(columns={'__code__': 'code_j'})

    try:
        joined = gpd.sjoin(left, right, how='inner', predicate='touches')
    except TypeError:
        joined = gpd.sjoin(left, right, how='inner', op='touches')

    edges = joined[['code_i', 'code_j']].copy()
    edges = edges[edges['code_i'] != edges['code_j']].drop_duplicates()

    if edges.empty:
        raise ValueError(
            'No Queen contiguity edges were constructed. '
            'Please check whether the SHP geometries contain gaps or whether the Code field is correct.'
        )

    edges['n_neighbors'] = edges.groupby('code_i')['code_j'].transform('count')
    edges['weight'] = 1.0 / edges['n_neighbors']

    return edges[['code_i', 'code_j', 'weight']].copy()


def build_knn_edges(gdf, k=6):
    from sklearn.neighbors import NearestNeighbors

    gdf2 = gdf.copy()

    try:
        target_crs = gdf2.estimate_utm_crs()
        if target_crs is not None:
            gdf2 = gdf2.to_crs(target_crs)
        else:
            gdf2 = gdf2.to_crs(epsg=3857)
    except Exception:
        gdf2 = gdf2.to_crs(epsg=3857)

    centroids = gdf2.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    codes = gdf2['__code__'].values

    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(coords)

    distances, indices = nn.kneighbors(coords)

    records = []

    for i, code_i in enumerate(codes):
        neigh_idx = indices[i][1:]
        neigh_dist = distances[i][1:]

        inv_dist = 1.0 / np.maximum(neigh_dist, 1e-09)
        weights = inv_dist / inv_dist.sum()

        for j_idx, w in zip(neigh_idx, weights):
            records.append({
                'code_i': code_i,
                'code_j': codes[j_idx],
                'weight': w
            })

    return pd.DataFrame(records)


def add_spatial_lags(panel_df, edges_df, code_col, year_col, value_cols):
    panel = panel_df.copy()

    value_cols = [c for c in value_cols if c in panel.columns]

    base = panel[[code_col, year_col] + value_cols].copy()
    base = base.rename(columns={code_col: 'code_j'})

    tmp = edges_df.merge(base, on='code_j', how='left')

    out_cols = []

    for c in value_cols:
        wc = f'W_{c}'
        tmp[wc] = tmp['weight'] * tmp[c]
        out_cols.append(wc)

    spatial_lag = (
        tmp.groupby(['code_i', year_col], as_index=False)[out_cols]
        .sum()
        .rename(columns={'code_i': code_col})
    )

    panel = panel.merge(spatial_lag, on=[code_col, year_col], how='left')

    for wc in out_cols:
        panel[wc] = panel[wc].fillna(0)

    return panel


def create_group_lags(df, group_col, time_col, cols, lag_list):
    df = df.sort_values([group_col, time_col]).copy()

    for c in cols:
        if c not in df.columns:
            continue

        for lag in lag_list:
            new_col = f'{c}_lag{lag}'
            df[new_col] = df.groupby(group_col)[c].shift(lag)

    return df


def create_delta_outcome(df, group_col, time_col, y_col):
    df = df.sort_values([group_col, time_col]).copy()
    df[f'Delta_{y_col}'] = df[y_col] - df.groupby(group_col)[y_col].shift(1)
    return df


def two_way_demean(df, cols, entity_col, time_col):
    x = df[cols].copy()

    entity_mean = df.groupby(entity_col)[cols].transform('mean')
    time_mean = df.groupby(time_col)[cols].transform('mean')
    overall_mean = df[cols].mean()

    return x - entity_mean - time_mean + overall_mean


def add_significance_stars(table):
    table['significance'] = ''

    table.loc[table['p_value'] < 0.1, 'significance'] = '*'
    table.loc[table['p_value'] < 0.05, 'significance'] = '**'
    table.loc[table['p_value'] < 0.01, 'significance'] = '***'

    return table


def make_ml_model(random_state=42):
    model = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            (
                'model',
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=random_state,
                    n_jobs=-1
                )
            )
        ]
    )

    return model


def remove_near_zero_variance_cols(df, cols, threshold=1e-12):
    kept = []
    removed = []

    for c in cols:
        if c not in df.columns:
            continue

        s = pd.to_numeric(df[c], errors='coerce')

        if s.std(skipna=True) > threshold:
            kept.append(c)
        else:
            removed.append(c)

    return kept, removed


def run_twfe_within(
    df,
    y_col,
    treatment_cols,
    control_cols,
    entity_col,
    time_col,
    output_prefix
):
    treatment_cols = [c for c in treatment_cols if c in df.columns]
    control_cols = [c for c in control_cols if c in df.columns]

    use_cols = [y_col] + treatment_cols + control_cols + [entity_col, time_col]
    use_cols = list(dict.fromkeys(use_cols))

    work = df[use_cols].copy()

    model_vars = [y_col] + treatment_cols + control_cols
    work = safe_numeric(work, model_vars)

    work = work.dropna(subset=[y_col] + treatment_cols).copy()

    for c in control_cols:
        work[c] = work[c].fillna(work[c].median()).fillna(0)

    model_vars = [y_col] + treatment_cols + control_cols
    dm = two_way_demean(work, model_vars, entity_col, time_col)

    y_dm = dm[y_col]
    x_cols = treatment_cols + control_cols

    x_cols, removed_cols = remove_near_zero_variance_cols(dm, x_cols)

    if removed_cols:
        pass

    X_dm = dm[x_cols]

    model = sm.OLS(y_dm, X_dm)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': work[entity_col]})

    coef_table = pd.DataFrame({
        'variable': res.params.index,
        'coef': res.params.values,
        'std_err_cluster': res.bse.values,
        't_value': res.tvalues.values,
        'p_value': res.pvalues.values
    })

    coef_table = add_significance_stars(coef_table)

    coef_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_TWFE_coefficients.csv')
    summary_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_TWFE_summary.txt')

    coef_table.to_csv(coef_path, index=False, encoding='utf-8-sig')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(str(res.summary()))
        f.write('\n\nTreatment variables:\n')
        f.write(str(treatment_cols))
        f.write('\n\nControl variables:\n')
        f.write(str(control_cols))
        f.write('\n\nRemoved near-zero variance variables:\n')
        f.write(str(removed_cols))

    treatment_result = coef_table[coef_table['variable'].isin(treatment_cols)].copy()

    return res, coef_table, treatment_result


def run_dynamic_spatial_dml(
    df,
    y_col,
    treatment_cols,
    control_cols,
    entity_col,
    time_col,
    output_prefix,
    n_splits=5,
    random_state=42
):
    treatment_cols = [c for c in treatment_cols if c in df.columns]
    control_cols = [c for c in control_cols if c in df.columns]

    use_cols = [y_col] + treatment_cols + control_cols + [entity_col, time_col]
    use_cols = list(dict.fromkeys(use_cols))

    work = df[use_cols].copy()

    model_vars = [y_col] + treatment_cols + control_cols
    work = safe_numeric(work, model_vars)

    work = work.dropna(subset=[y_col] + treatment_cols).copy()

    for c in control_cols:
        work[c] = work[c].fillna(work[c].median()).fillna(0)

    model_vars = [y_col] + treatment_cols + control_cols
    dm = two_way_demean(work, model_vars, entity_col, time_col)

    d_cols, removed_d = remove_near_zero_variance_cols(dm, treatment_cols)
    x_cols, removed_x = remove_near_zero_variance_cols(dm, control_cols)

    if removed_d:
        pass

    if removed_x:
        pass

    if len(d_cols) == 0:
        return None, None, None

    if len(x_cols) == 0:
        return None, None, None

    y = dm[y_col].values
    D = dm[d_cols].values
    X = dm[x_cols].values

    groups = work[entity_col].values
    unique_groups = pd.Series(groups).nunique()

    if unique_groups < n_splits:
        n_splits = unique_groups

    gkf = GroupKFold(n_splits=n_splits)

    y_hat = np.zeros_like(y, dtype=float)
    d_hat = np.zeros_like(D, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        model_y = make_ml_model(random_state=random_state)
        model_y.fit(X[train_idx], y[train_idx])
        y_hat[test_idx] = model_y.predict(X[test_idx])

        for j in range(D.shape[1]):
            model_d = make_ml_model(random_state=random_state + 100 + j)
            model_d.fit(X[train_idx], D[train_idx, j])
            d_hat[test_idx, j] = model_d.predict(X[test_idx])

    y_resid = y - y_hat
    d_resid = D - d_hat

    res_model = sm.OLS(y_resid, d_resid)
    res = res_model.fit(cov_type='cluster', cov_kwds={'groups': groups})

    coef_table = pd.DataFrame({
        'variable': d_cols,
        'coef': res.params,
        'std_err_cluster': res.bse,
        't_value': res.tvalues,
        'p_value': res.pvalues
    })

    coef_table = add_significance_stars(coef_table)

    coef_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_DML_coefficients.csv')
    residual_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_DML_residuals.csv')
    summary_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_DML_summary.txt')

    coef_table.to_csv(coef_path, index=False, encoding='utf-8-sig')

    residual_df = pd.DataFrame({
        entity_col: work[entity_col].values,
        time_col: work[time_col].values,
        'y_resid': y_resid
    })

    for j, c in enumerate(d_cols):
        residual_df[f'{c}_resid'] = d_resid[:, j]

    residual_df.to_csv(residual_path, index=False, encoding='utf-8-sig')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(str(res.summary()))
        f.write('\n\nTreatment variables:\n')
        f.write(str(d_cols))
        f.write('\n\nControl variables:\n')
        f.write(str(x_cols))
        f.write('\n\nRemoved treatment variables:\n')
        f.write(str(removed_d))
        f.write('\n\nRemoved control variables:\n')
        f.write(str(removed_x))

    return res, coef_table, residual_df


def add_policy_interactions(df):
    required = ['LCCP', 'ECERFP', 'ZWC']

    for c in required:
        if c not in df.columns:
            df[c] = 0

    df['INT_LCCP_ECERFP'] = df['LCCP'] * df['ECERFP']
    df['INT_LCCP_ZWC'] = df['LCCP'] * df['ZWC']
    df['INT_ECERFP_ZWC'] = df['ECERFP'] * df['ZWC']
    df['INT_ALL3'] = df['LCCP'] * df['ECERFP'] * df['ZWC']

    return df


def add_tnp_categories(df):
    if TNP_COL not in df.columns:
        return df

    for k in [1, 2, 3]:
        df[f'TNP_{k}'] = (df[TNP_COL] == k).astype(int)

    return df


def build_model_variables(panel):
    control_lag1 = [
        f'{c}_lag1'
        for c in CONTROL_COLS
        if f'{c}_lag1' in panel.columns
    ]

    spatial_control_lag1 = [
        f'W_{c}_lag1'
        for c in CONTROL_COLS
        if f'W_{c}_lag1' in panel.columns
    ]

    if ADD_SPATIAL_CONTROL_LAGS:
        controls = control_lag1 + spatial_control_lag1
    else:
        controls = control_lag1

    model1_treatments = []

    for p in POLICY_COLS:
        model1_treatments.append(f'{p}_lag1')

    for p in POLICY_COLS:
        model1_treatments.append(f'W_{p}_lag1')

    model1_treatments = [c for c in model1_treatments if c in panel.columns]

    model2_treatments = []

    for lag in [1, 2]:
        for p in POLICY_COLS:
            model2_treatments.append(f'{p}_lag{lag}')

        for p in POLICY_COLS:
            model2_treatments.append(f'W_{p}_lag{lag}')

    model2_treatments = [c for c in model2_treatments if c in panel.columns]

    model3_treatments = [f'{TNP_COL}_lag1', f'W_{TNP_COL}_lag1']
    model3_treatments = [c for c in model3_treatments if c in panel.columns]

    interaction_cols = [
        'INT_LCCP_ECERFP',
        'INT_LCCP_ZWC',
        'INT_ECERFP_ZWC',
        'INT_ALL3'
    ]

    model4_treatments = []

    for c in interaction_cols:
        model4_treatments.append(f'{c}_lag1')

    for c in interaction_cols:
        model4_treatments.append(f'W_{c}_lag1')

    model4_treatments = [c for c in model4_treatments if c in panel.columns]

    model5_treatments = [
        'TNP_1_lag1',
        'TNP_2_lag1',
        'TNP_3_lag1',
        'W_TNP_1_lag1',
        'W_TNP_2_lag1',
        'W_TNP_3_lag1'
    ]

    model5_treatments = [c for c in model5_treatments if c in panel.columns]

    return {
        'controls': controls,
        'model1_policy_lag1': model1_treatments,
        'model2_policy_lag1_lag2': model2_treatments,
        'model3_tnp_lag1': model3_treatments,
        'model4_policy_interaction_lag1': model4_treatments,
        'model5_tnp_category_lag1': model5_treatments
    }


def save_diagnostics(panel, edges):
    diag_path = os.path.join(OUTPUT_DIR, 'diagnostics_policy_distribution.txt')

    cols_to_check = []

    for c in POLICY_COLS + [TNP_COL]:
        if c in panel.columns:
            cols_to_check.append(c)

        wc = f'W_{c}'

        if wc in panel.columns:
            cols_to_check.append(wc)

    for c in POLICY_COLS:
        for lag in LAG_LIST:
            if f'{c}_lag{lag}' in panel.columns:
                cols_to_check.append(f'{c}_lag{lag}')

            if f'W_{c}_lag{lag}' in panel.columns:
                cols_to_check.append(f'W_{c}_lag{lag}')

    cols_to_check = list(dict.fromkeys(cols_to_check))

    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write('========== Basic information ==========\n')
        f.write(f'Sample size: {panel.shape}\n')
        f.write(f"Number of regions: {panel['__code__'].nunique()}\n")
        f.write(f'Year range: {panel[YEAR_COL].min()} - {panel[YEAR_COL].max()}\n')
        f.write(f'Number of spatial adjacency edges: {len(edges)}\n\n')

        f.write('========== Descriptive statistics of policy variables ==========\n')
        f.write(str(panel[cols_to_check].describe()))
        f.write('\n\n')

        f.write('========== Frequency distribution of original policy variables ==========\n')

        for c in POLICY_COLS + [TNP_COL]:
            if c in panel.columns:
                f.write(f'\n{c}\n')
                f.write(str(panel[c].value_counts(dropna=False).sort_index()))
                f.write('\n')

        f.write('\n========== Policy variables by year ==========\n')

        for c in POLICY_COLS + [TNP_COL]:
            if c in panel.columns:
                f.write(f'\n{c}\n')
                f.write(str(pd.crosstab(panel[YEAR_COL], panel[c])))
                f.write('\n')

        f.write('\n========== Standard deviation of variables ==========\n')

        for c in cols_to_check:
            f.write(f'{c}: std = {panel[c].std()}\n')


def main():
    panel = pd.read_csv(CSV_PATH)

    if CSV_CODE_COL not in panel.columns:
        raise ValueError(
            f'The column {CSV_CODE_COL} was not found in the CSV file. '
            f'Current columns are: {list(panel.columns)}'
        )

    if YEAR_COL not in panel.columns:
        raise ValueError(
            f'The column {YEAR_COL} was not found in the CSV file. '
            f'Current columns are: {list(panel.columns)}'
        )

    if TARGET_COL not in panel.columns:
        raise ValueError(
            f'The target variable {TARGET_COL} was not found in the CSV file. '
            f'Current columns are: {list(panel.columns)}'
        )

    panel['__code__'] = normalize_code(panel[CSV_CODE_COL])
    panel[YEAR_COL] = pd.to_numeric(panel[YEAR_COL], errors='coerce').astype('Int64')

    gdf = load_and_prepare_shp(SHP_PATH, SHP_CODE_COL)

    common_codes = sorted(set(panel['__code__']) & set(gdf['__code__']))

    if len(common_codes) == 0:
        raise ValueError(
            'No matching Code values were found between the CSV and SHP files. '
            'Please check the Code format.'
        )

    panel = panel[panel['__code__'].isin(common_codes)].copy()
    gdf = gdf[gdf['__code__'].isin(common_codes)].copy()

    if W_METHOD.lower() == 'queen':
        edges = build_queen_edges(gdf)
        edge_name = 'queen'
    elif W_METHOD.lower() == 'knn':
        edges = build_knn_edges(gdf, k=KNN_K)
        edge_name = f'knn{KNN_K}'
    else:
        raise ValueError("W_METHOD can only be set to 'queen' or 'knn'.")

    edge_path = os.path.join(OUTPUT_DIR, f'spatial_edges_{edge_name}.csv')
    edges.to_csv(edge_path, index=False, encoding='utf-8-sig')

    available_policy_cols = [c for c in POLICY_COLS if c in panel.columns]
    available_control_cols = [c for c in CONTROL_COLS if c in panel.columns]

    numeric_cols = [TARGET_COL, TNP_COL] + available_policy_cols + available_control_cols
    numeric_cols = [c for c in numeric_cols if c in panel.columns]

    panel = safe_numeric(panel, numeric_cols)

    for c in available_policy_cols + ([TNP_COL] if TNP_COL in panel.columns else []):
        panel[c] = panel[c].fillna(0)

    panel = fill_numeric_by_year_median(panel, available_control_cols, YEAR_COL)

    panel = add_policy_interactions(panel)
    panel = add_tnp_categories(panel)

    interaction_cols = [
        'INT_LCCP_ECERFP',
        'INT_LCCP_ZWC',
        'INT_ECERFP_ZWC',
        'INT_ALL3'
    ]

    tnp_category_cols = ['TNP_1', 'TNP_2', 'TNP_3']

    spatial_value_cols = []
    spatial_value_cols += available_policy_cols

    if TNP_COL in panel.columns:
        spatial_value_cols.append(TNP_COL)

    spatial_value_cols += interaction_cols
    spatial_value_cols += tnp_category_cols

    if ADD_SPATIAL_CONTROL_LAGS:
        spatial_value_cols += available_control_cols

    spatial_value_cols = list(dict.fromkeys([
        c for c in spatial_value_cols
        if c in panel.columns
    ]))

    panel = add_spatial_lags(
        panel_df=panel,
        edges_df=edges,
        code_col='__code__',
        year_col=YEAR_COL,
        value_cols=spatial_value_cols
    )

    panel = create_delta_outcome(
        df=panel,
        group_col='__code__',
        time_col=YEAR_COL,
        y_col=TARGET_COL
    )

    DELTA_TARGET_COL = f'Delta_{TARGET_COL}'

    lag_source_cols = []
    lag_source_cols += available_policy_cols

    if TNP_COL in panel.columns:
        lag_source_cols.append(TNP_COL)

    lag_source_cols += interaction_cols
    lag_source_cols += tnp_category_cols
    lag_source_cols += available_control_cols

    for c in spatial_value_cols:
        wc = f'W_{c}'
        if wc in panel.columns:
            lag_source_cols.append(wc)

    lag_source_cols = list(dict.fromkeys([
        c for c in lag_source_cols
        if c in panel.columns
    ]))

    panel = create_group_lags(
        df=panel,
        group_col='__code__',
        time_col=YEAR_COL,
        cols=lag_source_cols,
        lag_list=LAG_LIST
    )

    panel_path = os.path.join(OUTPUT_DIR, 'panel_dynamic_spatial_lagged.csv')
    panel.to_csv(panel_path, index=False, encoding='utf-8-sig')

    save_diagnostics(panel, edges)

    vars_dict = build_model_variables(panel)
    controls = vars_dict['controls']

    model_specs = {
        'model1_policy_lag1_local_spillover': vars_dict['model1_policy_lag1'],
        'model2_policy_lag1_lag2_dynamic': vars_dict['model2_policy_lag1_lag2'],
        'model3_tnp_lag1_local_spillover': vars_dict['model3_tnp_lag1'],
        'model4_policy_interaction_synergy': vars_dict['model4_policy_interaction_lag1'],
        'model5_tnp_category_lag1': vars_dict['model5_tnp_category_lag1']
    }

    all_results = []

    for model_name, treatments in model_specs.items():
        if len(treatments) == 0:
            continue

        try:
            twfe_res, twfe_coef, twfe_treat = run_twfe_within(
                df=panel,
                y_col=DELTA_TARGET_COL,
                treatment_cols=treatments,
                control_cols=controls,
                entity_col='__code__',
                time_col=YEAR_COL,
                output_prefix=model_name
            )

            twfe_treat['model'] = model_name
            twfe_treat['method'] = 'TWFE'
            all_results.append(twfe_treat)

        except Exception as e:
            pass

        try:
            dml_res, dml_coef, dml_residual = run_dynamic_spatial_dml(
                df=panel,
                y_col=DELTA_TARGET_COL,
                treatment_cols=treatments,
                control_cols=controls,
                entity_col='__code__',
                time_col=YEAR_COL,
                output_prefix=model_name,
                n_splits=N_SPLITS,
                random_state=RANDOM_STATE
            )

            if dml_coef is not None:
                dml_coef['model'] = model_name
                dml_coef['method'] = 'DML'
                all_results.append(dml_coef)

        except Exception as e:
            pass

    if all_results:
        final_result = pd.concat(all_results, ignore_index=True)
        final_path = os.path.join(OUTPUT_DIR, 'ALL_policy_effect_results.csv')
        final_result.to_csv(final_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()