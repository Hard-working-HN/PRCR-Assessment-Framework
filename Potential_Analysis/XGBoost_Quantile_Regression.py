from __future__ import annotations

import os
import json
import math
import random
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')


INPUT_CSV = 'Total_Data_with_QSstar_Strict.csv'
OUTPUT_DIR = 'QXGB_frontier_tuning_output_strict'

TARGET_COL = 'S_star'
GROUP_COL = 'Code'
YEAR_COL = 'Year'

KEEP_YEARS = list(range(2014, 2023))

FRONTIER_Q = 0.9
N_SPLITS = 5
RANDOM_STATE = 42

APPLY_LOG1P = True

N_RANDOM_SEARCH = 50
N_LOCAL_SEARCH = 50

FINAL_MODEL_NAME = 'QXGB'

TARGET_COVERAGE = FRONTIER_Q
COVERAGE_WEIGHT = 0.35
STABILITY_WEIGHT = 0.1

USE_EARLY_STOPPING = False
EARLY_STOPPING_ROUNDS = 80

OPTIONAL_META_COLS = [
    'Prefecture',
    'County',
    'Code',
    'Province_ID'
]

RAW_LOG_COLS = [
    'Registered_Population',
    'GDP_2013',
    'Added_value_of_primary_industry_2013',
    'Agriculture_forestry_animal_husbandry_fishing_2013',
    'Number_of_industrial_units_above_designated_size',
    'Gross_output_value_of_industrial_enterprises_above_designated_size_2013',
    'Road',
    'Total_retail_sales_of_consumer_goods_in_society_2013',
    'General_budget_revenue_of_local_finance_2013',
    'General_budget_expenditure_of_local_finance_2013',
    'Per-capita_Disposable_Income_of_Urban_Residents_2013',
    'Per-capita_disposable_income_of_rural_residents_2013',
    'Light_Data'
]

RAW_DIRECT_COLS = [
    'Cropland_Proportion',
    'Urban_Proportion',
    'Bare_areas_Proportion',
    'Forest_Proportion',
    'Grassland_Proportion',
    'Wetland_Proportion',
    'Water_bodies_Proportion'
]

RAW_EXTRA_DIRECT_COLS = [
    'Total number of policies',
    'Low-carbon City Pilot',
    'Comprehensive Demonstration City of Energy Conservation and Emission Reduction Fiscal Policies',
    'Zero-Waste City'
]

DPI = 600

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_text(text: str, path: str) -> None:
    with open(path, 'w', encoding='utf-8-sig') as f:
        f.write(text)


def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f'Missing required fields: {missing}')


def describe_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    desc = df[cols].describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.5,
            0.75,
            0.95,
            0.99
        ]
    ).T

    desc['missing_n'] = df[cols].isna().sum()
    desc['missing_pct'] = df[cols].isna().mean()

    return desc


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float
) -> float:
    err = y_true - y_pred

    return float(
        np.mean(
            np.maximum(
                q * err,
                (q - 1) * err
            )
        )
    )


def classify_four_types(
    region_df: pd.DataFrame,
    actual_col: str,
    gap_col: str
) -> pd.DataFrame:
    x_med = region_df[actual_col].median()
    y_med = region_df[gap_col].median()

    def judge(row):
        high_actual = row[actual_col] >= x_med
        high_gap = row[gap_col] >= y_med

        if high_actual and (not high_gap):
            return 'Consolidation'
        elif high_actual and high_gap:
            return 'Breakthrough'
        elif not high_actual and high_gap:
            return 'Priority intervention'
        else:
            return 'Structural constrained'

    out = region_df.copy()

    out['Actual_Median'] = x_med
    out['Gap_Median'] = y_med
    out['Potential_Type'] = out.apply(judge, axis=1)

    return out


def plot_frontier_scatter(
    df: pd.DataFrame,
    actual_col: str,
    frontier_col: str,
    save_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    ax.scatter(
        df[actual_col],
        df[frontier_col],
        s=8,
        alpha=0.35,
        linewidths=0
    )

    xy_min = min(
        df[actual_col].min(),
        df[frontier_col].min()
    )

    xy_max = max(
        df[actual_col].max(),
        df[frontier_col].max()
    )

    ax.plot(
        [xy_min, xy_max],
        [xy_min, xy_max],
        linewidth=1.2
    )

    ax.set_xlabel('Observed S_star')
    ax.set_ylabel(f'Conditional frontier (q={FRONTIER_Q:.2f})')

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_gap_hist(
    df: pd.DataFrame,
    gap_col: str,
    save_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    ax.hist(
        df[gap_col],
        bins=80
    )

    ax.set_xlabel(gap_col)
    ax.set_ylabel('Frequency')

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'
    )

    plt.close(fig)


def plot_quadrant(
    region_df: pd.DataFrame,
    actual_col: str,
    gap_col: str,
    save_path: str
) -> None:
    x_med = region_df[actual_col].median()
    y_med = region_df[gap_col].median()

    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    ax.scatter(
        region_df[actual_col],
        region_df[gap_col],
        s=14,
        alpha=0.7,
        linewidths=0
    )

    ax.axvline(
        x_med,
        linestyle='--',
        linewidth=1.1
    )

    ax.axhline(
        y_med,
        linestyle='--',
        linewidth=1.1
    )

    ax.set_xlabel('Regional mean observed S_star')
    ax.set_ylabel('Regional mean Gap')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.text(
        (x_med + x_max) / 2,
        (y_min + y_med) / 2,
        'Consolidation',
        ha='center',
        va='center',
        fontsize=10
    )

    ax.text(
        (x_med + x_max) / 2,
        (y_med + y_max) / 2,
        'Breakthrough',
        ha='center',
        va='center',
        fontsize=10
    )

    ax.text(
        (x_min + x_med) / 2,
        (y_med + y_max) / 2,
        'Priority intervention',
        ha='center',
        va='center',
        fontsize=10
    )

    ax.text(
        (x_min + x_med) / 2,
        (y_min + y_med) / 2,
        'Structural constrained',
        ha='center',
        va='center',
        fontsize=10
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight'
    )

    plt.close(fig)


def build_feature_columns(
    raw_log_cols: List[str],
    raw_direct_cols: List[str],
    extra_direct_cols: List[str],
    year_col: str,
    apply_log1p: bool
) -> Tuple[List[str], List[str]]:
    required_feature_input_cols = raw_log_cols + raw_direct_cols + extra_direct_cols

    if apply_log1p:
        log_feature_cols = [
            f'log_{c}'
            for c in raw_log_cols
        ]
    else:
        log_feature_cols = raw_log_cols.copy()

    feature_cols = (
        log_feature_cols
        + raw_direct_cols
        + extra_direct_cols
        + [year_col]
    )

    return required_feature_input_cols, feature_cols


def apply_feature_transformations(
    df: pd.DataFrame,
    raw_log_cols: List[str],
    apply_log1p: bool
) -> pd.DataFrame:
    df_out = df.copy()

    if apply_log1p:
        for c in raw_log_cols:
            if (df_out[c] < 0).any():
                raise ValueError(
                    f'The field {c} contains negative values, so log1p cannot be applied.'
                )

            df_out[f'log_{c}'] = np.log1p(df_out[c])

    return df_out


def check_xgboost():
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            'xgboost is not installed. Please install it first: pip install xgboost'
        ) from e

    version_str = getattr(
        xgb,
        '__version__',
        '0.0.0'
    )

    try:
        major = int(version_str.split('.')[0])
    except Exception:
        major = 0

    if major < 2:
        raise RuntimeError(
            f'The current xgboost version is {version_str}, which is too old and '
            f'does not support reg:quantileerror. Please upgrade to >= 2.0.'
        )

    return xgb, version_str


def make_qxgb_model(params: Dict):
    xgb, _ = check_xgboost()

    base_params = dict(
        objective='reg:quantileerror',
        quantile_alpha=FRONTIER_Q,
        tree_method='hist',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    if USE_EARLY_STOPPING:
        base_params['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS

    base_params.update(params)

    model = xgb.XGBRegressor(**base_params)

    return model


def to_float32_array(X: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        X,
        dtype=np.float32
    )


def sample_random_params(rng: random.Random) -> Dict:
    params = {
        'n_estimators': rng.randint(300, 1600),
        'learning_rate': 10 ** rng.uniform(
            math.log10(0.008),
            math.log10(0.12)
        ),
        'max_depth': rng.randint(3, 10),
        'min_child_weight': rng.randint(1, 20),
        'subsample': rng.uniform(0.6, 1.0),
        'colsample_bytree': rng.uniform(0.55, 1.0),
        'reg_lambda': 10 ** rng.uniform(
            math.log10(0.1),
            math.log10(30.0)
        ),
        'reg_alpha': 10 ** rng.uniform(
            math.log10(0.0001),
            math.log10(5.0)
        ),
        'gamma': 10 ** rng.uniform(
            math.log10(1e-06),
            math.log10(5.0)
        ),
        'max_bin': rng.choice([
            128,
            192,
            256,
            320,
            384,
            512
        ])
    }

    return params


def sample_local_params(
    best_params: Dict,
    rng: random.Random
) -> Dict:

    def clip(v, low, high):
        return max(
            low,
            min(high, v)
        )

    params = dict(best_params)

    params['n_estimators'] = int(
        clip(
            round(
                best_params['n_estimators']
                * rng.uniform(0.75, 1.3)
            ),
            200,
            2500
        )
    )

    params['learning_rate'] = clip(
        best_params['learning_rate']
        * 10 ** rng.uniform(-0.25, 0.25),
        0.003,
        0.2
    )

    params['max_depth'] = int(
        clip(
            best_params['max_depth']
            + rng.choice([-2, -1, 0, 1, 2]),
            2,
            12
        )
    )

    params['min_child_weight'] = int(
        clip(
            round(
                best_params['min_child_weight']
                * rng.uniform(0.6, 1.6)
            ),
            1,
            40
        )
    )

    params['subsample'] = clip(
        best_params['subsample']
        + rng.uniform(-0.12, 0.12),
        0.5,
        1.0
    )

    params['colsample_bytree'] = clip(
        best_params['colsample_bytree']
        + rng.uniform(-0.12, 0.12),
        0.5,
        1.0
    )

    params['reg_lambda'] = clip(
        best_params['reg_lambda']
        * 10 ** rng.uniform(-0.35, 0.35),
        0.01,
        100.0
    )

    params['reg_alpha'] = clip(
        best_params['reg_alpha']
        * 10 ** rng.uniform(-0.45, 0.45),
        1e-06,
        20.0
    )

    params['gamma'] = clip(
        best_params['gamma']
        * 10 ** rng.uniform(-0.45, 0.45),
        1e-08,
        20.0
    )

    params['max_bin'] = int(
        clip(
            best_params['max_bin']
            + rng.choice([-128, -64, 0, 64, 128]),
            64,
            512
        )
    )

    return params


def evaluate_params_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    group_col: str,
    params: Dict,
    n_splits: int
) -> Tuple[Dict, np.ndarray]:
    gkf = GroupKFold(n_splits=n_splits)

    X_df = df[feature_cols].copy()
    y = df[target_col].values
    groups = df[group_col].values

    oof_pred = np.full(
        len(df),
        np.nan
    )

    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_df, y, groups=groups),
        start=1
    ):
        X_tr_df, X_va_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        X_tr = to_float32_array(X_tr_df)
        X_va = to_float32_array(X_va_df)

        model = make_qxgb_model(params)

        if USE_EARLY_STOPPING:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
        else:
            model.fit(
                X_tr,
                y_tr,
                verbose=False
            )

        pred_va = model.predict(X_va)

        oof_pred[va_idx] = pred_va

        fold_pinball = pinball_loss(
            y_va,
            pred_va,
            FRONTIER_Q
        )

        fold_cov = float(
            np.mean(y_va <= pred_va)
        )

        fold_rows.append({
            'fold': fold,
            'pinball_loss': fold_pinball,
            'coverage': fold_cov,
            'frontier_mean': float(np.mean(pred_va)),
            'frontier_median': float(np.median(pred_va))
        })

    fold_df = pd.DataFrame(fold_rows)

    pinball_mean = float(
        fold_df['pinball_loss'].mean()
    )

    coverage_mean = float(
        fold_df['coverage'].mean()
    )

    coverage_std = float(
        fold_df['coverage'].std(ddof=0)
    )

    coverage_gap = abs(
        coverage_mean - TARGET_COVERAGE
    )

    objective_score = (
        pinball_mean
        + COVERAGE_WEIGHT * coverage_gap
        + STABILITY_WEIGHT * coverage_std
    )

    summary = {
        'pinball_mean': pinball_mean,
        'coverage_mean': coverage_mean,
        'coverage_std': coverage_std,
        'coverage_gap_abs': coverage_gap,
        'objective_score': objective_score,
        'frontier_mean': float(np.mean(oof_pred)),
        'frontier_median': float(np.median(oof_pred))
    }

    return summary, oof_pred


def fit_final_model_full(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Dict
):
    X_df = df[feature_cols].copy()
    y = df[target_col].values

    X = to_float32_array(X_df)

    model = make_qxgb_model(params)

    model.fit(
        X,
        y,
        verbose=False
    )

    pred_full = model.predict(X)

    return model, pred_full


def export_feature_importance(
    model,
    feature_cols: List[str]
) -> pd.DataFrame:
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame({
            'Feature': feature_cols,
            'Importance': [np.nan] * len(feature_cols),
            'Note': ['No feature_importances_'] * len(feature_cols)
        })

    importances = np.asarray(
        model.feature_importances_
    ).ravel()

    n_imp = len(importances)
    n_feat = len(feature_cols)

    if n_imp == n_feat:
        return pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances,
            'Note': ['Matched'] * n_feat
        }).sort_values(
            'Importance',
            ascending=False
        )

    fallback_names = [
        f'Feature_{i + 1}'
        for i in range(n_imp)
    ]

    return pd.DataFrame({
        'Feature': fallback_names,
        'Importance': importances,
        'Note': [
            f'Length mismatch: requested={n_feat}, actual={n_imp}'
        ] * n_imp
    }).sort_values(
        'Importance',
        ascending=False
    )


def main():
    ensure_dir(OUTPUT_DIR)

    _, xgb_version = check_xgboost()

    df = pd.read_csv(INPUT_CSV)

    raw_feature_input_cols, feature_cols = build_feature_columns(
        raw_log_cols=RAW_LOG_COLS,
        raw_direct_cols=RAW_DIRECT_COLS,
        extra_direct_cols=RAW_EXTRA_DIRECT_COLS,
        year_col=YEAR_COL,
        apply_log1p=APPLY_LOG1P
    )

    required_cols = list(
        dict.fromkeys(
            OPTIONAL_META_COLS
            + [YEAR_COL, TARGET_COL]
            + raw_feature_input_cols
        )
    )

    check_required_columns(
        df,
        required_cols
    )

    df = df[
        df[YEAR_COL].isin(KEEP_YEARS)
    ].copy()

    df.sort_values(
        [GROUP_COL, YEAR_COL],
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    dup_n = df.duplicated(
        subset=[
            GROUP_COL,
            YEAR_COL
        ]
    ).sum()

    if dup_n > 0:
        raise ValueError(
            f'Duplicate Code-Year records exist: {dup_n}. Please remove duplicates first.'
        )

    meta_cols = [
        c for c in OPTIONAL_META_COLS
        if c in df.columns
    ]

    desc_cols = raw_feature_input_cols + [TARGET_COL]
    desc_cols = list(dict.fromkeys(desc_cols))

    base_info = []

    base_info.append('=== Basic information ===')
    base_info.append(f'xgboost_version: {xgb_version}')
    base_info.append(f'Number of sample rows: {len(df)}')
    base_info.append(f'Number of regions: {df[GROUP_COL].nunique()}')
    base_info.append(f'Years: {sorted(df[YEAR_COL].unique().tolist())}')
    base_info.append(f'Number of duplicate Code-Year records: {dup_n}')
    base_info.append(f'Target quantile: {FRONTIER_Q}')
    base_info.append(f'APPLY_LOG1P: {APPLY_LOG1P}')
    base_info.append(f'RAW_LOG_COLS: {RAW_LOG_COLS}')
    base_info.append(f'RAW_DIRECT_COLS: {RAW_DIRECT_COLS}')
    base_info.append(f'RAW_EXTRA_DIRECT_COLS: {RAW_EXTRA_DIRECT_COLS}')
    base_info.append(f'FINAL feature_cols: {feature_cols}')

    save_text(
        '\n'.join(map(str, base_info)),
        os.path.join(OUTPUT_DIR, '00_basic_info.txt')
    )

    desc_df = describe_df(
        df,
        desc_cols
    )

    desc_df.to_csv(
        os.path.join(OUTPUT_DIR, '01_descriptive_statistics.csv'),
        encoding='utf-8-sig'
    )

    missing_df = (
        df[required_cols]
        .isna()
        .sum()
        .rename('missing_n')
        .to_frame()
    )

    missing_df['missing_pct'] = (
        missing_df['missing_n'] / len(df)
    )

    missing_df.to_csv(
        os.path.join(OUTPUT_DIR, '02_missing_report.csv'),
        encoding='utf-8-sig'
    )

    df_model = apply_feature_transformations(
        df=df,
        raw_log_cols=RAW_LOG_COLS,
        apply_log1p=APPLY_LOG1P
    )

    model_ready_cols = list(
        dict.fromkeys(
            meta_cols
            + [TARGET_COL]
            + feature_cols
        )
    )

    model_ready_df = df_model[
        model_ready_cols
    ].copy()

    model_ready_df.to_csv(
        os.path.join(OUTPUT_DIR, '03_model_ready_baseline.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    rng = random.Random(RANDOM_STATE)

    search_records = []

    best_score = np.inf
    best_params = None
    best_oof = None

    for i in range(1, N_RANDOM_SEARCH + 1):
        params = sample_random_params(rng)

        summary, oof_pred = evaluate_params_cv(
            df=model_ready_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            params=params,
            n_splits=N_SPLITS
        )

        record = {
            'stage': 'random',
            'iter': i,
            **params,
            **summary
        }

        search_records.append(record)

        if summary['objective_score'] < best_score:
            best_score = summary['objective_score']
            best_params = dict(params)
            best_oof = oof_pred.copy()

        if i % 10 == 0 or i == 1:
            pass

    for i in range(1, N_LOCAL_SEARCH + 1):
        params = sample_local_params(
            best_params,
            rng
        )

        summary, oof_pred = evaluate_params_cv(
            df=model_ready_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            params=params,
            n_splits=N_SPLITS
        )

        record = {
            'stage': 'local',
            'iter': i,
            **params,
            **summary
        }

        search_records.append(record)

        if summary['objective_score'] < best_score:
            best_score = summary['objective_score']
            best_params = dict(params)
            best_oof = oof_pred.copy()

        if i % 10 == 0 or i == 1:
            pass

    search_df = pd.DataFrame(
        search_records
    ).sort_values(
        'objective_score',
        ascending=True
    )

    search_df.to_csv(
        os.path.join(OUTPUT_DIR, '04_qxgb_tuning_history.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    search_df.head(30).to_csv(
        os.path.join(OUTPUT_DIR, '05_qxgb_top30_candidates.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    with open(
        os.path.join(OUTPUT_DIR, '06_best_params.json'),
        'w',
        encoding='utf-8'
    ) as f:
        json.dump(
            best_params,
            f,
            ensure_ascii=False,
            indent=2
        )

    best_summary, best_oof = evaluate_params_cv(
        df=model_ready_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        params=best_params,
        n_splits=N_SPLITS
    )

    final_model, pred_full = fit_final_model_full(
        df=model_ready_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        params=best_params
    )

    imp_df = export_feature_importance(
        final_model,
        feature_cols
    )

    imp_df.to_csv(
        os.path.join(OUTPUT_DIR, '07_feature_importance_final_model.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    result_df = model_ready_df.copy()

    result_df['Model'] = FINAL_MODEL_NAME
    result_df['Frontier_OOF'] = best_oof
    result_df['Gap_OOF'] = np.maximum(
        0.0,
        result_df['Frontier_OOF'] - result_df[TARGET_COL]
    )

    result_df['Frontier_Full'] = pred_full
    result_df['Gap_Full'] = np.maximum(
        0.0,
        result_df['Frontier_Full'] - result_df[TARGET_COL]
    )

    result_df.to_csv(
        os.path.join(OUTPUT_DIR, '08_region_year_frontier_gap_results.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    region_group_cols = [
        c for c in [
            GROUP_COL,
            'Province_ID',
            'Prefecture',
            'County'
        ]
        if c in result_df.columns
    ]

    region_mean_df = (
        result_df
        .groupby(region_group_cols, dropna=False)
        .agg(
            S_star_mean=(TARGET_COL, 'mean'),
            S_star_2022=(TARGET_COL, lambda x: x.iloc[-1]),
            Frontier_OOF_mean=('Frontier_OOF', 'mean'),
            Gap_OOF_mean=('Gap_OOF', 'mean'),
            Frontier_Full_mean=('Frontier_Full', 'mean'),
            Gap_Full_mean=('Gap_Full', 'mean')
        )
        .reset_index()
    )

    region_mean_df = classify_four_types(
        region_mean_df,
        actual_col='S_star_mean',
        gap_col='Gap_OOF_mean'
    )

    region_mean_df['Model'] = FINAL_MODEL_NAME

    region_mean_df.to_csv(
        os.path.join(OUTPUT_DIR, '09_region_mean_gap_and_type.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    type_summary = (
        region_mean_df['Potential_Type']
        .value_counts(dropna=False)
        .rename_axis('Potential_Type')
        .reset_index(name='Count')
    )

    type_summary['Percent'] = (
        type_summary['Count']
        / type_summary['Count'].sum()
    )

    type_summary.to_csv(
        os.path.join(OUTPUT_DIR, '10_potential_type_summary.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    plot_frontier_scatter(
        result_df,
        actual_col=TARGET_COL,
        frontier_col='Frontier_OOF',
        save_path=os.path.join(
            OUTPUT_DIR,
            '11_scatter_observed_vs_oof_frontier.png'
        )
    )

    plot_gap_hist(
        result_df,
        gap_col='Gap_OOF',
        save_path=os.path.join(
            OUTPUT_DIR,
            '12_hist_gap_oof.png'
        )
    )

    plot_quadrant(
        region_mean_df,
        actual_col='S_star_mean',
        gap_col='Gap_OOF_mean',
        save_path=os.path.join(
            OUTPUT_DIR,
            '13_quadrant_region_mean_actual_vs_gap.png'
        )
    )

    summary_lines = []

    summary_lines.append('=== QXGB conditional frontier enhanced tuning completed ===')
    summary_lines.append(f'xgboost_version = {xgb_version}')
    summary_lines.append(f'Target quantile q = {FRONTIER_Q}')
    summary_lines.append(f'Number of random search iterations = {N_RANDOM_SEARCH}')
    summary_lines.append(f'Number of local refinement iterations = {N_LOCAL_SEARCH}')
    summary_lines.append(f'APPLY_LOG1P = {APPLY_LOG1P}')
    summary_lines.append('')
    summary_lines.append('[Original log-transformed variables]')
    summary_lines.append(str(RAW_LOG_COLS))
    summary_lines.append('')
    summary_lines.append('[Original directly modeled variables]')
    summary_lines.append(str(RAW_DIRECT_COLS + RAW_EXTRA_DIRECT_COLS))
    summary_lines.append('')
    summary_lines.append('[Final modeled variables]')
    summary_lines.append(str(feature_cols))
    summary_lines.append('')
    summary_lines.append('[Best parameters]')
    summary_lines.append(
        json.dumps(
            best_params,
            ensure_ascii=False,
            indent=2
        )
    )
    summary_lines.append('')
    summary_lines.append('[Best CV results]')

    for k, v in best_summary.items():
        summary_lines.append(f'{k} = {v}')

    summary_lines.append('')
    summary_lines.append('[OOF Gap description]')
    summary_lines.append(
        result_df['Gap_OOF']
        .describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
                0.95
            ]
        )
        .to_string()
    )

    summary_lines.append('')
    summary_lines.append('[Full frontier reference description]')
    summary_lines.append(
        result_df['Gap_Full']
        .describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9,
                0.95
            ]
        )
        .to_string()
    )

    summary_lines.append('')
    summary_lines.append('[Potential type proportions]')
    summary_lines.append(
        type_summary.to_string(index=False)
    )

    save_text(
        '\n'.join(summary_lines),
        os.path.join(OUTPUT_DIR, '99_run_summary.txt')
    )


if __name__ == '__main__':
    main()