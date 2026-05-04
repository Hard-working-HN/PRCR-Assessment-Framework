import os
import gc
import time
import warnings
import hashlib
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


warnings.filterwarnings('ignore')


POLLUTANT_FILES = {
    'CO': 'Day_CO.csv',
    'NO2': 'Day_NO2.csv',
    'O3': 'Day_O3.csv',
    'PM10': 'Day_PM10.csv',
    'PM25': 'Day_PM25.csv',
    'SO2': 'Day_SO2.csv'
}

CLIMATE_FILES = {
    'temp_c': 'A_2m_temperature_C.csv',
    'rh': 'relative_humidity.csv',
    'wind_speed': 'wind_speed.csv',
    'wind_dir_sin': 'wind_direction_sin.csv',
    'wind_dir_cos': 'wind_direction_cos.csv',
    'pressure_hpa': 'surface_pressure_hPa.csv',
    'radiation_wm2': 'surface_solar_radiation_downwards_dailymean_Wm2.csv',
    'precip_mm': 'total_precipitation_mm.csv'
}

ROOT_RESULT_DIR = './Result'

RANDOM_STATE = 42
TEST_SIZE = 0.2

N_SPLINES = 8
LAM = 0.75
N_SAMPLES = 300

MIN_VALID_ROWS = 60
LOW_R2_THRESHOLD = 0.6

USE_PARALLEL = True
N_WORKERS = min(max(1, cpu_count() - 2), 24)
BATCH_SIZE = 400
MAX_REGIONS = None

GLOBAL_POLLUTANT_MONTHLY = None
GLOBAL_CLIMATE_MONTHLY = None
GLOBAL_POLLUTANT_NAME = None


def read_wide_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    df = pd.read_csv(path)

    if 'Date' not in df.columns:
        raise ValueError(f"The 'Date' column was not found in {path}")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    value_cols = [c for c in df.columns if c != 'Date']
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce')

    return df


def aggregate_wide_to_monthly_mean(df):
    out = df.copy()

    out['Month'] = out['Date'].dt.to_period('M').dt.to_timestamp()

    value_cols = [c for c in out.columns if c not in ['Date', 'Month']]

    out = out.groupby('Month', as_index=False)[value_cols].mean()
    out = out.rename(columns={'Month': 'Date'})

    return out


def get_common_dates(dfs):
    common = set(dfs[0]['Date'])

    for df in dfs[1:]:
        common &= set(df['Date'])

    return sorted(common)


def get_common_regions(dfs):
    common = set([c for c in dfs[0].columns if c != 'Date'])

    for df in dfs[1:]:
        common &= set([c for c in df.columns if c != 'Date'])

    return sorted(common)


def subset_wide_df(df, common_dates, common_regions):
    out = df[df['Date'].isin(common_dates)].copy()
    out = out[['Date'] + common_regions].sort_values('Date').reset_index(drop=True)

    return out


def stable_region_seed(region, base_seed=42):
    h = hashlib.md5(region.encode('utf-8')).hexdigest()
    region_int = int(h[:8], 16)

    return (base_seed + region_int) % (2 ** 32 - 1)


def split_into_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_region_monthly_df(region, pollutant_df, climate_df_dict, pollutant_col_name):
    data = {
        'Date': pollutant_df['Date'].values,
        pollutant_col_name: pollutant_df[region].values
    }

    for var_name, df in climate_df_dict.items():
        data[var_name] = df[region].values

    df = pd.DataFrame(data)

    df = df.dropna().sort_values('Date').reset_index(drop=True)

    if len(df) == 0:
        return df

    df['trend_month'] = np.arange(len(df), dtype=float)
    df['month'] = df['Date'].dt.month.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['year'] = df['Date'].dt.year.astype(int)

    return df


def fit_gam_model_monthly(df_region, pollutant_col_name):
    feature_cols = [
        'trend_month',
        'month_sin',
        'month_cos',
        'temp_c',
        'rh',
        'wind_speed',
        'wind_dir_sin',
        'wind_dir_cos',
        'pressure_hpa',
        'radiation_wm2',
        'precip_mm'
    ]

    X = df_region[feature_cols].to_numpy(dtype=float)

    y_raw = df_region[pollutant_col_name].to_numpy(dtype=float)
    y = np.log1p(y_raw)

    idx = np.arange(len(df_region))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    terms = (
        s(0, n_splines=N_SPLINES)
        + s(1, n_splines=N_SPLINES)
        + s(2, n_splines=N_SPLINES)
        + s(3, n_splines=N_SPLINES)
        + s(4, n_splines=N_SPLINES)
        + s(5, n_splines=N_SPLINES)
        + s(6, n_splines=N_SPLINES)
        + s(7, n_splines=N_SPLINES)
        + s(8, n_splines=N_SPLINES)
        + s(9, n_splines=N_SPLINES)
        + s(10, n_splines=N_SPLINES)
    )

    gam = LinearGAM(
        terms=terms,
        lam=LAM,
        fit_intercept=True
    )

    gam.fit(X_train, y_train)

    y_pred_test = np.expm1(gam.predict(X_test))
    y_pred_test = np.clip(y_pred_test, 0, None)

    y_test_raw = np.expm1(y_test)
    y_test_raw = np.clip(y_test_raw, 0, None)

    r2 = r2_score(y_test_raw, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_test))

    mean_val = float(df_region[pollutant_col_name].mean())
    std_val = float(df_region[pollutant_col_name].std(ddof=0))

    nrmse_mean = rmse / mean_val if mean_val > 0 else np.nan
    nrmse_sd = rmse / std_val if std_val > 0 else np.nan

    return gam, feature_cols, r2, rmse, nrmse_mean, nrmse_sd


def meteorological_normalisation_monthly_same_region_same_month(
    gam,
    df_region,
    feature_cols,
    n_samples=300,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    X_base = df_region[feature_cols].to_numpy(dtype=float)
    n = X_base.shape[0]

    met_col_idx = np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=int)

    month_arr = df_region['month'].to_numpy(dtype=int)
    month_to_indices = {
        m: np.where(month_arr == m)[0]
        for m in range(1, 13)
    }

    pred_sum = np.zeros(n, dtype=float)

    for _ in range(n_samples):
        sampled_idx = np.empty(n, dtype=int)

        for i in range(n):
            m = month_arr[i]
            pool_idx = month_to_indices[m]

            if len(pool_idx) == 0:
                sampled_idx[i] = i
            else:
                sampled_idx[i] = rng.choice(pool_idx)

        X_tmp = X_base.copy()
        X_tmp[:, met_col_idx] = X_base[sampled_idx][:, met_col_idx]

        pred_log = gam.predict(X_tmp)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)

        pred_sum += pred

    deweathered = pred_sum / n_samples

    return deweathered


def init_worker(pollutant_monthly, climate_monthly, pollutant_name):
    global GLOBAL_POLLUTANT_MONTHLY
    global GLOBAL_CLIMATE_MONTHLY
    global GLOBAL_POLLUTANT_NAME

    GLOBAL_POLLUTANT_MONTHLY = pollutant_monthly
    GLOBAL_CLIMATE_MONTHLY = climate_monthly
    GLOBAL_POLLUTANT_NAME = pollutant_name


def process_one_region(region):
    start = time.time()

    pollutant_col_name = GLOBAL_POLLUTANT_NAME

    df_region = build_region_monthly_df(
        region,
        GLOBAL_POLLUTANT_MONTHLY,
        GLOBAL_CLIMATE_MONTHLY,
        pollutant_col_name
    )

    if len(df_region) == 0:
        elapsed = time.time() - start

        return {
            'status': 'empty',
            'region': region,
            'elapsed_seconds': elapsed
        }

    df_region[f'{pollutant_col_name}_deweathered'] = df_region[pollutant_col_name].values

    r2 = np.nan
    rmse = np.nan
    nrmse_mean = np.nan
    nrmse_sd = np.nan

    fallback_reason = 'none'
    error_msg = ''

    if len(df_region) < MIN_VALID_ROWS:
        fallback_reason = 'insufficient_data'
    else:
        try:
            gam, feature_cols, r2, rmse, nrmse_mean, nrmse_sd = fit_gam_model_monthly(
                df_region,
                pollutant_col_name
            )

            if pd.notna(r2) and r2 < LOW_R2_THRESHOLD:
                fallback_reason = 'low_r2'
            else:
                region_seed = stable_region_seed(region, RANDOM_STATE)

                df_region[f'{pollutant_col_name}_deweathered'] = (
                    meteorological_normalisation_monthly_same_region_same_month(
                        gam=gam,
                        df_region=df_region,
                        feature_cols=feature_cols,
                        n_samples=N_SAMPLES,
                        random_state=region_seed
                    )
                )

                fallback_reason = 'none'

        except Exception as e:
            fallback_reason = 'model_error'
            error_msg = str(e)

    elapsed = time.time() - start

    monthly_series = pd.Series(
        data=df_region[f'{pollutant_col_name}_deweathered'].values,
        index=df_region['Date'].values
    )

    tmp_monthly = df_region[
        [
            'Date',
            'year',
            'month',
            pollutant_col_name,
            f'{pollutant_col_name}_deweathered'
        ]
    ].copy()

    tmp_monthly['Region'] = region
    tmp_monthly['R2_test'] = r2
    tmp_monthly['RMSE_test'] = rmse
    tmp_monthly['nRMSE_mean'] = nrmse_mean
    tmp_monthly['nRMSE_sd'] = nrmse_sd
    tmp_monthly['Fallback_Reason'] = fallback_reason
    tmp_monthly['Use_Observed_As_Result'] = 1 if fallback_reason != 'none' else 0
    tmp_monthly['Error_Message'] = error_msg

    tmp_annual = (
        tmp_monthly
        .groupby('year', as_index=False)
        .agg(
            Observed_Mean=(pollutant_col_name, 'mean'),
            Deweathered_Mean=(f'{pollutant_col_name}_deweathered', 'mean'),
            n_months=(pollutant_col_name, 'size')
        )
    )

    tmp_annual['Region'] = region
    tmp_annual['R2_test'] = r2
    tmp_annual['RMSE_test'] = rmse
    tmp_annual['nRMSE_mean'] = nrmse_mean
    tmp_annual['nRMSE_sd'] = nrmse_sd
    tmp_annual['Fallback_Reason'] = fallback_reason
    tmp_annual['Use_Observed_As_Result'] = 1 if fallback_reason != 'none' else 0
    tmp_annual['Error_Message'] = error_msg

    perf = {
        'Region': region,
        'n_valid_months': len(df_region),
        'R2_test': r2,
        'RMSE_test': rmse,
        'nRMSE_mean': nrmse_mean,
        'nRMSE_sd': nrmse_sd,
        'elapsed_seconds': elapsed,
        'Fallback_Reason': fallback_reason,
        'Use_Observed_As_Result': 1 if fallback_reason != 'none' else 0,
        'Low_R2_Fallback': 1 if fallback_reason == 'low_r2' else 0,
        'Error_Message': error_msg
    }

    return {
        'status': 'ok',
        'region': region,
        'elapsed_seconds': elapsed,
        'monthly_series': monthly_series,
        'monthly_long': tmp_monthly,
        'annual_long': tmp_annual,
        'perf': perf
    }


def run_one_batch(
    batch_regions,
    pollutant_monthly,
    climate_monthly,
    pollutant_name,
    batch_id,
    total_batches
):
    batch_results = {
        'monthly_dict': {},
        'monthly_long_list': [],
        'annual_long_list': [],
        'perf_list': []
    }

    total_in_batch = len(batch_regions)
    finished_in_batch = 0

    if USE_PARALLEL:
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=init_worker,
            initargs=(pollutant_monthly, climate_monthly, pollutant_name)
        ) as executor:
            future_to_region = {
                executor.submit(process_one_region, region): region
                for region in batch_regions
            }

            for future in as_completed(future_to_region):
                finished_in_batch += 1
                result = future.result()
                region = result['region']

                if result['status'] == 'ok':
                    batch_results['monthly_dict'][region] = result['monthly_series']
                    batch_results['monthly_long_list'].append(result['monthly_long'])
                    batch_results['annual_long_list'].append(result['annual_long'])
                    batch_results['perf_list'].append(result['perf'])
    else:
        init_worker(pollutant_monthly, climate_monthly, pollutant_name)

        for region in batch_regions:
            finished_in_batch += 1
            result = process_one_region(region)

            if result['status'] == 'ok':
                batch_results['monthly_dict'][region] = result['monthly_series']
                batch_results['monthly_long_list'].append(result['monthly_long'])
                batch_results['annual_long_list'].append(result['annual_long'])
                batch_results['perf_list'].append(result['perf'])

    return batch_results


def process_one_pollutant(pollutant_name, pollutant_file, climate_monthly_base):
    start_time = time.time()

    result_dir = os.path.join(ROOT_RESULT_DIR, pollutant_name)
    ensure_dir(result_dir)

    out_monthly_obs = os.path.join(
        result_dir,
        f'{pollutant_name}_monthly_observed.csv'
    )

    out_monthly_deweathered = os.path.join(
        result_dir,
        f'{pollutant_name}_monthly_deweathered_GAM.csv'
    )

    out_monthly_long = os.path.join(
        result_dir,
        f'{pollutant_name}_observed_deweathered_monthly_long_GAM.csv'
    )

    out_annual_long = os.path.join(
        result_dir,
        f'{pollutant_name}_observed_deweathered_annual_long_GAM.csv'
    )

    out_model_perf = os.path.join(
        result_dir,
        f'{pollutant_name}_deweather_model_performance_monthly_GAM.csv'
    )

    out_model_summary = os.path.join(
        result_dir,
        f'{pollutant_name}_deweather_model_summary_GAM.csv'
    )

    out_low_r2_regions = os.path.join(
        result_dir,
        f'{pollutant_name}_low_r2_fallback_regions_GAM.csv'
    )

    pollutant_daily = read_wide_csv(pollutant_file)
    pollutant_monthly = aggregate_wide_to_monthly_mean(pollutant_daily)

    all_dfs = [pollutant_monthly] + list(climate_monthly_base.values())

    common_dates = get_common_dates(all_dfs)
    common_regions = get_common_regions(all_dfs)

    if len(common_dates) == 0:
        raise ValueError(
            f'No common dates were found between {pollutant_name} and the climate files.'
        )

    if len(common_regions) == 0:
        raise ValueError(
            f'No common region columns were found between {pollutant_name} and the climate files.'
        )

    if MAX_REGIONS is not None:
        common_regions = common_regions[:MAX_REGIONS]

    pollutant_monthly = subset_wide_df(
        pollutant_monthly,
        common_dates,
        common_regions
    )

    climate_monthly = {
        k: subset_wide_df(df, common_dates, common_regions)
        for k, df in climate_monthly_base.items()
    }

    pollutant_monthly.to_csv(
        out_monthly_obs,
        index=False,
        encoding='utf-8-sig'
    )

    monthly_deweathered_dict = {}
    monthly_long_records = []
    annual_long_records = []
    perf_records = []

    region_batches = list(split_into_batches(common_regions, BATCH_SIZE))
    total_batches = len(region_batches)

    for batch_id, batch_regions in enumerate(region_batches, start=1):
        batch_start = time.time()

        batch_results = run_one_batch(
            batch_regions=batch_regions,
            pollutant_monthly=pollutant_monthly,
            climate_monthly=climate_monthly,
            pollutant_name=pollutant_name,
            batch_id=batch_id,
            total_batches=total_batches
        )

        monthly_deweathered_dict.update(batch_results['monthly_dict'])
        monthly_long_records.extend(batch_results['monthly_long_list'])
        annual_long_records.extend(batch_results['annual_long_list'])
        perf_records.extend(batch_results['perf_list'])

        del batch_results
        gc.collect()

        batch_elapsed = time.time() - batch_start

    monthly_deweathered_out = pd.DataFrame({
        'Date': common_dates
    })

    for region, s_region in monthly_deweathered_dict.items():
        monthly_deweathered_out[region] = pd.Series(s_region).reindex(common_dates).values

    monthly_deweathered_out = monthly_deweathered_out[
        ['Date'] + sorted([c for c in monthly_deweathered_out.columns if c != 'Date'])
    ]

    monthly_deweathered_out.to_csv(
        out_monthly_deweathered,
        index=False,
        encoding='utf-8-sig'
    )

    pollutant_col = pollutant_name
    pollutant_dew_col = f'{pollutant_name}_deweathered'

    if monthly_long_records:
        monthly_long_out = pd.concat(
            monthly_long_records,
            ignore_index=True
        )

        monthly_long_out = monthly_long_out[
            [
                'Region',
                'Date',
                'year',
                'month',
                pollutant_col,
                pollutant_dew_col,
                'R2_test',
                'RMSE_test',
                'nRMSE_mean',
                'nRMSE_sd',
                'Fallback_Reason',
                'Use_Observed_As_Result',
                'Error_Message'
            ]
        ].sort_values(['Region', 'Date'])
    else:
        monthly_long_out = pd.DataFrame()

    monthly_long_out.to_csv(
        out_monthly_long,
        index=False,
        encoding='utf-8-sig'
    )

    if annual_long_records:
        annual_long_out = pd.concat(
            annual_long_records,
            ignore_index=True
        )

        annual_long_out = annual_long_out[
            [
                'Region',
                'year',
                'Observed_Mean',
                'Deweathered_Mean',
                'n_months',
                'R2_test',
                'RMSE_test',
                'nRMSE_mean',
                'nRMSE_sd',
                'Fallback_Reason',
                'Use_Observed_As_Result',
                'Error_Message'
            ]
        ].sort_values(['Region', 'year'])

        annual_long_out = annual_long_out.rename(columns={'year': 'Year'})
    else:
        annual_long_out = pd.DataFrame()

    annual_long_out.to_csv(
        out_annual_long,
        index=False,
        encoding='utf-8-sig'
    )

    perf_out = pd.DataFrame(perf_records)

    if not perf_out.empty:
        perf_out = perf_out.sort_values('Region')

    perf_out.to_csv(
        out_model_perf,
        index=False,
        encoding='utf-8-sig'
    )

    if not perf_out.empty:
        low_r2_out = perf_out[perf_out['Fallback_Reason'] == 'low_r2'].copy()
    else:
        low_r2_out = pd.DataFrame()

    low_r2_out.to_csv(
        out_low_r2_regions,
        index=False,
        encoding='utf-8-sig'
    )

    if not perf_out.empty:
        summary_df = pd.DataFrame({
            'Metric': [
                'pollutant',
                'n_regions_total',
                'n_regions_low_r2_fallback',
                'n_regions_insufficient_data_fallback',
                'n_regions_model_error_fallback',
                'n_regions_all_fallback',
                'mean_R2_test',
                'median_R2_test',
                'mean_RMSE_test',
                'median_RMSE_test',
                'mean_nRMSE_mean',
                'median_nRMSE_mean',
                'mean_nRMSE_sd',
                'median_nRMSE_sd',
                'mean_elapsed_seconds',
                'median_elapsed_seconds'
            ],
            'Value': [
                pollutant_name,
                len(perf_out),
                (perf_out['Fallback_Reason'] == 'low_r2').sum(),
                (perf_out['Fallback_Reason'] == 'insufficient_data').sum(),
                (perf_out['Fallback_Reason'] == 'model_error').sum(),
                (perf_out['Use_Observed_As_Result'] == 1).sum(),
                perf_out['R2_test'].mean(),
                perf_out['R2_test'].median(),
                perf_out['RMSE_test'].mean(),
                perf_out['RMSE_test'].median(),
                perf_out['nRMSE_mean'].mean(),
                perf_out['nRMSE_mean'].median(),
                perf_out['nRMSE_sd'].mean(),
                perf_out['nRMSE_sd'].median(),
                perf_out['elapsed_seconds'].mean(),
                perf_out['elapsed_seconds'].median()
            ]
        })
    else:
        summary_df = pd.DataFrame(columns=['Metric', 'Value'])

    summary_df.to_csv(
        out_model_summary,
        index=False,
        encoding='utf-8-sig'
    )

    total_elapsed = time.time() - start_time

    if not perf_out.empty:
        pass

    del pollutant_daily
    del pollutant_monthly
    del climate_monthly
    del monthly_deweathered_dict
    del monthly_long_records
    del annual_long_records
    del perf_records
    del monthly_deweathered_out
    del monthly_long_out
    del annual_long_out
    del perf_out
    del summary_df

    gc.collect()


def main():
    total_start = time.time()

    ensure_dir(ROOT_RESULT_DIR)

    climate_daily = {
        k: read_wide_csv(v)
        for k, v in CLIMATE_FILES.items()
    }

    climate_monthly_base = {
        k: aggregate_wide_to_monthly_mean(df)
        for k, df in climate_daily.items()
    }

    del climate_daily
    gc.collect()

    for pollutant_name, pollutant_file in POLLUTANT_FILES.items():
        process_one_pollutant(
            pollutant_name=pollutant_name,
            pollutant_file=pollutant_file,
            climate_monthly_base=climate_monthly_base
        )

        gc.collect()

    del climate_monthly_base
    gc.collect()

    total_elapsed = time.time() - total_start


if __name__ == '__main__':
    main()