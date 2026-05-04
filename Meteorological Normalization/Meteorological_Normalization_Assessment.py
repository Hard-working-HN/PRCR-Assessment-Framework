import pandas as pd
import numpy as np
from scipy.stats import linregress


obs = pd.read_csv('annual_mean_PM25.csv')
dw = pd.read_csv('PM25_annual_deweathered_GAM.csv')

obs['Date'] = pd.to_datetime(obs['Date'])
dw['Date'] = pd.to_datetime(dw['Date'])

regions = sorted(
    list(
        set(obs.columns[1:]).intersection(set(dw.columns[1:]))
    )
)

obs = (
    obs[['Date'] + regions]
    .sort_values('Date')
    .set_index('Date')
)

dw = (
    dw[['Date'] + regions]
    .sort_values('Date')
    .set_index('Date')
)

common_dates = obs.index.intersection(dw.index)

obs = obs.loc[common_dates]
dw = dw.loc[common_dates]

results = []

for r in regions:
    o = obs[r].astype(float)
    d = dw[r].astype(float)

    valid = ~(o.isna() | d.isna())

    o = o[valid]
    d = d[valid]

    if len(o) < 4:
        continue

    mean_obs = o.mean()
    mean_dw = d.mean()

    mean_shift = (
        np.nan
        if mean_obs == 0
        else abs(mean_dw - mean_obs) / abs(mean_obs)
    )

    sd_obs = o.std(ddof=1)
    sd_dw = d.std(ddof=1)

    sd_ratio = (
        np.nan
        if sd_obs == 0
        else sd_dw / sd_obs
    )

    corr_level = o.corr(d)

    od = o.diff().dropna()
    dd = d.diff().dropna()

    corr_diff = (
        od.corr(dd)
        if len(od) >= 2
        else np.nan
    )

    x = np.arange(len(o))

    slope_obs = linregress(x, o).slope
    slope_dw = linregress(x, d).slope

    same_slope_sign = np.sign(slope_obs) == np.sign(slope_dw)

    sign_obs = np.sign(od)
    sign_dw = np.sign(dd)

    same_sign_ratio = (sign_obs == sign_dw).mean()

    has_negative_dw = (d < 0).any()

    score = 0

    if not has_negative_dw:
        score += 1

    if pd.notna(mean_shift) and mean_shift <= 0.2:
        score += 1

    if pd.notna(corr_level) and corr_level >= 0.8:
        score += 1

    if pd.notna(sd_ratio) and 0.5 <= sd_ratio <= 1.1:
        score += 1

    if same_slope_sign:
        score += 1

    if score >= 5:
        label = 'high_confidence'
    elif score >= 4:
        label = 'usable'
    elif score >= 3:
        label = 'cautious'
    else:
        label = 'questionable'

    results.append({
        'Region': r,
        'mean_obs': mean_obs,
        'mean_dw': mean_dw,
        'mean_shift': mean_shift,
        'sd_ratio': sd_ratio,
        'corr_level': corr_level,
        'corr_diff': corr_diff,
        'slope_obs': slope_obs,
        'slope_dw': slope_dw,
        'same_slope_sign': same_slope_sign,
        'same_sign_ratio': same_sign_ratio,
        'has_negative_dw': has_negative_dw,
        'score': score,
        'label': label
    })


res = pd.DataFrame(results).sort_values(
    ['score', 'corr_level'],
    ascending=[False, False]
)

res.to_csv(
    'deweathered_PM25_quality_check.csv',
    index=False,
    encoding='utf-8-sig'
)