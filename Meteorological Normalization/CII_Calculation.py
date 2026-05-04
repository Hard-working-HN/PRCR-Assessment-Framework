import pandas as pd
import numpy as np


df = pd.read_csv('Total_Data.csv')


def standardize_zero_anchored(series):
    s = series.copy()
    out = pd.Series(np.nan, index=s.index, dtype=float)

    valid = s.dropna()

    if valid.empty:
        return out

    pos = valid[valid > 0]
    neg = valid[valid < 0]

    pos_scale = np.nan

    if not pos.empty:
        pos_scale = pos.quantile(0.99)

    neg_scale = np.nan

    if not neg.empty:
        neg_scale = abs(neg.quantile(0.01))

    if pd.notna(pos_scale) and pos_scale > 0:
        mask_pos = s > 0
        out.loc[mask_pos] = s.loc[mask_pos] / pos_scale
    else:
        out.loc[s > 0] = 1.0

    if pd.notna(neg_scale) and neg_scale > 0:
        mask_neg = s < 0
        out.loc[mask_neg] = s.loc[mask_neg] / neg_scale
    else:
        out.loc[s < 0] = -1.0

    out.loc[s == 0] = 0.0
    out = out.clip(-1, 1)

    return out


df['Q_C'] = standardize_zero_anchored(df['V_C'])
df['Q_P'] = standardize_zero_anchored(df['V_P'])

eps = 1e-06
delta = 0.01
tau = 0.1

both_good = (df['Q_C'] > 0) & (df['Q_P'] > 0)

df['S_star'] = np.nan

df.loc[both_good, 'S_star'] = (
    2
    * df.loc[both_good, 'Q_C']
    * df.loc[both_good, 'Q_P']
    / (
        df.loc[both_good, 'Q_C']
        + df.loc[both_good, 'Q_P']
        + eps
    )
)

df.loc[~both_good, 'S_star'] = -(
    2
    * np.abs(
        df.loc[~both_good, 'Q_C']
        * df.loc[~both_good, 'Q_P']
    )
    / (
        np.abs(df.loc[~both_good, 'Q_C'])
        + np.abs(df.loc[~both_good, 'Q_P'])
        + eps
    )
)

df['L'] = np.nan

df.loc[both_good, 'L'] = np.log(
    (
        df.loc[both_good, 'Q_C']
        + delta
    )
    / (
        df.loc[both_good, 'Q_P']
        + delta
    )
)

df['collaboration_Type'] = 'Non-collaborative'

df.loc[
    both_good & (df['L'] > tau),
    'collaboration_Type'
] = 'Carbon-leading collaboration'

df.loc[
    both_good & (df['L'] < -tau),
    'collaboration_Type'
] = 'Pollution-leading collaboration'

df.loc[
    both_good & (df['L'].abs() <= tau),
    'collaboration_Type'
] = 'Balanced collaboration'


df.to_csv(
    'Total_Data_with_QSstar_zeroanchored.csv',
    index=False,
    encoding='utf-8-sig'
)