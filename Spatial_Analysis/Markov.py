import pandas as pd
import numpy as np


input_csv = 'Total_Data_with_QSstar.csv'

code_col = 'Code'
year_col = 'Year'
value_col = 'S_star'

weak_upper = 0.2
medium_upper = 0.5

state_panel_file = 'QSstar_state_panel.csv'
state_counts_file = 'QSstar_state_distribution_counts.csv'
state_shares_file = 'QSstar_state_distribution_shares.csv'
transition_pairs_file = 'QSstar_transition_pairs.csv'
transition_counts_file = 'QSstar_markov_transition_counts.csv'
transition_probs_file = 'QSstar_markov_transition_probs.csv'
transition_summary_file = 'QSstar_markov_summary.csv'
duplicate_file = 'QSstar_duplicate_code_year_rows.csv'


state_name_map = {
    1: 'State 1: Non-synergistic / imbalanced',
    2: 'State 2: Weak synergy',
    3: 'State 3: Moderate synergy',
    4: 'State 4: High synergy'
}


df = pd.read_csv(input_csv, encoding='utf-8-sig')


required_cols = [
    code_col,
    year_col,
    value_col
]

missing_cols = [
    c for c in required_cols
    if c not in df.columns
]

if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')


df[code_col] = (
    df[code_col]
    .astype(str)
    .str.replace('\\.0$', '', regex=True)
    .str.strip()
    .str.zfill(6)
)

df[year_col] = pd.to_numeric(
    df[year_col],
    errors='coerce'
)

df[value_col] = pd.to_numeric(
    df[value_col],
    errors='coerce'
)


df = df[required_cols].copy()

df = df[
    df[year_col] != 2013
].copy()

df = df.dropna(
    subset=[
        code_col,
        year_col,
        value_col
    ]
).copy()

df[year_col] = df[year_col].astype(int)


dups = df[
    df.duplicated(
        subset=[
            code_col,
            year_col
        ],
        keep=False
    )
].copy()


if not dups.empty:
    dups.to_csv(
        duplicate_file,
        index=False,
        encoding='utf-8-sig'
    )

    raise ValueError(
        f'Duplicate Code-Year combinations were found and exported to {duplicate_file}. '
        f'Please handle duplicate values before rerunning the script.'
    )


df = df.sort_values(
    [
        code_col,
        year_col
    ]
).reset_index(drop=True)


def assign_state(
    x,
    weak_upper=0.2,
    medium_upper=0.5
):
    if x <= 0:
        return 1
    elif x <= weak_upper:
        return 2
    elif x <= medium_upper:
        return 3
    else:
        return 4


df['state'] = df[value_col].apply(
    lambda x: assign_state(
        x,
        weak_upper=weak_upper,
        medium_upper=medium_upper
    )
)

df['state_name'] = df['state'].map(state_name_map)


df.to_csv(
    state_panel_file,
    index=False,
    encoding='utf-8-sig'
)


state_order = [
    1,
    2,
    3,
    4
]


state_counts = pd.crosstab(
    df[year_col],
    df['state']
).reindex(
    columns=state_order,
    fill_value=0
)

state_counts.columns = [
    state_name_map[c]
    for c in state_counts.columns
]

state_counts.index.name = 'Year'

state_counts.to_csv(
    state_counts_file,
    encoding='utf-8-sig'
)


state_shares = state_counts.div(
    state_counts.sum(axis=1),
    axis=0
)

state_shares.to_csv(
    state_shares_file,
    encoding='utf-8-sig'
)


df['Year_next'] = df.groupby(code_col)[year_col].shift(-1)
df['S_star_next'] = df.groupby(code_col)[value_col].shift(-1)
df['state_next'] = df.groupby(code_col)['state'].shift(-1)
df['state_name_next'] = df.groupby(code_col)['state_name'].shift(-1)


transitions = df[
    df['Year_next'] == df[year_col] + 1
].copy()


transitions = transitions.rename(
    columns={
        year_col: 'Year_t',
        value_col: 'S_star_t',
        'state': 'state_t',
        'state_name': 'state_name_t'
    }
)


transitions['Year_t1'] = transitions['Year_next'].astype(int)
transitions['state_t1'] = transitions['state_next'].astype(int)


transitions = transitions[
    [
        code_col,
        'Year_t',
        'Year_t1',
        'S_star_t',
        'S_star_next',
        'state_t',
        'state_name_t',
        'state_t1',
        'state_name_next'
    ]
].copy()


transitions.to_csv(
    transition_pairs_file,
    index=False,
    encoding='utf-8-sig'
)


transition_counts = pd.crosstab(
    transitions['state_t'],
    transitions['state_t1']
).reindex(
    index=state_order,
    columns=state_order,
    fill_value=0
)


transition_counts.index = [
    state_name_map[i]
    for i in transition_counts.index
]

transition_counts.columns = [
    state_name_map[i]
    for i in transition_counts.columns
]


transition_counts.to_csv(
    transition_counts_file,
    encoding='utf-8-sig'
)


row_sums = transition_counts.sum(axis=1).replace(
    0,
    np.nan
)

transition_probs = transition_counts.div(
    row_sums,
    axis=0
)

transition_probs.to_csv(
    transition_probs_file,
    encoding='utf-8-sig'
)


summary_rows = []


transition_counts_num = pd.crosstab(
    transitions['state_t'],
    transitions['state_t1']
).reindex(
    index=state_order,
    columns=state_order,
    fill_value=0
)


row_sums_num = transition_counts_num.sum(axis=1).replace(
    0,
    np.nan
)

transition_probs_num = transition_counts_num.div(
    row_sums_num,
    axis=0
)


for s in state_order:
    stay_prob = transition_probs_num.loc[s, s]

    up_prob = transition_probs_num.loc[
        s,
        [
            j for j in state_order
            if j > s
        ]
    ].sum()

    down_prob = transition_probs_num.loc[
        s,
        [
            j for j in state_order
            if j < s
        ]
    ].sum()

    n_from = transition_counts_num.loc[s].sum()

    summary_rows.append({
        'state': s,
        'state_name': state_name_map[s],
        'N_from_state': n_from,
        'stay_probability': stay_prob,
        'upward_probability': up_prob,
        'downward_probability': down_prob
    })


summary_df = pd.DataFrame(summary_rows)


summary_df.to_csv(
    transition_summary_file,
    index=False,
    encoding='utf-8-sig'
)