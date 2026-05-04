import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import copy
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


CSV_PATH = 'Total_Data_with_QSstar_merged_None_Name.csv'

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

RESULT_DIR = 'tabular_models_no_code_no_year_results'
METRICS_DIR = os.path.join(RESULT_DIR, 'metrics')
PRED_DIR = os.path.join(RESULT_DIR, 'predictions')
MODEL_DIR = os.path.join(RESULT_DIR, 'saved_models')

IGNORE_FEATURE_COLS = ['Code', 'Year']

NN_EPOCHS = 2000
NN_BATCH_SIZE = 256
NN_LR = 0.001
NN_WEIGHT_DECAY = 1e-05
NN_PATIENCE = 180
NN_LR_REDUCE_PATIENCE = 60
NN_LR_REDUCE_FACTOR = 0.8

MLP_HIDDEN_DIMS = [256, 128]
MLP_DROPOUT = 0.15

RNN_HIDDEN_DIM = 64
RNN_NUM_LAYERS = 1
RNN_DROPOUT = 0.0


def ensure_dirs():
    for d in [RESULT_DIR, METRICS_DIR, PRED_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_code(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    if s.endswith('.0'):
        s = s[:-2]

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


def random_split_indices(
    n,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-08:
        raise ValueError(
            'TRAIN_RATIO + VALID_RATIO + TEST_RATIO must equal 1.'
        )

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


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def detect_feature_types(df, feature_cols):
    numeric_cols = []
    categorical_cols = []

    for c in feature_cols:
        converted = pd.to_numeric(df[c], errors='coerce')
        orig_na = df[c].isna().sum()
        conv_na = converted.isna().sum()

        if conv_na == orig_na:
            df[c] = converted.astype(float)
            numeric_cols.append(c)
        else:
            df[c] = df[c].astype(str)
            categorical_cols.append(c)

    return df, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if len(numeric_cols) > 0:
        numeric_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_pipe, numeric_cols))

    if len(categorical_cols) > 0:
        categorical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', make_onehot_encoder())
        ])
        transformers.append(('cat', categorical_pipe, categorical_cols))

    if len(transformers) == 0:
        raise ValueError(
            'No available input features. Please check whether the CSV columns '
            'only contain the target column or whether all features have been excluded.'
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def safe_to_numpy_dense(x):
    if hasattr(x, 'toarray'):
        x = x.toarray()

    return np.asarray(x, dtype=np.float32)


def save_predictions(model_name, split_name, meta_df, y_true, y_pred):
    out_df = meta_df.copy().reset_index(drop=True)

    out_df['y_true'] = y_true.reshape(-1)
    out_df['y_pred'] = y_pred.reshape(-1)

    out_path = os.path.join(PRED_DIR, f'{model_name}_{split_name}_predictions.csv')
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    return out_path


ensure_dirs()
seed_everything(SEED)

df = pd.read_csv(CSV_PATH)

target_col = df.columns[-1]

df['source_row_index'] = np.arange(len(df))

if 'Code' in df.columns:
    df['Code'] = df['Code'].apply(clean_code)

if 'Year' in df.columns:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

before_n = len(df)

df = df[df[target_col].notna()].copy().reset_index(drop=True)

after_n = len(df)

if after_n < 3:
    raise ValueError(
        'The number of valid target samples is too small to split into '
        'train, valid, and test sets.'
    )

feature_cols = [
    c for c in df.columns
    if c != target_col
    and c != 'source_row_index'
    and (c not in IGNORE_FEATURE_COLS)
]

if len(feature_cols) == 0:
    raise ValueError(
        'After removing the target column, Code, and Year, '
        'there are no remaining input features.'
    )

working_df = df.copy()

working_df, numeric_cols, categorical_cols = detect_feature_types(
    df=working_df,
    feature_cols=feature_cols
)

train_idx, valid_idx, test_idx = random_split_indices(
    n=len(working_df),
    train_ratio=TRAIN_RATIO,
    valid_ratio=VALID_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED
)

working_df['split'] = 'unused'
working_df.loc[train_idx, 'split'] = 'train'
working_df.loc[valid_idx, 'split'] = 'valid'
working_df.loc[test_idx, 'split'] = 'test'

split_counts = working_df['split'].value_counts()

train_df = working_df.loc[working_df['split'] == 'train'].copy().reset_index(drop=True)
valid_df = working_df.loc[working_df['split'] == 'valid'].copy().reset_index(drop=True)
test_df = working_df.loc[working_df['split'] == 'test'].copy().reset_index(drop=True)

X_train_df = train_df[feature_cols].copy()
X_valid_df = valid_df[feature_cols].copy()
X_test_df = test_df[feature_cols].copy()

y_train = train_df[target_col].values.astype(np.float32)
y_valid = valid_df[target_col].values.astype(np.float32)
y_test = test_df[target_col].values.astype(np.float32)

meta_cols = ['source_row_index']

train_meta = train_df[meta_cols].copy()
valid_meta = valid_df[meta_cols].copy()
test_meta = test_df[meta_cols].copy()

preprocessor = build_preprocessor(numeric_cols, categorical_cols)

preprocessor.fit(X_train_df)

X_train = safe_to_numpy_dense(preprocessor.transform(X_train_df))
X_valid = safe_to_numpy_dense(preprocessor.transform(X_valid_df))
X_test = safe_to_numpy_dense(preprocessor.transform(X_test_df))

try:
    transformed_feature_names = preprocessor.get_feature_names_out()

    pd.DataFrame({
        'feature': transformed_feature_names
    }).to_csv(
        os.path.join(METRICS_DIR, 'transformed_feature_names.csv'),
        index=False,
        encoding='utf-8-sig'
    )

except Exception:
    pass


class MLPRegressorTorch(nn.Module):

    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.15):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hid in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hid),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hid

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RNNRegressorTorch(nn.Module):

    def __init__(self, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity='tanh',
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)

        _, h_n = self.rnn(x)

        last_hidden = h_n[-1]

        return self.head(last_hidden).squeeze(-1)


def make_tensor_loader(X, y, batch_size=256, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


def evaluate_torch_model(model, X, y, batch_size=1024):
    loader = make_tensor_loader(X, y, batch_size=batch_size, shuffle=False)

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            pred = model(xb).detach().cpu().numpy().reshape(-1)

            preds.append(pred)
            trues.append(yb.numpy().reshape(-1))

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)

    return y_true, y_pred


def train_torch_regressor(
    model_name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid
):
    train_loader = make_tensor_loader(
        X_train,
        y_train,
        batch_size=NN_BATCH_SIZE,
        shuffle=True
    )

    valid_loader = make_tensor_loader(
        X_valid,
        y_valid,
        batch_size=NN_BATCH_SIZE,
        shuffle=False
    )

    model = model.to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_LR,
        weight_decay=NN_WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=NN_LR_REDUCE_FACTOR,
        patience=NN_LR_REDUCE_PATIENCE,
        threshold=1e-06,
        threshold_mode='rel',
        min_lr=1e-06
    )

    best_valid_r2 = -np.inf
    best_epoch = -1
    best_state = None
    patience_count = 0

    for epoch in range(1, NN_EPOCHS + 1):
        model.train()

        batch_losses = []
        train_preds = []
        train_trues = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            batch_losses.append(float(loss.item()))
            train_preds.append(pred.detach().cpu().numpy().reshape(-1))
            train_trues.append(yb.detach().cpu().numpy().reshape(-1))

        train_loss = float(np.mean(batch_losses)) if len(batch_losses) > 0 else np.nan

        train_true = np.concatenate(train_trues)
        train_pred = np.concatenate(train_preds)

        train_r2 = r2_score(train_true, train_pred)

        model.eval()

        valid_losses = []
        valid_preds = []
        valid_trues = []

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                pred = model(xb)
                loss = criterion(pred, yb)

                valid_losses.append(float(loss.item()))
                valid_preds.append(pred.detach().cpu().numpy().reshape(-1))
                valid_trues.append(yb.detach().cpu().numpy().reshape(-1))

        valid_loss = float(np.mean(valid_losses)) if len(valid_losses) > 0 else np.nan

        valid_true = np.concatenate(valid_trues)
        valid_pred = np.concatenate(valid_preds)

        valid_r2 = r2_score(valid_true, valid_pred)

        scheduler.step(valid_r2)

        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            pass

        if valid_r2 > best_valid_r2:
            best_valid_r2 = valid_r2
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

            if patience_count >= NN_PATIENCE:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    torch.save(
        best_state,
        os.path.join(MODEL_DIR, f'{model_name}.pt')
    )

    return model


def fit_and_evaluate_sklearn_model(
    model_name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test
):
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train).reshape(-1)
    pred_valid = model.predict(X_valid).reshape(-1)
    pred_test = model.predict(X_test).reshape(-1)

    train_metrics = regression_metrics(y_train, pred_train)
    valid_metrics = regression_metrics(y_valid, pred_valid)
    test_metrics = regression_metrics(y_test, pred_test)

    save_predictions(model_name, 'train', train_meta, y_train, pred_train)
    save_predictions(model_name, 'valid', valid_meta, y_valid, pred_valid)
    save_predictions(model_name, 'test', test_meta, y_test, pred_test)

    return {
        'model': model_name,
        'train': train_metrics,
        'valid': valid_metrics,
        'test': test_metrics
    }


def fit_and_evaluate_torch_model(
    model_name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test
):
    model = train_torch_regressor(
        model_name,
        model,
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    y_train_true, pred_train = evaluate_torch_model(model, X_train, y_train)
    y_valid_true, pred_valid = evaluate_torch_model(model, X_valid, y_valid)
    y_test_true, pred_test = evaluate_torch_model(model, X_test, y_test)

    train_metrics = regression_metrics(y_train_true, pred_train)
    valid_metrics = regression_metrics(y_valid_true, pred_valid)
    test_metrics = regression_metrics(y_test_true, pred_test)

    save_predictions(model_name, 'train', train_meta, y_train_true, pred_train)
    save_predictions(model_name, 'valid', valid_meta, y_valid_true, pred_valid)
    save_predictions(model_name, 'test', test_meta, y_test_true, pred_test)

    return {
        'model': model_name,
        'train': train_metrics,
        'valid': valid_metrics,
        'test': test_metrics
    }


models_to_run = {
    'MultipleLinearRegression': LinearRegression(),
    'BayesianRegression': BayesianRidge(),
    'SVM': SVR(
        C=10.0,
        epsilon=0.1,
        kernel='rbf'
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=SEED
    )
}

if HAS_XGBOOST:
    models_to_run['XGBoost'] = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        objective='reg:squarederror',
        random_state=SEED,
        n_jobs=-1,
        tree_method='auto'
    )


all_results = []

for model_name, model in models_to_run.items():
    result = fit_and_evaluate_sklearn_model(
        model_name=model_name,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test
    )

    all_results.append(result)


mlp_model = MLPRegressorTorch(
    input_dim=X_train.shape[1],
    hidden_dims=tuple(MLP_HIDDEN_DIMS),
    dropout=MLP_DROPOUT
)

all_results.append(
    fit_and_evaluate_torch_model(
        model_name='MLP',
        model=mlp_model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test
    )
)


rnn_model = RNNRegressorTorch(
    hidden_dim=RNN_HIDDEN_DIM,
    num_layers=RNN_NUM_LAYERS,
    dropout=RNN_DROPOUT
)

all_results.append(
    fit_and_evaluate_torch_model(
        model_name='RNN',
        model=rnn_model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test
    )
)


summary_rows = []

for res in all_results:
    model_name = res['model']

    for split_name in ['train', 'valid', 'test']:
        row = {
            'model': model_name,
            'split': split_name
        }

        row.update(res[split_name])
        summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(METRICS_DIR, 'all_models_metrics.csv')

summary_df.to_csv(
    summary_path,
    index=False,
    encoding='utf-8-sig'
)