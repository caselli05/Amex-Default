"""Train the LightGBM model of 06_treebaselines on the raw statement rows and save it as the pkl the Flask app loads.

Same steps, parameters and seed as the notebook:
1. split the train customers into train / valid (optional stratified subsample first), stratified by target
2. load each side's statement rows: numerical features as float32, the categorical features as pandas `category`
   with the categories of the training side (values only seen in valid become NaN)
3. LGBMClassifier with early stopping on the validation Amex at customer level (prediction on each customer's
   last statement)
4. validation scores at row level and customer level (last statement, mean over statements)
5. pickle the classifier, reload it through `src.model.Model` and check it gives the same predictions

The run is logged to MLflow (mlflow.db at the project root, experiment `lightgbm_train`): parameters, the
validation scores and the pkl as an artifact.

Run from the project root:
    python -m src.train                    # 150,000 customers, as in the notebook
    python -m src.train --n-customers 0    # all 458,913 customers
"""
import argparse
import logging
import pickle
import sys
import time
import uuid
from pathlib import Path

import duckdb as ddb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.dataclass import InputData
from src.model import DEFAULT_MODEL_PATH, Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.amex_metric import amex_metric_np, normalized_weighted_gini, top_four_percent_captured
from src.utils.data import CATEGORICAL_FEATURES, PROJECT_DIR, TARGET, feature_columns, get_connection

logger = logging.getLogger(__name__)

LGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.1,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    min_child_samples=100,
    metric='None',  # the custom Amex metric is the only one evaluated and the one early stopping watches
    n_jobs=-1,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 50
MIN_DELTA = 1e-4  # an improvement of the validation Amex smaller than this does not reset the patience counter
MLFLOW_EXPERIMENT = 'lightgbm_train'


def split_customers(con: ddb.DuckDBPyConnection, n_customers: int | None, valid_size: float, seed: int) -> None:
    """Create table `split` (customer_ID, split = 'train' / 'valid'): optional stratified subsample of `n_customers`,
    then a split by customer stratified by target, so all the statements of a customer land on the same side."""
    customers = con.sql(f"SELECT customer_ID, ANY_VALUE({TARGET}) AS {TARGET} FROM train "
                        f"GROUP BY customer_ID ORDER BY customer_ID").df()
    if customers.empty:
        raise ValueError('the train table has no customers')
    if n_customers is not None:
        customers, _ = train_test_split(customers, train_size=n_customers, stratify=customers[TARGET],
                                        random_state=seed)
    cust_train, cust_valid = train_test_split(customers, test_size=valid_size, stratify=customers[TARGET],
                                              random_state=seed)
    split = pd.concat([cust_train.assign(split='train'), cust_valid.assign(split='valid')])[['customer_ID', 'split']]
    con.register('split_df', split)
    try:
        con.execute("CREATE OR REPLACE TABLE split AS SELECT * FROM split_df")
    finally:
        con.unregister('split_df')


def load_split(con: ddb.DuckDBPyConnection, name: str, features: list[str]
               ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X (features, categoricals still as strings), y and customer_ID of the statement rows of side `name` of the
    `split` table, sorted by customer_ID, S_2."""
    num_features = [c for c in features if c not in CATEGORICAL_FEATURES]
    cols = ', '.join([f't."{c}"::FLOAT AS "{c}"' for c in num_features]
                     + [f't."{c}"::VARCHAR AS "{c}"' for c in CATEGORICAL_FEATURES])
    df = con.sql(f"""
        SELECT t.customer_ID, t.{TARGET}, {cols}
        FROM train t JOIN split s USING (customer_ID)
        WHERE s.split = '{name}'
        ORDER BY t.customer_ID, t.S_2
    """).df()
    if df.empty:
        raise ValueError(f'no statement rows in split {name!r}')
    customer_id = df.pop('customer_ID')
    y = df.pop(TARGET).astype('int8')
    return df[features], y, customer_id


def last_statement_idx(customer_id: pd.Series) -> np.ndarray:
    """Row position of each customer's last statement (rows are sorted by customer_ID, S_2)."""
    return np.flatnonzero(customer_id.ne(customer_id.shift(-1)).to_numpy())


class AmexLast:
    """LightGBM eval metric: Amex at customer level, on the prediction of each customer's last statement.

    A class rather than a closure so it can be pickled, should LightGBM keep a reference to it in the model.
    """

    def __init__(self, last_idx: np.ndarray):
        self.last_idx = last_idx

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return 'amex', amex_metric_np(np.asarray(y_true)[self.last_idx], np.asarray(y_pred)[self.last_idx]), True


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """auc, gini, top4 and amex (the functions of src/utils/amex_metric) of predictions `p` against targets `y`."""
    y_true = pd.DataFrame({'target': np.asarray(y).astype(int)})
    y_pred = pd.DataFrame({'prediction': np.asarray(p)})
    gini = normalized_weighted_gini(y_true, y_pred)
    top4 = top_four_percent_captured(y_true, y_pred)
    return {'auc': roc_auc_score(y_true['target'], p), 'gini': gini, 'top4': top4, 'amex': 0.5 * (gini + top4)}


def validation_scores(y: pd.Series, p: np.ndarray, customer_id: pd.Series) -> pd.DataFrame:
    """auc, gini, top4 and amex at row level and at customer level (last statement, mean over statements)."""
    rows = pd.DataFrame({'customer_ID': customer_id.to_numpy(), 'target': y.to_numpy(), 'p': p})
    levels = {'rows': rows,
              'customer_last': rows.groupby('customer_ID').last(),  # rows are sorted by S_2 within customer
              'customer_mean': rows.groupby('customer_ID')[['target', 'p']].mean()}
    return pd.DataFrame({level: scores(df['target'], df['p']) for level, df in levels.items()}).T


def save_model(model: lgb.LGBMClassifier, path: Path) -> None:
    """Pickle `model` to `path`, through a temporary file renamed at the end, so a failed or interrupted save never
    leaves a half-written pkl that the app would try to load."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{uuid.uuid4().hex[:8]}.tmp')
    try:
        with open(tmp, 'wb') as f:
            pickle.dump(model, f)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def check_saved_model(path: Path, X: pd.DataFrame, expected: np.ndarray, n: int = 1000) -> None:
    """Reload the pkl the way the app does and check it predicts the same as the trained model on `n` rows,
    given as a plain object ndarray (what the app receives)."""
    X, expected = X.head(n), expected[:n]
    p = Model(path).load().predict(InputData(X.astype(object).to_numpy(), list(X.columns)))
    if not np.allclose(p, expected, atol=1e-6):
        raise RuntimeError(f'the saved model predicts differently from the trained one (max abs diff '
                           f'{np.abs(p - expected).max():.2e})')


def train(n_customers: int | None, valid_size: float, seed: int, output: Path, use_mlflow: bool) -> None:
    t0 = time.time()
    con = get_connection()
    features = feature_columns(con)

    split_customers(con, n_customers, valid_size, seed)
    X_train, y_train, _ = load_split(con, 'train', features)
    X_valid, y_valid, cid_valid = load_split(con, 'valid', features)
    con.close()

    cat_dtypes = {c: pd.CategoricalDtype(sorted(X_train[c].dropna().unique())) for c in CATEGORICAL_FEATURES}
    for X in (X_train, X_valid):
        for c, dt in cat_dtypes.items():
            X[c] = X[c].astype(dt)
    logger.info('loaded %d train rows, %d valid rows, %d features in %.0fs',
                len(X_train), len(X_valid), len(features), time.time() - t0)

    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    t1 = time.time()
    n_train_rows = len(X_train)
    model.fit(X_train, y_train, eval_X=(X_valid,), eval_y=(y_valid,),
              eval_metric=AmexLast(last_statement_idx(cid_valid)),
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, min_delta=MIN_DELTA), lgb.log_evaluation(100)])
    train_seconds = time.time() - t1
    del X_train, y_train
    logger.info('trained in %.0fs, best iteration %d, best valid Amex %.5f',
                train_seconds, model.best_iteration_, model.best_score_['valid_0']['amex'])

    p_valid = model.predict_proba(X_valid)[:, 1]  # uses the best iteration
    scores = validation_scores(y_valid, p_valid, cid_valid)
    print(scores.round(5).to_string())

    save_model(model, output)
    check_saved_model(output, X_valid, p_valid)
    logger.info('saved to %s (reload checked)', output)

    if use_mlflow:
        mlflow.set_tracking_uri(f'sqlite:///{(PROJECT_DIR / "mlflow.db").as_posix()}')
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name='lightgbm'):
            mlflow.log_params({**LGB_PARAMS, 'random_state': seed, 'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                               'min_delta': MIN_DELTA, 'early_stopping_metric': 'valid_amex_customer_last',
                               'n_customers': n_customers, 'valid_size': valid_size,
                               'n_train_rows': n_train_rows,
                               'n_valid_rows': len(X_valid), 'n_features': len(features)})
            mlflow.log_metrics({'best_iteration': model.best_iteration_, 'train_seconds': train_seconds,
                                **{f'valid_{level}_{metric}': v
                                   for level, row in scores.iterrows() for metric, v in row.items()}})
            mlflow.log_artifact(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-customers', type=int, default=150_000,
                        help='stratified subsample of train customers; 0 = all (default: 150000)')
    parser.add_argument('--valid-size', type=float, default=0.2, help='share of customers in valid (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL_PATH, help=f'default: {DEFAULT_MODEL_PATH}')
    parser.add_argument('--no-mlflow', action='store_true', help='do not log the run to MLflow')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        train(args.n_customers or None, args.valid_size, args.seed, args.output, not args.no_mlflow)
    except Exception:
        logger.exception('training failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
