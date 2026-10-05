import numpy as np
import pandas as pd


def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    df = (pd.concat([y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
    four_pct_cutoff = int(0.04 * df['weight'].sum())
    df['weight_cumsum'] = df['weight'].cumsum()
    df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
    return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()


def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    df = (pd.concat([y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
    df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
    total_pos = (df['target'] * df['weight']).sum()
    df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
    df['lorentz'] = df['cum_pos_found'] / total_pos
    df['gini'] = (df['lorentz'] - df['random']) * df['weight']
    return df['gini'].sum()


def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    y_true_pred = y_true.rename(columns={'target': 'prediction'})
    return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def amex_metric_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same value as amex_metric, on numpy arrays: one sort, no pandas. Fast enough for per-round evaluation."""
    y = np.asarray(y_true)[np.argsort(-np.asarray(y_pred), kind='stable')]
    w = np.where(y == 0, 20, 1)

    top4 = y[np.cumsum(w) <= int(0.04 * w.sum())].sum() / y.sum()

    def wgini(y_sorted):
        w_ = np.where(y_sorted == 0, 20, 1)
        random = np.cumsum(w_ / w_.sum())
        lorentz = np.cumsum(y_sorted * w_) / (y_sorted * w_).sum()
        return ((lorentz - random) * w_).sum()

    gini = wgini(y) / wgini(np.sort(y)[::-1])
    return 0.5 * (gini + top4)
