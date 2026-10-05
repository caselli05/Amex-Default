"""Feature preprocessing for the neural network models: denoising and log transform of skewed features."""
import numpy as np
import pandas as pd


def denoise(df: pd.DataFrame, cols: list[str], decimals: int = 2, inplace: bool = False) -> pd.DataFrame:
    """Floor `cols` to `decimals` decimal places: `np.floor(x * 10**decimals) / 10**decimals`.

    Removes noise in [0, 10**-decimals) added on top of discrete levels. Floors towards -inf, so a negative value
    moves down (-0.004 -> -0.01). NULLs stay NULL; the dtype of each column is kept.
    """
    out = df if inplace else df.copy()
    scale = 10 ** decimals
    for c in cols:
        x = out[c].to_numpy()
        out[c] = (np.floor(x * scale) / scale).astype(x.dtype, copy=False)
    return out


def skewed_columns(df: pd.DataFrame, cols: list[str], threshold: float = 2.0) -> list[str]:
    """Columns of `cols` whose skewness (non-null values) is above `threshold` in absolute value.

    Call it on the training rows only and reuse the list for validation and test.
    """
    skew = df[cols].skew()
    return skew.index[skew.abs() > threshold].tolist()


def signed_log1p(df: pd.DataFrame, cols: list[str], inplace: bool = False) -> pd.DataFrame:
    """`sign(x) * log1p(|x|)` on `cols`: compresses long tails on both sides and is defined for negative values.

    Monotonic and 0 -> 0. NULLs stay NULL; the dtype of each column is kept.
    """
    out = df if inplace else df.copy()
    for c in cols:
        x = out[c].to_numpy()
        out[c] = (np.sign(x) * np.log1p(np.abs(x))).astype(x.dtype, copy=False)
    return out
