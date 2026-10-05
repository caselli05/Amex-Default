"""Input data for inference: a 2D array of statement rows, optionally with the name of each column."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.errors import DataValidationError


@dataclass
class InputData:
    """Statement rows to score: `array` of shape (rows, features), column j named `columns[j]`.

    A 1D array is taken as a single row. `columns` can be left as None when the array is already in the order of the
    model's features (the model then names them). Raises DataValidationError if `array` cannot be read as a
    non-empty 2D array, if `columns` has duplicates, or if the number of columns does not match.
    """
    array: np.ndarray
    columns: list[str] | None = None

    def __post_init__(self):
        try:
            self.array = np.asarray(self.array)
        except (ValueError, TypeError) as e:  # e.g. nested lists of different lengths
            raise DataValidationError(f'array cannot be converted to an ndarray: {e}') from e
        if self.array.ndim == 1:
            self.array = self.array.reshape(1, -1)
        if self.array.ndim != 2:
            raise DataValidationError(f'array must be 1D or 2D, got shape {self.array.shape}')
        if self.array.shape[0] == 0:
            raise DataValidationError('array has no rows')

        if self.columns is not None:
            self.columns = list(self.columns)
            if len(set(self.columns)) != len(self.columns):
                raise DataValidationError('columns has duplicate names')
            if self.array.shape[1] != len(self.columns):
                raise DataValidationError(f'array has {self.array.shape[1]} columns but {len(self.columns)} '
                                          f'names were given')

    def __len__(self) -> int:
        return self.array.shape[0]

    @property
    def n_columns(self) -> int:
        return self.array.shape[1]

    @property
    def to_dataframe(self) -> pd.DataFrame:
        """`array` as a DataFrame, with `columns` as column names (0, 1, ... if None). A new frame on each access."""
        try:
            return pd.DataFrame(self.array, columns=self.columns)
        except (ValueError, TypeError) as e:
            raise DataValidationError(f'could not build a DataFrame from array: {e}') from e
