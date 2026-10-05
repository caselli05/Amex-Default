"""LightGBM model for inference: loading its pkl from src/models and predicting the probability of default."""
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.dataclass import InputData
from src.utils.errors import DataValidationError, ModelLoadError, PredictionError

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / 'models'
DEFAULT_MODEL_PATH = MODELS_DIR / 'lightgbm.pkl'


def _category_key(value) -> str:
    """Key under which a categorical value is compared: numbers by value, so 1, '1', '1.0' and 1.0 all match the
    training category '1.0' (and -1.0 matches D_64's '-1'); anything else as its string."""
    try:
        return str(float(value))
    except (ValueError, TypeError):
        return str(value)


def _match_categories(values: pd.Series, categories: list) -> pd.Series:
    """`values` as a category Series with the training `categories`; NULL and unseen values become NaN."""
    lookup = {_category_key(c): c for c in categories}
    mapping = {v: lookup.get(_category_key(v)) for v in values.dropna().unique()}
    return values.map(mapping).astype(pd.CategoricalDtype(categories))


class Model:
    """A pickled `LGBMClassifier` that gives the probability of default of each statement row.

    The classifier must have been fitted on a DataFrame, as in 06_treebaselines, so it stores its feature names and
    the categories of its categorical features. Only load pkls you trust: unpickling can run arbitrary code.

    Usage: `model = Model().load()`, then `model.predict(InputData(array, columns))`.
    """

    def __init__(self, path: str | Path = DEFAULT_MODEL_PATH):
        self.path = Path(path)
        self._model: lgb.LGBMClassifier | None = None

    def load(self) -> 'Model':
        """Unpickle the model from `path`. Raises ModelLoadError if the pkl is missing, unreadable or not LightGBM."""
        try:
            with open(self.path, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError as e:
            raise ModelLoadError(f'model file not found: {self.path}') from e
        except Exception as e:  # corrupt or truncated file, incompatible library version, ...
            raise ModelLoadError(f'could not unpickle {self.path}: {type(e).__name__}: {e}') from e

        if not isinstance(model, lgb.LGBMClassifier):
            raise ModelLoadError(f'{self.path} holds a {type(model).__name__}, expected an LGBMClassifier')
        try:
            model.booster_  # raises if the classifier was pickled before being fitted
        except Exception as e:
            raise ModelLoadError(f'the LGBMClassifier in {self.path} is not fitted') from e

        self._model = model
        logger.info('loaded LGBMClassifier with %d features from %s', len(self.feature_names), self.path)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def feature_names(self) -> list[str]:
        """Columns the model was trained on, in order."""
        if self._model is None:
            raise ModelLoadError('no model loaded: call load() first')
        return list(self._model.feature_name_)

    @property
    def categories(self) -> dict[str, list]:
        """Categorical feature -> the categories it had in training (LightGBM stores them in the order of the
        categorical columns of the training DataFrame)."""
        if self._model is None:
            raise ModelLoadError('no model loaded: call load() first')
        booster = self._model.booster_
        cat_idx = booster.params.get('categorical_column') or []
        if not cat_idx:  # sklearn API with category dtypes: indices are in the model's feature infos
            infos = booster.dump_model(num_iteration=0)['feature_infos']
            cat_idx = [i for i, name in enumerate(self.feature_names)
                       if infos.get(name, {}).get('values') is not None]
        cat_cols = [self.feature_names[i] for i in cat_idx]
        return dict(zip(cat_cols, booster.pandas_categorical or []))

    def _prepare(self, data: InputData) -> pd.DataFrame:
        """DataFrame with the model's columns in the model's order and the training dtypes: categorical features as
        category with the training categories (unseen values become NaN), the rest as float32."""
        features = self.feature_names
        df = data.to_dataframe
        if data.columns is None:
            if data.n_columns != len(features):
                raise DataValidationError(f'array has {data.n_columns} columns, the model expects {len(features)}')
            df.columns = features
        else:
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise DataValidationError(f'missing columns: {missing}')
            df = df[features]  # model order; extra columns are dropped

        categories = self.categories
        try:
            for c in features:
                if c in categories:
                    df[c] = _match_categories(df[c], categories[c])
                else:
                    df[c] = pd.to_numeric(df[c]).astype(np.float32)
        except (ValueError, TypeError) as e:
            raise DataValidationError(f'column {c!r} has values of the wrong type: {e}') from e
        return df

    def predict(self, data: InputData) -> np.ndarray:
        """Probability of default of each row of `data`, shape (rows,), in the order of the rows."""
        if self._model is None:
            raise ModelLoadError('no model loaded: call load() first')
        df = self._prepare(data)
        try:
            p = self._model.predict_proba(df)[:, 1]
        except Exception as e:
            raise PredictionError(f'LightGBM failed to predict: {type(e).__name__}: {e}') from e
        if len(p) != len(data):
            raise PredictionError(f'model returned {len(p)} predictions for {len(data)} rows')
        return p
