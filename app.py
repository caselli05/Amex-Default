"""Flask app that serves the LightGBM default model.

Run with `python app.py` (or `flask --app app run`), then open http://127.0.0.1:5000.

Endpoints:
- GET  /          page to upload a CSV of rows and get their predictions (and Amex score)
- GET  /health    whether the model is loaded
- GET  /features  the columns the model expects, in order, and the categories of each categorical feature
- POST /predict   JSON {"data": [[...], ...], "columns": [...] (optional)} -> {"predictions": [...]}
- POST /predict_csv  multipart form with a CSV `file` (header row with the feature names; other columns are
                     ignored) -> {"predictions": [...], "ids": {"customer_ID": [...], "S_2": [...]},
                     "target": [...] or null, "amex_score": float or null, "amex_score_info": str}
                     amex_score is computed when the CSV has a `target` column, null otherwise
"""
import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.dataclass import InputData
from src.model import Model
from src.utils.amex_metric import amex_metric_np
from src.utils.errors import DataValidationError, InferenceError, ModelLoadError, PredictionError

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # larger uploads get a 413
app.json.sort_keys = False  # keep the ID columns in the order of ID_COLUMNS

ID_COLUMNS = ['customer_ID', 'S_2']  # returned next to the predictions when the CSV has them
TARGET_COLUMN = 'target'  # when the CSV has it, the Amex score of the predictions is computed

# Load once at startup. If it fails the app still starts: /health reports it and /predict answers 503.
model = Model()
try:
    model.load()
except ModelLoadError:
    logger.exception('model not loaded')

STATUS = {DataValidationError: 400, ModelLoadError: 503, PredictionError: 500}


@app.errorhandler(InferenceError)
def handle_inference_error(e: InferenceError):
    status = STATUS.get(type(e), 500)
    if status >= 500:
        logger.error('%s: %s', type(e).__name__, e)
    return jsonify(error=type(e).__name__, message=str(e)), status


@app.errorhandler(Exception)
def handle_unexpected_error(e: Exception):
    if hasattr(e, 'code'):  # HTTP errors raised by Flask itself (404, 405, ...): keep their status
        return jsonify(error=type(e).__name__, message=str(e)), e.code
    logger.exception('unexpected error')
    return jsonify(error='InternalServerError', message='unexpected error, see the server logs'), 500


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/health')
def health():
    return jsonify(status='ok' if model.is_loaded else 'model not loaded', model_loaded=model.is_loaded,
                   model_path=str(model.path))


@app.get('/features')
def features():
    return jsonify(features=model.feature_names, categories=model.categories)


@app.post('/predict')
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or 'data' not in body:
        raise DataValidationError('send a JSON object with a "data" key: a list of rows (lists of values)')
    data = InputData(body['data'], body.get('columns'))
    predictions = model.predict(data)
    return jsonify(predictions=predictions.tolist(), n_rows=len(data))


def amex_score(target: pd.Series | None, predictions: np.ndarray) -> tuple[float | None, str]:
    """Amex metric of `predictions` against `target`, over the rows as sent (one prediction per row), and a note
    saying how it was computed or why it is None (no target column, invalid targets, a single class)."""
    if target is None:
        return None, f'no {TARGET_COLUMN!r} column in the CSV'
    y = pd.to_numeric(target, errors='coerce')
    if y.isna().any():
        return None, f'{TARGET_COLUMN!r} has {int(y.isna().sum())} empty or non-numeric values'
    if not y.isin([0, 1]).all():
        return None, f'{TARGET_COLUMN!r} must only contain 0 and 1'
    if y.nunique() < 2:
        return None, f'{TARGET_COLUMN!r} has a single class ({int(y.iloc[0])}); the Amex score needs both 0 and 1'
    try:
        score = float(amex_metric_np(y.to_numpy(int), predictions))
    except (ValueError, FloatingPointError, ZeroDivisionError) as e:
        logger.warning('amex score failed: %s', e)
        return None, f'could not compute the Amex score: {e}'
    return score, (f'computed over {len(y)} rows ({int(y.sum())} with target 1), one prediction per row as sent')


@app.post('/predict_csv')
def predict_csv():
    file = request.files.get('file')
    if file is None or file.filename == '':
        raise DataValidationError('no file uploaded: send a CSV in the "file" field of a multipart form')

    # categoricals and IDs as strings, so e.g. D_63 = "CO" or an ID with leading zeros is kept as written
    str_columns = {c: str for c in [*ID_COLUMNS, *(model.categories if model.is_loaded else {})]}
    try:
        df = pd.read_csv(file.stream, dtype=str_columns, low_memory=False)
    except pd.errors.EmptyDataError as e:
        raise DataValidationError('the CSV is empty') from e
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        raise DataValidationError(f'could not read {file.filename!r} as a CSV: {e}') from e

    data = InputData(df.to_numpy(dtype=object), list(df.columns))
    predictions = model.predict(data)
    ids = {c: df[c].where(df[c].notna(), None).tolist() for c in ID_COLUMNS if c in df.columns}
    target = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    score, score_info = amex_score(target, predictions)
    return jsonify(predictions=predictions.tolist(), ids=ids,
                   target=None if target is None else target.astype(object).where(target.notna(), None).tolist(),
                   amex_score=score, amex_score_info=score_info, n_rows=len(data), filename=file.filename)


if __name__ == '__main__':
    app.run(debug=True)
