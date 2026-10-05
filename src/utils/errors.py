"""Custom exceptions of the inference pipeline, one per kind of failure, so the Flask app can map each to an HTTP
status."""


class InferenceError(Exception):
    """Base class of every inference failure."""


class DataValidationError(InferenceError):
    """The input data is malformed: wrong shape, wrong number of columns (a client error, HTTP 400)."""


class ModelLoadError(InferenceError):
    """The model is not loaded, its pkl is missing, or it cannot be unpickled (service unavailable, HTTP 503)."""


class PredictionError(InferenceError):
    """The model raised or returned something unexpected while predicting (server error, HTTP 500)."""
