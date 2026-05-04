"""
ml_rhythmicity_classifier.py
=============================
Inference wrapper for the trained rhythmicity Random Forest.

Loads model artefacts produced by train_classifier.py and exposes a
single .predict() method compatible with the existing app.py call sites:

    classifier = MLRhythmicityClassifier(
        model_path="rhythmicity_classifier.pkl",
        feature_names_path="feature_names.pkl",
    )
    result = classifier.predict(signal_series, time_series)

result is a dict:
    {
        "is_rhythmic":            bool,
        "probability_rhythmic":   float,   # 0–1
        "probability_arrhythmic": float,   # 0–1
        "confidence":             str,     # "high" | "medium" | "low"
        "features":               dict,    # extracted feature values
    }

The classifier delegates feature extraction entirely to
RhythmicityFeatureExtractor — no wavelet, no PyBoat, no rpy2.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ML_classifier.rhythmicity_feature_extractor import RhythmicityFeatureExtractor, FEATURE_NAMES


class MLRhythmicityClassifier:
    """
    Rhythmicity classifier backed by a trained Random Forest.

    Parameters
    ----------
    model_path         : Path to rhythmicity_classifier.pkl
    feature_names_path : Path to feature_names.pkl
    metadata_path      : Path to model_metadata.pkl  (optional but recommended —
                         used for data-driven confidence thresholds and NaN imputation)
    period_min/max     : Circadian window in hours (default 18–30).
    """

    def __init__(
        self,
        model_path: str          = "rhythmicity_classifier.pkl",
        feature_names_path: str  = "feature_names.pkl",
        metadata_path: str       = "model_metadata.pkl",
        period_min: float        = 18.0,
        period_max: float        = 30.0,
    ):
        self.period_min = period_min
        self.period_max = period_max

        # Load model
        with open(model_path, "rb") as f:
            self._clf = pickle.load(f)

        # Load feature names (used for alignment / validation)
        with open(feature_names_path, "rb") as f:
            self._feature_names = pickle.load(f)

        # Load metadata (optional — fall back gracefully)
        self._metadata    = None
        self._col_medians = None
        self._p_high      = 0.70   # default confidence thresholds
        self._p_low       = 0.30

        meta_path = Path(metadata_path)
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            self._col_medians = self._metadata.get("col_medians", None)
            self._p_high      = self._metadata.get("p_high_threshold", 0.70)
            self._p_low       = self._metadata.get("p_low_threshold",  0.30)

        # Feature extractor
        self._extractor = RhythmicityFeatureExtractor(
            period_min=period_min,
            period_max=period_max,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        signal: "array-like",
        time:   "array-like",
    ) -> dict:
        """
        Classify a single time series.

        Parameters
        ----------
        signal : array-like  Signal values (pandas Series or numpy array).
        time   : array-like  Time axis in hours.

        Returns
        -------
        dict with keys:
            is_rhythmic            bool
            probability_rhythmic   float
            probability_arrhythmic float
            confidence             "high" | "medium" | "low"
            features               dict of raw feature values
        """
        x = np.asarray(signal, dtype=float)
        t = np.asarray(time,   dtype=float)

        # Extract features
        try:
            feats = self._extractor.extract(x, t)
        except Exception as e:
            warnings.warn(f"Feature extraction failed: {e}. Returning arrhythmic.")
            return self._fallback_result()

        # Build feature vector aligned to training order
        feat_vec = np.array(
            [feats.get(name, np.nan) for name in self._feature_names],
            dtype=float,
        )

        # NaN imputation: use column medians from training data if available,
        # otherwise fall back to 0 (neutral / uninformative)
        nan_mask = np.isnan(feat_vec) | np.isinf(feat_vec)
        if nan_mask.any():
            if self._col_medians is not None:
                feat_vec[nan_mask] = self._col_medians[nan_mask]
            else:
                feat_vec[nan_mask] = 0.0

        # Predict
        proba        = self._clf.predict_proba(feat_vec.reshape(1, -1))[0]
        # proba[1] = rhythmic, proba[0] = arrhythmic  (sklearn label order)
        p_rhythmic   = float(proba[1])
        p_arrhythmic = float(proba[0])
        is_rhythmic  = p_rhythmic >= 0.5

        confidence = self._confidence(p_rhythmic)

        return {
            "is_rhythmic":            is_rhythmic,
            "probability_rhythmic":   p_rhythmic,
            "probability_arrhythmic": p_arrhythmic,
            "confidence":             confidence,
            "features":               feats,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        t_col: str,
        data_cols: list,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Classify multiple signals from a DataFrame.

        Returns
        -------
        pd.DataFrame  One row per signal with classification columns.
        """
        records = []
        for col in data_cols:
            result = self.predict(df[col].values, df[t_col].values)
            records.append({
                "sample":                col,
                "is_rhythmic":           result["is_rhythmic"],
                "probability_rhythmic":  result["probability_rhythmic"],
                "probability_arrhythmic": result["probability_arrhythmic"],
                "confidence":            result["confidence"],
            })
        return pd.DataFrame(records).set_index("sample")

    @property
    def model_info(self) -> dict:
        """Return summary of training metadata (if available)."""
        if self._metadata is None:
            return {"status": "no metadata loaded"}
        return {
            "cv_roc_auc":   f"{self._metadata['cv_roc_auc_mean']:.4f} ± {self._metadata['cv_roc_auc_std']:.4f}",
            "n_train":      self._metadata.get("n_train_rhythmic", "?") + self._metadata.get("n_train_arrhythmic", 0),
            "n_estimators": self._metadata.get("n_estimators", "?"),
            "features":     self._feature_names,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _confidence(self, p_rhythmic: float) -> str:
        """
        Map probability to a three-tier confidence label.
        Thresholds are derived from the training distribution in train_classifier.py.
        """
        if p_rhythmic >= self._p_high or p_rhythmic <= self._p_low:
            return "high"
        # Medium band: probability clearly above or below 0.5 but not extreme
        p_medium = self._metadata.get("p_medium_threshold", 0.60) if self._metadata else 0.60
        if p_rhythmic >= p_medium or p_rhythmic <= (1.0 - p_medium):
            return "medium"
        return "low"

    def _fallback_result(self) -> dict:
        return {
            "is_rhythmic":            False,
            "probability_rhythmic":   0.0,
            "probability_arrhythmic": 1.0,
            "confidence":             "low",
            "features":               {k: np.nan for k in FEATURE_NAMES},
        }
