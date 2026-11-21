from __future__ import absolute_import, division, print_function

from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


class PerformancePreprocessor(object):
    def __init__(self, imputer, scaler, input_scaler, target_scaling,
                 y_center, y_scale):
        self.imputer = imputer
        self.scaler = scaler
        self.input_scaler = str(input_scaler)
        self.target_scaling = str(target_scaling)
        self.y_center = float(y_center)
        self.y_scale = float(y_scale)

    @classmethod
    def fit(cls, x_train, y_train, input_scaler="standard",
            target_scaling="standard"):
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        imputer = SimpleImputer(strategy="median")
        x_imputed = imputer.fit_transform(x_train)
        if input_scaler == "none":
            scaler = None
        elif input_scaler == "standard":
            scaler = StandardScaler().fit(x_imputed)
        elif input_scaler == "robust":
            scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(x_imputed)
        else:
            raise ValueError("Unknown input_scaler=%r" % input_scaler)

        if target_scaling == "none":
            y_center = 0.0
            y_scale = 1.0
        elif target_scaling == "standard":
            y_center = float(np.mean(y_train))
            y_scale = float(np.std(y_train))
            if not np.isfinite(y_scale) or y_scale < 1.0e-8:
                y_scale = 1.0
        else:
            raise ValueError("Unknown target_scaling=%r" % target_scaling)
        return cls(imputer, scaler, input_scaler, target_scaling,
                   y_center, y_scale)

    def transform_x(self, values):
        x = self.imputer.transform(np.asarray(values, dtype=np.float32))
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return np.asarray(x, dtype=np.float32)

    def transform_y(self, values):
        y = np.asarray(values, dtype=np.float32).reshape(-1)
        return ((y - self.y_center) / self.y_scale).astype(np.float32)

    def inverse_y(self, values):
        y = np.asarray(values, dtype=np.float32).reshape(-1)
        return (y * self.y_scale + self.y_center).astype(np.float32)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path):
        return joblib.load(str(path))
