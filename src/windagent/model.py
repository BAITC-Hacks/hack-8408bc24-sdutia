"""Forecast model: physics-informed gradient boosting with quantiles.

1. Per weather model, an isotonic "NWP power curve" maps forecast 100 m wind → power (fitted on
   training rows only) and is added as a feature.
2. scikit-learn HistGradientBoosting: squared error for the point forecast, quantile loss for
   P10/P50/P90. Turbine is a native categorical feature; missing models (e.g. AIFS before
   Feb 2025) are handled natively as NaN.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

QUANTILES = (0.1, 0.5, 0.9)
# lead_h is excluded on purpose: training issues are all at 00:00 (data clock), so lead time would be
# confounded with hour of day and would not transfer to v2 updates / live runs at other hours.
# Forecast age is still represented per weather model (<model>_lead_from_init_h). Validation: MAE 0.168 vs 0.169.
_EXCLUDE = {"issue_time_utc", "target_time_utc", "entity", "actual", "available", "ws_obs", "lead_h"}
_HGB = dict(max_iter=350, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=60,
            l2_regularization=1.0, early_stopping=False, random_state=0)


def _ws_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_ws100") and not c.startswith("ens_")]


@dataclass
class ForecastModel:
    turbine_ids: list[str]
    feature_names: list[str] = field(default_factory=list)
    power_curves: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    cutoff_utc: str | None = None
    n_train: int = 0
    interval_scale: float = 1.0          # P10/P90 distance from P50 × scale (calibrated on validation folds)
    calibration: dict = field(default_factory=dict)

    # ---- feature augmentation -------------------------------------------------------------
    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        pcs = []
        for col, iso in self.power_curves.items():
            name = col.replace("_ws100", "_pc")
            if col not in X.columns:                 # model excluded by the agent / missing → NaN (handled natively)
                X[col] = np.nan
            x = X[col].to_numpy(dtype=float)
            out = np.full(len(X), np.nan)
            ok = ~np.isnan(x)
            if ok.any():
                out[ok] = iso.predict(x[ok])
            X[name] = out
            pcs.append(name)
        if pcs:
            X["ens_pc_mean"] = X[pcs].mean(axis=1)
        X["turbine_idx"] = X["entity"].map({t: i for i, t in enumerate(self.turbine_ids)}).astype(float)
        return X

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        for c in self.feature_names:
            if c not in X.columns:
                X[c] = np.nan
        return X[self.feature_names].to_numpy(dtype=float)

    # ---- fit / predict ----------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "ForecastModel":
        """train: feature rows with columns entity, actual (only available hours)."""
        d = train.dropna(subset=["actual"])
        self.power_curves = {}
        for col in _ws_columns(d):
            m = d[[col, "actual"]].dropna()
            if len(m) < 500:
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(m[col].to_numpy(), m["actual"].to_numpy())
            self.power_curves[col] = iso
        X = self._augment(d)
        self.feature_names = sorted(c for c in X.columns if c not in _EXCLUDE and c != "turbine_idx") + ["turbine_idx"]
        M = self._matrix(X)
        cat = [len(self.feature_names) - 1]
        y = d["actual"].to_numpy(dtype=float)
        self.models = {"mean": HistGradientBoostingRegressor(loss="squared_error", categorical_features=cat, **_HGB).fit(M, y)}
        for q in QUANTILES:
            self.models[f"p{int(q * 100)}"] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, categorical_features=cat, **_HGB).fit(M, y)
        self.n_train = int(len(d))
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        X = self._augment(rows)
        M = self._matrix(X)
        out = pd.DataFrame(index=rows.index)
        for name, mdl in self.models.items():
            out[name] = np.clip(mdl.predict(M), 0.0, 1.0)
        q = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)   # no crossing quantiles
        k = float(getattr(self, "interval_scale", 1.0) or 1.0)
        out["p10"] = np.clip(q[:, 1] - k * (q[:, 1] - q[:, 0]), 0.0, 1.0)
        out["p50"] = q[:, 1]
        out["p90"] = np.clip(q[:, 1] + k * (q[:, 2] - q[:, 1]), 0.0, 1.0)
        out["mean"] = np.clip(out["mean"], out["p10"], out["p90"])   # point forecast stays inside its interval
        return out

    def predict_powercurve(self, rows: pd.DataFrame, prefer=("aifs", "ifs", "icon", "gfs")) -> pd.Series:
        """Baseline: raw NWP power curve of the best available single model."""
        res = pd.Series(np.nan, index=rows.index)
        for short in reversed(prefer):
            col = f"{short}_ws100"
            if col in self.power_curves and col in rows:
                x = rows[col].to_numpy(dtype=float)
                ok = ~np.isnan(x)
                vals = np.full(len(rows), np.nan)
                if ok.any():
                    vals[ok] = self.power_curves[col].predict(x[ok])
                res = pd.Series(np.where(ok, vals, res.to_numpy()), index=rows.index)
        return res

    # ---- persistence -------------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Path) -> "ForecastModel":
        with open(path, "rb") as f:
            return pickle.load(f)
