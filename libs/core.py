from collections.abc import Callable, Iterable
from typing import Any, Self, cast

import lightgbm as lgb
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore

from libs.interfaces import Model  # type: ignore


class LGBMultiClassifier(Model):
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_rounds: int = 10000,
        early_stopping_rounds: int = 10,
        log_eval_period: int = 1,
        random_state: int = 0,
    ) -> None:
        self._params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "random_state": random_state,
            "verbosity": -1,
            **(params or {}),
        }
        self._num_boost_rounds = num_boost_rounds
        self._callbacks: list[Callable[..., Any]] = [
            lgb.log_evaluation(period=log_eval_period),
            lgb.early_stopping(early_stopping_rounds, verbose=bool(log_eval_period)),
        ]

        self._is_fitted: bool = False

    def train(
        self, train: pd.DataFrame, calib: pd.DataFrame, valid: pd.DataFrame, label: str
    ) -> Self:
        train_dataset, valid_datasets = self._preprocess(train, calib, valid, label)

        self._model = lgb.train(
            train_set=train_dataset,
            valid_sets=valid_datasets,
            params=self._params,
            num_boost_round=self._num_boost_rounds,
            callbacks=self._callbacks,
        )

        self._is_fitted = True

        return self

    def _preprocess(
        self, train: pd.DataFrame, calib: pd.DataFrame, valid: pd.DataFrame, label: str
    ) -> tuple[lgb.Dataset, list[lgb.Dataset]]:
        self._label = label
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(train[self.label])
        self._features = train.drop(self.label, axis=1).columns

        self._params.update({"num_classes": len(self._label_encoder.classes_)})

        train_dataset = lgb.Dataset(
            train.filter(self.features),
            label=self._label_encoder.transform(train[self.label]),
        )
        calib_dataset = lgb.Dataset(
            calib.filter(self.features),
            label=self._label_encoder.transform(calib[self.label]),
            reference=train_dataset,
        )
        valid_dataset = lgb.Dataset(
            valid.filter(self.features),
            label=self._label_encoder.transform(valid[self.label]),
            reference=train_dataset,
        )

        return train_dataset, [train_dataset, calib_dataset, valid_dataset]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            data=self._model.predict(df.filter(self.features)),
            index=df.index,
            columns=self.classes,
        )

    def predict_point(self, df: pd.DataFrame) -> pd.Series:
        predictions = self.predict(df).values.argmax(axis=1)

        return pd.Series(
            data=self._label_encoder.inverse_transform(predictions),
            index=df.index,
            name=self.label,
        )

    @property
    def features(self) -> list[str]:
        return list(self._features)

    @property
    def label(self) -> str:
        return self._label

    @property
    def classes(self) -> list[str]:
        return list(self._label_encoder.classes_)


class ConformalClassifier:
    def __init__(
        self,
        model: Model,
        alphas: Iterable[float] = np.arange(0.05, 1, 0.05),
    ) -> None:
        """
        Initializes the ConformalPredictor.

        Args:
            model: A trained Model.
            alpha: The significance levels.
        """
        self._model = model
        self._alphas = [round(a, 2) for a in alphas]

        if self._model._is_fitted:
            self._features = self._model.features
            self._label = self._model.label
            self._classes = self._model.classes

        self._thresholds: dict[float, float] = {}
        self._calibration_scores: np.ndarray | None = None
        self._is_fitted: bool = self._model._is_fitted
        self._is_calibrated: bool = False

    def train(
        self, train: pd.DataFrame, calib: pd.DataFrame, valid: pd.DataFrame, label: str
    ) -> Self:
        self._model.train(train, calib, valid, label)
        self._features = self._model.features
        self._label = self._model.label
        self._classes = self._model.classes

        self._is_fitted = self._model._is_fitted

        return self

    def calibrate(self, df: pd.DataFrame) -> None:
        X, y = df.filter(self._features), df[self._label]
        n = X.shape[0]
        cal_smx = self._model.predict(X).values
        labels = self._model._label_encoder.transform(y)
        cal_scores = 1 - cal_smx[np.arange(n), labels]

        self._calibration_scores = cast(np.ndarray, cal_scores)
        self._thresholds = {alpha: self._calibrate(alpha) for alpha in self._alphas}
        self._is_calibrated = True

    def _calibrate(self, alpha: float) -> float:
        if self._calibration_scores is None:
            raise ValueError("Calibration scores aren't computed")

        n = self._calibration_scores.shape[0]
        q = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(self._calibration_scores, q, method="higher")

    def predict(
        self, df: pd.DataFrame, alpha: float = 0.1, include_actuals: bool = True
    ) -> pd.DataFrame:
        if not self._is_calibrated:
            raise ValueError(
                "The predictor must be calibrated before calling predict(). "
                "Call the 'calibrate' method first."
            )

        if alpha not in self._thresholds:
            raise ValueError(
                f"Alpha value {alpha} not found in calibrated thresholds.  "
                f"Available alpha values are: {list(self._thresholds.keys())}"
            )

        preds = self._predict(df, alpha)

        if self._label in df.columns and include_actuals:
            preds = preds.assign(Actuals=df[self._label])

        return preds

    def _predict(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        return self._model.predict(df) >= (1 - self._thresholds[alpha])

    def predict_set(
        self, df: pd.DataFrame, alpha: float = 0.1, include_actuals: bool = True
    ) -> pd.DataFrame | pd.Series:
        preds = self.predict(df, alpha, include_actuals)

        preds_sets = preds.apply(lambda row: self._predict_set(row), axis=1).rename(
            "Prediction Sets"
        )

        if self._label in df.columns and include_actuals:
            preds = preds_sets.to_frame().assign(Actuals=df[self._label])

        return preds

    def _predict_set(self, row: pd.Series) -> list[str]:
        return [col for col in self._classes if row[col]]

    def predict_point(
        self, df: pd.DataFrame, include_actuals: bool = True
    ) -> pd.DataFrame | pd.Series:
        preds = self._model.predict_point(df)

        if self._label in df.columns and include_actuals:
            preds = preds.to_frame().assign(Actuals=df[self._label])

        return preds


class AdaptativeConformalClassifier:
    def __init__(
        self,
        model: Model,
        alphas: Iterable[float] = np.arange(0.05, 1, 0.05),
    ) -> None:
        """
        Initializes the ConformalPredictor.

        Args:
            model: A trained Model.
            alpha: The significance levels.
        """
        self._model = model
        self._alphas = [round(a, 2) for a in alphas]

        if self._model._is_fitted:
            self._features = self._model.features
            self._label = self._model.label
            self._classes = self._model.classes

        self._thresholds: dict[float, float] = {}
        self._calibration_scores: np.ndarray | None = None
        self._is_fitted: bool = self._model._is_fitted
        self._is_calibrated: bool = False

    def train(
        self, train: pd.DataFrame, calib: pd.DataFrame, valid: pd.DataFrame, label: str
    ) -> Self:
        self._model.train(train, calib, valid, label)
        self._features = self._model.features
        self._label = self._model.label
        self._classes = self._model.classes

        self._is_fitted = self._model._is_fitted

        return self

    def calibrate(self, df: pd.DataFrame) -> None:
        X, y = df.filter(self._features), df[self._label]
        n = X.shape[0]
        cal_smx = self._model.predict(X).values
        labels = self._model._label_encoder.transform(y)

        cal_pi = cal_smx.argsort(axis=1)[:, ::-1]
        cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)

        cal_pi_srt = cal_pi.argsort(axis=1)
        cal_scores = np.take_along_axis(cal_srt, cal_pi_srt, axis=1)[range(n), labels]

        self._calibration_scores = cast(np.ndarray, cal_scores)
        self._thresholds = {alpha: self._calibrate(alpha) for alpha in self._alphas}
        self._is_calibrated = True

    def _calibrate(self, alpha: float) -> float:
        if self._calibration_scores is None:
            raise ValueError("Calibration scores aren't computed")

        n = self._calibration_scores.shape[0]
        q = np.ceil((n + 1) * (1 - alpha)) / n
        return np.quantile(self._calibration_scores, q, method="higher")

    def predict(
        self, df: pd.DataFrame, alpha: float = 0.1, include_actuals: bool = True
    ) -> pd.DataFrame:
        if not self._is_calibrated:
            raise ValueError(
                "The predictor must be calibrated before calling predict(). "
                "Call the 'calibrate' method first."
            )

        if alpha not in self._thresholds:
            raise ValueError(
                f"Alpha value {alpha} not found in calibrated thresholds.  "
                f"Available alpha values are: {list(self._thresholds.keys())}"
            )

        preds = self._predict(df, alpha)

        if self._label in df.columns and include_actuals:
            preds = preds.assign(Actuals=df[self._label])

        return preds

    def _predict(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        qhat = self._thresholds[alpha]
        preds = self._model.predict(df)
        preds_smx = preds.values

        preds_pi = preds_smx.argsort(1)[:, ::-1]
        preds_srt = np.take_along_axis(preds_smx, preds_pi, axis=1).cumsum(axis=1)

        preds_pi_srt = preds_pi.argsort(axis=1)
        prediction_sets = np.take_along_axis(preds_srt <= qhat, preds_pi_srt, axis=1)

        return pd.DataFrame(prediction_sets, index=df.index, columns=preds.columns)

    def predict_set(
        self, df: pd.DataFrame, alpha: float = 0.1, include_actuals: bool = True
    ) -> pd.DataFrame | pd.Series:
        preds = self.predict(df, alpha, include_actuals)

        preds_sets = preds.apply(lambda row: self._predict_set(row), axis=1).rename(
            "Prediction Sets"
        )

        if self._label in df.columns and include_actuals:
            preds = preds_sets.to_frame().assign(Actuals=df[self._label])

        return preds

    def _predict_set(self, row: pd.Series) -> list[str]:
        return [col for col in self._classes if row[col]]

    def predict_point(
        self, df: pd.DataFrame, include_actuals: bool = True
    ) -> pd.DataFrame | pd.Series:
        preds = self._model.predict_point(df)

        if self._label in df.columns and include_actuals:
            preds = preds.to_frame().assign(Actuals=df[self._label])

        return preds
