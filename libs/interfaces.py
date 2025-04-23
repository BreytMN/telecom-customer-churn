from typing import Protocol, Self

import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore


class Model(Protocol):
    _label_encoder: LabelEncoder
    _is_fitted: bool

    def train(
        self, train: pd.DataFrame, calib: pd.DataFrame, valid: pd.DataFrame, label: str
    ) -> Self: ...
    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def predict_point(self, df: pd.DataFrame) -> pd.Series: ...

    @property
    def features(self) -> list[str]: ...

    @property
    def label(self) -> str: ...

    @property
    def classes(self) -> list[str]: ...
