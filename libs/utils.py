import altair as alt
import pandas as pd  # type: ignore
from altair_upset import UpSetAltair  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


def train_calib_valid_split(
    df: pd.DataFrame, calib_size: float = 0.3, valid_size=0.3, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ts1 = calib_size + valid_size
    _ts2 = valid_size / _ts1

    train, _ = train_test_split(df, test_size=_ts1, random_state=random_state)
    calib, valid = train_test_split(_, test_size=_ts2, random_state=random_state)

    return train.sort_index(), calib.sort_index(), valid.sort_index()


def create_upset_plot(df: pd.DataFrame, title: str = "UpSet Plot") -> alt.Chart:
    return UpSetAltair(data=df.astype(int), sets=list(df.columns), title=title).chart
