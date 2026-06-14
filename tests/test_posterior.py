import pandas as pd
import pytest

from src.posterior import extract_p_draws


def test_extract_p_draws_orders_columns_by_lag_row_col() -> None:
    draws = pd.DataFrame(
        {
            "lp__": [0.0],
            "P[2,1,1]": [211.0],
            "P[1,2,1]": [121.0],
            "P[1,1,2]": [112.0],
            "P[1,1,1]": [111.0],
        }
    )

    assert extract_p_draws(draws).tolist() == [[111.0, 112.0, 121.0, 211.0]]


def test_extract_p_draws_requires_p_columns() -> None:
    with pytest.raises(ValueError, match=r"No P\[\.\.\.\] draw columns"):
        extract_p_draws(pd.DataFrame({"lp__": [0.0]}))
