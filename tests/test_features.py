import pytest
import numpy as np
import pandas as pd
from features import camber_geometry, add_geometry_features

@pytest.mark.parametrize("M, P, XX, L, Q, expected", [
    (2,    4,     12,     None,  None,  {"max_camber": 0.02,    "camber_pos": 0.4,   "thickness": 0.12,    "reflex": 0}),
    (0,    0,     12,     None,  None,  {"max_camber": 0.0,     "camber_pos": 0.0,   "thickness": 0.12,    "reflex": 0}),
    (None, 5.424, 16.283, 3.941, True,  {"max_camber": 0.04581, "camber_pos": 0.271, "thickness": 0.16283, "reflex": 1}),
    (None, 6.252, 11.422, 0.329, False, {"max_camber": 0.00398, "camber_pos": 0.313, "thickness": 0.11422, "reflex": 0}),
])
def test_camber_geometry(M,P,XX,L,Q,expected):
    result = camber_geometry(M=M,P=P,XX=XX,L=L,Q=Q)
    assert result == pytest.approx(expected,abs=1e-4)

def test_full_dataFrame():
    test_df = pd.DataFrame({
        "name": ["NACA 2412", "NACA 0012", "5-digit reflex", "5-digit normal"],
        "M":    [2.0,    0.0,    np.nan, np.nan],
        "L":    [np.nan, np.nan, 3.941,  0.329],
        "P":    [4.0,    0.0,    5.424,  6.252],
        "Q":    [np.nan, np.nan, True,   False],
        "XX":   [12.0,   12.0,   16.283, 11.422],
    })

    expected_df = pd.DataFrame({
        "name":         ["NACA 2412", "NACA 0012", "5-digit reflex", "5-digit normal"],
        "M":            [2.0,    0.0,    np.nan, np.nan],
        "L":            [np.nan, np.nan, 3.941,  0.329],
        "P":            [4.0,    0.0,    5.424,  6.252],
        "Q":            [np.nan, np.nan, True,   False],
        "XX":           [12.0,   12.0,   16.283, 11.422],
        "max_camber":   [0.02, 0.0, 0.045809, 0.003979],
        "camber_pos":   [0.4, 0.0, 0.271, 0.313],
        "thickness":    [0.12, 0.12, 0.16283, 0.11422],
        "reflex":       [0,0,1,0]
    })

    cols = ["max_camber", "camber_pos", "thickness", "reflex"]
    result = add_geometry_features(test_df)
    pd.testing.assert_frame_equal(
        result[cols], expected_df[cols],
        check_dtype=False, atol=1e-4,
    )

def test_subset_keeps_index():
    test_df = pd.DataFrame({
        "name": ["NACA 2412", "NACA 0012", "5-digit reflex", "5-digit normal"],
        "M":    [2.0,    0.0,    np.nan, np.nan],
        "L":    [np.nan, np.nan, 3.941,  0.329],
        "P":    [4.0,    0.0,    5.424,  6.252],
        "Q":    [np.nan, np.nan, True,   False],
        "XX":   [12.0,   12.0,   16.283, 11.422],
    })

    sub = test_df.iloc[[1,3]]
    result = add_geometry_features(sub)

    assert list(result.index) == [1,3]
    assert result.loc[1,"max_camber"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc[3,"max_camber"] == pytest.approx(0.00398, abs=1e-4)


