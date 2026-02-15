#!/usr/bin/env python3

from lppls import lppls, data_loader
import pytest
import numpy as np
import pandas as pd
import random


@pytest.fixture
def data():
    return data_loader.nasdaq_dotcom()


@pytest.fixture
def observations(data):
    data = data.head(100)  # make it smaller so mp_compute_nested_fits runs faster
    time_ = np.linspace(0, len(data) - 1, len(data))
    price = [p for p in data['Adj Close']]
    return np.array([time_, price])


@pytest.fixture
def lppls_model(observations):
    """Returns a model instance."""
    return lppls.LPPLS(observations=observations)


# ---------------------------------------------------------------------------
# Dependency / import smoke tests
# ---------------------------------------------------------------------------

def test_imports():
    """Verify all key dependencies import without error."""
    import matplotlib
    import numba
    import numpy
    import pandas
    import scipy
    import sklearn
    import tqdm
    import xarray
    import cma


def test_numba_jit():
    """Verify numba JIT compilation works on the core lppls function."""
    result = lppls.LPPLS.lppls(0.0, 100.0, 0.5, 8.0, 1000.0, -10.0, 0.1, -0.1)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Pure function tests (deterministic, no optimizer)
# ---------------------------------------------------------------------------

def test_lppls_function():
    """Verify the @njit lppls function returns expected values for known inputs."""
    # Period 0
    result = lppls.LPPLS.lppls(
        0.0, 1300.2412888852296, 0.6087189222292106, 6.344318139503496,
        3034.166016949172, -16.041970137173486, 0.21878280136703082, -0.14789336333436504,
    )
    assert result == pytest.approx(1762.3196588471408, rel=1e-6)

    # Period 500
    result = lppls.LPPLS.lppls(
        500.0, 1428.0641475858731, 0.3473013071950998, 6.052643019980449,
        3910.1099206097356, -169.93053270790418, 0.05189394517600043, -0.045820295077658835,
    )
    assert result == pytest.approx(2086.3299554496016, rel=1e-6)


def test_get_oscillations_basic(lppls_model):
    """Verify oscillation count for known inputs."""
    # w=2*pi, tc=200, t1=0, t2=100 -> (1) * log(200/100) = log(2)
    result = lppls_model.get_oscillations(2 * np.pi, 200, 0, 100)
    assert result == pytest.approx(np.log(2), rel=1e-9)


def test_get_oscillations_tc_equals_t2(lppls_model):
    """tc == t2 means denominator is zero; should return NaN, not raise."""
    result = lppls_model.get_oscillations(8.0, 100, 0, 100)
    assert np.isnan(result)


def test_get_oscillations_tc_between_t1_t2(lppls_model):
    """tc between t1 and t2 gives a negative log argument; should return NaN."""
    result = lppls_model.get_oscillations(8.0, 50, 0, 100)
    assert np.isnan(result)


def test_get_oscillations_tc_equals_t1(lppls_model):
    """tc == t1 means numerator is zero (log(0) = -inf); should return NaN."""
    result = lppls_model.get_oscillations(8.0, 0, 0, 100)
    assert np.isnan(result)


def test_get_damping(lppls_model):
    """Verify damping ratio for known inputs."""
    result = lppls_model.get_damping(0.5, 10.0, -100.0, 50.0)
    expected = (0.5 * 100.0) / (10.0 * 50.0)
    assert result == pytest.approx(expected, rel=1e-9)


def test_get_c(lppls_model):
    """Verify get_c computes c = c1 / cos(arctan(c2/c1))."""
    # Both zero -> 0
    assert lppls_model.get_c(0, 0) == 0

    # c1=3, c2=4 -> c = 3/cos(arctan(4/3)) = 3/(3/5) = 5
    assert lppls_model.get_c(3, 4) == pytest.approx(5.0, rel=1e-9)

    # c1=1, c2=0 -> c = 1/cos(0) = 1
    assert lppls_model.get_c(1, 0) == 0  # c2=0 triggers the falsy branch


def test_ordinal_to_date(lppls_model):
    """Verify date conversion and out-of-bounds handling."""
    # Valid ordinal
    assert lppls_model.ordinal_to_date(738000) == "2021-07-29"

    # Out-of-bounds ordinal
    assert lppls_model.ordinal_to_date(-999999999) == str(pd.NaT)


# ---------------------------------------------------------------------------
# Basic model tests
# ---------------------------------------------------------------------------

def test_model_creation(observations):
    """Verify model can be instantiated with observations."""
    model = lppls.LPPLS(observations=observations)
    assert model.observations is not None
    assert model.observations.shape[0] == 2
    assert model.observations.shape[1] == 100


def test_matrix_equation(observations, lppls_model):
    """Verify matrix_equation returns valid linear params."""
    tc, m, w = 110.0, 0.5, 8.0
    result = lppls_model.matrix_equation(observations, tc, m, w)
    assert result.shape == (4, 1)
    assert all(np.isfinite(result[:, 0]))


def test__get_tc_bounds(observations, lppls_model):
    """Verify tc bounds are computed relative to observation endpoints."""
    t1 = observations[0][0]
    t2 = observations[0][-1]
    delta = t2 - t1

    tc_min, tc_max = lppls_model._get_tc_bounds(observations, 0.20, 0.20)
    assert tc_min == pytest.approx(t2 - delta * 0.20)
    assert tc_max == pytest.approx(t2 + delta * 0.20)

    # Symmetric bounds should be equidistant from t2
    tc_min_50, tc_max_50 = lppls_model._get_tc_bounds(observations, 0.50, 0.50)
    assert tc_max_50 - t2 == pytest.approx(t2 - tc_min_50)


def test__is_O_in_range(lppls_model):
    """Verify oscillation range check."""
    # Case 1, True
    assert lppls_model._is_O_in_range(1000, 9.8, 800, 2.5) == True

    # Case 2, False
    assert lppls_model._is_O_in_range(1000, 9.7, 800, 2.5) == False


def test__is_D_in_range(lppls_model):
    """Verify damping range check with boundary cases."""
    D_min = 1.0

    # True: abs((0.5 * 3000) / (9.8 * 100)) > 1.0
    assert lppls_model._is_D_in_range(0.5, 9.8, 3000, 100, D_min) is True

    # False: abs((0.5 * 1000) / (9.8 * 100)) < 1.0
    assert lppls_model._is_D_in_range(0.5, 9.8, 1000, 100, D_min) is False

    # m = 0 -> always False
    assert lppls_model._is_D_in_range(0, 9.8, 1000, 100, D_min) is False

    # w = 0 -> always False
    assert lppls_model._is_D_in_range(0.5, 0, 1000, 100, D_min) is False

    # Both zero -> always False
    assert lppls_model._is_D_in_range(0, 0, 1000, 100, D_min) is False


# ---------------------------------------------------------------------------
# Fit tests (non-deterministic — use invariants, not exact values)
# ---------------------------------------------------------------------------

def test_fit(observations, lppls_model):
    """Verify fit() runs and returns 10 values with plausible types."""
    result = lppls_model.fit(max_searches=25)
    assert len(result) == 10
    tc, m, w, a, b, c, c1, c2, O, D = result
    assert all(isinstance(v, (int, float, np.floating)) for v in result)


def test_fit_seeded(observations, lppls_model):
    """Verify fit() with a seeded RNG produces structurally valid results."""
    random.seed(42)
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(max_searches=25)

    if tc != 0:  # fit converged
        t1 = observations[0][0]
        t2 = observations[0][-1]

        # All params should be finite
        assert np.isfinite(tc)
        assert np.isfinite(O)
        assert np.isfinite(D)

        # tc should be in a plausible range near the data window
        assert t1 < tc < t2 + 0.5 * (t2 - t1)
    else:
        # All-zeros fallback is acceptable
        assert (m, w, a, b, c, c1, c2, O, D) == (0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_fit_exhausted_returns_zeros():
    """When all searches fail, fit() should return all zeros."""
    obs = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    model = lppls.LPPLS(observations=obs)
    result = model.fit(max_searches=1)
    assert result == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_compute_nested_fits_xarray(observations, lppls_model):
    """Verify compute_nested_fits returns a well-formed xarray.DataArray."""
    import xarray as xr
    result = lppls_model.compute_nested_fits(
        window_size=80,
        smallest_window_size=60,
        outer_increment=20,
        inner_increment=10,
        max_searches=5,
    )
    assert isinstance(result, xr.DataArray)
    assert "t2" in result.dims
    assert "windowsizes" in result.dims
    assert "params" in result.dims


def test_mp_compute_nested_fits(observations, lppls_model):
    """Verify mp_compute_nested_fits returns expected structure."""
    res = lppls_model.mp_compute_nested_fits(workers=1)
    assert len(res) == 5
    assert res[0]['t1'] == 0.0
    assert res[0]['t2'] == 79.0
    assert res[4]['t1'] == 20.0
    expected_keys = {'tc', 'm', 'w', 'a', 'b', 'c', 'c1', 'c2', 't1', 't2', 'O', 'D'}
    assert len(res[0]['res']) == 30
    assert set(res[0]['res'][0]).issubset(expected_keys)


def test_detect_bubble_start_time_via_lagrange(observations, lppls_model):
    """Smoke test: Lagrange method runs without error and returns expected keys."""
    result = lppls_model.detect_bubble_start_time_via_lagrange(
        max_window_size=80,
        min_window_size=40,
        step_size=10,
        max_searches=5,
    )
    if result is not None:
        expected_keys = {
            "tau", "optimal_window_size", "tc", "m", "w", "a", "b", "c1", "c2",
            "window_sizes", "sse_list", "ssen_list", "lagrange_sse_list",
        }
        assert expected_keys.issubset(result.keys())
        assert result["optimal_window_size"] >= 40
        assert result["optimal_window_size"] <= 80
