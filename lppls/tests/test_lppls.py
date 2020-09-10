#!/usr/bin/env python3

import lppls
import data_loader
import pytest
import numpy as np


@pytest.fixture
def data():
    return data_loader.sp500()


@pytest.fixture
def lppls_model():
    """Returns a model instance"""
    return lppls.LPPLS(observations=data_loader.sp500())


def test_lppls(data):
    # Test that the base lppls function is giving expected results.
    lppls_model = lppls.LPPLS(observations=data)
    # Check integrity at period 0
    t, tc, m, w, a, b, c1, c2 = 0.0, 1300.2412888852296, 0.6087189222292106, 6.344318139503496, 3034.166016949172, \
                                -16.041970137173486, 0.21878280136703082, - 0.14789336333436504
    assert 1793.5438277142375 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    # Check integrity at period 500
    t, tc, m, w, a, b, c1, c2 = 500.0, 1428.0641475858731, 0.3473013071950998, 6.052643019980449, 3910.1099206097356, \
                                -169.93053270790418, 0.05189394517600043, -0.045820295077658835
    assert 2087.089947771132 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

@pytest.mark.skip()
def test_minimize(data):
    # Testing the minimizer is slow test but vital for confidence as dependencies are updated.
    lppls_model = lppls.LPPLS(observations=data)
    time_ = np.linspace(0, len(data) - 1, len(data))
    price = [p for p in data['Adj Close']]
    observations = np.array([time_, price])

    # Test that the minimizer is giving expected results
    seed = [1.34976187e+03, 2.80963765e-01, 6.49838027e+00]
    tc, m, w, a, b, c = lppls_model.minimize(observations, seed, "Nelder-Mead")

    # Test that coefficients were successfully saved to object memory (self.coef_)
    for coef in ["tc", "m", "w", "a", "b", "c"]:
        assert lppls_model.coef_[coef] == eval(coef)

    # Test that the minimizer function is raising an UnboundedError when solving the linear params fails.
    with pytest.raises(UnboundLocalError):
        seed = [1.41378435e+03, 1.60789624e-01, 7.39236331e+00]
        lppls_model.minimize(observations, seed, "Nelder-Mead")

    # Test that numpy.linalg.LinAlgError is raised when non-linear minimization fails.
    with pytest.raises(np.linalg.LinAlgError):
        seed = [1.28337021e+03, 5.20824525e-01, 1.26182622e+01]
        lppls_model.minimize(observations, seed, "Nelder-Mead")


def test_fit(data):
    # LPPLS.fit() uses random numbers which are not guaranteed to have parity across platforms.
    # The only test that will run every time is a check for exceptions.

    lppls_model = lppls.LPPLS(observations=data)
    time_ = np.linspace(0, len(data) - 1, len(data))
    price = [p for p in data['Adj Close']]
    observations = np.array([time_, price])
    MAX_SEARCHES = 25

    # fit the model to the data and get back the params
    tc, m, w, a, b, c = lppls_model.fit(observations, MAX_SEARCHES, minimizer='Nelder-Mead')


def test__get_tc_bounds(data):
    # Test that time-critical search interval is expected.
    lppls_model = lppls.LPPLS(observations=data)
    time_ = np.linspace(0, len(data) - 1, len(data))
    price = [p for p in data['Adj Close']]
    observations = np.array([time_, price])

    tc_init_min, tc_init_max = lppls_model._get_tc_bounds(observations, 0.20, 0.20)
    assert tc_init_min == 1005.6
    assert tc_init_max == 1508.4

@pytest.mark.skip(reason='np.linalg.lstsq is non-deterministic')
def test_matrix_equation(data):
    # Test that the linear params are generated in an expected way.

    lppls_model = lppls.LPPLS(observations=data)
    time_ = np.linspace(0, len(data) - 1, len(data))
    price = [p for p in data['Adj Close']]
    observations = np.array([time_, price])

    # Case 1, expected values
    tc, m, w = 1341.3258583124998, 0.28623183559375, 6.620224900062501
    lin_vals = lppls_model.matrix_equation(observations, tc, m, w)
    assert lin_vals.tolist() == [3935.3417335296936, -269.8802449167163, 0.5689486764385909, 0.24572805481553928]

    # Case 2, expected values
    tc, m, w = 1344.1378622083332, 0.2704276238124999, 6.796222699041667
    lin_vals = lppls_model.matrix_equation(observations, tc, m, w)
    assert lin_vals.tolist() == [4038.5204665897268, -316.6880507095347, -0.07145919131095202, 0.05217571734085247]

