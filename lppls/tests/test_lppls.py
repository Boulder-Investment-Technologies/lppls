#!/usr/bin/env python3

import lppls
import data_loader
import pytest
import numpy as np


@pytest.fixture
def data():
    return data_loader.sp500()


@pytest.fixture
def observations(data):
    time_ = np.linspace(0, len(data) - 1, len(data))
    price = [p for p in data['Adj Close']]
    return np.array([time_, price])


@pytest.fixture
def lppls_model(observations):
    """Returns a model instance"""
    return lppls.LPPLS(observations=observations)


def test_lppls(lppls_model):
    # Test that the base lppls function is giving expected results.

    # Check integrity at period 0
    t, tc, m, w, a, b, c1, c2 = 0.0, 1300.2412888852296, 0.6087189222292106, 6.344318139503496, 3034.166016949172, \
                                -16.041970137173486, 0.21878280136703082, - 0.14789336333436504
    assert 1762.3196588471408 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    # Check integrity at period 500
    t, tc, m, w, a, b, c1, c2 = 500.0, 1428.0641475858731, 0.3473013071950998, 6.052643019980449, 3910.1099206097356, \
                                -169.93053270790418, 0.05189394517600043, -0.045820295077658835
    assert 2086.3299554496016 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)


def test_minimize(observations, lppls_model):
    # Testing the minimizer is slow test but vital for confidence as dependencies are updated.
    # Test that the minimizer is giving expected results
    seed = [1346.2379633747132, 0.25299669770427197, 9.202480294316384]
    tc, m, w, a, b, c, c1, c2 = lppls_model.minimize(observations, seed, "SLSQP")

    # Test that coefficients were successfully saved to object memory (self.coef_)
    for coef in ["tc", "m", "w", "a", "b", "c", "c1", "c2"]:
        assert lppls_model.coef_[coef] == eval(coef)

    # Test that the minimizer function is raising an UnboundedError when solving the linear params fails.
    with pytest.raises(np.linalg.LinAlgError):
        seed = [np.nan, np.nan, np.nan]
        lppls_model.minimize(observations, seed, "SLSQP")


def test_fit(observations, lppls_model):
    # LPPLS.fit() uses random numbers which are not guaranteed to have parity across platforms.
    # The only test that will run every time is a check for exceptions.

    MAX_SEARCHES = 25

    # fit the model to the data and get back the params
    lppls_model.fit(observations, MAX_SEARCHES, minimizer='SLSQP')


def test__get_tc_bounds(observations, lppls_model):
    # Test that time-critical search interval is expected.

    tc_init_min, tc_init_max = lppls_model._get_tc_bounds(observations, 0.20, 0.20)
    assert tc_init_min == 1005.6
    assert tc_init_max == 1508.4


def test_matrix_equation(observations, lppls_model):
    # Test that the linear params are generated in an expected way (10-decimal precision)

    # Case 1, expected values
    tc, m, w = 1341.3258583124998, 0.28623183559375, 6.620224900062501
    lin_vals = lppls_model.matrix_equation(observations, tc, m, w)
    assert (np.round(lin_vals, 10) == np.round(
        [4022.6602773956506, -285.82229531206656, -5.534444109995974, 10.151437800554937], 10)).all()

    # Case 2, expected values
    tc, m, w = 1344.1378622083332, 0.2704276238124999, 6.796222699041667
    lin_vals = lppls_model.matrix_equation(observations, tc, m, w)
    assert (np.round(lin_vals, 10) == np.round(
        [4123.919805408301, -333.7726805698412, -12.107142946248267, -1.8589644488871784], 10)).all()
