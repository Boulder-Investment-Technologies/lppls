#!/usr/bin/env python3

import lppls
import data_loader
import pytest

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
    t, tc, m, w, a, b, c1, c2 = 0.0, 1300.2412888852296, 0.6087189222292106, 6.344318139503496, 3034.166016949172,\
                                -16.041970137173486, 0.21878280136703082, - 0.14789336333436504
    assert 1793.5438277142375 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    # Check integrity at period 500
    t, tc, m, w, a, b, c1, c2 = 500.0, 1428.0641475858731, 0.3473013071950998, 6.052643019980449, 3910.1099206097356,\
                                -169.93053270790418, 0.05189394517600043, -0.045820295077658835
    assert 2087.089947771132 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    return
