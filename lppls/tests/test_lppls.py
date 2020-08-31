#!/usr/bin/env python3

import lppls
import data_loader
import pytest

@pytest.fixture
def data():
    return data_loader.sp500()

@pytest.fixture
def lppls_model():
    """Returns a model instance with use_ln=True"""
    return lppls.LPPLS(use_ln=True, observations=data_loader.sp500())

def test_lppls(data):
    # Test that the base lppls function is giving expected results.
    lppls_model = lppls.LPPLS(use_ln=True,observations=data)

    # Using ln
    # Check integrity at period 0
    t, tc, m, w, a, b, c1, c2 = 0.0, 1279.0429001048128, 0.2133326239108304, 11.091557842299654, 8.496823547814529,\
                                -0.20816240802471664, 0.00478771542574161, 0.002854567990966797
    assert 7.514469730428095 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    # Check integrity at period 500
    t, tc, m, w, a, b, c1, c2 = 500.0, 1267.9408517287782, 0.2782773005077447, 8.48668944887598, 8.314317740049685,\
                                -0.10736015203099668, 0.0006372137521824459, 0.005475817732843753
    assert 7.6306097488441615 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    # Not using ln
    # Check integrity at period 0
    t, tc, m, w, a, b, c1, c2 = 0.0, 1300.2412888852296, 0.6087189222292106, 6.344318139503496, 3034.166016949172,\
                                - 16.041970137173486, 0.21878280136703082, - 0.14789336333436504
    assert 1762.3196588471408 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    # Check integrity at period 500
    t, tc, m, w, a, b, c1, c2 = 500.0, 1428.0641475858731, 0.3473013071950998, 6.052643019980449, 3910.1099206097356,\
                                -169.93053270790418, 0.05189394517600043, -0.045820295077658835
    assert 2086.3299554496016 == lppls_model.lppls(t, tc, m, w, a, b, c1, c2)

    return
