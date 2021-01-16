import data_loader
import pytest


def test_sp500():
    # Test that stream loads
    data = data_loader.sp500()
    actual = data.iloc[0].values.tolist()
    expected = ['2013-10-30', 1772.27002, 1775.219971, 1757.2399899999998, 1763.310059,
                1763.310059, 3523040000]
    assert all([a == pytest.approx(b, 1e6) for a, b in zip(actual, expected)])
