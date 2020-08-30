import data_loader
import pandas as pd


def test_sp500():
    # Test that stream loads
    data = data_loader.sp500()
    assert data.iloc[0].values.tolist() == ['2013-10-30', 1772.27002, 1775.219971, 1757.2399899999998, 1763.310059,
                                          1763.310059, 3523040000]


