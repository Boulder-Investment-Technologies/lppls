import data_loader
import pytest


def test_nasdaq_dotcom():
    # Test that stream loads
    data = data_loader.nasdaq_dotcom()
    actual = data.iloc[0].values.tolist()
    expected = ['1994-01-03', 774.109985, 777.289978, 768.409973, 770.760010, 770.760010, 253020000]
    assert all([a == pytest.approx(b, 1e6) for a, b in zip(actual, expected)])
