import pkg_resources
import pandas as pd


def sp500():
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/sp500.csv')
    return pd.read_csv(stream, encoding='utf-8')
