# Log Periodic Power Law Singularity Model
`lppls` is a Python module for fitting the LPPLS model to data.

## Installation
Dependencies

`lppls` requires:
 - Pandas (>= 0.18.0)
 - Python (>= 3.6)
 - NumPy (>= 1.13.3)
 - SciPy (>= 0.19.1)
 - Matplotlib (>= 2.1.1)


User installation
If you already have a working installation of numpy and scipy, the easiest way to install scikit-learn is using pip
```
pip install -U lppls
```

## Example Use:

```python
if __name__ == '__main__':
    start = time.time()
    data = pd.read_csv('data/crypto_assets_daily.csv', index_col='epoch_ts', parse_dates=True)
    signals_list = []
    asset_list = data.columns.tolist()
    window = 126
    for seq in tqdm(range(data.shape[0] - window)):
        window_data = data.iloc[seq:seq + window].copy()
        lppl_model = LPPL(window_data, asset_list)
        signals_list.append(lppl_model.fetch_indicators(126, 5, 21, 5))
    end = time.time()
    duration = end - start
    print(duration)
```