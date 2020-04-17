# Log Periodic Power Law Singularity (LPPLS) Model 
`lppls` is a Python module for fitting the LPPLS model to data.


## Overview
The LPPL model provides a flexible framework to detect bubbles and predict regime changes of a financial asset. A bubble is defined as a faster-than-exponential increase in asset price, that reflects positive feedback loop of higher return anticipations competing with negative feedback spirals of crash expectations. It models a bubble price as a power law with a finite-time singularity decorated by oscillations with a frequency increasing with time. 

## Installation
Dependencies

`lppls` requires:
 - Pandas (>= 0.25.0)
 - Python (>= 3.6)
 - NumPy (>= 1.17.0)
 - SciPy (>= 1.3.0)
 - Matplotlib (>= 3.1.1)

User installation
If you already have a working installation of numpy and scipy, the easiest way to install scikit-learn is using pip
```
pip install -U lppls
```

## Example Use
```python
from lppls import lppls
import pandas as pd
import tqdm
import time 

if __name__ == '__main__':
    start = time.time()
    data = pd.read_csv('<location>.csv', index_col='<index_col>', parse_dates=True)
    signals_list = []
    asset_list = data.columns.tolist()
    window = 126
    for seq in tqdm(range(data.shape[0] - window)):
        window_data = data.iloc[seq:seq + window].copy()
        lppl_model = lppls.LPPLS(window_data, asset_list)
        signals_list.append(lppl_model.fetch_indicators(126, 5, 21, 5))
    end = time.time()
    duration = end - start
    print(duration)
```

## References
 - Filimonov, V. and Sornette, D. A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model. Physica A: Statistical Mechanics and its Applications. 2013
 - Sornette, D. Why Stock Markets Crash: Critical Events in Complex Financial Systems. Princeton University Press. 2002.
