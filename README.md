![Publish 🐍 📦 to PyPI](https://github.com/Boulder-Investment-Technologies/lppls/workflows/Publish%20%F0%9F%90%8D%20%F0%9F%93%A6%20to%20PyPI/badge.svg?branch=master)

# Log Periodic Power Law Singularity (LPPLS) Model 
`lppls` is a Python module for fitting the LPPLS model to data.


## Overview
The LPPL model provides a flexible framework to detect bubbles and predict regime changes of a financial asset. A bubble is defined as a faster-than-exponential increase in asset price, that reflects positive feedback loop of higher return anticipations competing with negative feedback spirals of crash expectations. It models a bubble price as a power law with a finite-time singularity decorated by oscillations with a frequency increasing with time. 

Here is the model:

<img src="https://latex.codecogs.com/svg.latex?E[\text{ln&space;}p(t)]=A+B(t_c-t)^m&space;+C(t_c-t)^m&space;cos(\omega&space;ln(t_c-t)-\phi)" title="LPPLS Model" />

  where:

  - <img src="https://latex.codecogs.com/svg.latex?E[\text{ln&space;}p(t)]&space;:=" title="Expected Log Price" /> expected log price at the date of the termination of the bubble
  - <img src="https://latex.codecogs.com/svg.latex?t_c&space;:=" title="Critcal Time" /> critical time (date of termination of the bubble and transition in a new regime) 
  - <img src="https://latex.codecogs.com/svg.latex?A&space;:=" title="A" /> expected log price at the peak when the end of the bubble is reached at <img src="https://latex.codecogs.com/svg.latex?t_c" title="Critcal Time" />
  - <img src="https://latex.codecogs.com/svg.latex?B&space;:=" title="B" /> amplitude of the power law acceleration
  - <img src="https://latex.codecogs.com/svg.latex?C&space;:=" title="C" /> amplitude of the log-periodic oscillations
  - <img src="https://latex.codecogs.com/svg.latex?m&space;:=" title="m" /> degree of the super exponential growth
  - <img src="https://latex.codecogs.com/svg.latex?\omega&space;:=" title="Omega" /> scaling ratio of the temporal hierarchy of osciallations
  - <img src="https://latex.codecogs.com/svg.latex?\phi&space;:=" title="Phi" /> time scale of the oscillations
    
The model has three components representing a bubble. The first, <img src="https://latex.codecogs.com/svg.latex?A+B(t_c-t)^m" title="LPPLS Term 1" />, handles the hyperbolic power law. For <img src="https://latex.codecogs.com/svg.latex?m<1" title="M less than 1" /> when the price growth becomes unsustainable, and at <img src="https://latex.codecogs.com/svg.latex?t_c" title="Critcal Time" /> the growth rate becomes infinite. The second term, <img src="https://latex.codecogs.com/svg.latex?C(t_c-t)^m" title="LPPLS Term 2" />, controls the amplitude of the oscillations. It drops to zero at the critical time <img src="https://latex.codecogs.com/svg.latex?t_c" title="Critcal Time" />. The third term, <img src="https://latex.codecogs.com/svg.latex?cos(\omega&space;ln(t_c-t)-\phi)" title="LPPLS Term 3" />, models the frequency of the osciallations. They become infinite at <img src="https://latex.codecogs.com/svg.latex?t_c" title="Critcal Time" />.

## Important links
 - Official source code repo: https://github.com/Boulder-Investment-Technologies/lppls
 - Download releases: https://pypi.org/project/lppls/
 - Issue tracker: https://github.com/Boulder-Investment-Technologies/lppls/issues

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
