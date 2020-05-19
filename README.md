![PyPI 📦   ](https://github.com/Boulder-Investment-Technologies/lppls/workflows/PyPI%20%F0%9F%93%A6%20%20%20/badge.svg?branch=master)
# Log Periodic Power Law Singularity (LPPLS) Model 
`lppls` is a Python module for fitting the LPPLS model to data.


## Overview
The LPPLS model provides a flexible framework to detect bubbles and predict regime changes of a financial asset. A bubble is defined as a faster-than-exponential increase in asset price, that reflects positive feedback loop of higher return anticipations competing with negative feedback spirals of crash expectations. It models a bubble price as a power law with a finite-time singularity decorated by oscillations with a frequency increasing with time. 

Here is the model:

<img src="https://latex.codecogs.com/svg.latex?E[\text{ln&space;}p(t)]=A+B(t_c-t)^m&space;+C(t_c-t)^m&space;cos(\omega&space;ln(t_c-t)-\phi)" title="LPPLS Model" />

  where:

  - <img src="https://latex.codecogs.com/svg.latex?E[\text{ln&space;}p(t)]&space;:=" title="Expected Log Price" /> expected log price at the date of the termination of the bubble
  - <img src="https://latex.codecogs.com/svg.latex?t_c&space;:=" title="Critcal Time" /> critical time (date of termination of the bubble and transition in a new regime) 
  - <img src="https://latex.codecogs.com/svg.latex?A&space;:=" title="A" /> expected log price at the peak when the end of the bubble is reached at <img src="https://latex.codecogs.com/svg.latex?t_c" title="Critcal Time" />
  - <img src="https://latex.codecogs.com/svg.latex?B&space;:=" title="B" /> amplitude of the power law acceleration
  - <img src="https://latex.codecogs.com/svg.latex?C&space;:=" title="C" /> amplitude of the log-periodic oscillations
  - <img src="https://latex.codecogs.com/svg.latex?m&space;:=" title="m" /> degree of the super exponential growth
  - <img src="https://latex.codecogs.com/svg.latex?\omega&space;:=" title="Omega" /> scaling ratio of the temporal hierarchy of oscillations
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
```
pip install -U lppls
```

## Example Use
```python
from lppls import lppls
import numpy as np
import pandas as pd
%matplotlib inline

# read example dataset into df 
data = pd.read_csv('/Users/joshnielsen/projects/lppls/data/sp500.csv', index_col='Date')

# convert index col to evenly spaced numbers over a specified interval
time = np.linspace(0, len(data)-1, len(data))

# create list of observation data, in this case, 
# daily adjusted close prices of the S&P 500
price = [p for p in data['Adj Close']]

# create Mx2 matrix (expected format for LPPLS observations)
observations = np.array([time, price])

# set the max number for searches to perfrom before giving-up
# the literature suggests 25
MAX_SEARCHES = 25

# instantiate a new LPPLS model with the S&P 500 dataset
lppls_model = lppls.LPPLS(use_ln=True, observations=observations)

# fit the model to the data and get back the params
tc, m, w, a, b, c = lppls_model.fit(observations, MAX_SEARCHES, minimizer='Nelder-Mead')

# visualize the fit
lppls_model.plot_fit(observations, tc, m, w)

# should give a plot like the following...
```

![LPPLS Fit to the S&P500 Dataset](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/sp500_lppls_fit.png)

## References
 - Filimonov, V. and Sornette, D. A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model. Physica A: Statistical Mechanics and its Applications. 2013
 - Sornette, D. Why Stock Markets Crash: Critical Events in Complex Financial Systems. Princeton University Press. 2002.
