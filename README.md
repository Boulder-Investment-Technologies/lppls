![PyPI 📦   ](https://github.com/Boulder-Investment-Technologies/lppls/workflows/PyPI%20%F0%9F%93%A6%20%20%20/badge.svg?branch=master)
# Log Periodic Power Law Singularity (LPPLS) Model 
`lppls` is a Python module for fitting the LPPLS model to data.


## Overview
The LPPLS model provides a flexible framework to detect bubbles and predict regime changes of a financial asset. A bubble is defined as a faster-than-exponential increase in asset price, that reflects positive feedback loop of higher return anticipations competing with negative feedback spirals of crash expectations. It models a bubble price as a power law with a finite-time singularity decorated by oscillations with a frequency increasing with time. 

Here is the model:

![LPPLS Model](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/LPPLS_Model.svg)

  where:

  - ![Expected Log Price](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Expected_Log_Price.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) expected log price at the date of the termination of the bubble
  - ![Critical Time](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Critical_Time.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) critical time (date of termination of the bubble and transition in a new regime) 
  - ![A](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/A.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) expected log price at the peak when the end of the bubble is reached at <img src="https://latex.codecogs.com/svg.latex?t_c" title="Critcal Time" />
  - ![B](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/B.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) amplitude of the power law acceleration
  - ![C](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/C.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) amplitude of the log-periodic oscillations
  - ![m](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/m.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) degree of the super exponential growth
  - ![omega](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/omega.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) scaling ratio of the temporal hierarchy of oscillations
  - ![phi](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/phi.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) time scale of the oscillations
    
The model has three components representing a bubble. The first, ![LPPLS Term 1](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/LPPLS_Term_1.svg), handles the hyperbolic power law. For ![m](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/m.svg) < 1 when the price growth becomes unsustainable, and at ![Critical Time](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Critical_Time.svg) the growth rate becomes infinite. The second term, ![LPPLS Term 2](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/LPPLS_Term_2.svg), controls the amplitude of the oscillations. It drops to zero at the critical time ![Critical Time](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Critical_Time.svg). The third term, ![LPPLS Term 3](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/LPPLS_Term_3.svg), models the frequency of the osciallations. They become infinite at ![Critical Time](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Critical_Time.svg).

## Important links
 - Official source code repo: https://github.com/Boulder-Investment-Technologies/lppls
 - Download releases: https://pypi.org/project/lppls/
 - Issue tracker: https://github.com/Boulder-Investment-Technologies/lppls/issues

## Installation
Dependencies

`lppls` requires:
 - Matplotlib (>= 3.1.1)
 - NumPy (>= 1.17.0)
 - Pandas (>= 0.25.0)
 - Python (>= 3.6)
 - SciPy (>= 1.3.0)

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
data = pd.read_csv('data/sp500.csv', index_col='Date')

# convert index col to evenly spaced numbers over a specified interval
time = np.linspace(0, len(data)-1, len(data))

# create list of observation data, in this case, 
# daily adjusted close prices of the S&P 500
price = [p for p in data['Adj Close']]

# create Mx2 matrix (expected format for LPPLS observations)
observations = np.array([time, price])

# create reasonable bounds for critical time initialization
first = time[0]
last = time[-1]
pct_delta = (last - first) * 0.20
tc_init_min = last - pct_delta
tc_init_max = last + pct_delta

# set random initialization limits for non-linear params
init_limits = [
    (tc_init_min, tc_init_max), # tc : Critical Time 
    (0.1, 0.9),                 # m : 0.1 ≤ m ≤ 0.9
    (6, 13),                    # ω : 6 ≤ ω ≤ 13
]

# set the max number for searches to perform before giving-up
# the literature suggests 25
MAX_SEARCHES = 25

# instantiate a new LPPLS model with the S&P 500 dataset
lppls_model = lppls.LPPLS(use_ln=True, observations=observations)

# fit the model to the data and get back the params
tc, m, w, a, b, c = lppls_model.fit(observations, MAX_SEARCHES, init_limits, minimizer='Nelder-Mead')

# visualize the fit
lppls_model.plot_fit(tc, m, w, observations)

# should give a plot like the following...
```

![LPPLS Fit to the S&P500 Dataset](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/sp500_lppls_fit.png)

## References
 - Filimonov, V. and Sornette, D. A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model. Physica A: Statistical Mechanics and its Applications. 2013
 - Sornette, D. Why Stock Markets Crash: Critical Events in Complex Financial Systems. Princeton University Press. 2002.
