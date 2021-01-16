![PyPI 📦   ](https://github.com/Boulder-Investment-Technologies/lppls/workflows/PyPI%20%F0%9F%93%A6%20%20%20/badge.svg?branch=master)
![PyTests](https://github.com/Boulder-Investment-Technologies/lppls/workflows/PyTests/badge.svg?branch=master)

# Log Periodic Power Law Singularity (LPPLS) Model 
`lppls` is a Python module for fitting the LPPLS model to data.


## Overview
The LPPLS model provides a flexible framework to detect bubbles and predict regime changes of a financial asset. A bubble is defined as a faster-than-exponential increase in asset price, that reflects positive feedback loop of higher return anticipations competing with negative feedback spirals of crash expectations. It models a bubble price as a power law with a finite-time singularity decorated by oscillations with a frequency increasing with time. 

Here is the model:

![LPPLS Model](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/LPPLS_Model.svg)

  where:

  - ![Expected Log Price](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Expected_Log_Price.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) expected log price at the date of the termination of the bubble
  - ![Critical Time](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Critical_Time.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) critical time (date of termination of the bubble and transition in a new regime) 
  - ![A](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/A.svg) ![Colon Equals](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/coloneq.svg) expected log price at the peak when the end of the bubble is reached at ![Critical Time](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/latex/Critical_Time.svg)
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
 - Python (== 3.8)
 - Matplotlib (>= 3.1.1)
 - Numba (>= 0.51.2)
 - NumPy (>= 1.17.0)
 - Pandas (>= 0.25.0)
 - SciPy (>= 1.3.0)
 - Pytest (>= 6.2.1)

User installation
```
pip install -U lppls
```

## Example Use
```python
from lppls import lppls, data_loader
import numpy as np
import pandas as pd
%matplotlib inline

# read example dataset into df 
data = data_loader.sp500()

# convert index col to evenly spaced numbers over a specified interval
time = np.linspace(0, len(data)-1, len(data))

# create list of observation data, in this case, 
# daily adjusted close prices of the S&P 500
# use log price
price = np.log(data['Adj Close'].values)

# create Mx2 matrix (expected format for LPPLS observations)
observations = np.array([time, price])

# set the max number for searches to perform before giving-up
# the literature suggests 25
MAX_SEARCHES = 25

# instantiate a new LPPLS model with the S&P 500 dataset
lppls_model = lppls.LPPLS(observations=observations)

# fit the model to the data and get back the params
tc, m, w, a, b, c, c1, c2 = lppls_model.fit(observations, MAX_SEARCHES, minimizer='Nelder-Mead')

# visualize the fit
lppls_model.plot_fit()

# should give a plot like the following...
```

![LPPLS Fit to the S&P500 Dataset](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/sp500_lppls_fit.png)

```python
# define custom filter condition
filter_conditions_config = [
  {'condition_1':[
      (0.0, 0.1), # tc_range
      (0,1), # m_range
      (4,25), # w_range
      2.5, # O_min
      0.5, # D_min
  ]},
]

# compute the confidence indicator
res = lppls_model.mp_compute_indicator(
    workers=4, 
    window_size=120, 
    smallest_window_size=30, 
    increment=5, 
    max_searches=25,
    filter_conditions_config=filter_conditions_config
)

lppls_model.plot_confidence_indicators(res, condition_name='condition_1', title='Short Term Indicator 120-30')

# should give a plot like the following...
```
![LPPLS Confidnce Indicator](https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/sp500_confidence_indicator.png)

If you wish to store `res` as a pd.DataFrame, use `res_to_df`.
```python
res_df = lppls_model.res_to_df(res, condition_name='condition_1')
res_df.tail()
# gives the following...
```
<div style="height: 200px; overflow-y:scroll;">
  <img src="https://github.com/Boulder-Investment-Technologies/lppls/raw/master/img/res_to_df.png" width="500" />
</div>

## References
 - Filimonov, V. and Sornette, D. A Stable and Robust Calibration Scheme of the Log-Periodic Power Law Model. Physica A: Statistical Mechanics and its Applications. 2013
 - Sornette, D. Why Stock Markets Crash: Critical Events in Complex Financial Systems. Princeton University Press. 2002.
 - Sornette, D. and Demos, G. and Zhang, Q. and Cauwels, P. and Filimonov, V. and Zhang, Q., Real-Time Prediction and Post-Mortem Analysis of the Shanghai 2015 Stock Market Bubble and Crash (August 6, 2015). Swiss Finance Institute Research Paper No. 15-31.
