from lppls import lppls, data_loader
import jax.numpy as np
import pandas as pd

# read example dataset into df 
data = data_loader.sp500()

# convert index col to evenly spaced numbers over a specified interval
time = np.linspace(0, len(data)-1, len(data))

# create list of observation data, in this case, 
# daily adjusted close prices of the S&P 500
price = [p for p in data['Adj Close']]
price = np.array(price)

# A tuple we pass as fixed args for the lppl function
observations = [time, price]

# set the max number for searches to perform before giving-up
# the literature suggests 25
MAX_SEARCHES = 25

# instantiate a new LPPLS model with the S&P 500 dataset
lppls_model = lppls.LPPLS(use_ln=True, observations=observations)

print("Linear parameters are ", \
    lppls_model.matrix_equation(
        observations, 1500.00, 0.4, 7.0))
"""
Expect something like 
 [[ 8.48615856e+00]
 [-5.43618247e-02]
 [ 2.74893254e-03]
 [-1.24610178e-03]]
print("Time shape is ", time.shape)
print("Price shape is ", price.shape)
print("Value is ", lppls_model.func_restricted(time, price, np.array([1300, 0.1, 0.2])))
"""
# # fit the model to the data and get back the params
# tc, m, w, a, b, c = lppls_model.fit(observations, MAX_SEARCHES, minimizer='Nelder-Mead')

# # visualize the fit
# lppls_model.plot_fit(observations, tc, m, w)