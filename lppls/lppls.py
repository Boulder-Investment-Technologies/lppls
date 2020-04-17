import itertools
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
import random
from scipy.optimize import least_squares, minimize
from scipy import linalg
import statistics as stats
import time
import warnings
import time
import pdb
from tqdm import tqdm


# warnings.filterwarnings('ignore')

class LPPLS(object):
    def __init__(self, price_data, securities):
        self.price_data = price_data
        self.securities = securities

    def fetch_indicators(self, upper_bound, lower_bound, interval, num_workers, csv=False):
        pool = multiprocessing.Pool(processes=num_workers)
        result = pool.map(self.compute_ds_lppls_confidence,
                          [(self.price_data[symbol].iloc[-upper_bound:], symbol, upper_bound, lower_bound, interval) \
                           for symbol in self.securities]
                          )
        pool.close()
        results_df = pd.DataFrame(result)
        if csv:
            results_df.to_csv(f'{self.price_data.iloc[-1, :].index.values}')
        else:
            return results_df

    # matrix helpers
    def _yi(self, price_series):
        return [np.log(p) for p in price_series]

    def _fi(self, tc, m, time_series):
        return [np.power((tc - t), m) if str(np.power((tc - t), m)) != 'nan' else 0 for t in time_series]

    def _gi(self, tc, m, w, time_series):
        return [np.power((tc - t), m) * np.cos(w * np.log(tc - t)) if str(
            np.power((tc - t), m) * np.cos(w * np.log(tc - t))) != 'nan' else 0 for t in time_series]

    def _hi(self, tc, m, w, time_series):
        return [np.power((tc - t), m) * np.sin(w * np.log(tc - t)) if str(
            np.power((tc - t), m) * np.sin(w * np.log(tc - t))) != 'nan' else 0 for t in time_series]

    def _fi_pow_2(self, tc, m, time_series):
        return np.power(self._fi(tc, m, time_series), 2)

    def _gi_pow_2(self, tc, m, w, time_series):
        return np.power(self._gi(tc, m, w, time_series), 2)

    def _hi_pow_2(self, tc, m, w, time_series):
        return np.power(self._hi(tc, m, w, time_series), 2)

    def _figi(self, tc, m, w, time_series):
        return np.multiply(self._fi(tc, m, time_series), self._gi(tc, m, w, time_series))

    def _fihi(self, tc, m, w, time_series):
        return np.multiply(self._fi(tc, m, time_series), self._hi(tc, m, w, time_series))

    def _gihi(self, tc, m, w, time_series):
        return np.multiply(self._gi(tc, m, w, time_series), self._hi(tc, m, w, time_series))

    def _yifi(self, tc, m, time_series, price_series):
        return np.multiply(self._yi(price_series), self._fi(tc, m, time_series))

    def _yigi(self, tc, m, w, time_series, price_series):
        return np.multiply(self._yi(price_series), self._gi(tc, m, w, time_series))

    def _yihi(self, tc, m, w, time_series, price_series):
        return np.multiply(self._yi(price_series), self._hi(tc, m, w, time_series))

    def lppl(self, t, tc, m, w, a, b, c1, c2):
        '''
        revised version of the LPPL without φ
        found on page 11 as equation (13)
        '''
        return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * np.log(tc - t))) + (c2 * np.sin(w * np.log(tc - t)))))

    def func_restricted(self, x, *args):
        '''
        finds the least square difference
        '''
        tc = x[0]
        m = x[1]
        w = x[2]

        data_series = args[0]

        lin_vals = self.matrix_equation(tc, m, w, data_series)

        a = lin_vals[0]
        b = lin_vals[1]
        c1 = lin_vals[2]
        c2 = lin_vals[3]

        delta = [self.lppl(t, tc, m, w, a, b, c1, c2) for t in data_series[0]]
        delta = np.subtract(delta, data_series[1])
        delta = np.power(delta, 2)

        return np.sum(delta)

    def matrix_equation(self, tc, m, w, data_series):
        '''
        solve the matrix equation
        '''
        time_series = data_series[0]
        price_series = data_series[1]
        N = len(price_series)

        # --------------------------------
        fi = sum(self._fi(tc, m, time_series))
        gi = sum(self._gi(tc, m, w, time_series))
        hi = sum(self._hi(tc, m, w, time_series))

        # --------------------------------
        fi_pow_2 = sum(self._fi_pow_2(tc, m, time_series))
        gi_pow_2 = sum(self._gi_pow_2(tc, m, w, time_series))
        hi_pow_2 = sum(self._hi_pow_2(tc, m, w, time_series))

        # --------------------------------
        figi = sum(self._figi(tc, m, w, time_series))
        fihi = sum(self._fihi(tc, m, w, time_series))
        gihi = sum(self._gihi(tc, m, w, time_series))

        # --------------------------------
        yi = sum(self._yi(price_series))
        yifi = sum(self._yifi(tc, m, time_series, price_series))
        yigi = sum(self._yigi(tc, m, w, time_series, price_series))
        yihi = sum(self._yihi(tc, m, w, time_series, price_series))

        # --------------------------------
        matrix_1 = np.matrix([
            [N, fi, gi, hi],
            [fi, fi_pow_2, figi, fihi],
            [gi, figi, gi_pow_2, gihi],
            [hi, fihi, gihi, hi_pow_2]
        ])

        matrix_2 = np.matrix([
            [yi],
            [yifi],
            [yigi],
            [yihi]
        ])

        try:
            product = linalg.solve(matrix_1, matrix_2)
            return [i[0] for i in product]

        except Exception as e:
            print(e)

    def compute_ds_lppls_confidence(self, args, minimizer='Nelder-Mead'):
        """
        Parameters:
        -----------
        args : list
            df : pd.Series
                prices
            symbol : str
                security symbol
            upperbound : int
                126 # ~6 months (in trading days)
            lowerbound : int
                21 # ~1 month (in trading days)
            interval : int
                5
        """
        df, symbol, upperbound, lowerbound, interval = args

        df2 = pd.DataFrame(df).tail(upperbound).copy()

        ds_lppls = []

        number_of_fitting_windows = (upperbound - lowerbound) // interval

        for i in range(number_of_fitting_windows):

            tLen = upperbound - (i * interval)
            trading_days_data = df.tail(tLen)
            time = np.linspace(0, tLen - 1, tLen)
            price = np.array([trading_days_data[i] for i in range(len(trading_days_data))])
            data_series = np.array([time, price])

            MAX_SEARCHES = 5
            SEARCH_COUNT = 0

            # set limits for non-linear params
            bounds = [
                (tLen - (tLen * 0.2), tLen + (tLen * 0.2)),  # Critical Time + or - .2
                (0.1, 0.9),  # m : 0.1 ≤ m ≤ 0.9
                (6, 13),  # ω : 6 ≤ ω ≤ 13
            ]

            # find bubbles
            while SEARCH_COUNT < MAX_SEARCHES:

                # randomly choose vals for non-linear params
                non_lin_vals = [random.uniform(a[0], a[1]) for a in bounds]

                tc = non_lin_vals[0]
                m = non_lin_vals[1]
                w = non_lin_vals[2]

                # params to pass to scipy.optimize
                seed = np.array([tc, m, w])

                # scipy optimize minimize
                try:
                    # Nelder-Mead
                    cofs = minimize(
                        args=(data_series, bounds),
                        fun=self.func_restricted,
                        method=minimizer,
                        options={
                            'adaptive': True
                        },
                        x0=seed
                    )

                    if cofs.success:
                        # print('minimize ran succsessfully in {} search(es)'.format(SEARCH_COUNT+1))
                        # determine if it falls in range:

                        tc = cofs.x[0]
                        m = cofs.x[1]
                        w = cofs.x[2]

                        # calculate the linear vals again...
                        lin_vals = self.matrix_equation(tc, m, w, data_series)

                        a = lin_vals[0]
                        b = lin_vals[1]
                        c1 = lin_vals[2]
                        c2 = lin_vals[3]

                        # filtering conditions
                        tc_in_range = tLen - (tLen * 0.05) < tc < tLen + (tLen * 0.1)
                        m_in_range = 0.01 < m < 1.2
                        w_in_range = 2 < w < 25
                        n_oscillation = ((w / 2) * np.log(abs((tc - (i * 5)) / (tLen)))) > 2.5
                        # for bubble end flag
                        damping_bef = (m * abs(b)) / (w * abs(c1 + c2)) > 0.8
                        # for bubble early warning
                        damping_bew = (m * abs(b)) / (w * abs(c1 + c2)) > 0.0

                        if (tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bef):
                            ds_lppls_confidence_bef = True
                        else:
                            ds_lppls_confidence_bef = False

                        if (tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bew):
                            ds_lppls_confidence_bew = True
                        else:
                            ds_lppls_confidence_bew = False

                        ds_lppls.append({symbol: {
                            'ds_lppls_confidence_bef': ds_lppls_confidence_bef,
                            'ds_lppls_confidence_bew': ds_lppls_confidence_bew,
                            'cof': cofs.x,
                            'max_searches_exceeded': False
                        }})
                        break

                    else:
                        SEARCH_COUNT += 1
                        # print('minimize failed to find a solution, trying again')

                except Exception as e:
                    print('minimize failed: {}'.format(e))
                    SEARCH_COUNT += 1

            if SEARCH_COUNT >= MAX_SEARCHES:
                # no solution found in 5 tries, so just add this and move one
                # print('minimize failed in allotted attempts (5)')
                ds_lppls.append({symbol: {
                    'ds_lppls_confidence_bef': False,
                    'ds_lppls_confidence_bew': False,
                    'cof': None,
                    'max_searches_exceeded': True
                }})

        # calculate the actual ds lppls confidence value for end flag and early warning
        true_count_bef = 0
        true_count_bew = 0
        # total_count = len(ds_lppls)
        for i in ds_lppls:
            if i[symbol]['ds_lppls_confidence_bef'] == True:
                true_count_bef += 1
            if i[symbol]['ds_lppls_confidence_bew'] == True:
                true_count_bew += 1

        ds_lppls_confidence_bef_val = true_count_bef / number_of_fitting_windows
        ds_lppls_confidence_bew_val = true_count_bew / number_of_fitting_windows

        # find the sign of the median of cumulative returns since the time t1
        df2['ret'] = df2[symbol].pct_change()
        df2['cum_ret'] = df2['ret'].cumsum()
        median = stats.median(df2['cum_ret'].tolist())
        median_sign = 1 if median > 0 else -1

        return {
            'val_bef': ds_lppls_confidence_bef_val * median_sign,
            'val_bew': ds_lppls_confidence_bew_val * median_sign,
            'date': df.tail(1).index.values[0],
            'price': df.tail(1).values[0],
            'symbol': symbol,
        }
