import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
# from scipy import linalg
import statistics as stats


class LPPLS(object):

    def __init__(self, use_ln, observations):
        """
        Args:
            use_ln (bool): Whether to take the natural logarithm of the observations.
            observations (np.array): Mx2 matrix with timestamp and observed value.
        """
        self.use_ln = use_ln
        self.observations = observations
        # self.securities = securities

    # def fetch_indicators(self, upper_bound, lower_bound, interval, num_workers, csv=False):
    #     pool = multiprocessing.Pool(processes=num_workers)
    #     result = pool.map(self.compute_ds_lppls_confidence,
    #                       [(self.observations[symbol].iloc[-upper_bound:], symbol, upper_bound, lower_bound, interval) \
    #                        for symbol in self.securities]
    #                       )
    #     pool.close()
    #     results_df = pd.DataFrame(result)
    #     if csv:
    #         results_df.to_csv(f'{self.observations.iloc[-1, :].index.values}')
    #     else:
    #         return results_df

    # matrix helpers
    def _yi(self, price_series):
        if self.use_ln:
            return [np.log(p) for p in price_series]
        else:
            return [p for p in price_series]

    def _fi(self, tc, m, time_series):
        return [np.power((tc - t), m) for t in time_series]

    def _gi(self, tc, m, w, time_series):
        if self.use_ln:
            return [np.power((tc - t), m) * np.cos(w * np.log(tc - t)) for t in time_series]
        else:
            return [np.power((tc - t), m) * np.cos(w * (tc - t)) for t in time_series]

    def _hi(self, tc, m, w, time_series):
        if self.use_ln:
            return [np.power((tc - t), m) * np.sin(w * np.log(tc - t)) for t in time_series]
        else:
            return [np.power((tc - t), m) * np.sin(w * (tc - t)) for t in time_series]

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

    def lppls(self, t, tc, m, w, a, b, c1, c2):
        if self.use_ln:
            return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * np.log(tc - t))) + (c2 * np.sin(w * np.log(tc - t)))))
        else:
            return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * (tc - t))) + (c2 * np.sin(w * (tc - t)))))

    def func_restricted(self, x, *args):
        '''
        finds the least square difference
        '''
        tc = x[0]
        m = x[1]
        w = x[2]

        obs = args[0]
        # print('shape: {}'.format(obs.shape))

        lin_vals = self.matrix_equation(obs, tc, m, w)

        a = float(lin_vals[0])
        b = float(lin_vals[1])
        c1 = float(lin_vals[2])
        c2 = float(lin_vals[3])

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in obs[0, :]]
        delta = np.subtract(delta, obs[1, :])
        delta = np.power(delta, 2)

        return np.sum(delta)

    def matrix_equation(self, observations, tc, m, w):
        '''
        solve the matrix equation
        '''
        time = observations[0, :]
        obs = observations[1, :]
        N = len(obs)

        # --------------------------------
        fi = sum(self._fi(tc, m, time))
        gi = sum(self._gi(tc, m, w, time))
        hi = sum(self._hi(tc, m, w, time))

        # --------------------------------
        fi_pow_2 = sum(self._fi_pow_2(tc, m, time))
        gi_pow_2 = sum(self._gi_pow_2(tc, m, w, time))
        hi_pow_2 = sum(self._hi_pow_2(tc, m, w, time))

        # --------------------------------
        figi = sum(self._figi(tc, m, w, time))
        fihi = sum(self._fihi(tc, m, w, time))
        gihi = sum(self._gihi(tc, m, w, time))

        # --------------------------------
        yi = sum(self._yi(obs))
        yifi = sum(self._yifi(tc, m, time, obs))
        yigi = sum(self._yigi(tc, m, w, time, obs))
        yihi = sum(self._yihi(tc, m, w, time, obs))

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
            # product = linalg.solve(matrix_1, matrix_2)
            # return [i[0] for i in product]

            inverse = np.linalg.pinv(matrix_1)
            product = inverse * matrix_2
            return product

        except Exception as e:
            print(e)

        return np.array([0,
                         0,
                         0,
                         0])

    # def compute_ds_lppls_confidence(self, args, minimizer='Nelder-Mead'):
    #     """
    #     Parameters:
    #     -----------
    #     args : list
    #         df : pd.Series
    #             prices
    #         symbol : str
    #             security symbol
    #         upperbound : int
    #             126 # ~6 months (in trading days)
    #         lowerbound : int
    #             21 # ~1 month (in trading days)
    #         interval : int
    #             5
    #     """
    #     df, symbol, upperbound, lowerbound, interval = args
    #
    #     df2 = pd.DataFrame(df).tail(upperbound).copy()
    #
    #     ds_lppls = []
    #
    #     number_of_fitting_windows = (upperbound - lowerbound) // interval
    #
    #     for i in range(number_of_fitting_windows):
    #
    #         tLen = upperbound - (i * interval)
    #         trading_days_data = df.tail(tLen)
    #         time = np.linspace(0, tLen - 1, tLen)
    #         price = np.array([trading_days_data[i] for i in range(len(trading_days_data))])
    #         data_series = np.array([time, price])
    #
    #         MAX_SEARCHES = 5
    #         SEARCH_COUNT = 0
    #
    #         # set limits for non-linear params
    #         bounds = [
    #             (tLen - (tLen * 0.2), tLen + (tLen * 0.2)),  # Critical Time + or - .2
    #             (0.1, 0.9),  # m : 0.1 ≤ m ≤ 0.9
    #             (6, 13),  # ω : 6 ≤ ω ≤ 13
    #         ]
    #
    #         # find bubbles
    #         while SEARCH_COUNT < MAX_SEARCHES:
    #
    #             # randomly choose vals for non-linear params
    #             non_lin_vals = [random.uniform(a[0], a[1]) for a in bounds]
    #
    #             tc = non_lin_vals[0]
    #             m = non_lin_vals[1]
    #             w = non_lin_vals[2]
    #
    #             # params to pass to scipy.optimize
    #             seed = np.array([tc, m, w])
    #
    #             # scipy optimize minimize
    #             try:
    #                 # Nelder-Mead
    #                 cofs = minimize(
    #                     args=(data_series, bounds),
    #                     fun=self.func_restricted,
    #                     method=minimizer,
    #                     options={
    #                         'adaptive': True
    #                     },
    #                     x0=seed
    #                 )
    #
    #                 if cofs.success:
    #                     # print('minimize ran succsessfully in {} search(es)'.format(SEARCH_COUNT+1))
    #                     # determine if it falls in range:
    #
    #                     tc = cofs.x[0]
    #                     m = cofs.x[1]
    #                     w = cofs.x[2]
    #
    #                     # calculate the linear vals again...
    #                     lin_vals = self.matrix_equation(tc, m, w, data_series)
    #
    #                     a = lin_vals[0]
    #                     b = lin_vals[1]
    #                     c1 = lin_vals[2]
    #                     c2 = lin_vals[3]
    #
    #                     c = (c1 ** 2 + c2 ** 2) ** (0.5)
    #
    #                     # filtering conditions
    #                     tc_in_range = tLen - (tLen * 0.05) < tc < tLen + (tLen * 0.1)
    #                     m_in_range = 0.01 < m < 1.2
    #                     w_in_range = 2 < w < 25
    #                     n_oscillation = ((w / 2) * np.log(abs((tc - (i * 5)) / (tLen)))) > 2.5
    #                     # for bubble end flag
    #                     damping_bef = (m * abs(b)) / (w * abs(c)) > 0.8
    #                     # for bubble early warning
    #                     damping_bew = (m * abs(b)) / (w * abs(c)) > 0.0
    #
    #                     if (tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bef):
    #                         ds_lppls_confidence_bef = True
    #                     else:
    #                         ds_lppls_confidence_bef = False
    #
    #                     if (tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bew):
    #                         ds_lppls_confidence_bew = True
    #                     else:
    #                         ds_lppls_confidence_bew = False
    #
    #                     ds_lppls.append({symbol: {
    #                         'ds_lppls_confidence_bef': ds_lppls_confidence_bef,
    #                         'ds_lppls_confidence_bew': ds_lppls_confidence_bew,
    #                         'cof': cofs.x,
    #                         'max_searches_exceeded': False
    #                     }})
    #                     break
    #
    #                 else:
    #                     SEARCH_COUNT += 1
    #                     # print('minimize failed to find a solution, trying again')
    #
    #             except Exception as e:
    #                 print('minimize failed: {}'.format(e))
    #                 SEARCH_COUNT += 1
    #
    #         if SEARCH_COUNT >= MAX_SEARCHES:
    #             # no solution found in 5 tries, so just add this and move one
    #             # print('minimize failed in allotted attempts (5)')
    #             ds_lppls.append({symbol: {
    #                 'ds_lppls_confidence_bef': False,
    #                 'ds_lppls_confidence_bew': False,
    #                 'cof': None,
    #                 'max_searches_exceeded': True
    #             }})
    #
    #     # calculate the actual ds lppls confidence value for end flag and early warning
    #     true_count_bef = 0
    #     true_count_bew = 0
    #     # total_count = len(ds_lppls)
    #     for i in ds_lppls:
    #         if i[symbol]['ds_lppls_confidence_bef'] == True:
    #             true_count_bef += 1
    #         if i[symbol]['ds_lppls_confidence_bew'] == True:
    #             true_count_bew += 1
    #
    #     ds_lppls_confidence_bef_val = true_count_bef / number_of_fitting_windows
    #     ds_lppls_confidence_bew_val = true_count_bew / number_of_fitting_windows
    #
    #     # find the sign of the median of cumulative returns since the time t1
    #     df2['ret'] = df2[symbol].pct_change()
    #     df2['cum_ret'] = df2['ret'].cumsum()
    #     median = stats.median(df2['cum_ret'].tolist())
    #     median_sign = 1 if median > 0 else -1
    #
    #     return {
    #         'val_bef': ds_lppls_confidence_bef_val * median_sign,
    #         'val_bew': ds_lppls_confidence_bew_val * median_sign,
    #         'date': df.tail(1).index.values[0],
    #         'price': df.tail(1).values[0],
    #         'symbol': symbol,
    #     }

    def fit(self, observations, max_searches, init_limits, minimizer='Nelder-Mead'):
        """
        Args:
            observations (Mx2 numpy array): the observed data
            max_searches (int): The maxi amount of searches to perform before giving up. The literature suggests 25
            init_limits (list): random initialize bounds for tc, m, w. Should be formatted as a list of tuples in the order: [(tc_min, tc_max), (m_min, m_max), (w_min, w_max)].
            minimizer (str): See list of valid methods to pass to scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        Returns:
            tc, m, w, a, b, c1, c2
        """
        search_count = 0

        # find bubble
        while search_count < max_searches:

            # randomly choose vals within bounds for non-linear params
            non_lin_vals = [random.uniform(a[0], a[1]) for a in init_limits]

            tc = non_lin_vals[0]
            m = non_lin_vals[1]
            w = non_lin_vals[2]

            seed = np.array([tc, m, w])

            try:
                cofs = minimize(
                    args=observations,
                    fun=self.func_restricted,
                    method=minimizer,
                    x0=seed
                )

                if cofs.success:

                    tc = cofs.x[0]
                    m = cofs.x[1]
                    w = cofs.x[2]

                    # calculate the linear vals again...
                    lin_vals = self.matrix_equation(observations, tc, m, w)

                    a = float(lin_vals[0])
                    b = float(lin_vals[1])
                    c1 = float(lin_vals[2])
                    c2 = float(lin_vals[3])

                    c = (c1 ** 2 + c2 ** 2) ** (0.5)

                    return tc, m, w, a, b, c

                else:
                    search_count += 1
            except Exception as e:
                print('minimize failed: {}'.format(e))
                search_count += 1
        return 0, 0, 0, 0, 0, 0

    def plot_many_fits(self, observations, tc, m, w, overlay=True, path_out=False):
        """
        Args:
            tc (float): predicted critical time
            m (float): predicted degree of super-exponential growth
            w (float): predicted scaling ratio of the temporal hierarchy of oscillations
            observations (Mx2 numpy array): the observed data

        Returns:
            nothing, should plot the fit
        """
        lin_vals = self.matrix_equation(observations, tc, m, w)

        a = float(lin_vals[0])
        b = float(lin_vals[1])
        c1 = float(lin_vals[2])
        c2 = float(lin_vals[3])

        time_subset = observations[0, :]
        obs_subset = observations[1, :]
        obs_subset = np.log(obs_subset) if self.use_ln else obs_subset

        time_full = self.observations[0, :]
        obs_full = self.observations[1, :]
        obs_full = np.log(obs_full) if self.use_ln else obs_full

        fig, ax1 = plt.subplots(figsize=(14, 8))
        plt.xticks(rotation=45)

        color = 'tab:blue'
        color2 = 'tab:green'
        color3 = 'tab:pink'
        ax1.set_xlabel('time')
        ax1.set_ylabel('observations')
        ax1.plot(time_full, obs_full, color=color3)
        t_last = time_full[-1]
        if tc < t_last:
            plt.axvline(x=tc)
            lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in time_full]
            ax1.plot(time_full, lppls_fit, color=color)
        else:
            lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in time_subset]
            ax1.plot(time_subset, lppls_fit, color=color)
        ax1.plot(time_subset, obs_subset, color=color2)
        ax1.tick_params(axis='y', labelcolor=color)

        if path_out:
            fig.savefig(path_out)

    def plot_fit(self, observations, tc, m, w):
        """
        Args:
            tc (float): predicted critical time
            m (float): predicted degree of super-exponential growth
            w (float): predicted scaling ratio of the temporal hierarchy of oscillations
            observations (Mx2 numpy array): the observed data
        Returns:
            nothing, should plot the fit
        """
        lin_vals = self.matrix_equation(observations, tc, m, w)

        a = float(lin_vals[0])
        b = float(lin_vals[1])
        c1 = float(lin_vals[2])
        c2 = float(lin_vals[3])
        lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in observations[0]]
        original_observations = np.log(observations[1]) if self.use_ln else observations[1]

        data = pd.DataFrame({
            'Time': observations[0],
            'LPPLS Fit': lppls_fit,
            'Observations': original_observations,
        })
        data = data.set_index('Time')
        data.plot(figsize=(14, 8))

    def mp_compute_indicator(self, workers, window_size=80, smallest_window_size=20, increment=5, max_searches=25):
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0, :]) - window_size

        func = self._func_compute_indicator
        func_arg_map = [(
            obs_copy[:, i:window_size + i],  # obs
            i,                               # n_iter
            window_size,                     # window_size
            smallest_window_size,            # smallest_window_size
            increment,                       # increment
            max_searches                     # max_searches
        ) for i in range(obs_copy_len)]

        pool = multiprocessing.Pool(processes=workers)
        result = pool.map(func, func_arg_map)
        pool.close()

        return result

    def _func_compute_indicator(self, args):

        obs, n_iter, window_size, smallest_window_size, increment, max_searches = args

        n_fits = (window_size - smallest_window_size) // increment

        cofs = []

        # create reasonable bounds for critical time initialization
        t_first = obs[0, 0]
        t_last = obs[0, -1]
        pct_delta = (t_last - t_first) * 0.20
        tc_init_min = t_last - pct_delta
        tc_init_max = t_last + pct_delta

        # set random initialization limits for non-linear params
        init_limits = [
            (tc_init_min, tc_init_max),  # tc : Critical Time
            (0.1, 0.9),  # m : 0.1 ≤ m ≤ 0.9
            (6, 13),  # ω : 6 ≤ ω ≤ 13
        ]

        # run n fits on the observation slice.
        for j in range(n_fits):
            obs_shrinking_slice = obs[:, j * increment:window_size + n_iter]

            # fit the model to the data and get back the params
            tc, m, w, a, b, c = self.fit(obs_shrinking_slice, max_searches, init_limits, minimizer='Nelder-Mead')

            t_len = len(obs_shrinking_slice)
            # filtering conditions
            # @TODO - make filtering conditions configurable

            first = obs_shrinking_slice[0][0]
            last = obs_shrinking_slice[0][-1]
            pct_delta_05 = (last - first) * 0.05
            pct_delta_1 = (last - first) * 0.1
            tc_init_min = last - pct_delta_05
            tc_init_max = last + pct_delta_1

            tc_in_range = t_len - (tc_init_min) < tc < t_len + (tc_init_max)
            m_in_range = 0.01 < m < 1.2
            w_in_range = 2 < w < 25
            n_oscillation = ((w / 2) * np.log(abs((tc - first) / (last - first)))) > 2.5
            # for bubble end flag
            damping_bef = (m * abs(b)) / (w * abs(c)) > 0.8
            # for bubble early warning
            damping_bew = (m * abs(b)) / (w * abs(c)) > 0.0

            if tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bef:
                bef = True
            else:
                bef = False

            if tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bew:
                bew = True
            else:
                bew = False

            median = stats.median(pd.Series(obs_shrinking_slice[1, :]).pct_change().cumsum())
            median_sign = 1 if median > 0 else -1

            cofs.append({
                'tc': tc,
                'm': m,
                'w': w,
                'a': a,
                'b': b,
                'c': c,
                'bef': bef,
                'bew': bew,
                'median_sign': median_sign
            })

            # # visualize the fit
            # self.plot_many_fit(obs_shrinking_slice, tc, m, w, overlay=True,
            #               path_out='/Users/joshnielsen/Desktop/newfits_testing/{}-{}.png'.format(n_iter, j))
        return cofs
