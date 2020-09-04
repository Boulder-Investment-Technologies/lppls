import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize


class LPPLS(object):

    def __init__(self, observations):
        """
        Args:
            observations (np.array,pd.DataFrame): 2xM matrix with timestamp and observed value.
        """
        assert isinstance(observations, (np.ndarray,pd.DataFrame)), \
            f'Expected observations to be <pd.DataFrame> or <np.array>, got :{type(observations)}'

        self.observations = observations

    def lppls(self, t, tc, m, w, a, b, c1, c2):
        return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * (tc - t))) + (c2 * np.sin(w * (tc - t)))))

    def func_restricted(self, x, *args):
        '''
        Finds the least square difference.
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
        Derive linear parameters in LPPLs from nonlinear ones.
        '''
        T = observations[0]
        P = observations[1]
        deltaT = tc - T
        phase = deltaT
        fi = np.power(deltaT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)
        A = np.stack([np.ones_like(deltaT), fi, gi, hi])

        return np.linalg.lstsq(A.T, P, rcond=None)[0]

    def fit(self, observations, max_searches, minimizer='Nelder-Mead'):
        """
        Args:
            observations (Mx2 numpy array): the observed data
            max_searches (int): The maxi amount of searches to perform before giving up. The literature suggests 25
            minimizer (str): See list of valid methods to pass to scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize

        Returns:
            tc, m, w, a, b, c1, c2
        """
        search_count = 0
        # find bubble
        while search_count < max_searches:
            tc_init_min, tc_init_max = self._get_tc_bounds(observations, 0.20, 0.20)

            # set random initialization limits for non-linear params
            init_limits = [
                (tc_init_min, tc_init_max),  # tc : Critical Time
                (0.1, 0.9),                  # m : 0.1 ≤ m ≤ 0.9
                (6, 13),                     # ω : 6 ≤ ω ≤ 13
            ]

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

                    c = (c1 ** 2 + c2 ** 2) ** 0.5

                    # @TODO save these as variables local to the class so you can access them again
                    return tc, m, w, a, b, c

                else:
                    search_count += 1
            except Exception as e:
                print('minimize failed: {}'.format(e))
                search_count += 1
        return 0, 0, 0, 0, 0, 0

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
        original_observations = observations[1]

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

        # run n fits on the observation slice.
        for j in range(n_fits):
            obs_shrinking_slice = obs[:, j * increment:window_size + n_iter]

            # fit the model to the data and get back the params
            tc, m, w, a, b, c = self.fit(obs_shrinking_slice, max_searches, minimizer='Nelder-Mead')

            t_len = len(obs_shrinking_slice)
            # filtering conditions
            # @TODO - make filtering conditions configurable

            first = obs_shrinking_slice[0][0]
            last = obs_shrinking_slice[0][-1]
            tc_init_min, tc_init_max = self._get_tc_bounds(obs_shrinking_slice, 0.05, 0.10)

            tc_in_range = last - tc_init_min < tc < last + tc_init_max
            m_in_range = 0.01 < m < 1.2
            w_in_range = 2 < w < 25

            # n_oscillation = ((w / 2) * np.log(abs((tc - first) / (last - first)))) > 2.5
            # Use filtering conditions as implemented in the R 'bubble' package by Dean Fantazzini
            # (https://github.com/Boulder-Investment-Technologies/lppls/issues/url)
            n_oscillation = ((w / (2 * np.pi)) * np.log(abs(tc / (tc - last)))) > 2.5

            # for bubble end flag
            damping_bef = (m * abs(b)) / (w * abs(c)) > 0.8 if m > 0 and w > 0 else False
            # for bubble early warning
            damping_bew = (m * abs(b)) / (w * abs(c)) > 0.0 if m > 0 and w > 0 else False

            if tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bef:
                bef = True
            else:
                bef = False

            if tc_in_range and m_in_range and w_in_range and n_oscillation and damping_bew:
                bew = True
            else:
                bew = False

            cum_pct_change = pd.Series(obs_shrinking_slice[1, :]).pct_change().cumsum()

            # Compute the median, while ignoring NaNs (pct_change introduces a NaN).
            median = np.nanmedian(cum_pct_change)
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
                'median_sign': median_sign,
                't1': first,
                't2': last,
            })

            # # visualize the fit
            # self.plot_many_fit(obs_shrinking_slice, tc, m, w, overlay=True,
            #               path_out='/Users/joshnielsen/Desktop/newfits_testing/{}-{}.png'.format(n_iter, j))
        return cofs

    def _get_tc_bounds(self, obs, lower_bound_pct, upper_bound_pct):
        """
        Args:
            obs (Mx2 numpy array): the observed data
            lower_bound_pct (float): percent of (t_2 - t_1) to use as the LOWER bound initial value for the optimization
            upper_bound_pct (float): percent of (t_2 - t_1) to use as the UPPER bound initial value for the optimization
        Returns:
            tc_init_min, tc_init_max
        """
        t_first = obs[0, 0]
        t_last = obs[0, -1]
        t_delta = t_last - t_first
        pct_delta_min = t_delta * lower_bound_pct
        pct_delta_max = t_delta * upper_bound_pct
        tc_init_min = t_last - pct_delta_min
        tc_init_max = t_last + pct_delta_max
        return tc_init_min, tc_init_max
