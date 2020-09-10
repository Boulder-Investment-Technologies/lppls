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
        assert isinstance(observations, (np.ndarray, pd.DataFrame)), \
            f'Expected observations to be <pd.DataFrame> or <np.ndarray>, got :{type(observations)}'

        self.observations = observations
        self.coef_ = {}

    def lppls(self, t, tc, m, w, a, b, c1, c2):
        return a + np.power(tc - t, m) * (b + ((c1 * np.cos(w * (tc - t))) + (c2 * np.sin(w * (tc - t)))))

    def func_restricted(self, x, *args):
        """
        Finds the least square difference.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        Args:
            x(np.ndarray):  1-D array with shape (n,).
            args:           Tuple of the fixed parameters needed to completely specify the function.
        Returns:
            (float)
        """

        tc = x[0]
        m = x[1]
        w = x[2]
        obs = args[0]

        a, b, c1, c2 = self.matrix_equation(obs, tc, m, w).astype('float').tolist()

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in obs[0, :]]
        delta = np.subtract(delta, obs[1, :])
        delta = np.power(delta, 2)

        return np.sum(delta)

    def matrix_equation(self, observations, tc, m, w):
        """
        Derive linear parameters in LPPLs from nonlinear ones.
        """
        T = observations[0]
        P = observations[1]
        deltaT = tc - T
        phase = deltaT
        fi = np.power(deltaT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)
        A = np.stack([np.ones_like(deltaT), fi, gi, hi])

        return np.linalg.lstsq(A.T, P, rcond=None)[0]

    def fit(self, observations, max_searches, minimizer='Nelder-Mead', matrix_solution='Estimate'):
        """
        Args:
            observations (Mx2 numpy array): the observed time-series data.
            max_searches (int): The maxi amount of searches to perform before giving up. The literature suggests 25.
            minimizer (str): See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            matrix_solution (str): Solve or Estimate. Whether to solve the matrix equation directly to obtain the linear
                params or estimate via least square solution.

        Returns:
            tc, m, w, a, b, c1, c2
        """
        matrix_func = self.solve_matrix_equation if matrix_solution == 'Solve' else self.matrix_equation
        search_count = 0
        # find bubble
        while search_count < max_searches:
            tc_init_min, tc_init_max = self._get_tc_bounds(observations, 0.20, 0.20)

            # set random initialization limits for non-linear params
            init_limits = [
                (tc_init_min, tc_init_max),  # tc : Critical Time
                (0.1, 0.9),  # m : 0.1 ≤ m ≤ 0.9
                (6, 13),  # ω : 6 ≤ ω ≤ 13
            ]

            # randomly choose vals within bounds for non-linear params
            non_lin_vals = [random.uniform(a[0], a[1]) for a in init_limits]

            tc = non_lin_vals[0]
            m = non_lin_vals[1]
            w = non_lin_vals[2]
            seed = np.array([tc, m, w])

            # Increment search count on SVD convergence error, but raise all other exceptions.
            try:
                tc, m, w, a, b, c = self.minimize(observations, seed, minimizer, matrix_func)
                return tc, m, w, a, b, c
            except (np.linalg.LinAlgError, UnboundLocalError):
                search_count += 1

        return 0, 0, 0, 0, 0, 0

    def minimize(self, observations, seed, minimizer, matrix_func):
        """
        Args:
            observations (np.ndarray):  the observed time-series data.
            seed (list):  time-critical, omega, and m.
            minimizer (str):  See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            matrix_func (func): The function used to estimate or solve linear params a b, c1, c2.
        Returns:
            tc, m, w, a, b, c
        """

        cofs = minimize(
            args=observations,
            fun=self.func_restricted,
            x0=seed,
            method=minimizer
        )

        if cofs.success:

            tc = cofs.x[0]
            m = cofs.x[1]
            w = cofs.x[2]

            a, b, c1, c2 = matrix_func(observations, tc, m, w).astype('float').tolist()
            c = (c1 ** 2 + c2 ** 2) ** 0.5

            # Use sklearn format for storing fit params
            for coef in ['tc', 'm', 'w', 'a', 'b', 'c']:
                self.coef_[coef] = eval(coef)
            return tc, m, w, a, b, c
        else:
            raise UnboundLocalError

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

        a, b, c1, c2 = self.matrix_equation(observations, tc, m, w).astype('float').tolist()
        lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in observations[0]]
        original_observations = observations[1]

        data = pd.DataFrame({
            'Time': observations[0],
            'LPPLS Fit': lppls_fit,
            'Observations': original_observations,
        })
        data = data.set_index('Time')
        data.plot(figsize=(14, 8))

    def mp_compute_indicator(self, workers, window_size=80, smallest_window_size=20, increment=5, max_searches=25,
                             filter_conditions=[]):
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0, :]) - window_size

        func = self._func_compute_indicator
        func_arg_map = [(
            obs_copy[:, i:window_size + i],  # obs
            i,  # n_iter
            window_size,  # window_size
            smallest_window_size,  # smallest_window_size
            increment,  # increment
            max_searches, # max_searches
            filter_conditions,
        ) for i in range(obs_copy_len)]

        pool = multiprocessing.Pool(processes=workers)
        result = pool.map(func, func_arg_map)
        pool.close()

        return result

    def _func_compute_indicator(self, args):

        obs, n_iter, window_size, smallest_window_size, increment, max_searches, filter_conditions = args

        n_fits = (window_size - smallest_window_size) // increment

        cofs = []

        # run n fits on the observation slice.
        for j in range(n_fits):
            obs_shrinking_slice = obs[:, j * increment:window_size + n_iter]

            # fit the model to the data and get back the params
            tc, m, w, a, b, c = self.fit(obs_shrinking_slice, max_searches, minimizer='Nelder-Mead')

            first = obs_shrinking_slice[0][0]
            last = obs_shrinking_slice[0][-1]

            qualified = {}
            # filter_conditions = [
            #   {'condition_1':[tc_range, m_range, w_range, O_min, D_min]},
            #   {'condition_2':[tc_range, m_range, w_range, O_range, D_range]}
            # ]
            for condition in filter_conditions:
                for value in condition:
                    tc_min, tc_max = condition[value][0]
                    m_min, m_max = condition[value][1]
                    w_min, w_max = condition[value][2]
                    O_min = condition[value][3]
                    D_min = condition[value][4]

                    tc_init_min, tc_init_max = self._get_tc_bounds(obs_shrinking_slice, tc_min, tc_max)

                    tc_in_range = last - tc_init_min < tc < last + tc_init_max
                    m_in_range = m_min < m < m_max
                    w_in_range = w_min < w < w_max

                    O_in_range = ((w / (2 * np.pi)) * np.log(abs(tc / (tc - last)))) > O_min

                    D_in_range = (m * abs(b)) / (w * abs(c)) > D_min if m > 0 and w > 0 else False

                    if tc_in_range and m_in_range and w_in_range and O_in_range and D_in_range:
                        is_qualified = True
                    else:
                        is_qualified = False

                    qualified[condition_name] = is_qualified

            sign = 1 if b < 0 else -1

            cofs.append({
                'tc': tc,
                'm': m,
                'w': w,
                'a': a,
                'b': b,
                'c': c,
                'qualified': qualified,
                'sign': sign,
                't1': first,
                't2': last,
            })

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

    def solve_matrix_equation(self, observations, tc, m, w):
        '''
        Solve the matrix equation using the Filimonov trick
        '''
        time = observations[0, :]
        obs = observations[1, :]
        N = len(obs)
        zeros = np.array([0, 0, 0, 0])

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

            matrix_1_is_not_inf_or_nan = not np.isinf(matrix_1).any() and not np.isnan(matrix_1).any()
            matrix_2_is_not_inf_or_nan = not np.isinf(matrix_2).any() and not np.isnan(matrix_2).any()

            if matrix_1_is_not_inf_or_nan and matrix_2_is_not_inf_or_nan:
                inverse = np.linalg.pinv(matrix_1)
                product = inverse * matrix_2
                return product
            return zeros

        except Exception as e:
            print('matrix_equation failed: {}'.format(e))

        return zeros

    # matrix helpers
    def _yi(self, price_series):
        return [p for p in price_series]

    def _fi(self, tc, m, time_series):
        return [np.power((tc - t), m) for t in time_series]

    def _gi(self, tc, m, w, time_series):
        return [np.power((tc - t), m) * np.cos(w * (tc - t)) for t in time_series]

    def _hi(self, tc, m, w, time_series):
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