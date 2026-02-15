from __future__ import annotations

from multiprocessing import Pool
from matplotlib import pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import random
from datetime import datetime as date
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import xarray as xr
from typing import Any
import warnings


class LPPLS:
    def __init__(self, observations: np.ndarray | pd.DataFrame) -> None:
        """
        Args:
            observations (np.ndarray | pd.DataFrame): 2xM matrix with timestamp and observed value.
        """
        assert isinstance(observations, (np.ndarray, pd.DataFrame)), (
            f"Expected observations to be <pd.DataFrame> or <np.ndarray>, got :{type(observations)}"
        )

        self.observations: np.ndarray | pd.DataFrame = observations
        self.coef_: dict[str, float] = {}
        self.indicator_result: list[dict[str, Any]] = []

    @staticmethod
    @njit
    def lppls(
        t: float | np.ndarray,
        tc: float,
        m: float,
        w: float,
        a: float,
        b: float,
        c1: float,
        c2: float,
    ) -> float | np.ndarray:
        dt = np.abs(tc - t) + 1e-8
        return a + np.power(dt, m) * (
            b + ((c1 * np.cos(w * np.log(dt))) + (c2 * np.sin(w * np.log(dt))))
        )

    def func_restricted(self, x: np.ndarray, *args: np.ndarray) -> float:
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
        observations = args[0]

        rM = self.matrix_equation(observations, tc, m, w)
        a, b, c1, c2 = rM[:, 0].tolist()
        # print('type', type(res))
        # print('func_restricted', res)

        delta = self.lppls(observations[0, :], tc, m, w, a, b, c1, c2)
        delta = np.subtract(delta, observations[1, :])
        delta = np.power(delta, 2)
        return np.sum(delta)

    @staticmethod
    @njit
    def matrix_equation(
        observations: np.ndarray, tc: float, m: float, w: float
    ) -> np.ndarray:
        """
        Derive linear parameters in LPPLs from nonlinear ones.
        """
        T = observations[0]
        P = observations[1]
        N = len(T)

        dT = np.abs(tc - T) + 1e-8
        phase = np.log(dT)

        fi = np.power(dT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)

        fi_pow_2 = np.power(fi, 2)
        gi_pow_2 = np.power(gi, 2)
        hi_pow_2 = np.power(hi, 2)

        figi = np.multiply(fi, gi)
        fihi = np.multiply(fi, hi)
        gihi = np.multiply(gi, hi)

        yi = P
        yifi = np.multiply(yi, fi)
        yigi = np.multiply(yi, gi)
        yihi = np.multiply(yi, hi)

        matrix_1 = np.array(
            [
                [N, np.sum(fi), np.sum(gi), np.sum(hi)],
                [np.sum(fi), np.sum(fi_pow_2), np.sum(figi), np.sum(fihi)],
                [np.sum(gi), np.sum(figi), np.sum(gi_pow_2), np.sum(gihi)],
                [np.sum(hi), np.sum(fihi), np.sum(gihi), np.sum(hi_pow_2)],
            ]
        )

        matrix_2 = np.array(
            [[np.sum(yi)], [np.sum(yifi)], [np.sum(yigi)], [np.sum(yihi)]]
        )

        matrix_1 += 1e-8 * np.eye(matrix_1.shape[0])

        return np.linalg.solve(matrix_1, matrix_2)

    def fit(
        self,
        max_searches: int,
        minimizer: str = "Nelder-Mead",
        obs: np.ndarray | None = None,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        """
        Args:
            max_searches (int): The maxi amount of searches to perform before giving up. The literature suggests 25.
            minimizer (str): See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            obs (np.ndarray | None): the observed time-series data. Optional, if not included will use self.observations
        Returns:
            tuple: (tc, m, w, a, b, c, c1, c2, O, D)
        """
        if obs is None:
            obs = self.observations

        search_count = 0
        # find bubble
        while search_count < max_searches:
            # tc_init_min, tc_init_max = self._get_tc_bounds(obs, 0.50, 0.50)
            t1 = obs[0, 0]
            t2 = obs[0, -1]

            # @TODO make configurable
            # set random initialization limits for non-linear params
            init_limits = [
                # (tc_init_min, tc_init_max),
                (t2 - 0.2 * (t2 - t1), t2 + 0.2 * (t2 - t1)),  # tc
                (0.1, 1.0),  # m
                (6.0, 13.0),  # ω
            ]

            # randomly choose vals within bounds for non-linear params
            non_lin_vals = [random.uniform(a[0], a[1]) for a in init_limits]

            tc = non_lin_vals[0]
            m = non_lin_vals[1]
            w = non_lin_vals[2]
            seed = np.array([tc, m, w])

            # Increment search count on SVD convergence error, but raise all other exceptions.
            try:
                tc, m, w, a, b, c, c1, c2 = self.estimate_params(obs, seed, minimizer)
                O = self.get_oscillations(w, tc, t1, t2)
                D = self.get_damping(m, w, b, c)
                if not np.isfinite(O) or not np.isfinite(D):
                    search_count += 1
                    continue
                return tc, m, w, a, b, c, c1, c2, O, D
            except Exception:
                search_count += 1
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def estimate_params(
        self, observations: np.ndarray, seed: np.ndarray, minimizer: str
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """
        Args:
            observations (np.ndarray):  the observed time-series data.
            seed (np.ndarray):  initial values for tc, m, and w.
            minimizer (str):  See list of valid methods to pass to scipy.optimize.minimize:
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        Returns:
            tuple: (tc, m, w, a, b, c, c1, c2)
        """

        cofs = minimize(
            args=observations, fun=self.func_restricted, x0=seed, method=minimizer
        )

        if cofs.success:
            tc = cofs.x[0]
            m = cofs.x[1]
            w = cofs.x[2]
            # r =
            # m_f =

            rM = self.matrix_equation(observations, tc, m, w)
            a, b, c1, c2 = rM[:, 0].tolist()

            c = self.get_c(c1, c2)

            # Use sklearn format for storing fit params
            # @TODO only save when running single fits.
            for coef in ["tc", "m", "w", "a", "b", "c", "c1", "c2"]:
                self.coef_[coef] = eval(coef)
            return tc, m, w, a, b, c, c1, c2
        else:
            raise UnboundLocalError

    def plot_fit(self, show_tc: bool = False) -> None:
        """
        Args:
            observations (Mx2 numpy array): the observed data
        Returns:
            nothing, should plot the fit
        """
        tc, m, w, a, b, c, c1, c2 = self.coef_.values()
        time_ord = [
            pd.Timestamp.fromordinal(d) for d in self.observations[0, :].astype("int32")
        ]
        t_obs = self.observations[0, :]
        # ts = pd.to_datetime(t_obs*10**9)
        # compatible_date = np.array(ts, dtype=np.datetime64)

        lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in t_obs]
        price = self.observations[1, :]

        first = t_obs[0]
        last = t_obs[-1]

        _O = (w / (2.0 * np.pi)) * np.log((tc - first) / (tc - last))  # noqa: F841
        _D = (m * np.abs(b)) / (w * np.abs(c))  # noqa: F841

        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(14, 8))
        # fig.suptitle(
        #     'Single Fit\ntc: {:.2f}, m: {:.2f}, w: {:.2f}, a: {:.2f}, b: {:.2f}, c: {:.2f}, O: {:.2f}, D: {:.2f}'.format(tc, m, w, a, b, c, O, D),
        #     fontsize=16)

        ax1.plot(time_ord, price, label="price", color="black", linewidth=0.75)
        ax1.plot(time_ord, lppls_fit, label="lppls fit", color="blue", alpha=0.5)
        # if show_tc:
        #     ax1.axvline(x=np.array(tc_ts, dtype=np.datetime64), label='tc={}'.format(ts), color='red', alpha=0.5)
        # set grids
        ax1.grid(which="major", axis="both", linestyle="--")
        # set labels
        ax1.set_ylabel("ln(p)")
        ax1.legend(loc=2)

        plt.xticks(rotation=45)
        # ax1.xaxis.set_major_formatter(months)
        # # rotates and right aligns the x labels, and moves the bottom of the
        # # axes up to make room for them
        # fig.autofmt_xdate()

    def compute_indicators(
        self,
        res: list[dict[str, Any]],
        filter_conditions_config: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute LPPLS confidence indicators from nested fit results.

        Processes the raw output of mp_compute_nested_fits (or the
        multiprocessing-based nested fits) and classifies each fit as a
        positive bubble (b < 0) or negative bubble (b > 0). Fits are
        further filtered by quality conditions (tc range, m, w, O, D) to
        produce a confidence score between 0 and 1 for each time step.

        Args:
            res (list[dict]): The result list returned by
                mp_compute_nested_fits. Each element is a dict with keys
                't2', 'p2', and 'res' (a list of individual fit dicts).
            filter_conditions_config (dict, optional): Custom filter
                thresholds. Not yet implemented; when None, the following
                defaults are used:
                  - m: (0.0, 1.0)
                  - w: (2.0, 15.0)
                  - O (oscillations): > 2.5
                  - D (damping): > 0.5

        Returns:
            pd.DataFrame: A DataFrame with columns:
              - 'time': ordinal timestamp for each window end
              - 'price': observed log-price at each window end
              - 'pos_conf': positive bubble confidence (0-1)
              - 'neg_conf': negative bubble confidence (0-1)
              - '_fits': raw fit dicts (with added 'is_qualified' flag)
        """
        pos_conf_lst = []
        neg_conf_lst = []
        price = []
        ts = []
        _fits = []

        if filter_conditions_config is None:
            # TODO make configurable again!
            m_min, m_max = (0.0, 1.0)
            w_min, w_max = (2.0, 15.0)
            O_min = 2.5
            D_min = 0.5
        else:
            # TODO parse user provided conditions
            pass

        for r in res:
            ts.append(r["t2"])
            price.append(r["p2"])
            pos_qual_count = 0
            neg_qual_count = 0
            pos_count = 0
            neg_count = 0
            # _fits.append(r['res'])

            for idx, fits in enumerate(r["res"]):
                t1 = fits["t1"]
                t2 = fits["t2"]
                tc = fits["tc"]
                m = fits["m"]
                w = fits["w"]
                b = fits["b"]
                c = fits["c"]
                O = fits["O"]
                D = fits["D"]

                # t_delta = t2 - t1
                # pct_delta_min = t_delta * 0.5
                # pct_delta_max = t_delta * 0.5
                # tc_min = t2 - pct_delta_min
                # tc_max = t2 + pct_delta_max

                # [max(t2 - 60, t2 - 0.5 * (t2 - t1)), min(252, t2 + 0.5 * (t2 - t1))]

                # print('lb: max({}, {})={}'.format(t2 - 60, t2 - 0.5 * (t2 - t1), max(t2 - 60, t2 - 0.5 * (t2 - t1))))
                # print('ub: min({}, {})={}'.format(t2 + 252, t2 + 0.5 * (t2 - t1), min(t2 + 252, t2 + 0.5 * (t2 - t1))))
                #
                # print('{} < {} < {}'.format(max(t2 - 60, t2 - 0.5 * (t2 - t1)), tc, min(t2 + 252, t2 + 0.5 * (t2 - t1))))
                # print('______________')

                tc_in_range = (
                    max(t2 - 60, t2 - 0.5 * (t2 - t1))
                    < tc
                    < min(t2 + 252, t2 + 0.5 * (t2 - t1))
                )
                m_in_range = m_min < m < m_max
                w_in_range = w_min < w < w_max

                if b != 0 and c != 0:
                    O = O
                else:
                    O = np.inf

                O_in_range = O > O_min
                D_in_range = D > D_min  # if m > 0 and w > 0 else False

                if (
                    tc_in_range
                    and m_in_range
                    and w_in_range
                    and O_in_range
                    and D_in_range
                ):
                    is_qualified = True
                else:
                    is_qualified = False

                if b < 0:
                    pos_count += 1
                    if is_qualified:
                        pos_qual_count += 1
                if b > 0:
                    neg_count += 1
                    if is_qualified:
                        neg_qual_count += 1
                # add this to res to make life easier
                r["res"][idx]["is_qualified"] = is_qualified

            _fits.append(r["res"])

            pos_conf = pos_qual_count / pos_count if pos_count > 0 else 0
            neg_conf = neg_qual_count / neg_count if neg_count > 0 else 0
            pos_conf_lst.append(pos_conf)
            neg_conf_lst.append(neg_conf)

            # pos_lst.append(pos_count / (pos_count + neg_count))
            # neg_lst.append(neg_count / (pos_count + neg_count))

            # tc_lst.append(tc_cnt)
            # m_lst.append(m_cnt)
            # w_lst.append(w_cnt)
            # O_lst.append(O_cnt)
            # D_lst.append(D_cnt)

        res_df = pd.DataFrame(
            {
                "time": ts,
                "price": price,
                "pos_conf": pos_conf_lst,
                "neg_conf": neg_conf_lst,
                "_fits": _fits,
            }
        )
        return res_df
        # return ts, price, pos_lst, neg_lst, pos_conf_lst, neg_conf_lst, #tc_lst, m_lst, w_lst, O_lst, D_lst

    def plot_confidence_indicators(self, res: list[dict[str, Any]]) -> None:
        """
        Args:
            res (list): result from mp_compute_indicator
            condition_name (str): the name you assigned to the filter condition in your config
            title (str): super title for both subplots
        Returns:
            nothing, should plot the indicator
        """
        res_df = self.compute_indicators(res)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 10))

        ord = res_df["time"].astype("int32")
        ts = [pd.Timestamp.fromordinal(d) for d in ord]

        # plot pos bubbles
        ax1_0 = ax1.twinx()
        ax1.plot(ts, res_df["price"], color="black", linewidth=0.75)
        # ax1_0.plot(compatible_date, pos_lst, label='pos bubbles', color='gray', alpha=0.5)
        ax1_0.plot(
            ts,
            res_df["pos_conf"],
            label="bubble indicator (pos)",
            color="red",
            alpha=0.5,
        )

        # plot neg bubbles
        ax2_0 = ax2.twinx()
        ax2.plot(ts, res_df["price"], color="black", linewidth=0.75)
        # ax2_0.plot(compatible_date, neg_lst, label='neg bubbles', color='gray', alpha=0.5)
        ax2_0.plot(
            ts,
            res_df["neg_conf"],
            label="bubble indicator (neg)",
            color="green",
            alpha=0.5,
        )

        # if debug:
        #     ax3.plot(ts, tc_lst, label='tc count')
        #     ax3.plot(ts, m_lst, label='m count')
        #     ax3.plot(ts, w_lst, label='w count')
        #     ax3.plot(ts, O_lst, label='O count')
        #     ax3.plot(ts, D_lst, label='D count')

        # set grids
        ax1.grid(which="major", axis="both", linestyle="--")
        ax2.grid(which="major", axis="both", linestyle="--")

        # set labels
        ax1.set_ylabel("ln(p)")
        ax2.set_ylabel("ln(p)")

        ax1_0.set_ylabel("bubble indicator (pos)")
        ax2_0.set_ylabel("bubble indicator (neg)")

        ax1_0.legend(loc=2)
        ax2_0.legend(loc=2)

        plt.xticks(rotation=45)
        # format the ticks
        # ax1.xaxis.set_major_locator(years)
        # ax2.xaxis.set_major_locator(years)
        # ax1.xaxis.set_major_formatter(years_fmt)
        # ax2.xaxis.set_major_formatter(years_fmt)
        # ax1.xaxis.set_minor_locator(months)
        # ax2.xaxis.set_minor_locator(months)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

    def mp_compute_nested_fits(
        self,
        workers: int,
        window_size: int = 80,
        smallest_window_size: int = 20,
        outer_increment: int = 5,
        inner_increment: int = 2,
        max_searches: int = 25,
        filter_conditions_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Compute nested LPPLS fits across the time series using multiprocessing.

        This is the parallelized version of compute_nested_fits. It slides a
        fixed-size window across the observation data and, at each window
        position, fits the LPPLS model on progressively smaller sub-windows.
        The collection of fits is used to build the LPPLS confidence indicator
        (see plot_confidence_indicators / compute_indicators).

        The nested fitting procedure works as follows:
          1. A window of ``window_size`` observations is placed at the start of
             the time series and slid forward by ``outer_increment`` data points
             at each step.
          2. At each window position, the model is fit on sub-windows that
             shrink from ``window_size`` down to ``smallest_window_size`` by
             removing ``inner_increment`` data points from the left of the
             window at each inner step.
          3. Each individual fit uses up to ``max_searches`` random
             initializations of the non-linear parameters (tc, m, w).

        Performance notes:
          - **Computation time** scales roughly as
            ``(N / outer_increment) * ((window_size - smallest_window_size) /
            inner_increment) * max_searches``, where N is the length of the
            time series. Increasing any of these values will increase runtime.
          - Using more ``workers`` (up to the number of CPU cores) will speed
            up computation proportionally.
          - Larger ``outer_increment`` and ``inner_increment`` values reduce
            the number of fits but may lower the resolution / quality of the
            confidence indicator.

        Args:
            workers (int): Number of parallel worker processes to use.
                A good default is the number of available CPU cores.
            window_size (int): The number of observations in the sliding
                window. This is the largest fitting window used and defines
                the outer time scale of the analysis. Larger values capture
                longer-horizon bubbles but increase computation time.
                Default: 80.
            smallest_window_size (int): The smallest sub-window (in number of
                observations) to fit within each window position. This defines
                the shortest time horizon analysed. Must be smaller than
                ``window_size``. Default: 20.
            outer_increment (int): Step size (in number of data points) for
                sliding the main window across the time series. A value of 1
                means the window moves one observation at a time (highest
                resolution but slowest). Larger values skip observations and
                speed up computation at the cost of temporal resolution.
                Default: 5.
            inner_increment (int): Step size (in number of data points) for
                shrinking the sub-window within each window position. Controls
                the granularity of the multi-scale analysis. Smaller values
                produce more fits per window position. Default: 2.
            max_searches (int): Maximum number of random initializations to
                attempt when fitting the LPPLS model for each sub-window. The
                literature suggests 25. Higher values improve the chance of
                finding a good fit but increase computation time. Default: 25.
            filter_conditions_config (dict): Reserved for future use.
                Not implemented in 0.6.x.

        Returns:
            list[dict]: A list of result dicts, one per outer window position.
                Each dict contains:
                  - 't1' (float): ordinal timestamp of the window start
                  - 't2' (float): ordinal timestamp of the window end
                  - 'p2' (float): observed log-price at the window end
                  - 'res' (list[dict]): list of fit results for each
                    sub-window, each containing fitted parameters
                    (tc, m, w, a, b, c, c1, c2, t1, t2, O, D).
        """
        obs_copy = self.observations
        obs_opy_len = len(obs_copy[0]) - window_size
        func = self._func_compute_nested_fits

        # print('obs_copy', obs_copy)
        # print('obs_opy_len', obs_opy_len)

        func_arg_map = [
            (
                obs_copy[:, i : window_size + i],
                window_size,
                i,
                smallest_window_size,
                outer_increment,
                inner_increment,
                max_searches,
            )
            for i in range(0, obs_opy_len + 1, outer_increment)
        ]

        with Pool(processes=workers) as pool:
            self.indicator_result = list(
                tqdm(pool.imap(func, func_arg_map), total=len(func_arg_map))
            )

        return self.indicator_result

    def compute_nested_fits(
        self,
        window_size: int = 80,
        smallest_window_size: int = 20,
        outer_increment: int = 5,
        inner_increment: int = 2,
        max_searches: int = 25,
    ) -> xr.DataArray:
        """Compute nested LPPLS fits across the time series (single-process).

        This is the single-threaded version of mp_compute_nested_fits. It
        performs the same nested fitting procedure but runs sequentially
        without multiprocessing. Use mp_compute_nested_fits for faster
        execution on multi-core machines.

        The nested fitting procedure works as follows:
          1. A window of ``window_size`` observations is placed at the start of
             the time series and slid forward by ``outer_increment`` data points
             at each step.
          2. At each window position, the model is fit on sub-windows that
             shrink from ``window_size`` down to ``smallest_window_size`` by
             removing ``inner_increment`` data points from the left of the
             window at each inner step.
          3. Each individual fit uses up to ``max_searches`` random
             initializations of the non-linear parameters (tc, m, w).

        Args:
            window_size (int): The number of observations in the sliding
                window. This is the largest fitting window used and defines
                the outer time scale of the analysis. Larger values capture
                longer-horizon bubbles but increase computation time.
                Default: 80.
            smallest_window_size (int): The smallest sub-window (in number of
                observations) to fit within each window position. This defines
                the shortest time horizon analysed. Must be smaller than
                ``window_size``. Default: 20.
            outer_increment (int): Step size (in number of data points) for
                sliding the main window across the time series. A value of 1
                means the window moves one observation at a time (highest
                resolution but slowest). Larger values skip observations and
                speed up computation at the cost of temporal resolution.
                Default: 5.
            inner_increment (int): Step size (in number of data points) for
                shrinking the sub-window within each window position. Controls
                the granularity of the multi-scale analysis. Smaller values
                produce more fits per window position. Default: 2.
            max_searches (int): Maximum number of random initializations to
                attempt when fitting the LPPLS model for each sub-window. The
                literature suggests 25. Higher values improve the chance of
                finding a good fit but increase computation time. Default: 25.

        Returns:
            xr.DataArray: A 3D xarray DataArray with dimensions
                (t2, windowsizes, params) containing fitted parameters for
                each combination of window position and sub-window size.
        """
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0]) - window_size
        window_delta = window_size - smallest_window_size
        res = []
        i_idx = 0
        for i in range(0, obs_copy_len + 1, outer_increment):
            j_idx = 0
            obs = obs_copy[:, i : window_size + i]
            t1 = obs[0][0]
            t2 = obs[0][-1]
            res.append([])
            i_idx += 1
            for j in range(0, window_delta, inner_increment):
                obs_shrinking_slice = obs[:, j:window_size]
                tc, m, w, a, b, c, _, _, _, _ = self.fit(
                    max_searches, obs=obs_shrinking_slice
                )
                res[i_idx - 1].append([])
                j_idx += 1
                for k in [t2, t1, a, b, c, m, 0, tc]:
                    res[i_idx - 1][j_idx - 1].append(k)
        return xr.DataArray(
            data=res,
            dims=("t2", "windowsizes", "params"),
            coords=dict(
                t2=obs_copy[0][(window_size - 1) :: outer_increment],
                windowsizes=range(smallest_window_size, window_size, inner_increment),
                params=["t2", "t1", "a", "b", "c", "m", "0", "tc"],
            ),
        )

    def _func_compute_nested_fits(
        self, args: tuple[np.ndarray, int, int, int, int, int, int]
    ) -> dict[str, Any]:
        """Worker function for mp_compute_nested_fits.

        Fits the LPPLS model on progressively smaller sub-windows within a
        single window position. This method is called in parallel by
        mp_compute_nested_fits via multiprocessing.Pool.

        Args:
            args (tuple): A tuple of (obs, window_size, n_iter,
                smallest_window_size, outer_increment, inner_increment,
                max_searches) where:
                  - obs (np.ndarray): 2xN observation slice for this window
                  - window_size (int): size of the full window
                  - n_iter (int): current outer iteration index
                  - smallest_window_size (int): minimum sub-window size
                  - outer_increment (int): outer step size (unused here,
                    passed through from the caller)
                  - inner_increment (int): step size for shrinking sub-windows
                  - max_searches (int): max random search attempts per fit

        Returns:
            dict: A dict with keys 't1', 't2', 'p2', and 'res' (a list of
                fit result dicts for each sub-window).
        """

        (
            obs,
            window_size,
            n_iter,
            smallest_window_size,
            outer_increment,
            inner_increment,
            max_searches,
        ) = args

        window_delta = window_size - smallest_window_size

        res = []

        # print('obs', obs)
        t1 = obs[0][0]
        t2 = obs[0][-1]
        p2 = obs[1][-1]

        # tc_init_min, tc_init_max = self._get_tc_bounds(obs_shrinking_slice, tc_min, tc_max)
        #
        # tc_in_range = last - tc_init_min < tc < last + tc_init_max
        # m_in_range = m_min < m < m_max
        # w_in_range = w_min < w < w_max
        # O_in_range = self._is_O_in_range(tc, w, last, O_min)
        # D_in_range = self._is_D_in_range(m, w, b, c, D_min)
        #
        # qualified[value] = tc_in_range and m_in_range and w_in_range and O_in_range and D_in_range

        # run n fits on the observation slice.
        for j in range(0, window_delta, inner_increment):
            obs_shrinking_slice = obs[:, j:window_size]

            # fit the model to the data and get back the params
            if self.__class__.__name__ == "LPPLSCMAES":
                # print('cmaes fit is running!')
                tc, m, w, a, b, c, c1, c2, O, D = self.fit(
                    max_iteration=2500, pop_size=4, obs=obs_shrinking_slice
                )
            else:
                tc, m, w, a, b, c, c1, c2, O, D = self.fit(
                    max_searches, obs=obs_shrinking_slice
                )

            nested_t1 = obs_shrinking_slice[0][0]
            nested_t2 = obs_shrinking_slice[0][-1]

            res.append(
                {
                    # "tc_d": self.ordinal_to_date(tc),
                    "tc": tc,
                    "m": m,
                    "w": w,
                    "a": a,
                    "b": b,
                    "c": c,
                    "c1": c1,
                    "c2": c2,
                    # "t1_d": self.ordinal_to_date(nested_t1),
                    # "t2_d": self.ordinal_to_date(nested_t2),
                    "t1": nested_t1,
                    "t2": nested_t2,
                    "O": O,
                    "D": D,
                }
            )

        # return {'t1': self.ordinal_to_date(t1), 't2': self.ordinal_to_date(t2), 'p2': p2, 'res': res}
        return {"t1": t1, "t2": t2, "p2": p2, "res": res}

    def _get_tc_bounds(
        self, obs: np.ndarray, lower_bound_pct: float, upper_bound_pct: float
    ) -> tuple[float, float]:
        """
        Args:
            obs (Mx2 numpy array): the observed data
            lower_bound_pct (float): percent of (t_2 - t_1) to use as the LOWER bound initial value for the optimization
            upper_bound_pct (float): percent of (t_2 - t_1) to use as the UPPER bound initial value for the optimization
        Returns:
            tc_init_min, tc_init_max
        """
        t_first = obs[0][0]
        t_last = obs[0][-1]
        t_delta = t_last - t_first
        pct_delta_min = t_delta * lower_bound_pct
        pct_delta_max = t_delta * upper_bound_pct
        tc_init_min = t_last - pct_delta_min
        tc_init_max = t_last + pct_delta_max
        return tc_init_min, tc_init_max

    def _is_O_in_range(self, tc: float, w: float, last: float, O_min: float) -> bool:
        return ((w / (2 * np.pi)) * np.log(abs(tc / (tc - last)))) > O_min

    def _is_D_in_range(
        self, m: float, w: float, b: float, c: float, D_min: float
    ) -> bool:
        return False if m <= 0 or w <= 0 else abs((m * b) / (w * c)) > D_min

    def get_oscillations(self, w: float, tc: float, t1: float, t2: float) -> float:
        denom = tc - t2
        numer = tc - t1
        ratio = numer / denom if denom != 0 else 0
        if ratio <= 0:
            return np.nan
        return (w / (2.0 * np.pi)) * np.log(ratio)

    def get_damping(self, m: float, w: float, b: float, c: float) -> float:
        return (m * np.abs(b)) / (w * np.abs(c))

    def get_c(self, c1: float, c2: float) -> float:
        if c1 and c2:
            # c = (c1 ** 2 + c2 ** 2) ** 0.5
            return c1 / np.cos(np.arctan(c2 / c1))
        else:
            return 0

    def ordinal_to_date(self, ordinal: float) -> str:
        # Since pandas represents timestamps in nanosecond resolution,
        # the time span that can be represented using a 64-bit integer
        # is limited to approximately 584 years
        try:
            return date.fromordinal(int(ordinal)).strftime("%Y-%m-%d")
        except (ValueError, OutOfBoundsDatetime):
            return str(pd.NaT)

    def detect_bubble_start_time_via_lagrange(
        self,
        max_window_size: int,
        min_window_size: int,
        step_size: int = 1,
        max_searches: int = 25,
    ) -> dict[str, Any] | None:

        window_sizes = []
        sse_list = []
        ssen_list = []
        lagrange_sse_list = []
        start_times = []
        n_params = 7  # The number of degrees of freedom used for this exercise as well as for the real-world time series is p = 8, which includes the 7 parameters of the LPPLS model augmented by the extra parameter t1

        total_obs = len(self.observations[0])

        lppls_params_list = []

        for window_size in range(max_window_size, min_window_size - 1, -step_size):
            start_idx = total_obs - window_size
            end_idx = total_obs
            obs_window = self.observations[:, start_idx:end_idx]

            start_time = self.observations[0][start_idx]
            start_times.append(start_time)
            try:
                tc, m, w, a, b, _, c1, c2, _, _ = self.fit(max_searches, obs=obs_window)
                if tc == 0.0:
                    continue

                # compute predictions and residuals
                Yhat = self.lppls(obs_window[0], tc, m, w, a, b, c1, c2)
                residuals = obs_window[1] - Yhat

                # compute SSE and normalized SSE
                sse = np.sum(residuals**2)
                n = len(obs_window[0])
                if n - n_params <= 0:
                    continue  # avoid division by zero or negative degrees of freedom
                ssen = sse / (n - n_params)

                window_sizes.append(window_size)
                sse_list.append(sse)
                ssen_list.append(ssen)
                lppls_params_list.append(
                    {
                        "tc": tc,
                        "m": m,
                        "w": w,
                        "a": a,
                        "b": b,
                        "c1": c1,
                        "c2": c2,
                        "obs_window": obs_window,  # may be useful later
                    }
                )
            except Exception as e:
                print(e)
                continue

        if len(ssen_list) < 2:
            warnings.warn("Not enough data points to compute Lagrange regularization.")
            return None

        window_sizes_np = np.array(window_sizes).reshape(-1, 1)
        ssen_list_np = np.array(ssen_list)

        # fit linear regression to normalized SSE vs. window sizes
        reg = LinearRegression().fit(window_sizes_np, ssen_list_np)
        slope = reg.coef_[0]
        _intercept = reg.intercept_  # noqa: F841

        # compute Lagrange-regularized SSE
        for i in range(len(sse_list)):
            lagrange_sse = ssen_list[i] - slope * window_sizes[i]
            lagrange_sse_list.append(lagrange_sse)

        # find the optimal window size
        min_index = np.argmin(lagrange_sse_list)
        optimal_window_size = window_sizes[min_index]
        optimal_params = lppls_params_list[
            min_index
        ]  # get LPPLS parameters for optimal window

        # get tau (start time of the bubble)
        tau_idx = total_obs - optimal_window_size
        tau = self.observations[0][tau_idx]

        return {
            "tau": tau,
            "optimal_window_size": optimal_window_size,
            "tc": optimal_params["tc"],
            "m": optimal_params["m"],
            "w": optimal_params["w"],
            "a": optimal_params["a"],
            "b": optimal_params["b"],
            "c1": optimal_params["c1"],
            "c2": optimal_params["c2"],
            "window_sizes": window_sizes,
            "sse_list": sse_list,
            "ssen_list": ssen_list,
            "lagrange_sse_list": lagrange_sse_list,
            "start_times": start_times,
        }
