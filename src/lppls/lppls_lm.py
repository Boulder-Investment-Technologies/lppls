from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from lppls.lppls import LPPLS


class LPPLS_LM(LPPLS):
    def func_restricted(self, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        tc = x[0]
        m = x[1]
        w = x[2]

        rM = self.matrix_equation(obs, tc, m, w)
        a, b, c1, c2 = rM[:, 0].tolist()

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in obs[0, :]]
        residuals = np.subtract(delta, obs[1, :])
        return residuals  # return the array of residuals

    def estimate_params(
        self, observations: np.ndarray, seed: np.ndarray, minimizer: str | None = None
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """
        Overrides the estimate_params method to use least_squares with 'lm' method.
        Args:
            observations (np.ndarray): The observed time-series data.
            seed (np.ndarray): Initial guess for tc, m, and w.
        Returns:
            tuple: (tc, m, w, a, b, c, c1, c2)
        """

        # Define a wrapper function for least_squares
        def wrapper(x):
            return self.func_restricted(x, observations)

        # Use least_squares with the Levenberg-Marquardt method
        result = least_squares(wrapper, seed, method="lm")

        if result.success:
            tc, m, w = result.x
            rM = self.matrix_equation(observations, tc, m, w)
            a, b, c1, c2 = rM[:, 0].tolist()
            c = self.get_c(c1, c2)

            # Store fitted parameters
            for coef in ["tc", "m", "w", "a", "b", "c", "c1", "c2"]:
                self.coef_[coef] = eval(coef)

            return tc, m, w, a, b, c, c1, c2
        else:
            raise ValueError("Parameter estimation failed.")
