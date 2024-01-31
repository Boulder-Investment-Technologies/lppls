from lppls.lppls import LPPLS
import numpy as np


class QLPPLS(LPPLS):
    def __init__(self, observations, q=0.5):
        super().__init__(observations)
        self.q = q

    def func_restricted(self, x, *args):
        """
        Finds the least absolute differences adjusted for the q-dependent loss function.
        Args:
            x(np.ndarray):  1-D array with shape (n,).
            args: Tuple of the fixed parameters needed to completely specify the function.
        Returns:
            (float)
        """

        tc = x[0]
        m = x[1]
        w = x[2]
        observations = args[0]

        rM = self.matrix_equation(observations, tc, m, w)
        a, b, c1, c2 = rM[:, 0].tolist()

        delta = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in observations[0, :]]
        delta = np.subtract(delta, observations[1, :])

        # Use the L1 norm (sum of absolute differences) instead of the L2 norm
        # Apply the q-dependent loss function using the given quantile
        loss = np.sum([-(1 - self.q) * e if e < 0 else self.q * e for e in np.abs(delta)])

        return loss