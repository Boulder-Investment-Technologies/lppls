import cma as cm
from lppls.lppls import LPPLS
import multiprocessing as mp
import numpy as np
from scipy.stats import chisquare


class LPPLSCMAES(LPPLS):

    def __init__(self, observations):
        super().__init__(observations)
        self.observations = observations

    def fun_restricted(self, x):
        """
        Define the objective function for the CMA-ES optimizer

        Args:
            x (List): objective variable

        Returns:
            float: error of the objective function
        """
        print(x)
        tc, m, w = x
        a, b, c1, c2 = super().matrix_equation(self.observations, tc, m, w)

        t = self.observations[0, :]
        res = super().lppls(t, tc, m, w, a, b, c1, c2)

        # make nan or inf to zero
        res[np.isnan(res)] = 0.
        res[np.isinf(res)] = 0.

        # calculate the chi square
        error, _ = chisquare(f_obs=res, f_exp=self.observations[1, :])
        return error

    def fit(self, max_iteration=2500, factor_sigma=0.1, pop_size=1):
        """
        Runs the optimazation loop

        Args:
            max_iteration (int, optional): maximum number of iterations. Defaults to 2500.
            factor_sigma (float, optiona): factor to multiplying the range of the bounded values
            pop_size (int, optional): population size for CMA ES
            cores (int, optional): number of parallel runs
        Returns:
            [List]: all optimized and calculated values for tc, m, w, a, b, c, c1, c2
        """
        # best guess of the starting values
        m = 0.5
        w = 9.
        # INFO: so far as I've understand the tc time this cannot be smaller als the max time of the time series
        tc = np.max(self.observations[0, :])

        # define options for CMAES
        opts = cm.CMAOptions()
        # here we define the initial search steps for CMAES usually I use to calculate the range of the
        # max and min bounds of the value and then apply a factor for sigma
        opts.set('CMA_stds', [factor_sigma * tc, factor_sigma * (0.9 - 0.1), factor_sigma * (13. - 6.)])
        opts.set('bounds', [(tc, 0.1, 6.), (np.inf, 0.9, 13.)])
        opts.set('popsize', 10 * 2 ** pop_size)

        es = cm.CMAEvolutionStrategy(x0=[tc, m, w], sigma0=1., inopts=opts)

        # here we go
        while not es.stop() and es.countiter <= max_iteration:
            solutions = es.ask()
            solution = [self.fun_restricted(s) for s in solutions]
            es.tell(solutions, solution)
            es.logger.add()  # write data to disc to be plotted
            es.disp()

        # after while loop print infos and plot the final
        # es.result_pretty()
        # cm.plot()
        # plt.savefig('cmaes.png', dpi=300)

        # get best results
        tc, m, w = es.result.xbest
        a, b, c1, c2 = super().matrix_equation(self.observations, tc, m, w)
        c = (c1 ** 2 + c2 ** 2) ** 0.5

        # Use sklearn format for storing fit params -> original code from lppls package
        for coef in ['tc', 'm', 'w', 'a', 'b', 'c', 'c1', 'c2']:
            self.coef_[coef] = eval(coef)

        return tc, m, w, a, b, c, c1, c2


# if __name__ == '__main__':
#     # read example dataset into df
#     data = pd.read_csv('./data/AAPL.csv')
#
#     # convert index col to evenly spaced numbers over a specified interval
#     time = np.linspace(0, len(data) - 1, len(data))
#
#     # create list of observation data, in this case,
#     # daily adjusted close prices of the S&P 500
#     # use log price
#     price = np.log(data['Adj Close'].values)
#
#     # create Mx2 matrix (expected format for LPPLS observations)
#     observations = np.array([time, price])
#
#     # instantiate a new LPPLS model with the S&P 500 dataset
#     lpplscmaes_model = LPPLSCMAES(observations=observations)
#
#     # fit the model to the data and get back the params
#     tc, m, w, a, b, c, c1, c2 = lpplscmaes_model.optimize(max_iteration=2500, pop_size=4)
#       tc, m, w, a, b, c, c1, c2
#     # do hier some plotting stuff after the optimazation
#     lpplscmaes_model.plot_fit()
#     plt.savefig('opti_plot_fit.png')
#     print(lpplscmaes_model.coef_.values())
