from lppls.curve_fit import CurveFit
from keras.initializers import Constant
import tensorflow as tf
from keras import backend as K
import numpy as np


class LPPLSLayer(CurveFit):
    # original model:
    #  dt = tc - t
    #  dtPm = dt ^ m
    #  A + B * dtPm + C * dtPm * cos(w * ln(dt) - phi)
    def __init__(self):
        super().__init__(3, LPPLSLayer.lppls, Constant(0.5))

    def get_tc(self):
        return self.get_weights()[0][0]

    def get_m(self):
        return self.get_weights()[0][1]

    def get_w(self):
        return self.get_weights()[0][2]

    # def get_weights(self):
    #     return self.weights

    @staticmethod
    def get_t_tc_m_w_abcc(x, args):
        N = K.constant(int(x.shape[-1]), dtype=x.dtype)
        t = K.arange(0, int(x.shape[-1]), 1, dtype=x.dtype)
        # note that we need to get the variables to be centered
        # around 0 so to correct the magnitude we offset them by
        # constants.
        # w just has a mangitude of 10s from empirical results
        # for tc we apply a factor of 20 which should be
        # interpreted as a month (~20 trading days).
        # A tc of 0.5 means half a month in the future
        tc = args[0] * K.constant(20, dtype=x.dtype) + N
        m = args[1]
        w = args[2] * K.constant(10, dtype=x.dtype)

        dt = (tc - t)
        dtPm = K.pow(dt, m)
        dtln = K.log(dt)
        abcc = LPPLSLayer.matrix_equation(x, dtPm, dtln, w, N)
        a, b, c1, c2 = (abcc[0], abcc[1], abcc[2], abcc[3])

        return t, tc, m, w, a, b, c1, c2

    @staticmethod
    def lppls(x, args):
        t, tc, m, w, a, b, c1, c2 = LPPLSLayer.get_t_tc_m_w_abcc(x, args)
        dt = (tc - t)
        dtPm = K.pow(dt, m)
        dtln = K.log(dt)
        # then we calculate the lppls with the given parameters
        return a + b * dtPm + c1 * dtPm * K.cos(w * dtln) + c2 * dtPm * K.sin(w * dtln)


    # nothing to see here, this is just used to simplify the
    # parameter space and fit for the LPPL function
    @staticmethod
    def matrix_equation(x, dtPm, dtln, w, N):
        fi = dtPm
        gi = dtPm * K.cos(w * dtln)
        hi = dtPm * K.sin(w * dtln)
        fi_pow_2 = K.sum(fi * fi)
        gi_pow_2 = K.sum(gi * gi)
        hi_pow_2 = K.sum(hi * hi)
        figi = K.sum(fi * gi)
        fihi = K.sum(fi * hi)
        gihi = K.sum(gi * hi)
        yi = x
        yifi = K.sum(yi * fi)
        yigi = K.sum(yi * gi)
        yihi = K.sum(yi * hi)
        fi = K.sum(fi)
        gi = K.sum(gi)
        hi = K.sum(hi)
        yi = K.sum(yi)
        A = K.stack([
            K.stack([N, fi, gi, hi]),
            K.stack([fi, fi_pow_2, figi, fihi]),
            K.stack([gi, figi, gi_pow_2, gihi]),
            K.stack([hi, fihi, gihi, hi_pow_2])
        ], axis=0)
        b = K.stack([yi, yifi, yigi, yihi])
        # do a classic x = (A'A)⁻¹A' b
        return tf.linalg.solve(A, K.reshape(b, (4, 1)))

    # @staticmethod
    # def compute_indicator(x, largest_window_size=120, smallest_window_size=30, increment=5):
    #     n_windows = len(x[0, :]) - smallest_window_size
    #     #     n_fits = (largest_window_size - smallest_window_size) // increment
    #     #     print(n_windows)
    #     cofs = []
    #
    #     for i in range(n_windows):
    #         start_idx = 0 + i
    #         end_index = largest_window_size + i
    #         x_slice = x[0][start_idx:end_index]
    #         #         print(x_slice)
    #         #         print(len(x_slice))
    #         j = smallest_window_size
    #
    #         bew_true_count = 0
    #         bef_true_count = 0
    #
    #         while j <= largest_window_size:
    #             x_sub_slice = x_slice[0:j]
    #             x_sub_slice = x_sub_slice.reshape(1, -1)
    #             j = j + increment
    #             #             print(x_sub_slice)
    #             #             print(len(x_sub_slice))
    #
    #             # create the model
    #             model = Sequential([lppls_layer.LPPLSLayer()])
    #             model.compile(loss='mse', optimizer=Adagrad(0.011))
    #             model.fit(x_sub_slice, x_sub_slice, epochs=8000, verbose=0)
    #
    #             tc = model.layers[0].get_tc()
    #             m = model.layers[0].get_m()
    #             w = model.layers[0].get_w()
    #             args = (tc, m, w)
    #
    #             t, tc, m, w, a, b, c1, c2 = model.layers[0].get_t_tc_m_w_abcc(x_sub_slice, args)
    #             c = (c1 ** 2 + c2 ** 2) ** 0.5
    #
    #             first = t.numpy()[0]
    #             last = t.numpy()[-1]
    #             dt = last - first
    #             dmin = last - dt * 0.05
    #             dmax = last + dt * 0.1
    #
    #             n_oscillation = (w / (2 * np.pi)) * np.log(abs((tc - first) / (tc - last)))
    #             damping = (m * abs(b)) / (w * abs(c))
    #
    #             # for bubble early warning
    #             bew_true = (damping > 0.8) and (0.01 < m < 1.2) and (2 < w < 25) and (dmin < tc < dmax) and (
    #                         n_oscillation > 2.5)
    #             bew_true_count = (bew_true_count + 1) if bew_true else bew_true_count
    #             #             print('bew_true_count: {}'.format(bew_true_count))
    #
    #             # for bubble end flag
    #             bef_true = (damping > 1) and (0.01 < m < 0.99) and (2 < w < 25) and (dmin < tc < dmax) and (
    #                         n_oscillation > 2.5)
    #             bef_true_count = (bef_true_count + 1) if bef_true else bef_true_count
    #
    #         #             print('----------------')
    #
    #         cofs.append({end_index: {
    #             'bew': bew_true_count / (largest_window_size - smallest_window_size),
    #             'bef': bef_true_count / (largest_window_size - smallest_window_size)
    #         }})
    #
    #         print(cofs)
    #
    #     return cofs