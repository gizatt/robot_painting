import numpy as np
from numba import njit

def submix_power(i):
    # From https://github.com/ctmakro/opencv_playground/blob/master/colormixer.py
    i = np.array(i)

    @njit
    def BGR2PWR(c):
        c = np.clip(c, a_max=1.-1e-6, a_min=1e-6)  # no overflow allowed
        c = np.power(c, 2.2/i)
        u = 1. - c
        return u  # unit absorbance

    @njit
    def PWR2BGR(u):
        c = 1. - u
        c = np.power(c, i/2.2)
        return c  # rgb color

    @njit
    def submix(c1, c2, ratio):
        uabs1, uabs2 = BGR2PWR(c1), BGR2PWR(c2)
        mixuabs = (uabs1 * ratio) + (uabs2*(1-ratio))
        return PWR2BGR(mixuabs)

    return submix, BGR2PWR, PWR2BGR