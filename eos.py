from scipy.optimize import brentq
import numpy as np
from scipy.optimize import curve_fit

def BM4_TH(x, K0, Kp, Kdp, a, b, c, V0, T0):
    Temp, vol = x
    eta = (V0 / vol)**(1/3)
    Pc = 3/2 * K0 * (eta**7 - eta**5) * (1 + 3/4 * (Kp - 4) * (eta**2 - 1) + 1/24 * (9 * Kp**2 - 63 * Kp + 9 * K0 * Kdp + 143) * (eta**2 - 1)**2)
    Pth = (a - b * (vol/V0) + c * ((vol/V0 )**2))/1000. * (Temp - T0)
    return Pc + Pth

def BM4(vol, K0, Kp, Kdp, V0):
    eta = (V0 / vol)**(1/3)
    Pc = 3/2 * K0 * (eta**7 - eta**5) * (1 + 3/4 * (Kp - 4) * (eta**2 - 1) + 1/24 * (9 * Kp**2 - 63 * Kp + 9 * K0 * Kdp + 143) * (eta**2 - 1)**2)
    return Pc

def fit_BM4_TH(T_data, V_data, P_data, V0, T0):
    popt, pcov = curve_fit(lambda x, K0, Kp, Kdp, a, b, c: BM4_TH(x, K0, Kp, Kdp, a, b, c, V0, T0), (T_data, V_data), P_data, maxfev=100000)
    return popt, pcov

def fit_BM4(V_data, P_data, V0):
    popt, pcov = curve_fit(lambda vol, K0, Kp, Kdp: BM4(vol, K0, Kp, Kdp, V0), V_data, P_data, maxfev=100000)
    return popt, pcov

def BM4_TH_pt2v(Temp, Press, K0, Kp, Kdp, a, b, c, V0, T0, Vmin, Vmax):
    vol_list = np.linspace(Vmin, Vmax, 100000)
    for vol in vol_list:
        P_fit = BM4_TH((Temp, vol), K0, Kp, Kdp, a, b, c, V0, T0)
        if np.isclose(Press, P_fit, rtol=1e-5):
            return vol
    print('Not found!')

def BM4_pt2v(Press, K0, Kp, Kdp, V0):
    vol_list = np.linspace(0.5 * V0, 5 * V0, 100000)
    for vol in vol_list:
        P_fit = BM4(vol, K0, Kp, Kdp, V0)
        if np.isclose(Press, P_fit, rtol=5e-5):
            return vol
    print('Not found!')

def BM4_pt2v_range(Press, K0, Kp, Kdp, V0, Vmin, Vmax):
    vol_list = np.linspace(Vmin, Vmax, 1000000)
    for vol in vol_list:
        P_fit = BM4(vol, K0, Kp, Kdp, V0)
        if np.isclose(Press, P_fit, rtol=5e-5):
            return vol
    print('Not found!')