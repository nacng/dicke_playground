import numpy as np
from scipy import integrate

from approx_2body.bath import Bath

def cosr(x, x0 = 1e-12):
    if np.abs(x) < x0:
        return 0.5 - (x**2 / 24.0) + (x**4 / 720)
    else:
        return (1 - np.cos(x))/(x**2)

class ExponentialBath(Bath):
    '''
    Abstract class defining baths with exponentially cut-off spectral densities of the form J(ω) = (α/2) * ω_c * (ω/ω_c)^s * exp(-ω/ω_c). Inherits attributes from the abstract Bath class.

    Attributes
    ----------
    alpha: float
        System-bath coupling strength
    s: float
        Exponent of the low-frequency part of the spectral density
    wc: float
        Frequency cutoff scale of the spectral density
    jw: Callable[[float], float]
        Spectral density of the spin-boson problem
    '''
    def __init__(self, alpha: float, s: float, wc: float, T: float):
        
        self.alpha = alpha
        self.s = s
        self.wc = wc
        self.T = T
        
        # J(w) defined only for w >= 0
        self.jw = lambda w: (alpha/2) * wc * (np.abs(w/wc)**s) * np.exp(-np.abs(w)/wc)


    def eta_pp_tt_kk(self, dt: float, d: int) -> complex:

        if self.T == 0.0:
            return self._eta_pp_tt_kk_zeroT(dt, d)
        else:
            return self._eta_pp_tt_kk(dt, d)
        
    def eta_pp_tt_k(self, dt: float) -> complex:
        
        if self.T == 0.0:
            return self._eta_pp_tt_k_zeroT(dt)
        else:
            return self._eta_pp_tt_k(dt)
            

    def _eta_pp_tt_kk(self, dt: float, d: int) -> complex:

        x0 = 1e-12
        cut = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*dt) * np.cos(w*dt*d)/ np.tanh(self.b*w/2),
            0.0,x0,
            points=[0.0]
        )[0]
        
        if dt == 0 or self.t == 0 or d == 0:
            res_re = integrate.quad(
                lambda w: 2 * self.jw(w) * cosr(w*dt) / np.tanh(self.b*w/2),
                x0,np.inf
            )[0]
            return 2*(dt**2)*(cut+res_re + 0.0 * 1j)

        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*dt) / np.tanh(self.b*w/2),
            x0,np.inf,
            weight='cos',
            wvar=dt*d
        )[0]
        res_im = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*dt),
            0.00,np.inf,
            weight='sin',
            wvar=dt*d
        )[0]

        return (dt**2)*(cut+res_re + 1j * res_im)


    def _eta_pp_tt_k(self, dt: float) -> complex:

        x0 = 1e-12
        cut_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*dt/(2*np.pi))**2 / np.tanh(self.b*w/2),
            0.0, x0,
            points=[0.0]
        )[0]
        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*dt/(2*np.pi))**2 / np.tanh(self.b*w/2),
            x0, np.inf
        )[0]
        cut_im = integrate.quad(
            lambda w: dt * self.jw(w) * (1 - np.sinc(w*dt/np.pi)) / w,
            0.0, x0,
            points=[0.0]
        )[0]
        res_im = integrate.quad(
            lambda w: dt * self.jw(w) * (1 - np.sinc(w*dt/np.pi)) / w,
            x0, np.inf
        )[0]

        return (dt/2)**2 * (cut_re+res_re) + 1j*(cut_im+res_im)


    ###################
    ###################

    def _eta_pp_tt_kk_zeroT(self, dt: float, d: int) -> complex:

        x0 = 1e-12
        cut = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*dt) * np.cos(w*dt*d),
            0.0,x0,
            points=[0.0]
        )[0]
        
        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*dt),
            x0,np.inf,
            weight='cos',
            wvar=dt*d
        )[0]
        res_im = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*dt),
            0.00,np.inf,
            weight='sin',
            wvar=dt*d
        )[0]

        return (dt**2)*(cut+res_re - 1j * res_im)
        
    def _eta_pp_tt_k_zeroT(self, dt: float) -> complex:

        x0 = 1e-12
        cut_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*dt/(2*np.pi))**2,
            0.0, x0,
            points=[0.0]
        )[0]
        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*dt/(2*np.pi))**2,
            x0, np.inf
        )[0]
        cut_im = integrate.quad(
            lambda w: dt * self.jw(w) * (1 - np.sinc(w*dt/np.pi)) / w,
            0.0, x0,
            points=[0.0]
        )[0]
        res_im = integrate.quad(
            lambda w: dt * self.jw(w) * (1 - np.sinc(w*dt/np.pi)) / w,
            x0, np.inf
        )[0]

        return (dt/2)**2 * (cut_re+res_re) - 1j*(cut_im+res_im)
