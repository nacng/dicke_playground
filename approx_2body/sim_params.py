from abc import ABC
from typing import Optional, Union, List
import numpy as np

class SimulationParams(ABC):

    def __init__(self):

        return

class SimulationParamsMarkov(SimulationParams):
    '''
    Defines the set of parameters required for a Markovian open Dicke simulation

    Attributes
    ----------
    dt: float
        Timestep along the real-time part of the contour
    t_max: float
        Maximum (real) time difference in the measurement of the correlation function
    n_sim: int
        Number of times at which to measure the correlation function
    t_list: ndarray
        List of real times, separated by `dt`, at which to measure the correlation function
    h_s: ndarray
        System Hamiltonian
    '''

    def __init__(self, dt: float,
                       t_max: float,
                       h_s: np.ndarray,
                       omega_cav: float,
                       kappa: float,
                       g: float,
                       gamma_deph: float,
                       gamma_diss: float
    ):

        self.dt = dt #delta
        self.t_max = t_max
        self.n_sim = int(np.ceil(t_max / dt))
        self.t_list = dt * np.linspace(0, self.n_sim, self.n_sim+1)

        self.kappa = kappa
        self.g = g
        self.omega_cav = omega_cav
        self.h_s = h_s

        self.gamma_deph = gamma_deph
        self.gamma_diss = gamma_diss


class SimulationParamsTEMPO(SimulationParams):
    """
    Defines the set of parameters required for an open Dicke simulation with local coupling to harmonic baths

    Attributes
    ----------
    N: int
        Total memory length to include into the calculation of the steady state influence functional, given by the memory time `tmem` divided by the timestep size `dt`. Defaults to an `N` such that the memory time `tmem` is 400 in units of the tunnelling matrix element.
    T: float
        Temperature of the initial bath state, assumed to equal the temperature of the steady state
    dt: float
        Timestep along the real-time part of the contour
    t_max: float
        Maximum (real) time difference in the measurement of the correlation function
    n_sim: int
        Number of times at which to measure the correlation function
    t_list: ndarray
        List of real times, separated by `dt`, at which to measure the correlation function
    cutoff: float
        Truncation threshold
    maxdim: int
        Maximum bond dimension
    h_s: ndarray
        System Hamiltonian
    alg: str
        Algorithm to use for iTEBD. Can be "mbh_tebd", "ov_tebd", or "qr_tebd". Default="mbh_tebd"
    tcut: float
        Timescale over which a smooth cutoff starts to take effect. Default=None
    """

    def __init__(self, dt: float,
                       t_max: float,
                       h_s: np.ndarray,
                       omega_cav: float,
                       kappa: float,
                       g: float,
                       n_c: int,
                       cutoff: float, maxdim: Optional[int] = 1000,
                       alg: Optional[str] = "mbh_tebd",
                       tmem: Optional[float] = 400.0,
                       tcut: Optional[float] = None
    ):

        self.dt = dt
        self.t_max = t_max
        self.n_sim = int(np.ceil(t_max / dt))
        self.t_list = dt * np.linspace(0, self.n_sim, self.n_sim+1)

        self.alg = alg
        self.cutoff = cutoff
        self.maxdim = maxdim

        self.n_c = n_c

        self.kappa = kappa
        self.g = g
        self.omega_cav = omega_cav
        self.h_s = h_s
