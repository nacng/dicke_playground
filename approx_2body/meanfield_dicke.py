from abc import ABC, abstractmethod

import numpy as np
from typing import Union, List, Callable

from approx_2body.sim_params import SimulationParams

class MeanFieldDicke(ABC):

    def __init__(self, sim_params: SimulationParams):

        self.dt = sim_params.dt
        self.n_sim = sim_params.n_sim
        
        self.omega_cav = sim_params.omega_cav
        self.kappa = sim_params.kappa
        self.g = sim_params.g
        self.h_s = sim_params.h_s
        self.dh = self.h_s.shape[1]
                
        return

    @abstractmethod
    def meanfield_dynamics(self, f_damp: Callable[[float, float], complex], rho_0: np.ndarray, a_initial: complex) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        raise NotImplementedError()
            
    @abstractmethod
    def system_evo(self, dt: float, g: float, a0: complex, dadt: complex) -> np.ndarray:

        raise NotImplementedError()
    
    #################################
    ## Models of damped boson mode ##
    #################################

    # Damping originating from coupling to a zero temperature reservoir
    def field_deriv_cavity_damp(self, a: complex, x: float) -> complex:

        return -(1j*self.omega_cav + self.kappa) * a - 1j * self.g * x

    # Damping from Stokes drag, ∝ velocity
    def field_deriv_friction_damp(self, a: complex, x: float) -> complex:
    
        return -(1j*self.omega_cav * a) - 1j * self.kappa * np.imag(a) - 1j * self.g * x

    def meanfield_cavity_damp(self, rho_0: np.ndarray, a_initial: complex) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        return self.meanfield_dynamics(self.field_deriv_cavity_damp, rho_0, a_initial)

    def meanfield_friction_damp(self, rho_0: np.ndarray, a_initial: complex) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        return self.meanfield_dynamics(self.field_deriv_friction_damp, rho_0, a_initial)
