import sys
from time import time
import numpy as np
from scipy.linalg import expm
from typing import Union, List, Callable
from ncon import ncon
from approx_2body.meanfield_dicke import MeanFieldDicke
from approx_2body.sim_params import SimulationParamsMarkov

class MeanFieldDickeMarkov(MeanFieldDicke):

    def __init__(self, sim_params: SimulationParamsMarkov):

        super().__init__(sim_params)
        
        self.gamma_deph = sim_params.gamma_deph
        self.gamma_diss = sim_params.gamma_diss
        
    def meanfield_dynamics(self, f_damp: Callable[[float, float], complex], rho_0: np.ndarray, a_initial: complex) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the mean field dynamics (using a damping on the central boson mode modeled by f_damp) starting from rho_0 as the spin state and <a> = a_initial for the central boson mode

        Time stepping done by Heun's method
        """
        
        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz

        n_sim = self.n_sim
        
        a_t = np.zeros(n_sim+1, dtype=np.complex128)
        x_t = np.zeros(n_sim+1, dtype=np.float64)
        y_t = np.zeros(n_sim+1, dtype=np.float64)
        z_t = np.zeros(n_sim+1, dtype=np.float64)
        
        a_t[0] = a_initial
        x_t[0] = np.trace(rho_0 @ sigmax).real
        y_t[0] = np.trace(rho_0 @ sigmay).real
        z_t[0] = np.trace(rho_0 @ sigmaz).real
        a_m = a_t[0]
        x_m = x_t[0]
        
        k1_m = f_damp(a_m, x_m)
        u1 = self.system_evo(self.dt/2, self.g, a_m, k1_m)
        u2 = self.system_evo(self.dt/2, self.g, a_m + (self.dt/2)*k1_m, k1_m)
        evol_tens = np.einsum('ab,bc->ac', u1.T, u2.T)

        state = rho_0.flatten()
        
        k2_m = 0.0 + 1j * 0.0

        tstart = time()
        
        for i in range(0, n_sim):

            sys.stdout.write(f"\rProgress: {100*i/(n_sim-1):.2f}%, Elapsed time (s): {time()-tstart:.2f}")
            sys.stdout.flush()
            
            state = ncon([state, evol_tens], [[2], [2, -2]])
            rho_mplus1 = state.reshape(rho_0.shape)
            
            Z = np.trace(rho_mplus1)
            x_t[i + 1] = (np.trace(rho_mplus1 @ sigmax) / Z).real
            y_t[i + 1] = (np.trace(rho_mplus1 @ sigmay) / Z).real
            z_t[i + 1] = (np.trace(rho_mplus1 @ sigmaz) / Z).real
            x_m = x_t[i + 1]
            k2_m = f_damp(a_m + self.dt*k1_m, x_m)
            a_t[i + 1] = a_m + self.dt*(k1_m + k2_m)/2
            a_m = a_t[i + 1]
            
            k1_m = f_damp(a_m, x_m)
            u1 = self.system_evo(self.dt/2, self.g, a_m, k1_m)
            u2 = self.system_evo(self.dt/2, self.g, a_m + (self.dt/2)*k1_m, k1_m)
            evol_tens = np.einsum('ab,bc->ac', u1.T, u2.T)

        print()
        
        return a_t, x_t, y_t, z_t

    
    ###########################
    ## Mean field propagator ##
    ###########################
    
    def system_evo(self, dt: float, g: float, a0: complex, dadt: complex) -> np.ndarray: 
        """
        Approximates the (Liouvillian) time evolution of U(t0 + dt, t0) under the action of the Hamiltonian
        H_s(t) = h_s + (a0 + (t-t0)*dadt) X
        as
        exp(-i*dt*H_s(t+dt/2))
        """

        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz
        sigma_plus = 0.5 * (sigmax + 1j*sigmay)
        sigma_minus = 0.5 * (sigmax - 1j*sigmay)
        
        a_mid = a0 + dadt*dt/2
        
        h_eff = self.h_s + g * (2 * np.real(a_mid)) * sigmax
        liouvillian = np.kron(h_eff, np.eye(self.dh)) - np.kron(np.eye(self.dh), np.transpose(h_eff))
        liouvillian *= -1j
        liouvillian += self.gamma_deph * (np.kron(sigmaz, np.transpose(sigmaz)) - np.kron(np.eye(self.dh), np.eye(self.dh)))
        liouvillian += self.gamma_diss * (np.kron(sigma_minus, np.transpose(sigma_plus)) - 0.5*np.kron(sigma_plus@sigma_minus, np.eye(self.dh)) - 0.5*np.kron(np.eye(self.dh), np.transpose(sigma_plus@sigma_minus)))
        
        u = expm(liouvillian * dt)
        
        return u
