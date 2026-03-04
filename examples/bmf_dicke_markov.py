import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from approx_2body.sim_params import SimulationParamsMarkov
from approx_2body.bmf_dicke_markov import BMFDickeMarkov

if __name__=='__main__':

    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(outpath, exist_ok=True)

    sigmax = np.complex128(np.array([[0.0, 1.0], [1.0, 0.0]]))
    sigmaz = np.complex128(np.array([[1.0, 0.0], [0.0, -1.0]]))
    sigmay = 1j * sigmax@sigmaz

    wz = 0.025
    omega_cav = 1.0
    kappa = 1.0
    g = 0.251

    gamma_deph = 1.0
    gamma_diss = 0.0
    
    h_s = wz * sigmaz
    sp = SimulationParamsMarkov(dt = 0.04,
                                t_max = 10.0,
                                h_s = h_s,
                                omega_cav = omega_cav,
                                kappa = kappa,
                                g = g,
                                gamma_deph = gamma_deph,
                                gamma_diss = gamma_diss)

    dicke = BMFDickeMarkov(sp)

    theta = 0.825*np.pi
    phi = 0.0
    
    mach_eps = np.finfo(float).eps
    if np.abs(theta - 0.0) < mach_eps:
        sx0 = 0.0
        sy0 = 0.0
        sz0 = 1.0
    elif np.abs(theta - np.pi) < mach_eps:
        sx0 = 0.0
        sy0 = 0.0
        sz0 = -1.0
    else:
        sx0 = np.sin(theta) * np.cos(phi)
        sy0 = np.sin(theta) * np.sin(phi)
        sz0 = np.cos(theta)

    rho0 = 0.5 * (np.eye(2, dtype=complex) + (sx0 * sigmax) + (sy0 * sigmay) + (sz0 * sigmaz))
    a0 = 0.0j
    n0 = 0.0
    aa0 = 0.0j

    max_steps = int(np.ceil(sp.t_max / sp.dt))
    atol = 1e-4
    
    # N_spins = 50
    
    for N_spins in [10, 50]:

        print(f"N_spins = {N_spins}:")
        t_list, a_t, n_t, x_t, y_t, z_t, q2_t, pq_qp_t = dicke.bmf_cavity_damp(rho0, a0, n0, aa0, max_steps, N_spins, atol)

        np.savetxt(os.path.join(outpath, f"bmf_dynamics_markovian_N={N_spins}.csv"), np.column_stack((t_list, np.real(a_t), np.imag(a_t), n_t, x_t, y_t, z_t)), delimiter=',')

