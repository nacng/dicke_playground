import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5py
import numpy as np

from approx_2body.baths.expbath import ExponentialBath
from approx_2body.sim_params import SimulationParamsTEMPO
from approx_2body.bmf_dicke_tempo import BMFDickeTEMPO

if __name__=='__main__':

    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(outpath, exist_ok=True)

    sigmax = np.complex128(np.array([[0.0, 1.0], [1.0, 0.0]]))
    sigmaz = np.complex128(np.array([[1.0, 0.0], [0.0, -1.0]]))
    sigmay = 1j * sigmax@sigmaz

    wz = 0.025
    omega_cav = 1.0
    kappa = 0.5
    g = 0.251

    expbath = ExponentialBath(alpha = 0.3,
                              s = 1.0,
                              wc = 1.0,
                              T = 0.0)

    h_s = wz * sigmaz
    
    dt = 0.04
    tmax = 100.0
    tmem = 800.0
    n_c = int(tmem / dt)
    sp = SimulationParamsTEMPO(dt = dt,
                               t_max = tmax,
                               h_s = h_s,
                               omega_cav = omega_cav,
                               kappa = kappa,
                               g = g,
                               n_c = n_c,
                               cutoff = 10**(-8.5),
                               alg="mbh_tebd", 
                               tmem=tmem, tcut=None)

    try:
        fi = h5py.File(os.path.join(outpath, f"F_inf_a{expbath.alpha}_s{expbath.s}_wc{expbath.wc}_T{expbath.T}_dt{sp.dt}_r{-np.log10(sp.cutoff):.2f}.hdf5"), 'r')
        gr = fi['IF']

        finf = np.zeros(gr['F_inf'].shape, dtype=complex)
        gr['F_inf'].read_direct(finf)

        bath_tr = np.zeros(gr['v_r'].shape, dtype=complex)
        bath_0 = np.zeros(gr['v_l'].shape, dtype=complex)
        gr['v_r'].read_direct(bath_tr)
        gr['v_l'].read_direct(bath_0)
        fi.close()

        dicke = BMFDickeTEMPO(sp, expbath, finf = finf, bath_tr = bath_tr, bath_0 = bath_0)
    except:
        dicke = BMFDickeTEMPO(sp, expbath)
        
        fi = h5py.File(os.path.join(outpath, f"F_inf_a{expbath.alpha}_s{expbath.s}_wc{expbath.wc}_T{expbath.T}_dt{sp.dt}_r{-np.log10(sp.cutoff):.2f}.hdf5"), 'w')
        gr = fi.create_group('IF')
        gr.attrs['dt'] = sp.dt
        gr.create_dataset('F_inf', data=dicke.finf)
        gr.create_dataset('v_r', data=dicke.bath_tr)
        gr.create_dataset('v_l', data=dicke.bath_0)
        fi.close()

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
    
    for N_spins in [10, 50]:

        print(f"N_spins = {N_spins}:")
        a_t, n_t, x_t, y_t, z_t, q2_t, pq_qp_t = dicke.bmf_cavity_damp(rho0, a0, n0, aa0, max_steps, N_spins)

        np.savetxt(os.path.join(outpath, f"bmf_dynamics_ohmic_N={N_spins}.csv"), np.column_stack((sp.t_list, np.real(a_t), np.imag(a_t), n_t, x_t, y_t, z_t)), delimiter=',')
