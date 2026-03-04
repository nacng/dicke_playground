# Script for computing the dynamics of the Dicke model with local Lindblad jump operators using QuTiP's permutationally invariant quantum solver (PIQS)
# Initial state assumed to have no photons

## !! NOTE: PIQS fails for large numbers of spins (eg N = 50)

import numpy as np
from qutip import *
from qutip.piqs.piqs import *

import argparse
from pathlib import Path
import h5py
import os
from datetime import datetime
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('wz', type=float)
parser.add_argument('wcav', type=float)
parser.add_argument('g_r', type=float)
parser.add_argument('g_cr', type=float)
parser.add_argument('kappa', type=float)

parser.add_argument('g_diss', type=float)
parser.add_argument('g_deph', type=float)
parser.add_argument('g_pump', type=float)

parser.add_argument('dt', type=float)
parser.add_argument('tsim', type=float)
parser.add_argument('n_ph', type=int)
parser.add_argument('n_spins', type=int)
# Initial state has all spins fully polarized in direction specified by theta, phi. Theta describes the angle from the positive x-axis (+z = theta=pi/2)
parser.add_argument('theta', type=float)
parser.add_argument('phi', type=float)


args = parser.parse_args()


n_ph = args.n_ph
n_sp = args.n_spins
omega_cav = args.wcav
kappa = args.kappa
g_r = args.g_r
g_cr = args.g_cr
w0 = args.wz
gamma_diss = args.g_diss
gamma_pump = args.g_pump
gamma_deph = args.g_deph

dt = args.dt
nsteps = int(np.ceil(args.tsim/dt))

basepath = Path('/home/nathan/dicke_exact_SRPT/')
outpath = basepath / f'dis={gamma_diss},dep={gamma_deph},pum={gamma_pump}' / f'nph={n_ph},nsp={n_sp}' / f'dt={dt}' / f'w0={w0}'
(outpath).mkdir(parents=True, exist_ok=True)


######################
## Model definition ##
######################

nds = num_dicke_states(n_sp)
[jx, jy, jz] = jspin(n_sp)
jp = jspin(n_sp, "+")
jm = jp.dag()
h_spin = 2 * w0 * jz

g = (g_r+g_cr)/2/np.sqrt(n_sp)
a = destroy(n_ph)

system = Dicke(hamiltonian = h_spin,
               N = n_sp,
               emission = gamma_diss,
               pumping = gamma_pump,
               dephasing = 4*gamma_deph, 
               collective_emission = 0.0,
               collective_pumping = 0.0,
               collective_dephasing = 0.0)
liouv_spin = system.liouvillian()

h_phot = omega_cav * a.dag() * a
c_ops_phot = [np.sqrt(2*kappa) * a]
liouv_phot = liouvillian(h_phot, c_ops_phot)

h_int = g_r * tensor(a.dag(), jm)
h_int += g_r * tensor(a, jp)
h_int += g_cr * tensor(a.dag(), jp)
h_int += g_cr * tensor(a, jm)
h_int /= np.sqrt(n_sp)
liouv_int = -1j* spre(h_int) + 1j* spost(h_int)

id_tls = to_super(qeye(nds))
id_phot = to_super(qeye(n_ph))

liouv_sum = super_tensor(liouv_phot, id_tls) + super_tensor(id_phot, liouv_spin)
liouv_tot = liouv_sum + liouv_int

#################
## Observables ##
#################

jx_tot = tensor(qeye(n_ph), jx)
jy_tot = tensor(qeye(n_ph), jy)
jz_tot = tensor(qeye(n_ph), jz)
nphot_tot = tensor(a.dag()*a, qeye(nds))
a_tot = tensor(a, qeye(nds))

jmax = (0.5 * n_sp)
j2max = (0.5 * n_sp + 1) * (0.5 * n_sp)

#####################
## Set up dynamics ##
#####################

t = dt * np.arange(0, nsteps)

# initial states 
theta = args.theta
rho0 = css(n_sp, (-theta + 0.5*np.pi), 0.0, coordinates='polar')
rho0_phot = fock_dm(n_ph, 0) # No photons
rho0_tot = tensor(rho0_phot, rho0)

##################
## Run dynamics ##
##################

result = mesolve(liouv_tot, rho0_tot, t, [], e_ops = [a_tot, nphot_tot, jx_tot, jy_tot, jz_tot], 
                 options = Options(store_states=False))
a_t1 = result.expect[0] / np.sqrt(n_sp)
n_t1 = np.real(result.expect[1]) / n_sp
jx_t1 = np.real(result.expect[2]) / jmax
jy_t1 = np.real(result.expect[3]) / jmax
jz_t1 = np.real(result.expect[4]) / jmax


np.savetxt(str(outpath / f'dynamics_gr={g_r}_gcr={g_cr}_k={kappa}.csv'), np.column_stack((t, np.real(a_t1), np.imag(a_t1), n_t1, jx_t1, jy_t1, jz_t1)), delimiter=',')
