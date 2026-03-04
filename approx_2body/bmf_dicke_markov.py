import sys
from time import time
import numpy as np
from scipy import linalg
from scipy.linalg import expm, norm
from typing import Union, List
from opt_einsum import contract
from ncon import ncon
from copy import deepcopy

from approx_2body.meanfield_dicke_markov import MeanFieldDickeMarkov
from approx_2body.sim_params import SimulationParamsMarkov

class BMFDickeMarkov(MeanFieldDickeMarkov):

    def __init__(self, sim_params: SimulationParamsMarkov):

        super().__init__(sim_params)

        self.dh = sim_params.h_s.shape[0]
        self.system_tr = np.eye(self.dh).flatten('C')
        self.iL_S = self._make_iL_S()
        
    def _make_iL_S(self) -> np.ndarray: 
        
        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz

        sigma_plus = 0.5 * (sigmax + 1j*sigmay)
        sigma_minus = 0.5 * (sigmax - 1j*sigmay)
        
        L = np.kron(self.h_s, np.eye(self.dh)) - np.kron(np.eye(self.dh), (self.h_s).T)
        L *= -1j
        L += self.gamma_deph * (np.kron(sigmaz, (sigmaz).T) - np.kron(np.eye(self.dh), np.eye(self.dh)))
        L += self.gamma_diss * (np.kron(sigma_minus, (sigma_plus).T) - 0.5*np.kron(sigma_plus@sigma_minus, np.eye(self.dh)) - 0.5*np.kron(np.eye(self.dh), (sigma_plus@sigma_minus).T))
        
        return L
        
    def _site_prop(self, dt: float) -> np.ndarray:

        return expm(self.iL_S * dt)

    def bmf_cavity_damp(self, rho_0: np.ndarray, a_initial: complex, n_initial: float, aa_initial: complex, max_steps: int, N_spins: int, atol: float = 1e-4, rtol: float = 0.0):

        return self.bmf_dynamics(self.field_deriv_cavity_damp, rho_0, a_initial, n_initial, aa_initial, max_steps, N_spins, atol, rtol)

    def bmf_friction_damp(self, rho_0: np.ndarray, a_initial: complex, n_initial: float, aa_initial: complex, max_steps: int, N_spins: int, atol: float = 1e-4, rtol: float = 0.0):

        return self.bmf_dynamics(self.field_deriv_friction_damp, rho_0, a_initial, n_initial, aa_initial, max_steps, N_spins, atol, rtol)
        
    def bmf_dynamics(self, f_damp, rho_0: np.ndarray, a_initial: complex, n_initial: float, aa_initial: complex, max_steps: int, N_spins: int, atol: float = 1e-4, rtol: float = 0.0):
        """
        Compute the beyond mean field dynamics of N_spins under a cumulant approximation (using a damping on the central boson mode modeled by f_damp).

        rtol and atol specify tolerances for the adaptive timestepping (Dormand-Prince 5(4))
        """
        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz

        # Convention:
        # y = [χ_{ij}, ⟨⟨q⟩⟩, ⟨⟨p⟩⟩, ⟨⟨n⟩⟩, ⟨⟨q²⟩⟩, ⟨⟨{p,q}/2⟩⟩]
        _chi, _q, _p, _n, _q2, _pq = 0, 1, 2, 3, 4, 5
        y_k = [None] * 6
        
        res_a = [a_initial]
        res_x = [np.trace(rho_0 @ sigmax).real]
        res_y = [np.trace(rho_0 @ sigmay).real]
        res_z = [np.trace(rho_0 @ sigmaz).real]
        res_n = []
        res_q2 = []
        res_pq_qp = []
        res_nz_diff = [0.0]
        res_pqz_diff = [0.0]
        res_j2 = []
        res_xx = []
        res_err = [0.0]
        
        Z = np.trace(rho_0)**2
        q2_0 = (2*aa_initial.real + 2*n_initial - (2 * a_initial.real)**2) + (1/N_spins) # (⟨aa+a†a†⟩+2⟨a†a⟩+1-⟨a+a†⟩²)/N
        assert q2_0 >= 0.0, "Invalid specification for initial fluctuation of q^2"
        pq_qp0 = (2*aa_initial.imag - 4*(a_initial.real)*(a_initial.imag)) #(⟨-i(aa-a†a†)⟩-⟨a+a†⟩⟨-i(a-a†)⟩)/N

        y_k[_chi] = np.einsum('a,b->ab', rho_0.flatten('C'), rho_0.flatten('C')) / Z
        y_k[_q] = (2*a_initial.real) * rho_0.flatten('C')
        y_k[_p] = (2*a_initial.imag) * rho_0.flatten('C')
        y_k[_n] = n_initial * rho_0.flatten('C')
        y_k[_q2] = q2_0 * rho_0.flatten('C')
        y_k[_pq] = pq_qp0 * rho_0.flatten('C')
        
        
        res_n.append(ncon([y_k[_n], self.system_tr], [[2], [2]])[()].real)
        res_q2.append(ncon([y_k[_q2], self.system_tr], [[2], [2]])[()].real)
        res_pq_qp.append(ncon([y_k[_pq], self.system_tr], [[2], [2]])[()].real)

        res_xx.append( ncon([y_k[_chi], self._system_comm(1,shift=-res_x[-1],sgn=1)@self.system_tr, self._system_comm(1,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real )
        res_xx[-1] *= (4*self.g**2)
        res_j2.append( ncon([y_k[_chi], self._system_comm(1,sgn=1)@self.system_tr, self._system_comm(1,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real )
        res_j2[-1] += ncon([y_k[_chi], self._system_comm(2,sgn=1)@self.system_tr, self._system_comm(2,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real
        res_j2[-1] += ncon([y_k[_chi], self._system_comm(3,sgn=1)@self.system_tr, self._system_comm(3,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real
        res_j2[-1] /= 4
        
        err_prev = atol
        res_t = [0.0]
        dt_prev = self.dt
        max_time = self.dt * max_steps
        current_time = 0.0
        current_step = 0
        lastsaved = 0

        tstart = time()
        
        while current_time < max_time or current_step < max_steps:

            sys.stdout.write(f"\rProgress: {100*current_time/max_time:.2f}%, Elapsed time (s): {time()-tstart:.2f}")
            sys.stdout.flush()
            
            y_k, dt_new, dt_current, err_current = self._stepper_DP5(y_k, dt_prev, err_prev, N_spins, atol=atol, rtol=0.0)
            res_err.append(err_current)
            
            # Measure single time spin observables
            rho_j = ncon([y_k[_chi], self.system_tr], [[2,-1], [2]]).reshape((self.dh, self.dh))
            rho_i = ncon([y_k[_chi], self.system_tr], [[-1,2], [2]]).reshape((self.dh, self.dh))
            
            Z = np.trace(rho_j)
            y_k[_chi] /= Z
            rho_j /= Z
            res_x.append(np.trace(rho_j @ sigmax).real)
            res_y.append(np.trace(rho_j @ sigmay).real)
            res_z.append(np.trace(rho_j @ sigmaz).real)
            
            q_m = ncon([y_k[_q], self.system_tr], [[2], [2]])[()].real
            p_m = ncon([y_k[_p], self.system_tr], [[2], [2]])[()].real

            res_a.append((q_m + 1j * p_m)/2 )
            res_n.append(ncon([y_k[_n], self.system_tr], [[2], [2]])[()].real)
            res_q2.append(ncon([y_k[_q2], self.system_tr], [[2], [2]])[()].real)
            res_pq_qp.append(ncon([y_k[_pq], self.system_tr], [[2], [2]])[()].real)
            
            # check factorization <n*z> = <n><z> and <{p,q}/2 *z> = <{p,q}/2><z>
            #res_nz_diff.append(0.5*ncon([y_k[_n], self._system_comm(3,sgn=1)@self.system_tr], [[2], [2]])[()].real - res_n[-1]*res_z[-1])
            #res_pqz_diff.append(0.5*ncon([y_k[_pq], self._system_comm(3,sgn=1)@self.system_tr], [[2], [2]])[()].real - res_pq_qp[-1]*res_z[-1])
            
            res_xx.append( ncon([y_k[_chi], self._system_comm(1,shift=-res_x[-1],sgn=1)@self.system_tr, self._system_comm(1,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real )
            res_xx[-1] *= (4*self.g**2)
            res_j2.append( ncon([y_k[_chi], self._system_comm(1,sgn=1)@self.system_tr, self._system_comm(1,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real )
            res_j2[-1] += ncon([y_k[_chi], self._system_comm(2,sgn=1)@self.system_tr, self._system_comm(2,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real
            res_j2[-1] += ncon([y_k[_chi], self._system_comm(3,sgn=1)@self.system_tr, self._system_comm(3,sgn=1)@self.system_tr], [[2,4], [2], [4]])[()].real
            res_j2[-1] /= 4
            
            current_time += dt_current
            current_step += 1
            res_t.append(current_time)
            
            dt_prev = dt_new
            err_prev = err_current
            
        print()
        
        return res_t, res_a, res_n, res_x, res_y, res_z, res_q2, res_pq_qp#, res_nz_diff, res_pqz_diff

    def _tableau_Heun(self):

        return np.ndarray([
            [0, 0],
            [1.0, 0]
        ]), np.ndarray([
            [0.5, 0.5]
        ]), np.ndarray([0.0, 1.0])
        
    def _stepper_Heun(self, y_k: List[np.ndarray], N_spins: int):

        #_a, _b, _c = self._tableau_Heun()
            
        Uf = self._prop_map(evol_tens, self._deriv_map(y_k, N_spins)) # = Uf
            
        y_kp1 = [(dt/2) * Uf[m] for m in range(0, len(y_k))]
        Uy_k = self._prop_map(evol_tens, y_k)
        Uy2 = [Uy_k[m] + dt*Uf[m] for m in range(0, len(y_k))]
        for m in range(0, len(y_k)):
            y_kp1[m] += Uy_k[m]

        fUy2 = self._deriv_map(Uy2, N_spins)
        for m in range(0, len(y_k)):
            y_kp1[m] += (dt/2) * fUy2[m]
            y_k[m] = deepcopy(y_kp1[m])

        # Now, y_k represents y(t_{k+1})}

        return y_k
    
    def _tableau_DP5(self):
        """
        Returns the Butcher tableau a_{ij}, b_i, c_i for the Dormand-Prince 5(4) method
        
        Outputs:
        - a
        - b: Row 1 = order 5, Row 2 = order 4
        - c
        """
        return np.array(
            [
                [0,               0,              0,               0,            0,              0,         0],
                [1.0/5.0,         0,              0,               0,            0,              0,         0],
                [3.0/40.0,        9.0/40.0,       0,               0,            0,              0,         0],
                [44.0/45.0,      -56.0/15.0,      32.0/9.0,        0,            0,              0,         0],
                [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,  0,              0,         0],
                [9017.0/3168.0,  -355.0/33.0,     46732.0/5247.0,  49.0/176.0,  -5103.0/18656.0, 0,         0],
                [35.0/384.0,      0.0,            500.0/1113.0,    125.0/192.0, -2187.0/6784.0,  11.0/84.0, 0]
            ]
        ), np.array(
            [
                [35.0/384.0,     0.0, 500.0/1113.0,   125.0/192.0, -2187.0/6784.0,    11.0/84.0,    0.0],
                [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0]
            ]
        ), np.array(
            [0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0]
        )
    
    def _stepper_DP5(self, y_k: List[np.ndarray], dt_prev: float, err_prev: float, N_spins: int, atol: float = 1e-4, rtol: float = 0.0):
        """
        Returns the Dormand-Prince 5(4) estimate for the next step, along with an updated step size 
        Adaptive timestepping using the PI step size control scheme described in Section IV.2 Hairer+Wanner's Solving ODEs II, eqs 2.42 and 2.48

        Note: This assumes that _deriv_map describes only the terms that come from nonzero light-matter coupling g. The uncoupled parts of the evolution are assumed to be exactly solved by the tensors from _ph_prop and from _site_prop
              Doing this makes DP5 lose its FSAL property, and introduces 40 times more contractions
        """
        _a, _b, _c = self._tableau_DP5()
        s = len(_c)

        num_obj = len(y_k)
        zrs = [np.zeros_like(y_k[n]) for n in range(0, num_obj)]

        reject = True
        dt = dt_prev

        while reject: 
            _k = [ deepcopy(zrs) for i in range(0, s) ]
            # Compute _k[0][n] first
            _k[0] = self._deriv_map(y_k, N_spins)

            for i in range(1, s):

                w_i = deepcopy(zrs)
                for j in range(0, i):
                    if _a[i, j] != 0.0:
                        temp = self._prop_map((_c[i] - _c[j])*dt, _k[j])
                        for n in range(0, num_obj):
                            w_i[n] += (dt * _a[i, j]) * temp[n]
                temp = self._prop_map(_c[i]*dt, y_k)
                for n in range(0, num_obj):
                    w_i[n] += temp[n]

                _k[i] = self._deriv_map(w_i, N_spins)

            y_kp1 = deepcopy(zrs)
            y_kp1_err = deepcopy(zrs)
            for i in range(0, s):
                temp = self._prop_map((1.0 - _c[i])*dt, _k[i])
                if _b[0,i] != 0.0:
                    for n in range(0, num_obj):
                        y_kp1[n] += (dt*_b[0,i]) * temp[n]
                if _b[1,i] != 0.0:
                    for n in range(0, num_obj):
                        y_kp1_err[n] += (dt*_b[1,i]) * temp[n]
            temp = self._prop_map(dt, y_k)
            err_kp1 = deepcopy(zrs)
            for n in range(0, num_obj):
                err_kp1[n] = y_kp1[n] - y_kp1_err[n]
                y_kp1[n] += temp[n]
                y_kp1_err[n] += temp[n]

            # Calcuate Tr ρ_{ij}² and Tr ρ_{i0}²
            err_current = norm(norm(err_kp1[0], axis=(0,1)))
            for n in range(1, num_obj):
                err_n = norm(norm(err_kp1[n]))
                err_current = max(err_current, err_n)
            sc = atol
            if rtol > 0.0:
                ymax_current = norm(norm(y_kp1[0], axis=(0,1)))
                for n in range(1, num_obj):
                    ymax_n = norm(norm(y_kp1[n]))
                    ymax_current = max(ymax_current, ymax_n)
                sc += rtol * ymax_current
            # print(err_current / sc)

            # Follows section II.V in Hairer I
            # With PI control following Eq 2.48 in Hairer II
            α = 0.7/5.0
            β = 0.4/5.0
            fac = 0.8 # 0.9 # (0.25)**(1/5) # (0.38)**(1/5)
            facmax = 5.0
            facmin = 0.01
            dt_new = dt * min(facmax, max(facmin, fac * (sc / err_current)**(α) * (err_prev / sc)**(β)) )

            # dt shouldnt be smaller than 1e-5, but not larger than 0.01
            # max(min(dt_new, 0.01), 1e-5)

            if err_current < sc:
                reject = False
                # print("pct err:", err_current / sc, ", new dt=", min(dt_new, 0.1))
            else:
                dt = min(dt_new, 0.01)
                print("Error estimate ", err_current, " versus tolerance ", sc, ". Retrying with new timestep dt =", dt_new) 

        return y_kp1, min(dt_new, 0.01), dt, err_current
    
    def _prop_map(self, dt: float, y: List[np.ndarray]):
        # Convention:
        # y = [χ_{ij}, ⟨⟨q⟩⟩, ⟨⟨p⟩⟩, ⟨⟨n⟩⟩, ⟨⟨q²⟩⟩, ⟨⟨{p,q}/2⟩⟩]
        _chi, _q, _p, _n, _q2, _pq = 0, 1, 2, 3, 4, 5
        res_y = [np.zeros_like(y[n]) for n in range(0, len(y))]
        
        evol_tens = self._site_prop(dt)
        
        res_y[_chi] = ncon([y[_chi], evol_tens, evol_tens], [[2,4], [2,-2], [4,-4]])
        res_y[_q] = ncon([y[_q], evol_tens], [[2], [2,-2]])
        res_y[_p] = ncon([y[_p], evol_tens], [[2], [2,-2]])
        res_y[_n] = ncon([y[_n], evol_tens], [[2], [2,-2]])
        res_y[_q2] = ncon([y[_q2], evol_tens], [[2], [2,-2]])
        res_y[_pq] = ncon([y[_pq], evol_tens], [[2], [2,-2]])

        return res_y
    
    def _deriv_map(self, y: List[np.ndarray], N_spins: int):
        # Convention:
        # y = [χ_{ij}, ⟨⟨q⟩⟩, ⟨⟨p⟩⟩, ⟨⟨n⟩⟩, ⟨⟨q²⟩⟩, ⟨⟨{p,q}/2⟩⟩]
        _chi, _q, _p, _n, _q2, _pq = 0, 1, 2, 3, 4, 5

        ka = self.kappa
        om = self.omega_cav
        g = self.g
        
        acomm = self._system_comm(1, shift=0.0, sgn=1)
        comm = self._system_comm(1, shift=0.0, sgn=-1)

        q = ncon([y[_q], self.system_tr], [[2], [2]])[()].real #⟨q⟩/\sqrt{N}
        p = ncon([y[_p], self.system_tr], [[2], [2]])[()].real #⟨p⟩/\sqrt{N}
        q2 = ncon([y[_q2], self.system_tr], [[2], [2]])[()].real #⟨q⟩/N
        pq = ncon([y[_pq], self.system_tr], [[2], [2]])[()].real #⟨{p,q}/2⟩/N
        rho = ncon([y[_chi], self.system_tr], [[-2,4], [4]])
        sx = ncon([rho, acomm@self.system_tr], [[2], [2]])[()].real / 2 #⟨σˣ⟩
        xi_j = ncon([y[_chi], acomm@self.system_tr], [[2,-4], [2]])

        
        dy = [None] * len(y)

        dy[_chi] = (-1j * g) * q * self._apply_supop_sum(comm, comm, y[_chi])
        dq = y[_q] - q*rho
        temp = np.einsum('b,d->bd', ncon([dq, comm], [[2], [2,-2]]), rho)
        dy[_chi] += (-1j * g) * (temp + np.moveaxis(temp, [0,1], np.argsort([1,0])))
        temp = np.einsum('b,d->bd', dq, ncon([rho, comm], [[2], [2,-2]]))
        dy[_chi] += (-1j * g) * (temp + np.moveaxis(temp, [0,1], np.argsort([1,0])))
        
        dy[_q] = -ka * y[_q]
        dy[_q] += om * y[_p]
        dy[_q] += (-1j * g) * q2 * ncon([rho, comm], [[2], [2,-2]]) # [σˣⱼ, rho]
        dy[_q] += (-1j * g) * 2 * q * ncon([y[_q], comm], [[2], [2,-2]]) # [σˣⱼ, y[_q]]
        dy[_q] += (1j * g) * 2 * (q**2) * ncon([rho, comm], [[2], [2,-2]]) # [σˣⱼ, rho]

        dy[_p] = -ka * y[_p]
        dy[_p] += (-om) * y[_q]
        dy[_p] += (-g / N_spins) * ncon([rho, acomm], [[2], [2,-2]]) # {σˣᵢ, ρᵢ}
        dy[_p] += (-g * (N_spins - 1)/N_spins) * xi_j
        dy[_p] += (-1j * g) * pq * ncon([rho, comm], [[2], [2,-2]])
        dy[_p] += (1j * g) * 2 * p * q * ncon([rho, comm], [[2], [2,-2]])
        dy[_p] += (-1j * g) * q * ncon([y[_p], comm], [[2], [2,-2]])
        dy[_p] += (-1j * g) * p * ncon([y[_q], comm], [[2], [2,-2]])

        dy[_n] = -2*ka * y[_n]
        dy[_n] += (-0.5*g) * ncon([y[_p], acomm], [[2], [2,-2]]) # {σˣᵢ, y[_p]}

        dy[_q2] = y[_q2] - (rho / N_spins)
        dy[_q2] *= (-2*ka)
        dy[_q2] += (2*om) * y[_pq]

        dy[_pq] = y[_q2] - (rho / N_spins)
        dy[_pq] *= (-2*om)
        dy[_pq] += (-2*ka) * y[_pq]
        dy[_pq] += (4*om) * y[_n]
        dy[_pq] += (-g) * ncon([y[_q], acomm], [[2], [2,-2]])
        
        return dy
    
    def _apply_supop_sum(self, sop_i: np.ndarray, sop_j: np.ndarray, rho: np.ndarray):
        """
        Returns (Oᵢ + Oⱼ) ρ
        """

        xi_j = ncon([rho, sop_i], [[2,-4], [2,-2]])

        # lazy way, assumes symmetric
        i_xj = np.einsum('bd->db', xi_j)

        return xi_j + i_xj

    def _apply_xcomm_prop(self, x_comm: np.ndarray, evol_tot: np.ndarray, rho: np.ndarray):
        """
        Returns [σᵢ+ σⱼ, UᵢUⱼ(ρ) ]

        Assumes that x_comm represents [σᵢ, Uᵢ]
        """

        xi_j = ncon([rho, x_comm, evol_tot], [[2,4], [2,-2], [4,-4]])

        # lazy way, assumes symmetric
        i_xj = np.einsum('bd->db', xi_j)

        return xi_j + i_xj

    def _system_comm(self, mu, shift = 0.0, sgn = -1):
        """
        Returns a representation of the commutator superoperator [σ^{mu}_i, ...], where
        mu = (1, 2, 3) == (x, y, z)

        The elements of this representation are given by
        ⟨s'_f| [σ^μ, |s_f⟩⟨s_b|] |s'_b⟩ = ⟨s'_f|σ^μ|s_f⟩ δ_{sb,s'b} - δ_{sf,s'f} ⟨s_b|σ^μ|s'_b⟩
        where states are ordered s_f = (0, 1) == (+, -)
        """

        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz

        if mu == 0:
            sigma = np.eye(self.dh)
        elif mu == 1:
            sigma = sigmax
        elif mu == 2:
            sigma = sigmay
        elif mu == 3:
            sigma = sigmaz
        else:
            raise RuntimeError("Invalid Pauli operator specification")

        # Transpose result so that 1st index = in, 2nd index = out
        return np.transpose( np.kron(sigma+shift*np.eye(self.dh), np.eye(self.dh)) + np.sign(sgn) * np.kron(np.eye(self.dh), (sigma+shift*np.eye(self.dh)).T) ) 
