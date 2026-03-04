import sys
from time import time
import numpy as np
from scipy import linalg
from scipy.linalg import expm, norm
from scipy.sparse.linalg import eigs
from typing import Optional, Union, List
from opt_einsum import contract
from ncon import ncon
import copy
from approx_2body.bath import Bath
from approx_2body.sim_params import SimulationParamsTEMPO
from approx_2body.meanfield_dicke_tempo import MeanFieldDickeTEMPO
        
    
class BMFDickeTEMPO(MeanFieldDickeTEMPO):

    def __init__(self, sim_params: SimulationParamsTEMPO, bath: Bath, finf: Optional[np.ndarray] = None, bath_tr: Optional[np.ndarray] = None, bath_0: Optional[np.ndarray] = None):

        super().__init__(sim_params, bath, finf, bath_tr, bath_0)

        self.system_tr = np.eye(self.dh).flatten('C')
        
    def bmf_cavity_damp(self, rho_0: np.ndarray, a_initial: complex, n_initial: float, aa_initial: complex, n_steps: int, N_spins: int):

        return self.bmf_dynamics(self.field_deriv_cavity_damp, rho_0, a_initial, n_initial, aa_initial, n_steps, N_spins)

    def bmf_friction_damp(self, rho_0: np.ndarray, a_initial: complex, n_initial: float, aa_initial: complex, n_steps: int, N_spins: int):

        return self.bmf_dynamics(self.field_deriv_friction_damp, rho_0, a_initial, n_initial, aa_initial, n_steps, N_spins)

    def bmf_dynamics(self, f_damp, rho_0: np.ndarray, a_initial: complex, n_initial: float, aa_initial: complex, n_steps: int, N_spins: int):

        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz

        dt = self.dt
        
        # Convention:
        # y = [χ_{ij}, ⟨⟨q⟩⟩, ⟨⟨p⟩⟩, ⟨⟨n⟩⟩, ⟨⟨q²⟩⟩, ⟨⟨{p,q}/2⟩⟩]
        _chi, _q, _p, _n, _q2, _pq = 0, 1, 2, 3, 4, 5
        y_k = [None] * 6
        
        res_a = np.zeros(n_steps+1, dtype=np.complex128)
        res_x = np.zeros(n_steps+1, dtype=np.float64)
        res_y = np.zeros(n_steps+1, dtype=np.float64)
        res_z = np.zeros(n_steps+1, dtype=np.float64)
        res_n = np.zeros(n_steps+1, dtype=np.float64)
        res_q2 = np.zeros(n_steps+1, dtype=np.float64)
        res_pq_qp = np.zeros(n_steps+1, dtype=np.float64)
        res_nz_diff = np.zeros(n_steps+1, dtype=np.float64)
        res_pqz_diff = np.zeros(n_steps+1, dtype=np.float64)
        res_j2 = np.zeros(n_steps+1, dtype=np.float64)
        res_xx = np.zeros(n_steps+1, dtype=np.float64)
        res_yy = np.zeros(n_steps+1, dtype=np.float64)
        res_zz = np.zeros(n_steps+1, dtype=np.float64)
        res_xy = np.zeros(n_steps+1, dtype=np.float64)
        res_xz = np.zeros(n_steps+1, dtype=np.float64)
        res_yz = np.zeros(n_steps+1, dtype=np.float64)
        
        res_a[0] = a_initial
        res_x[0] = np.trace(rho_0 @ sigmax).real
        res_y[0] = np.trace(rho_0 @ sigmay).real
        res_z[0] = np.trace(rho_0 @ sigmaz).real
        Z = (np.trace(rho_0) * ncon([self.bath_0, self.bath_tr], [[1], [1]])[()])
        q2_0 = (2*aa_initial.real + 2*n_initial - (2 * a_initial.real)**2) + (1/N_spins) # (⟨aa+a†a†⟩+2⟨a†a⟩+1-⟨a+a†⟩²)/N
        assert q2_0 >= 0.0, "Invalid specification for initial fluctuation of q^2"
        pq_qp0 = (2*aa_initial.imag - 4*(a_initial.real)*(a_initial.imag)) #(⟨-i(aa-a†a†)⟩-⟨a+a†⟩⟨-i(a-a†)⟩)/N

        rho1 = np.einsum('a,b->ab', self.bath_0, rho_0.flatten('C')) / Z
        y_k[_chi] = np.einsum('ab,cd->abcd', rho1, rho1)
        y_k[_q] = (2*a_initial.real) * rho1
        y_k[_p] = (2*a_initial.imag) * rho1
        y_k[_n] = n_initial * rho1
        y_k[_q2] = q2_0 * rho1
        y_k[_pq] = pq_qp0 * rho1
        
        
        res_n[0] = ncon([y_k[_n], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
        res_q2[0] = ncon([y_k[_q2], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
        res_pq_qp[0] = ncon([y_k[_pq], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real

        acomm_x = self._system_comm(1, shift=0.0, sgn=1)
        acomm_y = self._system_comm(2, shift=0.0, sgn=1)
        acomm_z = self._system_comm(3, shift=0.0, sgn=1)
        
        evol_tens = self._get_U(0.0 + 0.0j, 0.0 + 0.0j)
        
        print("Propagating...")
        tstart = time()
        
        for i in range(0, self.n_sim):

            sys.stdout.write(f"\rProgress: {100*i/(self.n_sim-1):.2f}%, Elapsed time (s): {time()-tstart:.2f}")
            sys.stdout.flush()
            
            # At this point, y_k represents y(t_k)
            
            # temp = self._prop_map(evol_tens, self._deriv_map(y_k, N_spins)) # = Uf

            # Uy2 = [dt * temp[m] for m in range(0, len(y_k))]
            # for m in range(0, len(y_k)):
            #     temp[m] = (dt/2) * temp[m]

            # y_k = self._prop_map(evol_tens, y_k) # = Uy_k
            # for m in range(0, len(y_k)):
            #     Uy2[m] += y_k[m] # Now Uy2 = U(y_k + dt*f)
            #     temp[m] += y_k[m]

            # y_k = self._deriv_map(Uy2, N_spins)
            # for m in range(0, len(y_k)):
            #      temp[m] += (dt/2) * y_k[m]
            #      y_k[m] = copy.deepcopy(temp[m])

            #if i == 0:
            #    print("dq2: ", ncon([self._deriv_map(y_k, N_spins)[_q2], self.system_tr], [[2], [2]]))
            Uf = self._prop_map(evol_tens, self._deriv_map(y_k, N_spins)) # = Uf

            
            y_kp1 = [(dt/2) * Uf[m] for m in range(0, len(y_k))]
            Uy_k = self._prop_map(evol_tens, y_k)
            Uy2 = [Uy_k[m] + dt*Uf[m] for m in range(0, len(y_k))]
            for m in range(0, len(y_k)):
                y_kp1[m] += Uy_k[m]

            fUy2 = self._deriv_map(Uy2, N_spins)
            for m in range(0, len(y_k)):
                y_kp1[m] += (dt/2) * fUy2[m]
                y_k[m] = copy.deepcopy(y_kp1[m])

            # Now, y_k represents y(t_{k+1})}
            
            # Measure single time spin observables
            rho_mp1 = ncon([y_k[_chi], self.bath_tr, self.bath_tr], [[1,-2,3,-4], [1], [3]])
            rho_j = ncon([y_k[_chi], self.bath_tr, self.system_tr, self.bath_tr], [[1,2,3,-1], [1], [2], [3]]).reshape((self.dh, self.dh))
            rho_i = ncon([y_k[_chi], self.bath_tr, self.bath_tr, self.system_tr], [[1,-1,2,3], [1], [2], [3]]).reshape((self.dh, self.dh))
            
            Z = np.trace(rho_j)
            #print("Z_j:", Z, ", Z_i:", np.trace(rho_i))
            y_k[_chi] /= Z
            res_x[i + 1] = (np.trace(rho_j @ sigmax) / Z).real
            res_y[i + 1] = (np.trace(rho_j @ sigmay) / Z).real
            res_z[i + 1] = (np.trace(rho_j @ sigmaz) / Z).real
            q_m = ncon([y_k[_q], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
            p_m = ncon([y_k[_p], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
            res_a[i + 1] = (q_m + 1j * p_m)/2
            res_n[i + 1] = ncon([y_k[_n], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
            res_q2[i + 1] = ncon([y_k[_q2], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
            res_pq_qp[i + 1] = ncon([y_k[_pq], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real
            
            # check factorization <n*z> = <n><z> and <{p,q}/2 *z> = <{p,q}/2><z>
            #res_nz_diff[i + 1] = 0.5*ncon([y_k[_n], self._system_comm(3,sgn=1)@self.system_tr], [[2], [2]])[()].real - res_n[i+1]*res_z[i+1]
            #res_pqz_diff[i + 1] = 0.5*ncon([y_k[_pq], self._system_comm(3,sgn=1)@self.system_tr], [[2], [2]])[()].real - res_pq_qp[i+1]*res_z[i+1]

            #res_xx[i + 1] = ncon([y_k[_chi], self.bath_tr, self._system_comm(1,shift=-res_x[i+1],sgn=1)@self.system_tr, self.bath_tr, self._system_comm(1,sgn=1)@self.system_tr], [[1,2,3,4], [1], [2], [3], [4]])[()].real
            #res_xx[i + 1] *= (4*self.g**2)
            res_j2[i + 1] = ncon([y_k[_chi], self.bath_tr, self._system_comm(1,sgn=1)@self.system_tr, self.bath_tr, self._system_comm(1,sgn=1)@self.system_tr], [[1,2,3,4], [1], [2], [3], [4]])[()].real
            res_j2[i + 1] += ncon([y_k[_chi], self.bath_tr, self._system_comm(2,sgn=1)@self.system_tr, self.bath_tr, self._system_comm(2,sgn=1)@self.system_tr], [[1,2,3,4], [1], [2], [3], [4]])[()].real
            res_j2[i + 1] += ncon([y_k[_chi], self.bath_tr, self._system_comm(3,sgn=1)@self.system_tr, self.bath_tr, self._system_comm(3,sgn=1)@self.system_tr], [[1,2,3,4], [1], [2], [3], [4]])[()].real
            res_j2[i + 1] /= 4

            res_xx[i + 1] = (ncon([rho_mp1, acomm_x@self.system_tr, acomm_x@self.system_tr], [[1,2], [1], [2]])[()] / (4*Z)).real
            res_xx[i + 1] -= ((np.trace(rho_i @ sigmax) / Z).real)*(res_x[i+1])
            res_yy[i + 1] = (ncon([rho_mp1, acomm_y@self.system_tr, acomm_y@self.system_tr], [[1,2], [1], [2]])[()] / (4*Z)).real
            res_yy[i + 1] -= ((np.trace(rho_i @ sigmay) / Z).real)*(res_y[i+1])
            res_zz[i + 1] = (ncon([rho_mp1, acomm_z@self.system_tr, acomm_z@self.system_tr], [[1,2], [1], [2]])[()] / (4*Z)).real
            res_zz[i + 1] -= ((np.trace(rho_i @ sigmaz) / Z).real)*(res_z[i+1])
            res_xy[i + 1] = (ncon([rho_mp1, acomm_x@self.system_tr, acomm_y@self.system_tr], [[1,2], [1], [2]])[()] / (4*Z)).real
            res_xy[i + 1] -= ((np.trace(rho_i @ sigmax) / Z).real)*(res_y[i+1])
            res_xz[i + 1] = (ncon([rho_mp1, acomm_x@self.system_tr, acomm_z@self.system_tr], [[1,2], [1], [2]])[()] / (4*Z)).real
            res_xz[i + 1] -= ((np.trace(rho_i @ sigmax) / Z).real)*(res_z[i+1])
            res_yz[i + 1] = (ncon([rho_mp1, acomm_y@self.system_tr, acomm_z@self.system_tr], [[1,2], [1], [2]])[()] / (4*Z)).real
            res_yz[i + 1] -= ((np.trace(rho_i @ sigmay) / Z).real)*(res_z[i+1])
            
        print()
        
        return res_a, res_n, res_x, res_y, res_z, res_q2, res_pq_qp#, res_nz_diff, res_pqz_diff, res_j2

    def _prop_map(self, evol_tens: np.ndarray, y: List[np.ndarray]):
        # Convention:
        # y = [χ_{ij}, ⟨⟨q⟩⟩, ⟨⟨p⟩⟩, ⟨⟨n⟩⟩, ⟨⟨q²⟩⟩, ⟨⟨{p,q}/2⟩⟩]
        _chi, _q, _p, _n, _q2, _pq = 0, 1, 2, 3, 4, 5
        
        y[_chi] = ncon([y[_chi], evol_tens, evol_tens], [[1,2,3,4], [1,2,-1,-2], [3,4,-3,-4]])
        y[_q] = ncon([y[_q], evol_tens], [[1,2], [1,2,-1,-2]])
        y[_p] = ncon([y[_p], evol_tens], [[1,2], [1,2,-1,-2]])
        #y[_n] = ncon([y[_n], evol_tens], [[1,2], [1,2,-1,-2]])
        #y[_q2] = ncon([y[_q2], evol_tens], [[1,2], [1,2,-1,-2]])
        #y[_pq] = ncon([y[_pq], evol_tens], [[1,2], [1,2,-1,-2]])

        return y
    
    def _deriv_map(self, y: List[np.ndarray], N_spins: int):
        # Convention:
        # y = [χ_{ij}, ⟨⟨q⟩⟩, ⟨⟨p⟩⟩, ⟨⟨n⟩⟩, ⟨⟨q²⟩⟩, ⟨⟨{p,q}/2⟩⟩]
        _chi, _q, _p, _n, _q2, _pq = 0, 1, 2, 3, 4, 5

        ka = self.kappa
        om = self.omega_cav
        g = self.g
        
        acomm = self._system_comm(1, shift=0.0, sgn=1)
        comm = self._system_comm(1, shift=0.0, sgn=-1)

        q = ncon([y[_q], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real #⟨q⟩/\sqrt{N}
        p = ncon([y[_p], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real #⟨p⟩/\sqrt{N}
        q2 = ncon([y[_q2], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real #⟨q²⟩/N
        pq = ncon([y[_pq], self.bath_tr, self.system_tr], [[1,2], [1], [2]])[()].real #⟨{p,q}/2⟩/N
        rho = ncon([y[_chi], self.bath_tr, self.system_tr], [[-1,-2,3,4], [3], [4]])
        sx = ncon([rho, self.bath_tr, acomm@self.system_tr], [[1,2], [1], [2]])[()].real / 2 #⟨σˣ⟩
        xi_j = ncon([y[_chi], self.bath_tr, acomm@self.system_tr], [[1,2,-3,-4], [1], [2]])

        # q3 = (q**3) * rho
        # q3 += 3 * q * y[_q2]
        # q3 += (-3 * q**2) * y[_q]

        # pqq = 0.5 * p * y[_q2]
        # pqq += 1.5 * q * y[_pq]
        # pqq += (-q * p) * y[_q]
        # pqq += (-0.5 * q**2) * y[_p]
        # pqq += (0.5 * p * q**2) * rho

        # nq = q * y[_n]
        # nq += 0.25 * (q**2 + p**2) * y[_q]
        # nq += -0.25 * q * (q**2 + p**2) * rho
        
        dy = [None] * len(y)

        dy[_chi] = (-1j * g) * q * self._apply_supop_sum(comm, comm, y[_chi])
        dq = y[_q] - q*rho
        temp = np.einsum('ab,cd->abcd', ncon([dq, comm], [[-1,2], [2,-2]]), rho)
        dy[_chi] += (-1j * g) * (temp + np.moveaxis(temp, [0,1,2,3], np.argsort([2,3,0,1])))
        temp = np.einsum('ab,cd->abcd', dq, ncon([rho, comm], [[-1,2], [2,-2]]))
        dy[_chi] += (-1j * g) * (temp + np.moveaxis(temp, [0,1,2,3], np.argsort([2,3,0,1])))
        
        dy[_q] = -ka * y[_q]
        dy[_q] += om * y[_p]
        dy[_q] += (-1j * g) * q2 * ncon([rho, comm], [[-1,2], [2,-2]]) # [σˣⱼ, rho]
        dy[_q] += (-1j * g) * 2 * q * ncon([y[_q], comm], [[-1,2], [2,-2]]) # [σˣⱼ, y[_q]]
        dy[_q] += (1j * g) * 2 * (q**2) * ncon([rho, comm], [[-1,2], [2,-2]]) # [σˣⱼ, rho]

        dy[_p] = -ka * y[_p]
        dy[_p] += (-om) * y[_q]
        dy[_p] += (-g / N_spins) * ncon([rho, acomm], [[-1, 2], [2,-2]]) # {σˣᵢ, ρᵢ}
        dy[_p] += (-g * (N_spins - 1)/N_spins) * xi_j
        dy[_p] += (-1j * g) * pq * ncon([rho, comm], [[-1, 2], [2,-2]])
        dy[_p] += (1j * g) * 2 * p * q * ncon([rho, comm], [[-1, 2], [2,-2]])
        dy[_p] += (-1j * g) * q * ncon([y[_p], comm], [[-1, 2], [2,-2]])
        dy[_p] += (-1j * g) * p * ncon([y[_q], comm], [[-1, 2], [2,-2]])

        dy[_n] = -2*ka * y[_n]
        dy[_n] += (-0.5*g) * ncon([y[_p], acomm], [[-1,2], [2,-2]]) # {σˣᵢ, y[_p]}

        dy[_q2] = y[_q2] - (rho / N_spins)
        dy[_q2] *= (-2*ka)
        dy[_q2] += (2*om) * y[_pq]

        dy[_pq] = y[_q2] - (rho / N_spins)
        dy[_pq] *= (-2*om)
        dy[_pq] += (-2*ka) * y[_pq]
        dy[_pq] += (4*om) * y[_n]
        dy[_pq] += (-g) * ncon([y[_q], acomm] ,[[-1,2], [2,-2]])
        
        return dy
    
    def _apply_supop_sum(self, sop_i: np.ndarray, sop_j: np.ndarray, rho: np.ndarray):
        """
        Returns (Oᵢ + Oⱼ) ρ
        """

        xi_j = ncon([rho, sop_i], [[-1,2,-3,-4], [2,-2]])

        # lazy way, assumes symmetric
        i_xj = np.einsum('abcd->cdab', xi_j)

        return xi_j + i_xj

    def _apply_xcomm_prop(self, x_comm: np.ndarray, evol_tot: np.ndarray, rho: np.ndarray):
        """
        Returns [σᵢ+ σⱼ, UᵢUⱼ(ρ) ]

        Assumes that x_comm represents [σᵢ, Uᵢ]
        """

        xi_j = ncon([rho, x_comm, evol_tot], [[1,2,3,4], [1,2,-1,-2], [3,4,-3,-4]])

        # lazy way, assumes symmetric
        i_xj = np.einsum('abcd->cdab', xi_j)

        return xi_j + i_xj

    def _get_U(self, a: complex, da: complex):

        dt = self.dt

        u1 = self.system_evo(dt/2, self.g, a, da)
        u2 = self.system_evo(dt/2, self.g, a + (dt/2)*da, da)
        u = np.einsum('ab,bc->abc', u1.T, u2.T)

        # return u
        return ncon([self.finf[:, 1:, :], u], [[-1, 2, -3], [-2, 2, -4]])

    
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
        return np.transpose( np.kron(sigma+shift*np.eye(self.dh), np.eye(self.dh)) + np.sign(sgn) * np.kron(np.eye(self.dh), np.transpose(sigma+shift*np.eye(self.dh))) )
