import sys
from time import time
import numpy as np
from scipy.linalg import expm, norm, svd
from scipy.sparse.linalg import eigs
from typing import Union, List, Optional
from opt_einsum import contract
import approx_2body.mps as mps
from approx_2body.bath import Bath
from approx_2body.sim_params import SimulationParamsTEMPO
from approx_2body.meanfield_dicke import MeanFieldDicke

class MeanFieldDickeTEMPO(MeanFieldDicke):

    def __init__(self, sim_params: SimulationParamsTEMPO, bath: Bath, finf: Optional[np.ndarray] = None, bath_tr: Optional[np.ndarray] = None, bath_0: Optional[np.ndarray] = None):

        super().__init__(sim_params)

        self.bath = bath

        self.n_c = sim_params.n_c
        self.dh = 2

        if finf is not None and bath_tr is not None and bath_0 is not None:
            self.finf = finf
            self.bath_tr = bath_tr
            self.bath_0 = bath_0
        else:
            self.finf, self.bath_tr, self.bath_0 = self.make_finf(sim_params.alg, sim_params.cutoff)

        
    def meanfield_dynamics(self, f_damp, rho_0: np.ndarray, a_initial: complex) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        u = np.einsum('ab,bc->abc', u1.T, u2.T)

        evol_tens = contract('aic,bid->badc', u, self.finf[:, 1:, :])
        state = contract('a,b->ba', rho_0.flatten(), self.bath_0)

        k2_m = 0.0 + 1j * 0.0

        print("Propagating...")
        tstart = time()
        
        for i in range(0, n_sim):

            sys.stdout.write(f"\rProgress: {100*i/(n_sim-1):.2f}%, Elapsed time (s): {time()-tstart:.2f}")
            sys.stdout.flush()
            
            state = contract('ba,badc->dc', state, evol_tens)
            rho_mplus1 = contract('b,ba->a', self.bath_tr, state).reshape(rho_0.shape)
            
            Z = np.trace(rho_mplus1)
            x_t[i + 1] = (np.trace(rho_mplus1 @ sigmax) / Z).real
            y_t[i + 1] = (np.trace(rho_mplus1 @ sigmay) / Z).real
            z_t[i + 1] = (np.trace(rho_mplus1 @ sigmaz) / Z).real
            x_m = x_t[i + 1]
            k2_m = f_damp(a_m + self.dt*k1_m, x_m)
            a_t[i + 1] = a_m + self.dt*(k1_m + k2_m)/2
            a_m = a_t[i + 1]
            # print("imag part of sx:", (np.trace(rho_mplus1 @ sigmax) / Z).imag)
            
            k1_m = f_damp(a_m, x_m)
            u1 = self.system_evo(self.dt/2, self.g, a_m, k1_m)
            u2 = self.system_evo(self.dt/2, self.g, a_m + (self.dt/2)*k1_m, k1_m)
            u = np.einsum('ab,bc->abc', u1.T, u2.T)
            
            evol_tens = contract('aic,bid->badc', u, self.finf[:, 1:, :])

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
        
        a_mid = a0 + dadt*dt/2
        
        u_for = expm(-1j * dt * (self.h_s + g * (2 * np.real(a_mid)) * sigmax))
        u_bac = u_for.conj().T
        
        return np.kron(u_for, u_bac.T)

    ####################################
    ## Construct influence functional ##
    ####################################

    def make_finf(self, alg: str, cutoff: float, maxdim: Optional[int] = None):
        """
        Constructs the time translationally invariant influence functional F_inf using the iTEBD construction by Link et al
        """

        eta = np.zeros(self.n_c, dtype=complex)

        print("Computing eta coefficients...")
        eta[0] = self.bath.eta_pp_tt_k(self.dt)

        tstart = time()
        
        for i in range(1, self.n_c):
            sys.stdout.write(f"\rProgress: {100*i/self.n_c:.2f}%, Elapsed time (s): {time()-tstart:.2f}")
            sys.stdout.flush()

            eta[i] = self.bath.eta_pp_tt_kk(self.dt, i)

        print()
        
        augdim = self.dh**2 + 1

        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])
        sigmay = 1j * sigmax@sigmaz        
        states = np.diagonal(sigmaz)
        
        iden = np.eye(augdim, dtype=complex)

        # log I_k = - (sf - sb) * (Reη O^- + 1j*Imη O^+)
        # log I_k = - η [(sf - sb) * sf] - η† [(sf - sb) * sb]

        # Inefficient way:
        # s_f = np.kron( np.diag(self.states), np.eye(dh) )
        # s_b = np.kron( np.eye(dh), np.diag(self.states) )
        # o_minus = s_f - s_b
        # o_plus = s_f + s_b

        # Store only diagonals, since all the operators are diagonal:
        # Put auxiliary index first:
        s_f = np.zeros(augdim, dtype=complex)
        s_b = np.zeros(augdim, dtype=complex)
        s_f[1:] = np.repeat(states, self.dh) # a (dl x dl) representation of the operator S_f ⊗ I_b
        s_b[1:] = np.tile(states, self.dh) # I_f ⊗ S_b

        o_minus = s_f - s_b
        o_plus = s_f + s_b
        
        o_fm = np.outer(s_f, o_minus)
        o_bm = np.outer(s_b, o_minus)
                
        a = np.ones((augdim, 1, 1))
        b = np.ones((augdim, 1, 1))
        s = np.array([1.0])

        if alg == "mbh_tebd":
            psi = mps.uMPS([a, b], False)
        elif alg == "ov_tebd":
            psi = mps.uMPS([a, s, b, s], True)
        else:
            raise RuntimeError("Invalid alg specification")

        print("Building IF...")
        
        tstart = time()
        
        for k in range(1, self.n_c):

            sys.stdout.write(f"\rProgress: {100*k/self.n_c:.2f}%, Elapsed time (s): {time()-tstart:.2f}")
            sys.stdout.flush()
            
            phi_k = -eta[-k] * o_fm + np.conj(eta[-k]) * o_bm
            I_k = np.exp(phi_k)

            gate = np.einsum('ix,ij,xy->ixjy', I_k, iden, iden)

            if alg == "mbh_tebd":
                s = psi.step_itebd_mbh(gate, s, cutoff, p=1.0)
            elif alg == "ov_tebd":
                psi.step_itebd_ov(gate, cutoff, p=1.0)

            if psi.istrivial():
                psi.trivialize()
                s = np.array([1.0])
                
        print()
        
        phi_k = -eta[0] * np.diagonal(o_fm) + np.conj(eta[0]) * np.diagonal(o_bm)
        I_k = np.exp(phi_k)

        gate = contract('i,ix,ij->xij', I_k, iden, iden)
        
        if alg == "ov_tebd":
            finf = contract('ikl,uv,kvw,wx,lxy->uiy', gate, np.diag(psi.tensors[3]), psi.tensors[0], np.diag(psi.tensors[1]), psi.tensors[2])
        elif alg == "mbh_tebd":
            finf = contract('ikl,kxy,lyz->xiz', gate, psi.tensors[0], psi.tensors[1])

        print('Bond dimension: ', finf.shape[0])
        
        #nvecs = 50
        nvecs = max(100, int(np.ceil(0.1 * finf.shape[0])))

        ws, v_r_vecs = eigs(finf[:, 0, :], nvecs, which='LM')
        r_max = np.argmax(np.abs(ws))
        w = ws[r_max]
        ws, v_l_vecs = eigs(finf[:, 0, :].T, nvecs, which='LM')
        l_max = np.argmax(np.abs(ws))
        w = ws[l_max]
        bath_tr = v_r_vecs[:, r_max]
        bath_0 = v_l_vecs[:, l_max] / (v_l_vecs[:, l_max] @ v_r_vecs[:, l_max])

        return finf, bath_tr, bath_0
    
    ##################
    ## Steady state ##
    ##################

    def analytic_gc(self, alpha: float, s: float, wc: float, T: float, wz: float, omega_cav: float, kappa: float):
        """
        Computes the analytically predicted value of g_c for the superradiant transition in the mean field open Dicke model with a local dephasing bath specified by the coupling strength (alpha), Ohmicity (s), and exponential cut off frequency (wc)
        """
        assert T == 0.0, "Finite temperatures not supported yet"

        e = 2*wz/wc
        if s == 1.0:
            chi = integrate.quad(
                lambda t: (1 + t)**(-2*alpha) * np.exp(-e*t),
                0.0, np.inf
            )[0]
            chi *= (2/wc)
        else:
            f = 2 * alpha * gamma(s) / (1-s)
            chi = integrate.quad(
                lambda t: np.exp(-e*t) * np.exp(-f * (1+t)**(1-s)),
                0.0, np.inf
            )[0]
            chi *= (2 / wc) * np.exp(f)
            
        return np.sqrt((omega_cav**2 + kappa**2) / (2 * omega_cav * chi))


    ##########################################
    ## Constrained phase boundary iteration ##
    ##########################################

    def a_mag_adiabatic_cavity_constrained(self, sx: float, omega_cav: float, kappa: float, g: float) -> float:
        """
        Computes the steady state value of <sigma_x> from its relation to the cavity degree of freedom
        """

        return -g * sx  / np.sqrt((omega_cav**2) + (kappa**2))
    
    def dicke_steadystate_constrained(self, b0: Optional[float] = 5.0) -> float:
        
        # Assume that g > g_c
        
        a_mag = brentq(lambda x: self.dicke_steadystate_constrained_diff(x), 0.01, b0)
        print("Steady state photon number =", a_mag**2)
        
        return a_mag
    
    def dicke_steadystate_constrained_diff(self, a_mag_cur: float, g: float) -> float:
       
        unit_vec = (self.omega_cav + 1j*self.kappa)/np.sqrt((self.omega_cav**2) + (self.kappa**2))
        sigmax = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        u1 = self.system_evo(self.dt/2, g, a_mag_cur * unit_vec, 0.0 + 1j*0.0)
        u = np.einsum('ab,bc->abc', u1.T, u1.T)
        
        evol_tens = contract('aic,bid->badc', u, self.finf[:, 1:, :])
        
        ws, v_vecs = eigs(evol_tens.reshape([evol_tens.shape[0] * evol_tens.shape[1], evol_tens.shape[2] * evol_tens.shape[3]]).T, 50, which='LM')
        l_max = np.argmax(np.abs(ws))
        w = ws[l_max]
        v = v_vecs[:, l_max]
        
        rho_ss = (self.bath_tr @ v.reshape([self.bath_tr.size, self.dh**2])).reshape([self.dh, self.dh])

        sx = (np.trace(sigmax @ rho_ss) / np.trace(rho_ss)).real
            
        a_mag_next = self.a_mag_adiabatic_cavity_constrained(sx, self.omega_cav, self.kappa, self.g)
        
        return a_mag_next - a_mag_cur
        
    def get_interval(self, ff, b0, bf):
    
        brange = np.linspace(b0, bf, 50)
        
        res = np.array([ff(x) for x in brange])
        
        i_min = np.argmin(res)
        i_max = np.argmax(res)
        
        if res[i_min] * res[i_max] < 0:
            return brange[i_min], brange[i_max]
        else:
            raise RuntimeError("cant compute gc")
        
    def dicke_steadystate_constrained_phase_boundary(self, g0: float, niters: Optional[int] = 6, b0: Optional[float] = 5.0) -> Union[np.ndarray, np.ndarray]:

        # Assume that g0 > g_c

        gList = [g0+0.1, g0]
        nList = [0.0, 0.0]

        left, right = self.get_interval(lambda x: self.dicke_steadystate_constrained_diff(x, gList[-2]), 0.01, b0)
        nList[0] = brentq(lambda x: self.dicke_steadystate_constrained_diff(x, gList[-2]), left, right) ** 2

        left, right = self.get_interval(lambda x: self.dicke_steadystate_constrained_diff(x, gList[-1]), 0.01, b0)
        nList[1] = brentq(lambda x: self.dicke_steadystate_constrained_diff(x, gList[-1]), left, right) ** 2

        # Linear extrapolation
        for i in range(0, niters):

            g1 = gList[-2]
            g2 = gList[-1]
            n1 = nList[-2]
            n2 = nList[-1]
            
            g_c_pred = (n2*g1 - n1*g2)/(n2 - n1)
            print("Predicted gc:", g_c_pred)

            gList.append( (g2 + g_c_pred)/2 )
            left, right = self.get_interval(lambda x: self.dicke_steadystate_constrained_diff(x, gList[-1]), 0.01, b0)
            
            nList.append( brentq(lambda x: self.dicke_steadystate_constrained_diff(x, gList[-1]), left, right) ** 2)
            print(gList[-1], ",", nList[-1])
        
        slope, intercept = np.polyfit(gList[-3:-1], nList[-3:-1], deg = 1)
        return -intercept/slope, gList, nList
