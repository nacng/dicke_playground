# Permutation invariant solver not using the PIQS approach of using a Dicke state basis. This allows for general qudits rather than qubits
# Works in the basis of operators for the qudits as |m><n| where m,n = 1,...,d
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import expm
import scipy.special as sc

import argparse
from typing import List

import os
import copy
from datetime import datetime
from pathlib import Path

# Useful helper functions
def ctz(v: int):
    # https://stackoverflow.com/a/63552117
    """
    Count trailing zeros in the binary representation
    """
    return (v & -v).bit_length() - 1

def nextperm(v: int):
    # credit to Dario Sneidermanis
    # https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    t = v | (v-1)
    return (t + 1) | (((~t & -~t) - 1) >> (ctz(v) + 1))

def num_to_part(n: int, nb: int, v: int):
    # Inspiration from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    """
    For a value v, number of qudits n, and number of bars nb, get a list of the number of objects (=qudits) in each of the nb+1 partitions
    `Bars` are represented by `1` and objects are represented by `0`, i.e., v should have nb number of `1` in its binary representation
    """
    # nb = d^2 - 1
    res = [0] * (nb+1)
    res[-1] = v
    t = v
    for i in range(2, nb+1):
        res[-i] = (t & (t-1))
        t = res[-i]
    for i in range(1, nb):
        res[-i] = res[-i] - res[-(i+1)]
    res[0] = (n+nb-1) - ctz(res[1])
    for i in range(1, nb):
        res[i] = ctz(res[i]) - ctz(res[i+1]) - 1
    res[nb] = ctz(res[nb])

    return res

def part_to_num(partition: List[int]):
    # Assumed convention: [n_1, ..., n_{D^2-1}]
    """
    For a given partition, return the unique integer corresponding to it
    """
    acc = partition[-1]
    res = 1 << acc
    for i in range(2, len(partition)):
        acc += partition[-i] + 1
        res += (1 << acc)
    return res

def multinomial(lst: List[int]):
    # https://stackoverflow.com/a/46378809
    """
    Returns the multinomial coefficient (a1 + ... + aN)
                                        ( a1, ..., aN )
    """
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def log_multinomial(lst: List[int]):
    # https://stackoverflow.com/a/46378809
    """
    Returns the log of the multinomial coefficient
    """
    res, i = 0.0, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res += np.log(i) - np.log(j)
            i -= 1
    return res

# def reversed_num_to_part(n: int, nb: int, v: int):
#     # Inspiration from https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
#     # nb = d^2 - 1
#     res = [0] * (nb+1)
#     res[0] = v
#     t = v
#     for i in range(1, nb):
#         res[i] = (t & (t-1))
#         t = res[i]
#     for i in range(0, nb-1):
#         res[i] = res[i] - res[i+1]
#     res[nb] = (n+nb-1) - ctz(res[nb-1])
#     for i in range(nb-1, 0, -1):
#         res[i] = ctz(res[i]) - ctz(res[i-1]) - 1
#     res[0] = ctz(res[0])
#     return res

class CQED_Markovian():

    # Allow for generalized Dicke model (unequal rotating and counterrotating terms)
    def __init__(self, n_ph: int, n_sp: int, omega_cav: float, kappa: float, g_r: float, g_cr: float, h_0: np.array, gamma_diss: float, gamma_pump: float, gamma_deph: float):

        self.omega_cav = omega_cav
        self.g_r = g_r
        self.g_cr = g_cr
        self.kappa = kappa
        
        self.h_0 = h_0
        self.d = h_0.shape[0]
        self.n_sp = n_sp
        self.gamma_diss = gamma_diss
        self.gamma_pump = gamma_pump
        self.gamma_deph = gamma_deph
        
        # Maximum photon occupation number
        self.n_ph = n_ph
        self.ph_tr = np.eye(n_ph+1).flatten('C')

        self._a = np.diag([np.sqrt(i) for i in range(self.n_ph, 0, -1)], k=-1)
        self._q = self._a + (self._a).T
        self._p = -1j * (self._a - (self._a).T)
        self._n = np.diag([i for i in range(self.n_ph, -1, -1)])

        self.basis, self.prebasis = self.op_basis()

        # Convention: kron(sites, photon)
        self.tr_vec = self.full_trace()
        
        
    ############################
    # Liouvillian construction #
    ############################

    def _ph_comm(self, mu: str, sgn = -1):
        """
        Returns a representation of the commutator superoperator [O_0, ...], where
        O_0 is an observable (hermitian) on the photon hilbert space

        The elements of this representation are given by
        ⟨s'_f| [σ^μ, |s_f⟩⟨s_b|] |s'_b⟩ = ⟨s'_f|σ^μ|s_f⟩ δ_{sb,s'b} - δ_{sf,s'f} ⟨s_b|σ^μ|s'_b⟩
        where states are ordered s_f = (0, 1) == (+, -)
        """
        dp = self.n_ph + 1 #+1 to include the 0 state
        #a, q, p, n = self._make_ph_ops

        if mu.lower() == "q":
            op = self._q
        elif mu.lower() == "p":
            op = self._p
        elif mu.lower() == "n":
            op = self._n
        else:
            raise RuntimeError("Invalid photon observable specification")

        # Transpose result so that 1st index = in, 2nd index = out
        return np.transpose( np.kron(op, np.eye(dp)) + np.sign(sgn) * np.kron(np.eye(dp), np.transpose(op)) )

    def _ph_prop(self):

        L_unit = (-1j*self.omega) * self._ph_comm("n", sgn=-1)
        L_lind = np.transpose( np.kron(self._a, (self._a) ) ) #two transposes
        L_lind += -0.5*self._ph_comm("n", sgn=1)
        L_lind *= 2*self.kappa

        L = L_unit + L_lind

        return expm(L*self.delta)
    
    def make_j(self, d: int, mu: int):

        s = (d-1)/2
        
        if mu == 1:
            dg = [0.5*np.sqrt(2*(s+1)*(i-1) - i*(i-1)) for i in range(2, d+1)]
            return np.diag(dg, k=-1) + np.diag(dg, k=1)
        elif mu == 2:
            dg = [1j*0.5*np.sqrt(2*(s+1)*(i-1) - i*(i-1)) for i in range(2, d+1)]
            return np.diag(dg, k=-1) - np.diag(dg, k=1)
        elif mu == 3:
            return np.diag([s + 1 - i for i in range(1, d+1)])
        else:
            raise RuntimeError("Invalid Pauli operator specification")

    def op_basis(self):

        # Idea: generate all possible partitions of N (including zeros) using the stars-and-bars representation
        # These partitions have bit representations that can be generated by looping over all lexicographically ordered permutations

        N = self.n_sp
        nb = (self.d**2) - 1
        n_states = sc.comb(N+nb, nb, exact=True) #(N+3)*(N+2)*(N+1)//6
        v = 2**nb - 1
        map_states = [0] * n_states
        
        for i in range(0, n_states):
            map_states[i] = v
            v = nextperm(v)
        
        return map_states, {map_states[i]: i for i in range(0, n_states)}
        
    def site_liou(self, liou: np.ndarray, thresh: float = 1e-14):
        """
        Liouvillian for a single site, represented as a dense matrix
        """
        N = self.n_sp
        nb = (self.d**2) - 1
        
        # As dense matrix, column format
        dL = len(self.basis)
        res = np.zeros((dL, dL), dtype=liou.dtype)
        for i in range(0, dL):
            irep = num_to_part(N, nb, self.basis[i])
            for s1 in range(0, nb+1):
                if irep[s1] == 0:
                    continue
                for s2 in range(0, nb+1):
                    if np.abs(liou[s1, s2]) < thresh:
                        continue
                    if s1 == s2:
                        res[i, i] += (liou[s1, s1] * (irep[s1]/N))
                        continue
                    if irep[s2] == N:
                        continue
                    newrep = irep.copy()
                    newrep[s1] -= 1
                    newrep[s2] += 1
                    res[i, self.prebasis[part_to_num(newrep)]] += (liou[s1, s2] * np.sqrt(newrep[s2]*irep[s1])/N)

        # Sparsity data:
        #  - For the anticommutator of a 2x2 hamiltonian, the Liouvillian has a density of 10.7% for N=5, 2.5% for N=10, 0.45% for N=20
        return res

    def site_liou_sparse(self, liou: np.ndarray, thresh: float = 1e-14):
        """
        Liouvillian for a single site, represented as a sparse matrix
        """
        N = self.n_sp
        nb = (self.d**2) - 1
        
        # As sparse matrix, constructing in CSR format
        dL = len(self.basis)
        indices = []
        indptr = [0]
        data = []
        for i in range(0, dL):
            irep = num_to_part(N, nb, self.basis[i])
            tmpind = []
            tmpdata = []
            tr = 0 * liou[0,0]
            for s1 in range(0, nb+1):
                if irep[s1] == 0:
                    continue
                for s2 in range(0, nb+1):
                    if np.abs(liou[s1, s2]) < thresh:
                        continue
                    if s1 == s2:
                        tr += (liou[s1, s1] * (irep[s1]/N))
                        continue
                    if irep[s2] == N:
                        continue
                    newrep = irep.copy()
                    newrep[s1] -= 1
                    newrep[s2] += 1
                    tmpind += [self.prebasis[part_to_num(newrep)]]
                    tmpdata += [liou[s1, s2] * np.sqrt(newrep[s2]*irep[s1])/N]
            if np.abs(tr) >= thresh: 
                tmpind += [i]
                tmpdata += [tr]
            indsort = np.argsort(tmpind).tolist()
            indices += [tmpind[k] for k in indsort]
            data += [tmpdata[k] for k in indsort]
            indptr += [indptr[-1] + len(indsort)]

        return sp.csr_array((data, indices, indptr), shape=(dL, dL))

    def total_liou(self, s_plus: np.ndarray, s_minus: np.ndarray, s_z: np.ndarray):
        """
        Construct the total Liouvillian (qudits + truncated photon) as a dense matrix
        """
        d = self.d
        dp = self.n_ph + 1
        dL = len(self.basis)
        
        comm_sp = self.site_liou(np.kron(s_plus, np.eye(d)) - np.kron(np.eye(d), np.transpose(s_plus)) )
        acomm_sp = self.site_liou(np.kron(s_plus, np.eye(d)) + np.kron(np.eye(d), np.transpose(s_plus)) )
        acomm_sp /= 2

        comm_sm = self.site_liou(np.kron(s_minus, np.eye(d)) - np.kron(np.eye(d), np.transpose(s_minus)) )
        acomm_sm = self.site_liou(np.kron(s_minus, np.eye(d)) + np.kron(np.eye(d), np.transpose(s_minus)) )
        acomm_sm /= 2

        Lop = self.gamma_diss * (np.kron(s_minus, np.transpose(s_plus)) - np.kron(s_plus@s_minus, 0.5*np.eye(d)) - np.kron(0.5*np.eye(d), np.transpose(s_plus@s_minus)))
        Lop += self.gamma_pump * (np.kron(s_plus, np.transpose(s_minus)) - np.kron(s_minus@s_plus, 0.5*np.eye(d)) - np.kron(0.5*np.eye(d), np.transpose(s_minus@s_plus)))
        Lop += self.gamma_deph * (np.kron(s_z, np.transpose(s_z)) - np.kron(s_z@s_z, 0.5*np.eye(d)) - np.kron(0.5*np.eye(d), np.transpose(s_z@s_z)))
        Lop *= 1j
        Lop += np.kron(self.h_0, np.eye(d)) - np.kron(np.eye(d), np.transpose(self.h_0))
        L_site = self.site_liou(Lop)
        L_site *= self.n_sp # in cases without permutational invariance, this would have to sum over all the different L_site

        L_ph = (self.kappa * 1j) * (2*np.kron(self._a, self._a) - np.kron(self._n, np.eye(dp)) - np.kron(np.eye(dp), self._n))
        L_ph += np.kron(self._n, self.omega_cav*np.eye(dp)) - np.kron(self.omega_cav*np.eye(dp), self._n)

        # Light matter coupling terms
        tmp_site = self.g_r*acomm_sp + self.g_cr*acomm_sm
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = np.kron(self._a, np.eye(dp, dtype=np.complex128)) - np.kron(np.eye(dp, dtype=np.complex128), np.transpose(self._a))
        res = np.kron(tmp_site, tmp_ph)
        tmp_site = self.g_r*comm_sp + self.g_cr*comm_sm
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = np.kron(self._a, np.eye(dp)) + np.kron(np.eye(dp), np.transpose(self._a))
        tmp_ph /= 2
        res += np.kron(tmp_site, tmp_ph)

        tmp_site = self.g_r*acomm_sm + self.g_cr*acomm_sp
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = np.kron(np.transpose(self._a), np.eye(dp)) - np.kron(np.eye(dp), self._a)
        res += np.kron(tmp_site, tmp_ph)
        tmp_site = self.g_r*comm_sm + self.g_cr*comm_sp
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = np.kron(np.transpose(self._a), np.eye(dp)) + np.kron(np.eye(dp), self._a)
        tmp_ph /= 2
        res += np.kron(tmp_site, tmp_ph)

        # Site only terms
        res += np.kron(L_site, np.eye(dp**2))

        # Photon only term
        res += np.kron(np.eye(dL), L_ph)

        return res

    def total_liou_sparse(self, s_plus: np.ndarray, s_minus: np.ndarray, s_z: np.ndarray):
        """
        Construct the total Liouvillian (qudits + truncated photon) as a sparse matrix
        """
        
        d = self.d
        dp = self.n_ph + 1
        dL = len(self.basis)

        a = sp.csr_matrix(self._a)
        a_dag = sp.csr_matrix(np.transpose(self._a))
        n = sp.csr_matrix(self._n)

        eye_ph = sp.eye(dp, dtype=np.complex128, format='csr')
        
        comm_sp = self.site_liou_sparse(np.kron(s_plus, np.eye(d)) - np.kron(np.eye(d), np.transpose(s_plus)) )
        acomm_sp = self.site_liou_sparse(np.kron(s_plus, np.eye(d)) + np.kron(np.eye(d), np.transpose(s_plus)) )
        acomm_sp /= 2

        comm_sm = self.site_liou_sparse(np.kron(s_minus, np.eye(d)) - np.kron(np.eye(d), np.transpose(s_minus)) )
        acomm_sm = self.site_liou_sparse(np.kron(s_minus, np.eye(d)) + np.kron(np.eye(d), np.transpose(s_minus)) )
        acomm_sm /= 2

        Lop = self.gamma_diss * (np.kron(s_minus, np.transpose(s_plus)) - np.kron(s_plus@s_minus, 0.5*np.eye(d)) - np.kron(0.5*np.eye(d), np.transpose(s_plus@s_minus)))
        Lop += self.gamma_pump * (np.kron(s_plus, np.transpose(s_minus)) - np.kron(s_minus@s_plus, 0.5*np.eye(d)) - np.kron(0.5*np.eye(d), np.transpose(s_minus@s_plus)))
        Lop += self.gamma_deph * (np.kron(s_z, np.transpose(s_z)) - np.kron(s_z@s_z, 0.5*np.eye(d)) - np.kron(0.5*np.eye(d), np.transpose(s_z@s_z)))
        Lop *= 1j
        Lop += np.kron(self.h_0, np.eye(d)) - np.kron(np.eye(d), np.transpose(self.h_0))
        L_site = self.site_liou_sparse(Lop)
        L_site *= self.n_sp # in cases without permutational invariance, this would have to sum over all the different L_site

        L_ph = (self.kappa * 1j) * (2*sp.kron(a, a) - sp.kron(n, eye_ph) - sp.kron(eye_ph, n))
        L_ph += sp.kron(n, self.omega_cav*eye_ph) - sp.kron(self.omega_cav*eye_ph, n)

        # Light matter coupling terms
        tmp_site = self.g_r*acomm_sp + self.g_cr*acomm_sm
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = sp.kron(a, eye_ph) - sp.kron(eye_ph, a_dag)
        res = sp.kron(tmp_site, tmp_ph)
        tmp_site = self.g_r*comm_sp + self.g_cr*comm_sm
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = sp.kron(a, eye_ph) + sp.kron(eye_ph, a_dag)
        tmp_ph /= 2
        res += sp.kron(tmp_site, tmp_ph)

        tmp_site = self.g_r*acomm_sm + self.g_cr*acomm_sp
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = sp.kron(a_dag, eye_ph) - sp.kron(eye_ph, a)
        res += sp.kron(tmp_site, tmp_ph)
        tmp_site = self.g_r*comm_sm + self.g_cr*comm_sp
        tmp_site *= self.n_sp # from sum over N sites
        tmp_site /= np.sqrt(self.n_sp)
        tmp_ph = sp.kron(a_dag, eye_ph) + sp.kron(eye_ph, a)
        tmp_ph /= 2
        res += sp.kron(tmp_site, tmp_ph)

        # Site only terms
        res += sp.kron(L_site, sp.eye(dp**2, dtype=np.complex128, format='csr'))

        # Photon only term
        res += sp.kron(sp.eye(dL, dtype=np.complex128, format='csr'), L_ph)

        return res

    
    ################
    # Measurements #
    ################

    def trace_spins(self, rho: sp.csr_array):
        """
        Trace over the spins to give the photon reduced density matrix
        """
        N = self.n_sp
        d = self.d
        dp = self.n_ph + 1
        dL = len(self.basis)
        
        diag_pos = [n*(d+1) for n in range(0,d)]

        n_diag = sc.comb(N+d-1, d-1, exact=True)
        #diag_states = [0] * n_diag

        res = np.zeros((dp, dp), dtype=np.complex128)
        v = 2**(d-1) - 1
        for i in range(0, n_diag):
            diag_occs = num_to_part(N, d-1, v)
            factor = multinomial(diag_occs)
            part = [0] * (d*d)
            for j in range(0, d):
                part[diag_pos[j]] = diag_occs[j]
            spin_state = self.prebasis[part_to_num(part)]
            # spinvec = sp.csr_matrix(([1], [part_to_num(part)], [0, 1]), shape=(1, dL))
            # diag_states[i] = self.prebasis[part_to_num(part)]

            # Loop over photon density matrix elements
            for m in range(0, dp):
                for n in range(0, dp):
                    # phvec = sp.csr_matrix(([1], [m*dp + n], [0, 1]), shape=(1, dp**2))
                    # totvec = sp.kron(spinvec, phvec)
                    # totvec = sp.csr_matrix(([1], [spin_state*(dp**2) + (m*dp + n)], [0, 1]), shape=(1, dL*(dp**2)))
                    res[m, n] += factor * rho[0, spin_state*(dp**2) + (m*dp + n)]
                    
            
            v = nextperm(v)

        return res

    def trace_ph(self, rho: sp.csr_array):
        """
        Trace over the photon to give the qudits reduced density matrix
        """
        N = self.n_sp
        d = self.d
        dp = self.n_ph + 1
        dL = len(self.basis)
        
        diag_pos = [n*(d+1) for n in range(0,d)]

        n_diag = sc.comb(N+d-1, d-1, exact=True)
        #diag_states = [0] * n_diag

        traced = np.zeros(dL, dtype=np.complex128)
        v = 2**(d-1) - 1
        # Loop over photon density matrix elements
        for i in range(0, dL):
            for m in range(0, dp):
                traced[i] += rho[0, spin_state*(dp**2) + (m*dp + m)]

        # Unfold vector into a density matrix
        for i in range(0, dL):
            diag_occs = num_to_part(N, d-1, v)
            factor = multinomial(diag_occs)
            part = [0] * (d*d)
            for j in range(0, d):
                part[diag_pos[j]] = diag_occs[j]
            spin_state = self.prebasis[part_to_num(part)]
            # spinvec = sp.csr_matrix(([1], [part_to_num(part)], [0, 1]), shape=(1, dL))
            # diag_states[i] = self.prebasis[part_to_num(part)]

            for i in range(0, n_diag):
                # TODO
                continue
            
            v = nextperm(v)

        return res
    
    def full_trace(self):
        """
        Construct the full trace over the photon and qudits as a (sparse) vector dual to operator vectors
        """
        N = self.n_sp
        d = self.d
        dp = self.n_ph + 1
        dL = len(self.basis)
        
        diag_pos = [n*(d+1) for n in range(0,d)]

        n_diag = sc.comb(N+d-1, d-1, exact=True)
        #diag_states = [0] * n_diag

        #tr_vec = np.zeros((dp**2)*dL, dtype=float)
        tr_vec = np.zeros(dL, dtype=float)
        v = 2**(d-1) - 1
        for i in range(0, n_diag):
            diag_occs = num_to_part(N, d-1, v)
            #factor = multinomial(diag_occs)
            logfactor = log_multinomial(diag_occs)
            part = [0] * (d*d)
            for j in range(0, d):
                part[diag_pos[j]] = diag_occs[j]
            spin_state = self.prebasis[part_to_num(part)]
            # spinvec = sp.csr_matrix(([1], [part_to_num(part)], [0, 1]), shape=(1, dL))
            # diag_states[i] = self.prebasis[part_to_num(part)]


            """
            rewrite for large N so that tr_vec keeps logfactor
            inner products will have to be specialized so that logfactors are added and then exponentiated
            """
            
            #tr_vec[spin_state] = np.sqrt(factor)
            tr_vec[spin_state] = np.exp(0.5 * logfactor)
            # # Loop over photon density matrix elements
            # for m in range(0, dp):
            #     tr_vec[spin_state*(dp**2) + (m*dp + m)] = np.sqrt(factor)

            v = nextperm(v)

        tr_vec_ph = np.eye(dp).flatten()
        
        return sp.kron(sp.csr_array(tr_vec), sp.csr_array(tr_vec_ph)) #sp.csr_array(tr_vec) # tr_vec

    def obs(self, op_site: np.ndarray, op_ph: np.ndarray):
        """
        Implements a symmetric superoperator {A⊗ B, ...} = ( [A, [B, ...]] + {A, {B, ...}} )/2
        
        Constructs it as a vector, dual to the operator vector
        """
        d = self.d
        dp = self.n_ph + 1
        dL = len(self.basis)

        eye_ph = sp.eye(dp, dtype=np.complex128, format='csr')

        comm_site = self.site_liou_sparse(np.kron(op_site, np.eye(d)) - np.kron(np.eye(d), np.transpose(op_site)) )
        acomm_site = self.site_liou_sparse(np.kron(op_site, np.eye(d)) + np.kron(np.eye(d), np.transpose(op_site)) )

        comm_ph = sp.kron(op_ph, eye_ph) - sp.kron(eye_ph, np.transpose(op_ph))
        acomm_ph = sp.kron(op_ph, eye_ph) + sp.kron(eye_ph, np.transpose(op_ph))
        
        # Light matter coupling terms
        res = sp.kron(comm_site, comm_ph)
        res += sp.kron(acomm_site, acomm_ph)
        res /= 4
        
        # # Photon only term
        # res = sp.kron(sp.eye(dL, dtype=np.complex128, format='csr'), L_ph)

        # return res # Superoperator
        return (self.tr_vec).dot(res) # Dual operator vector

    ####################
    # Time propagation #
    ####################

    def init_state(self, rho_sp: np.array, rho_ph: np.array, thresh: float = 1e-14):
        """
        Returns the initial state in this site-local permutationally invariant basis
        """
        N = self.n_sp
        d = self.d
        nb = (self.d**2) - 1
        dp = self.n_ph + 1
        dL = len(self.basis)

        nonzeros = np.nonzero(np.abs(rho_sp.flatten()) > thresh)[0]
        rho_mag = np.log(np.abs(rho_sp.flatten()[nonzeros]))
        rho_phase = np.angle(rho_sp.flatten()[nonzeros])
        
        ρ_sp = np.zeros(dL, dtype=np.complex128)
        
        # Construct rho vector for spins in the operator occupation basis |n₁, ..., n_d²)
        for i in range(0, dL):
            irep = num_to_part(N, nb, self.basis[i])
            tmp_mag = 0.5 * log_multinomial(irep)
            tmp_phase = 0.0

            if sum([irep[k] for k in nonzeros]) != N:
                continue

            for k in range(0, len(nonzeros)):
                tmp_mag += irep[nonzeros[k]] * rho_mag[k]
                tmp_phase += irep[nonzeros[k]] * rho_phase[k]

            ρ_sp[i] = np.exp(1j * tmp_phase)
            ρ_sp[i] *= np.exp(tmp_mag)        

        
        return np.kron(ρ_sp, rho_ph.flatten())

    def propagate(self, liou, rho_init, dt: float, n_steps: int, measurements: List, filename: str = None):
        # `measurements` is assumed to be a list of dual operator vectors

        # Measurements are assumed to be of Hermitian operators
        res = np.zeros((n_steps+1, len(measurements)), dtype=float)
        t_list = dt * np.arange(0, n_steps+1)

        rho = copy.copy(rho_init)

        def f_ode(t, rho_vec):
            return liou @ rho_vec

        ntgr = scipy.integrate.ode(f_ode)

        # qutip defaults
        ntgr.set_integrator('zvode', method='bdf', order=12,
                              atol=1e-8, rtol=1e-6, nsteps=2500,
                              first_step=0, min_step=0,
                              max_step=0)
        ntgr.set_initial_value(rho_init)

        for m in range(0, len(measurements)):
            res[0, m] = np.real(measurements[m].dot(rho_init))
        
        savelist = [n*int(np.floor(n_steps/10)) for n in range(10)] + [n_steps-1]
        for i in range(0, n_steps):

            ntgr.integrate(ntgr.t + dt)
            
            # Measure
            for m in range(0, len(measurements)):
                res[i+1, m] = np.real(measurements[m].dot(ntgr.y))

            if filename != None:

                if len(savelist) > 1 and i == savelist[1]:

                    with open(filename, 'a') as f:
                        np.savetxt(f, np.column_stack((t_list[savelist[0]:savelist[1]], res[savelist[0]:savelist[1], :])), delimiter=',')

                    savelist.pop(0)
                
                
        return res
    
    
if __name__=="__main__":
    
    print("main")
