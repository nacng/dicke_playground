import numpy as np
from numpy import linalg
from typing import List, Optional
from opt_einsum import contract

from approx_2body.utilities import svd_truncate, reshape_rq, reshape_qr

class uMPS():
    '''
    Infinite/uniform matrix product state, assumed to have a two-site unit cell

    Attributes
    ----------
    tensors: List[np.ndarray]
        List of tensors, defining the uniform matrix product state. Must either have two elements or four elements, see `vidal_form`
    vidal_form: bool
        Boolean specifying whether the list of tensors `tensors` is in Vidal's representation or not
    '''
    def __init__(self, tensors: List[np.ndarray], vidal_form: bool):
        
        # Convention: indices ordered anticlockwise from top (= site index)
        #      0          0
        #      |          |
        #    |---|      |---|
        #  1-| A |-2  1-| B |-2
        #    |---|      |---|
        self.tensors = tensors
        self.vidal_form = vidal_form
        # If vidal_form, expect tensors = [a, s_ab, b, s_bc, c, ...]

        if vidal_form:
            assert len(tensors) == 4, "Only 2 site unit cells are supported right now"
        else:
            assert len(tensors) == 2, "Only 2 site unit cells are supported right now"

    def istrivial(self):

        if self.vidal_form:
            return np.prod(self.tensors[1].shape) == 1 and np.prod(self.tensors[3].shape) == 1
        else:
            return np.prod(self.tensors[0].shape[1:]) == 1 and np.prod(self.tensors[1].shape[1:]) == 1
            
    def trivialize(self):
        dh = self.tensors[0].shape[0]
        if self.vidal_form:
            self.tensors[0] = np.ones((dh, 1, 1), dtype=complex)
            self.tensors[1] = np.ones(1, dtype=complex)
            self.tensors[2] = np.ones((dh, 1, 1), dtype=complex)
            self.tensors[3] = np.ones(1, dtype=complex)
        else:
            self.tensors[0] = np.ones((dh, 1, 1), dtype=complex)
            self.tensors[1] = np.ones((dh, 1, 1), dtype=complex)
            
    def step_itebd_ov(self, gate: np.ndarray, cutoff: float, p: Optional[float]=1.0):
        '''
        Performs one step of iTEBD using the Orus-Vidal algorithm (arXiv:0711.3960)

        Parameters
        ----------
        gate: np.ndarray
            Nearest-neighboring two-site gate to be contracted
        cutoff: float
            Truncation threshold
        p: float, optional
            p-norm for singular value truncation

        '''

        #
        #      0      3
        #      |      |
        #   |------------|
        #   |    gate    |
        #   |------------|
        #      |      |
        #      1      2
        #

        assert self.vidal_form == True, "Convert the uMPS to Vidal's representation to use the Orus-Vidal formulation of iTEBD"
        
        # a = self.tensors[0]
        # s_ab = self.tensors[1]
        # b = self.tensors[2]
        # s_ba = self.tensors[3]

        threshold = 1e-13
        
        s_ab = self.tensors[1] * linalg.norm(self.tensors[3])
        s_ba = self.tensors[3] / linalg.norm(self.tensors[3])
        s_ba[np.abs(s_ba) < threshold] = threshold
        
        inv_s_ab = 1/s_ba
        
        c = contract('iklj,uv,kvw,wx,lxy,yz->iuzj', gate, np.diag(s_ba), self.tensors[0], np.diag(s_ab), self.tensors[2], np.diag(s_ba))

        U, s_a, Vh = svd_truncate(c, cutoff, [0, 1], [3, 2], p=p)
        
        new_b = np.einsum('bjz,zy->jby', Vh, np.diag(inv_s_ab))
        new_a = np.einsum('xu,iub->ixb', np.diag(inv_s_ab), U)

        self.tensors[0] = new_b
        self.tensors[1] = s_ba
        self.tensors[2] = new_a
        self.tensors[3] = s_a
        
        return
    
    def step_itebd_mbh(self, gate: np.ndarray, s_b: np.ndarray, cutoff: float, p: Optional[float]=2.0):
        '''
        Performs one step of iTEBD using the Hastings modification (arXiv:0903.3253) to the Orus-Vidal algorithm

        Parameters
        ----------
        gate: np.ndarray
            Nearest-neighboring two-site gate to be contracted
        cutoff: float
            Truncation threshold
        p: float, optional
            p-norm for singular value truncation
        '''
        
        #
        #      0      3
        #      |      |
        #   |------------|
        #   |    gate    |
        #   |------------|
        #      |      |
        #      1      2
        #

        assert self.vidal_form == False, "Convert the uMPS from Vidal's representation to use Hasting's modification of iTEBD"
        
        # a = self.tensors[0]
        # b = self.tensors[1]
        
        c = contract('iklj,kxy,lyz->ixzj', gate, self.tensors[0], self.tensors[1])
        theta = contract('x,ixzj->ixzj', s_b, c)

        U, s_a, Vh = svd_truncate(theta, cutoff, [0, 1], [3, 2], p=p)
        s_a_norm = linalg.norm(s_a)
        
        new_b = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
        new_a = contract('ixyj,jzy->ixz', c, np.conj(new_b))

        self.tensors[0] = new_b
        self.tensors[1] = new_a
        
        return s_a / s_a_norm
