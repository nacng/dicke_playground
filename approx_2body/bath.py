from abc import ABC, abstractmethod
from approx_2body.sim_params import SimulationParamsTEMPO

class Bath(ABC):
    '''
    Abstract class defining a bath spectral density, used for calculations of eta values defining the influence functional

    Attributes
    ----------
    alpha: float
        System-bath coupling strength
    s: float
        Exponent of the low-frequency part of the spectral density
    wc: float
        Frequency cutoff scale of the spectral density
    jw: Callable[[float], float]
        Spectral density of the spin-boson problem
    '''
    def __init__(self, sim_params: SimulationParamsTEMPO):
    
        assert isinstance(sp, SimulationParamsTEMPO)
        
        self.sim_params = sim_params
    
    @abstractmethod
    def eta_pp_tt_kk(self, dt: float, d: int):
        raise NotImplementedError()

    @abstractmethod
    def eta_pp_tt_k(self, dt: float):
        raise NotImplementedError()
