from dataclasses import dataclass, field
import numpy as np


@dataclass
class LstmStates:
    mu: float = 0.0
    var: float = 0.0


@dataclass
class LstmInput:
    look_back_len: int
    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mu = np.zeros(self.look_back_len, dtype=np.float32)
        self.var = np.zeros(self.look_back_len, dtype=np.float32)


@dataclass
class SmootherStates:
    mu_priors: list = field(default_factory=list)
    var_priors: list = field(default_factory=list)
    mu_posteriors: list = field(default_factory=list)
    var_posteriors: list = field(default_factory=list)
    mu_smooths: list = field(default_factory=list)
    var_smooths: list = field(default_factory=list)
    cov_states: list = field(default_factory=list)
