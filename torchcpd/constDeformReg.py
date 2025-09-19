from builtins import super
import numpy as np
import numbers
from .deformReg import DeformableRegistration
import torch as th

class ConstrainedDeformableRegistration(DeformableRegistration):
    """
    Constrained deformable registration.
    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.
    beta: float(positive)
        Width of the Gaussian kernel.
    e_alpha: float (positive)
        Reliability of correspondence priors. Between 1e-8 (very reliable) and 1 (very unreliable)
    
    source_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the source array
    target_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the target array
    """

    def __init__(self, e_alpha = None, source_id = None, target_id= None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if e_alpha is not None and (not isinstance(e_alpha, numbers.Number) or e_alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(e_alpha))
        
        if not isinstance(source_id, np.ndarray) or source_id.ndim != 1:
            raise ValueError(
                "The source ids (source_id) must be a 1D numpy array of ints.")
        
        if not isinstance(target_id, np.ndarray) or target_id.ndim != 1:
            raise ValueError(
                "The target ids (target_id) must be a 1D numpy array of ints.")

        self.e_alpha = th.tensor(1e-8, dtype=th.float64).float().to(self.device) if e_alpha is None else e_alpha
        if not isinstance(self.e_alpha, th.Tensor):
            self.e_alpha = th.tensor(self.e_alpha, dtype=th.float64).float().to(self.device)
        self.source_id = th.tensor(source_id, dtype=th.int64).to(self.device)
        self.target_id = th.tensor(target_id, dtype=th.int64).to(self.device)
        self.P_tilde = th.zeros((self.M, self.N), dtype=th.float64).float().to(self.device)
        self.P_tilde[self.source_id, self.target_id] = 1.
        self.P1_tilde = th.sum(self.P_tilde, dim=1)
        self.PX_tilde = th.mm(self.P_tilde, self.X)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if self.low_rank is False:
            A = th.mm(th.diag(self.P1.reshape(-1, )), self.G) + \
                self.sigma2*(th.div(1., self.e_alpha))*th.mm(th.diag(self.P1_tilde), self.G) + \
                self.alpha * self.sigma2 * (th.eye(self.M, dtype=th.float64).float().to(self.device))
            B = self.PX - th.mm(th.diag(self.P1.reshape(-1, )), self.Y) + self.sigma2*(th.div(1., self.e_alpha))*(self.PX_tilde - th.mm(th.diag(self.P1_tilde), self.Y)) 
            self.W = th.linalg.solve(A, B)

        else:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = th.diag(self.P1.reshape(-1, )) + self.sigma2*(th.div(1.,self.e_alpha))*th.diag(self.P1_tilde)
            dPQ = th.mm(dP, self.Q)
            F = self.PX - th.mm(th.diag(self.P1.reshape(-1, )), self.Y) + self.sigma2*(th.div(1.,self.e_alpha))*(self.PX_tilde - th.mm(th.diag(self.P1_tilde), self.Y)) 

            self.W = th.div(1., (self.alpha * self.sigma2)) * (F - th.mm(dPQ, (
                th.linalg.solve((self.alpha * self.sigma2 * self.inv_S + th.mm(self.Q.T, dPQ)),
                                (th.mm(self.Q.T, F))))))
            QtW = th.mm(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * th.trace(th.mm(QtW.T, th.mm(self.S, QtW)))