from builtins import super
import numpy as np
import numbers
import torch as th
from .emregistration import EMRegistration
from .utils import gaussian_kernel, low_rank_eigen

class DeformableRegistration(EMRegistration):
    """
    Deformable registration.
    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.
    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.alpha = th.tensor(self.alpha, dtype=self.dtype).to(self.device)
        self.beta = th.tensor(self.beta, dtype=self.dtype).to(self.device)
        self.W = th.zeros((self.M, self.D), dtype=self.dtype).to(self.device)
        self.G = gaussian_kernel(self.Y, self.beta).to(self.device)
        self.num_eig = th.tensor(num_eig, dtype=th.int64).to(self.device)
        self.low_rank = low_rank
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = th.diag(th.div(1., self.S))
            self.S = th.diag(self.S)
            self.E = th.tensor(0., dtype=self.dtype).to(self.device)

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if self.low_rank is False:
            A = th.mm(th.diag(self.P1.reshape(-1, )), self.G) + \
                self.alpha * self.sigma2 * th.eye(self.M, dtype=self.dtype).to(self.device)
            B = self.PX - th.mm(th.diag(self.P1.reshape(-1, )), self.Y)
            self.W = th.linalg.solve(A, B)
        else:
            dP = th.diag(self.P1.reshape(-1, ))
            dPQ = th.mm(dP, self.Q)
            F = th.sub(self.PX, th.mm(dP, self.Y))

            self.W = 1. / (self.alpha * self.sigma2) * (F - th.mm(dPQ, (
                th.linalg.solve((self.alpha * self.sigma2 * self.inv_S + th.mm(self.Q.T, dPQ)),
                                (th.mm(self.Q.T, F))))))
            QtW = th.mm(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2. * th.trace(th.mm(QtW.T, th.mm(self.S, QtW)))



    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.
        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                
        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + th.mm(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + th.mm(self.G, self.W)
            else:
                self.TY = self.Y + th.mm(self.Q, th.mm(self.S, th.mm(self.Q.T, self.W)))
            return


    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = th.tensor(np.inf, dtype=self.dtype).to(self.device)
        xPx = th.mm(self.Pt1.permute(1, 0), th.sum(th.mul(self.X, self.X), dim=1).reshape(-1, 1))
        yPy = th.mm(self.P1.permute(1, 0), th.sum(th.mul(self.TY, self.TY), dim=1).reshape(-1, 1))
        trPXY = th.sum(th.mul(self.TY, self.PX))

        self.sigma2 = th.div((xPx - 2. * trPXY + yPy), (self.Np * self.D))

        if self.sigma2 <= 0.:
            self.sigma2 = (self.tolerance / 10.).clone()

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = th.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.
        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.
        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W
