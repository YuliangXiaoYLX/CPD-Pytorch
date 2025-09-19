from builtins import super
import torch as th
import numpy as np
from .emregistration import EMRegistration
from .utils import is_positive_semi_definite


class AffineRegistration(EMRegistration):
    """
    Affine registration.
    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.
    t: numpy array
        1xD initial translation vector.
    """
    # Additional parameters used in this class, but not inputs.
    # YPY: float
    #     Denominator value used to update the scale factor.
    #     Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    # X_hat: numpy array
    #     Centered target point cloud.
    #     Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if B is not None and ((B.ndim != 2) or (B.shape[0] != self.D) or (B.shape[1] != self.D) or not is_positive_semi_definite(B)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, B))

        if t is not None and ((t.ndim != 2) or (t.shape[0] != 1) or (t.shape[1] != self.D)):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        
        self.B = th.eye(self.D, dtype=self.dtype).to(self.device) if B is None else B
        if not isinstance(self.B, th.Tensor):
            self.B = th.tensor(self.B, dtype=self.dtype).to(self.device)
        self.t = th.atleast_2d(th.zeros((1, self.D), dtype=self.dtype)).to(self.device) if t is None else t
        if not isinstance(self.t, th.Tensor):
            self.t = th.tensor(self.t, dtype=self.dtype).to(self.device)
        self.YPY = None
        self.X_hat = None
        self.A = None

    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.
        """

        # source and target point cloud means
        muX = th.div(th.sum(self.PX, dim=0), self.Np)
        muY = th.div(th.sum(th.mm(self.P.permute(1, 0), self.Y), dim=0), self.Np)

        self.X_hat = th.sub(self.X, muX.repeat(self.N, 1))
        Y_hat = th.sub(self.Y, muY.repeat(self.M, 1))

        self.A = th.mm(self.X_hat.permute(1,0), self.P.permute(1,0))
        self.A = th.mm(self.A, Y_hat)

        self.YPY = th.mm(Y_hat.permute(1, 0), th.diag(self.P1.reshape(-1, )))
        self.YPY = th.mm(self.YPY, Y_hat)

        # Calculate the new estimate of affine parameters using update rules for (B, t)
        # as defined in Fig. 3 of https://arxiv.org/pdf/0905.2635.pdf.
        self.B = th.linalg.solve(self.YPY.permute(1,0), self.A.permute(1,0))
        self.t = (muX.reshape(-1, 1) - th.mm(self.B.permute(1, 0), muY.reshape(-1, 1))).permute(1, 0)

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.
        
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
        if Y is None:
            #print(self.Y.shape, self.B.shape, self.t.shape)
            self.TY = th.mm(self.Y, self.B) + self.t.repeat(self.M, 1)
            return
        else:
            return th.mm(Y, self.B) + self.t.repeat(Y.shape[0], 1)

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the affine transformation.
        See the update rule for sigma2 in Fig. 3 of of https://arxiv.org/pdf/0905.2635.pdf.
        """
        qprev = self.q

        trAB = th.trace(th.mm(self.A, self.B))
        xPx = th.mm(self.Pt1.permute(1,0), th.sum(th.mul(self.X_hat, self.X_hat), dim=1).reshape(-1, 1)).reshape(-1, )
        trBYPYP = th.trace(th.mm(th.mm(self.B, self.YPY), self.B))
        self.q = th.div((xPx - 2. * trAB + trBYPYP), (2. * self.sigma2)) + self.D * self.Np/2. * th.log(self.sigma2)
        self.diff = th.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)

        if self.sigma2 <= 0.:
            self.sigma2 = (self.tolerance / 10.).clone()

    def get_registration_parameters(self):
        """
        Return the current estimate of the affine transformation parameters.
        Returns
        -------
        B: numpy array
            DxD affine transformation matrix.
        t: numpy array
            1xD translation vector.
        """
        return self.B, self.t