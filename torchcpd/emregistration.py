import numpy as np
import numbers
from warnings import warn
import torch as th
import math

def initialize_sigma2(X, Y):
    """
    Initialize the variance (sigma2).
    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = th.sub(X[None, :, :], Y[:, None, :])
    err = th.pow(diff, 2)
    return th.sum(err) / (D * M * N)

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.
    Attributes
    ----------
    X: numpy array
        NxD array of target points.
    Y: numpy array
        MxD array of source points.
    TY: numpy array
        MxD array of transformed source points.
    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.
    N: int
        Number of target points.
    M: int
        Number of source points.
    D: int
        Dimensionality of source and target points
    iteration: int
        The current iteration throughout registration.
    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.
    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.
    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).
    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.
    diff: float (positive)
        The absolute difference between the current and previous objective function values.
    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.
    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.
    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.
    Np: float (positive)
        The sum of all elements in P.
    """

    def __init__(self, X, Y, device, sigma2=None, max_iterations=None, tolerance=None, w=None, dtype=th.float64, *args, **kwargs):
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D tensor array.")

        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        self.device = th.device(device)
        self.dtype = dtype
        self.X = th.tensor(X, dtype=dtype).to(self.device)
        self.Y = th.tensor(Y, dtype=dtype).to(self.device)
        self.TY = th.tensor(Y, dtype=dtype).to(self.device)
        self.sigma2 = initialize_sigma2(self.X, self.Y) if sigma2 is None else sigma2
        if not isinstance(self.sigma2, th.Tensor):
            self.sigma2 = th.tensor(self.sigma2, dtype=dtype).to(self.device)
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = th.tensor(0.001, dtype=dtype).to(self.device) if tolerance is None else tolerance
        if not isinstance(self.tolerance, th.Tensor):
            self.tolerance = th.tensor(self.tolerance, dtype=dtype).to(self.device)
        self.w = th.tensor(0.0, dtype=dtype).to(self.device) if w is None else w
        if not isinstance(self.w, th.Tensor):
            self.w = th.tensor(self.w, dtype=dtype).to(self.device)
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = th.tensor(np.inf, dtype=dtype).to(self.device)
        self.q = th.tensor(np.inf, dtype=dtype).to(self.device)
        self.P = th.zeros((self.M, self.N), dtype=dtype).to(self.device)
        self.Pt1 = th.zeros((self.N, 1), dtype=dtype).to(self.device)
        self.P1 = th.zeros((self.M, 1), dtype=dtype).to(self.device)
        self.PX = th.zeros((self.M, self.D), dtype=dtype).to(self.device)
        self.Np = th.tensor(0., dtype=dtype).to(self.device)

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.
        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.
        
        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.
        
        registration_parameters:
            Returned params dependent on registration method used. 
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q.detach().cpu().numpy(), 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.
        """
        P = th.sum(th.pow(self.X[None, :, :] - self.TY[:, None, :], 2), dim=2) # (M, N)
        P = th.exp(th.div(-P, (2.*self.sigma2)))
        c = th.pow(2.*th.tensor(math.pi, dtype=self.dtype)*self.sigma2, (self.D/2.))*self.w/(1. - self.w)*self.M/self.N

        den = th.sum(P, dim = 0, keepdims = True) # (1, N)
        den = th.clamp(den, th.finfo(self.X.dtype).eps, None) + c

        self.P = th.div(P, den)
        self.Pt1 = th.sum(self.P, dim=0).reshape(-1, 1)
        self.P1 = th.sum(self.P, dim=1).reshape(-1, 1)
        self.Np = th.sum(self.P1)
        self.PX = th.mm(self.P, self.X)

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()