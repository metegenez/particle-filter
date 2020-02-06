import torch

class ParticleFilter(object):
    r""" Implements a Monte Carlo (Particle) filter with PyTorch. You are responsible for setting the
        various state variables to reasonable values; the defaults  will
        not give you a functional filter.
    """
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = torch.zeros((dim_x, 1))  # state
        self.P = torch.eye(dim_x)  # uncertainty covariance
        self.Q = torch.eye(dim_x)  # process uncertainty
        self.F = None  # state transition matrix
        self.H = None  # Measurement function
        self.R = torch.eye(dim_z)  # state uncertainty
        self._alpha_sq = 1.  # fading memory control
        self.M = torch.zeros((dim_z, dim_z))  # process-measurement cross correlation
        self.z = torch.zeros((dim_z,1))

