from scipy.stats.morestats import binom_test
import torch
import numpy as np
import scipy.io as sio

# non-parametric sampling distribution from "Non-parametric priors for GANs"
# by Singh et al. 
class ZDist:
    def __init__(self, dim, bins=1024, width=2, device=None):
        self.dim = dim
        self.bins = bins
        self.width = width
        self.device = device

        mat = sio.loadmat('/home/jfb4/SeeingWithSound/code/giraffe/im2scene/giraffe/models/non_param_pdf.mat')
        self.pdf = torch.from_numpy(np.reshape(mat['X_final'], self.bins))
        self.points = torch.from_numpy(np.linspace(-self.width, self.width, self.bins))
        self.epsilon = self.width * 2 / self.bins

    def sample(self, dim=None):
        if dim is not None: self.dim = dim
        
        idx = self.pdf.multinomial(num_samples=dim, replacement=True)
        z = self.points[idx] + torch.rand(*dim) * self.epsilon
        z = z.type(torch.float32)
        z = z.to(self.device)

        return z
