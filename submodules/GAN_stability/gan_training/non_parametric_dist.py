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

        if self.width is None:
            self.width = 4
        if self.bins is None:
            self.bins = 1024

        mat = sio.loadmat('/home/jfb4/SeeingWithSound/code/giraffe/im2scene/giraffe/models/non_param_pdf.mat')
        self.pdf = torch.from_numpy(np.reshape(mat['X_final'], self.bins))
        self.points = torch.from_numpy(np.linspace(-self.width, self.width, self.bins))
        self.epsilon = self.width * 2 / self.bins

    def sample(self, num_codes):
        if len(num_codes) == 1:
            num_samples = num_codes[0] * self.dim
        if len(num_codes) == 2:
            num_samples = num_codes[0] * num_codes[1] * self.dim
        z = self.points[self.pdf.multinomial(num_samples=num_samples, replacement=True)]
        z = z.to(self.device)
        z += torch.rand(num_samples) * self.epsilon
        z = z.reshape((*num_codes, self.dim))
        z = z.type(torch.float32)
        z = z.to(self.device)

        return z
