import torch
from torch import distributions
from submodules.GAN_stability.gan_training import non_parametric_dist


def get_zdist(dist_name, dim, bins=None, width=None, device=None):
    # Get distribution
    # just needs 'sample' method
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    elif dist_name == 'non_parametric':
        zdist = non_parametric_dist.ZDist(dim, bins, width, device=device)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def get_ydist(nlabels, device=None):
    logits = torch.zeros(nlabels, device=device)
    ydist = distributions.categorical.Categorical(logits=logits)

    # Add nlabels attribute
    ydist.nlabels = nlabels

    return ydist


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2

    return z
