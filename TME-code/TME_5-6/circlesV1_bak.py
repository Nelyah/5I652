import torch

def init_parmas(nx, nh, ny):
    """Initialises parameters

    :nx: TODO
    :nh: TODO
    :ny: TODO
    :returns: TODO

    """
    d_params = {}
    d_params["wh"] = torch.randn(nh, nx)
    d_params["bh"] = torch.randn(nh)
    d_params["wy"] = torch.randn(ny, nh)
    d_params["by"] = torch.randn(ny)

    return d_params

def forward(d_params, X):
    """forward function for neural network

    :d_params: parameters
    :X: input
    :returns: TODO

    """
    pass


