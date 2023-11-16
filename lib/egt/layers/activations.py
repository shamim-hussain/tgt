import torch
import torch.nn.functional as F

@torch.jit.script
def geglu(x: torch.Tensor):
    g, e = x.chunk(2, dim=-1)
    return e * F.gelu(g)

@torch.jit.script
def glu(x: torch.Tensor):
    g, e = x.chunk(2, dim=-1)
    return e * torch.sigmoid(g)

@torch.jit.script
def swiglu(x: torch.Tensor):
    g, e = x.chunk(2, dim=-1)
    return e * torch.sigmoid(g) * g

glu_dict = {'geglu': geglu, 'glu': glu, 'swiglu': swiglu}

def get_activation(activation):
    if activation in glu_dict:
        return glu_dict[activation], 2
    else:
        return getattr(F, activation), 1
