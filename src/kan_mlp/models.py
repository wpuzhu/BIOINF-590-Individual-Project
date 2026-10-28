"""Dense models; every KAN connection, including output edges, is nonlinear."""
from functools import lru_cache
import math
import torch
from torch import nn
from torch.nn import functional as F

class KANLayer(nn.Module):
    def __init__(self, inputs, outputs, family="spline", terms=8):
        super().__init__()
        if terms < 4 or terms % 2:
            raise ValueError("Use an even number of basis terms >=4")
        self.family, self.terms = family, terms
        self.register_buffer("centers", torch.linspace(-1, 1, terms))
        self.register_buffer("frequencies", math.pi * torch.arange(1, terms//2+1))
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        self.scale = nn.Parameter(torch.ones(outputs, inputs))
        self.coeff = nn.Parameter(torch.randn(outputs, inputs, terms)*0.03/math.sqrt(inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.normal_(self.base, std=0.5/math.sqrt(inputs))

    def basis(self, x):
        if self.family == "fourier":
            phase = x.unsqueeze(-1)*self.frequencies
            return torch.cat((phase.sin(), phase.cos()), -1)
        z = (x.unsqueeze(-1)-self.centers)/(2/(self.terms-1))
        if self.family == "rbf":
            return torch.exp(-0.5*z.square())
        if self.family != "spline":
            raise ValueError(self.family)
        a = z.abs()
        # Uniform cardinal cubic B-spline, evaluated without input clipping.
        return torch.where(a<1, 2/3-a.square()+0.5*a.pow(3),
                           torch.where(a<2, (2-a).pow(3)/6, torch.zeros_like(a)))

    def forward(self, x):
        return F.linear(F.silu(x), self.base) + torch.einsum(
            "bir,oir->bo", self.basis(x), self.coeff*self.scale.unsqueeze(-1)) + self.bias


def build(kind, dims, terms=8):
    layers=[]
    if kind.startswith("mlp-"):
        act={"silu":nn.SiLU,"tanh":nn.Tanh,"relu":nn.ReLU}[kind[4:]]
        for n,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(nn.Linear(a,b))
            if n < len(dims)-2: layers.append(act())
    else:
        for a,b in zip(dims[:-1],dims[1:]):
            layers.append(KANLayer(a,b,kind[4:],terms))
    return nn.Sequential(*layers)


def parameter_count(kind, dims, terms=8):
    factor = 1 if kind.startswith("mlp-") else terms+2
    return sum(factor*a*b+b for a,b in zip(dims[:-1],dims[1:]))


@lru_cache(maxsize=512)
def architecture(kind, inputs, outputs, depth, budget, terms=8):
    """Closest active parameter count; no dummy parameters. Depth=hidden layers."""
    best=None
    factor=1 if kind.startswith("mlp-") else terms+2
    max_width=max(256,math.ceil((budget-outputs)/(factor*(inputs+outputs)+1))+2) if depth==1 else 256
    for w in range(1,max_width+1):
        tails=[w] if depth==1 else range(max(1,w//2),min(256,2*w)+1)
        for tail in tails:
            hidden=[w]*(depth-1)+[tail]
            dims=[inputs]+hidden+[outputs]
            p=parameter_count(kind,dims,terms)
            key=(abs(p-budget),max(hidden)/min(hidden),p)
            if best is None or key<best[0]: best=(key,dims,p)
    return best[1],best[2]

MODELS=["mlp-tanh","mlp-relu","mlp-silu","kan-spline","kan-fourier","kan-rbf"]
