from __future__ import annotations

import torch

from lddmm_2d import (discrete_geodesic_energy, integrate_time_dependent,
                      jacobian_determinants, validate_fields)


def main():
    torch.manual_seed(0);dtype=torch.float64;bounds=(0.,1.,-.5,.5);p=torch.tensor([[.2,.1],[.8,-.1]],dtype=dtype)
    zero=torch.zeros((4,12,12,2),dtype=dtype,requires_grad=True);validate_fields(zero)
    out=integrate_time_dependent(p,zero,bounds,4);assert torch.equal(out,p)
    jac=jacobian_determinants(zero,bounds,4,11);assert torch.allclose(jac,torch.ones_like(jac),atol=1e-12)
    const=torch.zeros((4,12,12,2),dtype=dtype);const[:,:,:,0]=.03;const[:,:,:,1]=-.02
    moved=integrate_time_dependent(p,const,bounds,4);assert torch.allclose(moved,p+torch.tensor([.03,-.02],dtype=dtype),atol=1e-12)
    loss=integrate_time_dependent(p,zero,bounds,2).square().sum()+discrete_geodesic_energy(zero,bounds);loss.backward();assert zero.grad is not None and torch.isfinite(zero.grad).all()
    first=integrate_time_dependent(p,const,bounds,4);second=integrate_time_dependent(p,const,bounds,4);assert torch.equal(first,second)
    print("airfoil LDDMM-style validation checks passed")


if __name__=="__main__":main()
