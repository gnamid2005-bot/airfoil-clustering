from __future__ import annotations

import torch
import torch.nn.functional as F


def validate_fields(fields: torch.Tensor) -> None:
    if fields.ndim != 4 or fields.shape[-1] != 2:
        raise ValueError(f"fields must have shape (T,Gx,Gy,2), got {tuple(fields.shape)}")


def _grid_coordinates(points, bounds):
    x0,x1,y0,y1=bounds
    return torch.stack((2*(points[:,0]-x0)/(x1-x0)-1,2*(points[:,1]-y0)/(y1-y0)-1),1)


def evaluate_slice(points, field, bounds):
    image=field.permute(2,1,0).unsqueeze(0)
    grid=_grid_coordinates(points,bounds).reshape(1,-1,1,2)
    return F.grid_sample(image,grid,mode="bilinear",padding_mode="border",align_corners=True)[0,:,:,0].T


def integrate_time_dependent(points, fields, bounds, substeps_per_slice=4):
    """Piecewise-constant v_t with equal time intervals and Euler substeps."""
    validate_fields(fields)
    if substeps_per_slice <= 0: raise ValueError("substeps_per_slice must be positive")
    x=points; dt=1.0/(fields.shape[0]*substeps_per_slice)
    for m in range(fields.shape[0]):
        for _ in range(substeps_per_slice): x=x+dt*evaluate_slice(x,fields[m],bounds)
    return x


def dirichlet_energy(field,bounds):
    x0,x1,y0,y1=bounds;dx=(x1-x0)/(field.shape[0]-1);dy=(y1-y0)/(field.shape[1]-1)
    return .5*(((field[1:]-field[:-1])/dx).square().sum(2).mean()+((field[:,1:]-field[:,:-1])/dy).square().sum(2).mean())


def discrete_geodesic_energy(fields,bounds):
    """Transparent surrogate: mean spatial Dirichlet energy over time slices."""
    validate_fields(fields)
    return torch.stack([dirichlet_energy(f,bounds) for f in fields]).mean()


def kinetic_l2_diagnostic(fields):
    validate_fields(fields)
    return fields.square().sum(-1).mean()


def jacobian_determinants(fields,bounds,substeps_per_slice=4,resolution=21):
    x0,x1,y0,y1=bounds;xs=torch.linspace(x0,x1,resolution,dtype=fields.dtype,device=fields.device);ys=torch.linspace(y0,y1,resolution,dtype=fields.dtype,device=fields.device)
    yy,xx=torch.meshgrid(ys,xs,indexing="ij");p=torch.stack((xx.flatten(),yy.flatten()),1);f=integrate_time_dependent(p,fields,bounds,substeps_per_slice).reshape(resolution,resolution,2)
    dx=(x1-x0)/(resolution-1);dy=(y1-y0)/(resolution-1);px=(f[1:-1,2:]-f[1:-1,:-2])/(2*dx);py=(f[2:,1:-1]-f[:-2,1:-1])/(2*dy)
    return (px[:,:,0]*py[:,:,1]-px[:,:,1]*py[:,:,0]).flatten()


def smooth_jacobian_barrier(jac,tau=.05,beta=50.):
    return (F.softplus(beta*(tau-jac))/beta).square().mean()
