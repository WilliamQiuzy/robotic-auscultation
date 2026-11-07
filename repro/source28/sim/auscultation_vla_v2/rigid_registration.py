"""Experimental bounded point-to-plane registration; not a deployed tracker.

Inputs must be measured RGB-D support points selected from the same rigid
surface. Normal diversity is tested; a plane cannot identify six-axis motion.
Nearest-neighbor agreement is not semantic identity or an occlusion proof.
The caller must enforce timestamps, RGB consistency and fail-latched tracking.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def rgbd_cloud(rgb,depth,valid,k,polygon,*,stride=8):
    """Unproject measured support; polygon is an initial image annotation."""
    y,x=np.mgrid[:depth.shape[0],:depth.shape[1]]
    safe=np.where(valid&np.isfinite(depth)&(depth>0),depth,0.)
    rays=np.stack([x,y,np.ones_like(x)],axis=-1)@np.linalg.inv(k).T
    xyz=rays*safe[...,None]
    # Wide-baseline normals suppress independent depth noise; raw one-pixel
    # derivatives can otherwise manufacture six-axis rank on a flat wall.
    smoothed=cv2.GaussianBlur(xyz,(9,9),2.)
    a=smoothed[8:,4:-4]-smoothed[:-8,4:-4]
    b=smoothed[4:-4,8:]-smoothed[4:-4,:-8]
    n=np.cross(a,b);length=np.linalg.norm(n,axis=-1)
    n/=np.maximum(length[...,None],1e-12)
    mask=np.zeros(depth.shape,np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
    neighborhood=cv2.erode(valid.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
    spread=cv2.dilate(safe,np.ones((17,17),np.uint8))-cv2.erode(safe,np.ones((17,17),np.uint8))
    ok=mask[4:-4,4:-4].astype(bool)&neighborhood[4:-4,4:-4]
    ok&=(length>1e-10)&(spread[4:-4,4:-4]<.06)
    keep=ok[::stride,::stride]
    return (xyz[4:-4,4:-4][::stride,::stride][keep],
            n[::stride,::stride][keep],rgb[4:-4,4:-4][::stride,::stride][keep])


def _cloud(points,normals):
    p=np.asarray(points,float);n=np.asarray(normals,float)
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or len(p)<100:
        raise ValueError('insufficient three-dimensional surface support')
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError('nonfinite surface support')
    if not np.allclose(np.linalg.norm(n,axis=1),1.,atol=.01):
        raise ValueError('surface normals must be unit vectors')
    return p,n


def register(reference,reference_normals,current,current_normals,*,initial=None):
    p,pn=_cloud(reference,reference_normals);q,qn=_cloud(current,current_normals)
    T=np.eye(4) if initial is None else np.array(initial,float,copy=True)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        raise ValueError('invalid initial transform')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<0:
        raise ValueError('initial rotation must be proper')
    tree=cKDTree(q)
    # Visible fraction alone does not establish or rule out six-axis motion.
    # Require a broad two-dimensional footprint as well as geometric rank.
    _,basis=np.linalg.eigh(np.cov(p.T))
    footprint=(p-p.mean(0))@basis[:,-2:]
    footprint_variance=np.var(footprint,axis=0)
    def correspond(transform):
        moved=p@transform[:3,:3].T+transform[:3,3]
        dist,index=tree.query(moved)
        normals=qn[index]
        ok=(dist<.015)&(np.sum((pn@transform[:3,:3].T)*normals,axis=1)>.85)
        if np.mean(ok)<.4 or len(np.unique(index[ok]))<100:
            raise ValueError('insufficient visible rigid support')
        coverage=np.var(footprint[ok],axis=0)/np.maximum(footprint_variance,1e-12)
        if np.min(coverage)<.16:
            raise ValueError('visible support has insufficient spatial coverage')
        x=moved[ok];n=normals[ok];y=q[index[ok]];center=x.mean(0)
        residual=np.sum((x-y)*n,axis=1)
        # Center and scale rotations by support radius for dimensionless rank testing.
        radius=max(float(np.sqrt(np.mean(np.sum((x-center)**2,axis=1)))),.01)
        J=np.column_stack([n,np.cross(x-center,n)])
        scaled=J.copy();scaled[:,3:]/=radius
        eigen=np.linalg.eigvalsh(scaled.T@scaled/len(x))
        if eigen[0]<1e-3 or eigen[-1]/eigen[0]>500:
            raise ValueError('six-axis surface motion is not observable')
        return x,n,center,residual,J,ok,eigen
    for iteration in range(30):
        x,n,center,residual,J,ok,eigen=correspond(T)
        # Bound a single occlusion-edge residual's influence to 0.2 mm.
        # All correspondences still contribute; this Huber solve scale is not
        # the acceptance threshold and does not discard poorly fitted points.
        weights=np.minimum(1.,.0002/np.maximum(np.abs(residual),1e-9))
        delta=np.linalg.lstsq(J*np.sqrt(weights[:,None]),-residual*np.sqrt(weights),rcond=None)[0]
        if np.linalg.norm(delta[:3])>.03 or np.linalg.norm(delta[3:])>.12:
            raise ValueError('registration increment exceeds bounded local motion')
        R=Rotation.from_rotvec(delta[3:]).as_matrix()
        step=np.eye(4);step[:3,:3]=R;step[:3,3]=center+delta[:3]-R@center
        T=step@T
        # Nearest-neighbor membership at an occluder boundary is discontinuous.
        # Stop at 0.05 mm maximum support-point movement, below the depth fit
        # tolerance, instead of requiring sub-micron parameter convergence.
        # This only sets optimizer precision: support, rank and residual gates
        # are recomputed below and must still pass without any relaxation.
        step_motion=np.linalg.norm(x@R.T+step[:3,3]-x,axis=1)
        if np.max(step_motion)<.00005:break
    else:raise ValueError('rigid registration did not converge')
    x,n,center,residual,J,ok,eigen=correspond(T)
    rmse=float(np.sqrt(np.mean(residual**2)))
    if rmse>.002 or np.quantile(np.abs(residual),.95)>.004:
        raise ValueError('surface fit exceeds residual threshold')
    moved=p@T[:3,:3].T+T[:3,3]
    _,indices=tree.query(moved)
    return dict(transform=T,rmse_m=rmse,support_fraction=float(np.mean(ok)),
                reference_indices=np.flatnonzero(ok),current_indices=indices[ok],
                conditioning=float(eigen[-1]/eigen[0]),iterations=iteration+1)
