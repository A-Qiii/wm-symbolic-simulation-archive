from common import *
def semantic_faces(v,f):
 tri=v[f];eps=5e-6;front=tri[:,:,2].min(1)>=.2-eps;handle=tri[:,:,2].max(1)>.225+eps
 # Cavity-facing inner wall of the front board is excluded by the original
 # 'exclude cavity/interior' definition; its exposed outer edges stay front.
 inner=(np.abs(tri[:,:,2]-.2).max(1)<eps)&(tri[:,:,0].min(1)>=-.225-eps)&(tri[:,:,0].max(1)<=.225+eps)&(tri[:,:,1].min(1)>=.025-eps)
 labels=np.full(len(f),3,np.uint8);labels[front&~inner]=1;labels[handle]=2
 assert not np.any((tri[:,:,2].min(1)<.2-eps)&(tri[:,:,2].max(1)>.2+eps));assert np.all(front[handle]);assert inner.sum()==2
 return labels,inner
def intersect_rays(origins,directions,triangles):
 # Chunked two-sided Moller-Trumbore; ray directions have unit world length.
 t0=triangles[:,0];e1=triangles[:,1]-t0;e2=triangles[:,2]-t0
 n=len(origins);best=np.full(n,np.inf);idx=np.full(n,-1,np.int32);second=np.full(n,np.inf);idx2=np.full(n,-1,np.int32)
 for k in range(0,n,512):
  o=origins[k:k+512];d=directions[k:k+512];h=np.cross(d[:,None,:],e2[None]);det=np.einsum('tj,ntj->nt',e1,h);valid=np.abs(det)>1e-12;inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
  ss=o[:,None,:]-t0;u=np.einsum('ntj,ntj->nt',ss,h)*inv;q=np.cross(ss,e1);vv=np.einsum('nj,ntj->nt',d,q)*inv;tt=np.einsum('tj,ntj->nt',e2,q)*inv
  ok=valid&(u>=-1e-9)&(vv>=-1e-9)&((u+vv)<=1+1e-9)&(tt>=0);tt=np.where(ok,tt,np.inf)
  ix=np.argmin(tt,axis=1);b=tt[np.arange(len(o)),ix];idx[k:k+len(o)]=np.where(np.isfinite(b),ix,-1);best[k:k+len(o)]=b
  tt[np.arange(len(o)),ix]=np.inf;ix2=np.argmin(tt,axis=1);b2=tt[np.arange(len(o)),ix2];idx2[k:k+len(o)]=np.where(np.isfinite(b2),ix2,-1);second[k:k+len(o)]=b2
 return best,idx,second,idx2
def classify(seg,zbuf,view,projection,tri_world,face_classes,table_uid,drawer_link,near,far):
 h,w=seg.shape;native=(seg>=0)&((seg&((1<<24)-1))==table_uid)&(((seg>>24)-1)==drawer_link);y,x=np.where(native)
 view=np.asarray(view).reshape(4,4).T;proj=np.asarray(projection).reshape(4,4).T;inv=np.linalg.inv(proj@view)
 ndcx=2*(x+.5)/w-1;ndcy=1-2*(y+.5)/h
 a=np.stack([ndcx,ndcy,np.full(len(x),-1.),np.ones(len(x))],1)@inv.T;b=np.stack([ndcx,ndcy,np.ones(len(x)),np.ones(len(x))],1)@inv.T
 a=a[:,:3]/a[:,3:];b=b[:,:3]/b[:,3:];directions=b-a;directions/=np.linalg.norm(directions,axis=1,keepdims=True)
 t,idx,t2,idx2=intersect_rays(a,directions,tri_world);hit=idx>=0
 points=a+np.where(hit,t,0)[:,None]*directions;camera=np.c_[points,np.ones(len(points))]@view.T;ray_depth=-camera[:,2];render_depth=far*near/(far-(far-near)*zbuf[y,x].astype(np.float64));err=np.abs(ray_depth-render_depth)
 classes=face_classes[np.maximum(idx,0)];ambig=hit&(idx2>=0)&(abs(t2-t)<=1e-7)&(face_classes[np.maximum(idx2,0)]!=classes)
 valid=hit&(err<=.0005)&~ambig
 sem=np.zeros(seg.shape,np.uint8);sem[y,x]=np.where(valid,classes,255);face_id=np.full(seg.shape,-1,np.int32);face_id[y,x]=np.where(valid,idx,-1)
 error=np.full(seg.shape,np.nan,np.float32);error[y,x]=err
 stats={'native_drawer_pixels':len(x),'ray_hit_pixels':int(hit.sum()),'depth_valid_pixels':int((hit&(err<=.0005)).sum()),'cross_semantic_tie_pixels':int(ambig.sum()),'assigned_pixels':int(valid.sum()),'unknown_pixels':int((~valid).sum()),'depth_error_median_m':float(np.median(err[hit]))if hit.any()else None,'depth_error_max_m':float(err[hit].max())if hit.any()else None,'perpart_pixels':{str(c):int((sem==c).sum())for c in [1,2,3,255]},'all_drawer_pixels_classified':bool(valid.all())}
 return sem,face_id,error,stats
