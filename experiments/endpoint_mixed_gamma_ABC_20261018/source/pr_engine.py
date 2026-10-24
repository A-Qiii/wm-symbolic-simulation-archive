from common import *
import sys,os,time,csv,torch
def unpack(raw):
 z=torch.as_tensor(raw).reshape(1024,3,5);scores=z[...,0].sigmoid();g=torch.cat([.5*z[...,1:3].tanh(),z[...,3:5].sigmoid()],-1);return scores.numpy(),toboxes(g).clamp(0,1).numpy()
def toboxes(g):
 p=torch.arange(1024)[:,None].expand(-1,3);xy=torch.stack([p%32+.5,p//32+.5],-1).to(g.dtype);center=(xy+g[...,:2])/32;return torch.cat([center-g[...,2:]/2,center+g[...,2:]/2],-1)
def overlaps(box,boxes):
 inter=np.maximum(np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2]),0).prod(1);union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)-inter;return inter/np.maximum(union,1e-9)
def nms(boxes,order):
 keep=[];remaining=order.copy()
 while len(remaining):
  i=int(remaining[0]);keep.append(i);rest=remaining[1:]
  if not len(rest):break
  remaining=rest[overlaps(boxes[i],boxes[rest])<=.5]
 return np.array(keep,np.int32)
def reference_nms(boxes,order):
 def scalar_iou(a,b):
  inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();u=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter;return float(inter/max(float(u),1e-9))
 keep=[]
 for i in order:
  if not any(scalar_iou(boxes[i],boxes[j])>.5 for j in keep):keep.append(int(i))
 return np.array(keep,np.int32)
def match(boxes,gt):
 used=set();tp=np.zeros(len(boxes),np.uint8);matched=np.full(len(boxes),-1,np.int32);miou=np.zeros(len(boxes),np.float32)
 for j,box in enumerate(boxes):
  ovs=overlaps(box,gt);best=max(((float(v),k)for k,v in enumerate(ovs)if k not in used),default=(0.,-1))
  if best[0]>=.5:used.add(best[1]);tp[j]=1;matched[j]=best[1];miou[j]=best[0]
 return tp,matched,miou
def point(scores,tp,n_gt,threshold):
 selected=scores>=threshold;nt=int(tp[selected].sum());nf=int(selected.sum())-nt;return {'threshold':float(threshold),'TP':nt,'FP':nf,'FN':n_gt-nt,'precision':nt/(nt+nf)if nt+nf else None,'recall':nt/n_gt,'F1':2*nt/(n_gt+nt+nf)}
def pr(scores,tp,image,patch,slot,n_gt):
 order=np.lexsort((slot,patch,image,-scores));s=scores[order];t=tp[order];ct=np.cumsum(t,dtype=np.int64);cf=np.arange(1,len(t)+1)-ct;ends=np.r_[np.where(s[:-1]!=s[1:])[0],len(s)-1];threshold=s[ends];precision=ct[ends]/(ct[ends]+cf[ends]);recall=ct[ends]/n_gt;mrec=np.r_[0.,recall,1.];mpre=np.r_[0.,precision,0.];mpre=np.maximum.accumulate(mpre[::-1])[::-1];change=np.where(mrec[1:]!=mrec[:-1])[0];ap=float(np.sum((mrec[change+1]-mrec[change])*mpre[change+1]));return {'threshold':threshold,'precision':precision,'recall':recall,'TP':ct[ends],'FP':cf[ends],'precision_envelope':mpre[1:-1]},ap
