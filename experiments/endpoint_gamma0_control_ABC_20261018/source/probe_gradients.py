from common import *
from net import *
import time
def components(raw,y,v,b,gamma):
 logits,g=unpack(raw);p=logits.sigmoid();w=torch.where(y>0,.25*(1-p).square(),.75*p.square())if gamma==2 else torch.where(y>0,.25,.75);cls=((F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w*v[:,None,:]).sum((1,2))/y.sum((1,2)).clamp(min=1)).mean();ii,pp,ss=torch.where(y>0);reg=5*F.smooth_l1_loss(g[ii,pp,ss],b[ii,pp,ss])+2*(1-giou(toboxes(g[ii,pp,ss],pp),toboxes(b[ii,pp,ss],pp))).mean();return cls,reg
def probe(m,data,ids,gamma):
 x,y,v,b=[data[k][ids]for k in ['x','y','v','b']];raw=m(x);cls,reg=components(raw,y,v,b,gamma);pars=list(m.parameters());gc=torch.autograd.grad(cls,pars,retain_graph=True);gr=torch.autograd.grad(reg,pars);a=torch.cat([g.flatten()for g in gc]);z=torch.cat([g.flatten()for g in gr]);nc=float(a.norm());nr=float(z.norm());nt=float((a+z).norm());return {'classification_loss':float(cls.detach()),'weighted_regression_loss':float(reg.detach()),'classification_gradient_norm':nc,'weighted_regression_gradient_norm':nr,'total_preclip_gradient_norm':nt,'classification_to_regression_norm_ratio':nc/nr,'classification_regression_gradient_cosine':float(torch.dot(a,z)/(a.norm()*z.norm()).clamp(min=1e-12)),'implied_clip_multiplier':min(1.,1/(nt+1e-6))}
def main():
 assert(E/'results/checkpoints_frozen.json').exists();assert not(E/'results/gradient_probes.json').exists();opened=install_guard([BASE/'data/train/train.npz']+[BASE/f'models/final_seed{s}.pt'for s in [0,1]]);torch.set_num_threads(4);z=np.load(BASE/'data/train/train.npz');data={k:torch.from_numpy(z[k])for k in ['x','y','v','b']};batches=read(E/'configs/probe_batches.json');results=[]
 for seed in [0,1]:
  for condition,path,gammas in [('shared_initial',E/f'models/initial_seed{seed}.pt',[2,0]),('focal_final',BASE/f'models/final_seed{seed}.pt',[2]),('BCE_final',E/f'models/final_seed{seed}.pt',[0])]:
   if path.is_relative_to(BASE):assert sha(path)==read(E/'configs/input_frozen.json')['sha256'][str(path)]
   m=make().eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);before=tensor_hash(m.state_dict())
   for gamma in gammas:
    for name,ids in batches.items():results.append(dict(seed=seed,checkpoint=condition,gamma=gamma,batch=name,image_ids=ids,**probe(m,data,ids,gamma)))
   assert tensor_hash(m.state_dict())==before
 save(E/'results/gradient_probes.json',{'utc':now(),'rows':results,'device':'CPU','optimizer_updates':0,'purpose':'matched fixed AB train batches, no-update gradient diagnostics; not historical baseline gradients','all_parameter_hashes_unchanged':True,'old_files_opened':sorted(opened),'D_or_dev_reads':0});print('GRADIENT_PROBES_COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
