from common import *
from net import *
import torch

def components(raw,y,v,b,variant):
 log,g=unpack(raw);p=log.sigmoid();f=torch.where(y>0,.25*(1-p).square(),.75*p.square());w=torch.where(y>0,.25,.75)if variant=='BCE'else f
 if variant=='mixed':w=torch.where(torch.arange(3)[None,None,:]==0,torch.where(y>0,.25,.75),f)
 cs=(F.binary_cross_entropy_with_logits(log,y,reduction='none')*w*v[:,None,:]).sum(1)/y.sum((1,2)).clamp(min=1)[:,None];front=cs[:,0].mean();finger=cs[:,1:].sum(1).mean();ii,pp,ss=torch.where(y>0);reg=5*F.smooth_l1_loss(g[ii,pp,ss],b[ii,pp,ss])+2*(1-giou(toboxes(g[ii,pp,ss],pp),toboxes(b[ii,pp,ss],pp))).mean();return front,finger,reg

def groups(gs):
 return {'shared':torch.cat([gs[0].flatten(),gs[1].flatten()]),'front_output':torch.cat([gs[2][:5].flatten(),gs[3][:5]]),'finger_output':torch.cat([gs[2][5:].flatten(),gs[3][5:]])}
def stats(g):return {k:float(v.norm())for k,v in groups(g).items()}
def main():
 assert not(E/'results/shared_layer_diagnostics.json').exists();allowed=[BASE/'data/train/train.npz']+[p/f'models/final_seed{s}.pt'for p in [BASE,BCEBASE]for s in [0,1]];opened=install_guard(allowed);frozen=read(E/'configs/input_frozen.json')['sha256'];assert all(sha(p)==frozen[str(p)]for p in allowed);torch.set_num_threads(4);z=np.load(BASE/'data/train/train.npz');data={k:torch.tensor(z[k])for k in ['x','y','v','b']};batches=read(E/'configs/probe_batches.json');rows=[];pairs=[];deltas=[]
 for seed in [0,1]:
  m=make().eval();m.load_state_dict(torch.load(E/f'models/initial_seed{seed}.pt',map_location='cpu',weights_only=True)['state']);before=tensor_hash(m.state_dict())
  for bn,ids in batches.items():
   caches={}
   for variant in ['focal','BCE','mixed']:
    vals=components(m(data['x'][ids]),data['y'][ids],data['v'][ids],data['b'][ids],variant);grads=[torch.autograd.grad(val,list(m.parameters()),retain_graph=True)for val in vals];total=[sum(g[i]for g in grads)for i in range(4)];gn=float(torch.cat([g.flatten()for g in total]).norm());caches[variant]=grads
    rows.append({'seed':seed,'batch':bn,'variant':variant,'front_loss':float(vals[0].detach()),'finger_loss':float(vals[1].detach()),'regression_loss':float(vals[2].detach()),'front_gradient_norm':stats(grads[0]),'finger_gradient_norm':stats(grads[1]),'regression_gradient_norm':stats(grads[2]),'total_gradient_norm':stats(total),'full_head_total_gradient_norm':gn,'clip1_multiplier':min(1.,1./(gn+1e-6))})
   fg,mg=caches['focal'],caches['mixed'];assert all(torch.equal(a,b)for a,b in zip(fg[1],mg[1]));assert all(torch.equal(a,b)for a,b in zip(fg[2],mg[2]));difference=[mg[0][i]-fg[0][i]for i in range(4)];pairs.append({'seed':seed,'batch':bn,'finger_direct_gradient_bitwise_equal':True,'regression_gradient_bitwise_equal':True,'changed_front_gradient_norm_by_group':stats(difference),'meaning':'same raw finger objective before clip, changed front path into shared hidden layer; global clipping can also rescale all gradients'})
  assert tensor_hash(m.state_dict())==before
  states={v:torch.load(p/f'models/final_seed{seed}.pt',map_location='cpu',weights_only=True)['state']for v,p in [('focal',BASE),('BCE',BCEBASE),('mixed',E)]}
  for variant in ['BCE','mixed']:
   d=[states[variant][k]-states['focal'][k]for k in ['0.weight','0.bias','2.weight','2.bias']];ref=[states['focal'][k]for k in ['0.weight','0.bias','2.weight','2.bias']];deltas.append({'seed':seed,'variant':variant,'final_parameter_distance_from_focal':stats(d),'relative_distance':{k:float(v.norm()/groups(ref)[k].norm())for k,v in groups(d).items()},'no_isolated_causal_claim':'final shared and output changes co-evolve; no layer swapping or extra model training'})
 save(E/'results/shared_layer_diagnostics.json',{'utc':now(),'initial_probes':rows,'same_initial_pairs':pairs,'final_parameter_distances':deltas,'same_initial_finger_direct_gradient_equal':True,'optimizer_updates':0,'all_parameter_hashes_unchanged':True,'old_files_opened':sorted(opened),'D_or_dev_sample_reads':0});print('SHARED_LAYER_DIAGNOSTICS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
