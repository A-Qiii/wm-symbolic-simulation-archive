from study import *
from wm_symbolic.world_model import symbol_nll,kl
import torch.nn.functional as F

def cosine(a,b):
    den=a.norm()*b.norm()
    return float((a*b).sum()/den) if float(den)>1e-12 else None

def main():
    guard('train');start=time.monotonic();signal.alarm(180)
    lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
    _,data,_,norm=prepared();weights=[torch.tensor(v,device='cuda')for v in norm['class_weights']];rows=[]
    for seed in [0,1]:
      for step in [500,1000]:
        model=WorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(C/f'models/paired_seed{seed}_step{step}.pt',map_location='cuda',weights_only=False);model.load_state_dict(cp['model']);model.eval();prior_by_root={}
        for ix,ep in enumerate(data.episodes):
          batch=data.batch([ix],'cuda');torch.manual_seed(81831+int(ep['root']['seed']))
          with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot']);aa=model.action(batch['skill'],batch['arguments'],batch['budget']).repeat_interleave(8,0)
            hh=encoded.repeat_interleave(8,0);root,_=model.initial(hh[:,0]);p1,_=model.imagine(root,aa[:,0]);h1,_=model.observe(p1,hh[:,1]);prior,p=model.imagine(h1,aa[:,1])
          prior=prior.detach();p=p.detach();e=hh[:,2].detach().requires_grad_(True)
          with torch.autocast('cuda',dtype=torch.bfloat16):
            post,q=model.observe(prior,e);sym=model.symbols(post)
            args=[batch[k][:,2].repeat_interleave(8,0)for k in ['cat','cat_valid','binary','binary_valid']]
            current=symbol_nll(sym,*args,weights).mean();regularizer=.1*kl(q,p).clamp_min(1).mean()
          gcur=torch.autograd.grad(current,e,retain_graph=True)[0].float();gkl=torch.autograd.grad(regularizer,e)[0].float()
          target=COLORS.index(ep['root']['matched_color']);probs=sym['objects'].float().softmax(-1).mean(0);held=probs[:,6].detach().cpu().numpy();b=ep['branch'];rid=ep['root_id']
          info={'e':e.detach().float().cpu().numpy(),'h':post.h.detach().float().cpu().numpy(),'q':q.detach().float().cpu().numpy(),'held':held}
          row={'seed':seed,'checkpoint':step,'root_id':rid,'branch':b,'posterior_target_Held':float(held[target]),'current_symbol_nll':float(current),'weighted_representation_kl':float(regularizer),'E_current_grad_norm':float(gcur.norm()),'E_KL_grad_norm':float(gkl.norm()),'current_vs_KL_E_cosine':cosine(gcur,gkl),'h_observation_change_max':float((post.h-prior.h).abs().max())}
          if b==1:prior_by_root[rid]=info
          else:
            other=prior_by_root[rid];assert np.array_equal(other['h'],info['h'])
            row.update({'paired_E_l2':float(np.linalg.norm(other['e']-info['e'])),'paired_q_mean_abs':float(np.abs(other['q']-info['q']).mean()),'paired_target_Held_gap':float(other['held'][target]-held[target]),'paired_h_exact_same':True})
          rows.append(row)
        del model,cp;torch.cuda.empty_cache()
    csvwrite(U/'results/prechange_diagnostic.csv',rows)
    summaries=[]
    for seed in [0,1]:
      for step in [500,1000]:
        rr=[r for r in rows if r['seed']==seed and r['checkpoint']==step];pair=[r for r in rr if r['branch']==2];cos=[r['current_vs_KL_E_cosine']for r in rr if r['current_vs_KL_E_cosine']is not None]
        summaries.append({'seed':seed,'checkpoint':step,'branches':len(rr),'all_h_unchanged':all(r['h_observation_change_max']==0 for r in rr),'pair_E_l2_min':min(r['paired_E_l2']for r in pair),'pair_q_mean_abs_mean':float(np.mean([r['paired_q_mean_abs']for r in pair])),'pair_Held_gap_mean':float(np.mean([r['paired_target_Held_gap']for r in pair])),'nonzero_current_KL_grad_comparisons':len(cos),'negative_cosine_count':sum(v<0 for v in cos),'cosine_mean':float(np.mean(cos))if cos else None})
    save(U/'results/prechange_diagnostic.json',{'utc':now(),'seconds':time.monotonic()-start,'summaries':summaries,'actual_same_prefix_observation_pairs_only':True,'no_cross_state_splicing':True,'no_training':True,'scope':'Train roots only. Local gradient directions are descriptive, not independent causal proof. Prior ambiguity is unavoidable under the hidden exogenous disturbance; no claim that probabilistic prior NLL is inherently invalid.'});signal.alarm(0);print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
