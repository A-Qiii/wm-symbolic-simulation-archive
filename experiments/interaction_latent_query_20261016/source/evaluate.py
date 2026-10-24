from common import *
from model import probs,feature_error
from train_queries import load_query,normalize
import torch,fcntl,argparse
def discrete(p):
 a=p>=.5;a[:,6:]=False;a[np.arange(len(p)),6+p[:,6:].argmax(1)]=True;return a
def score(pred,y,v,roots):
 pred=discrete(pred);y=np.array(y,bool);v=np.array(v,bool);ok=((pred==y)|~v);valid=v.any(1);full=ok.all(1)&valid;out={'states':len(y),'roots':len(set(roots)),'valid_facts':int(v.sum()),'correct_facts':int(((pred==y)&v).sum()),'all_valid_facts_correct_states':int(full.sum()),'root_macro_all_facts_accuracy':float(np.mean([full[np.array(roots)==r].mean()for r in set(roots)]))if roots else None,'all_states_correct_roots':sum(bool(full[np.array(roots)==r].all())for r in set(roots))}
 for lo,hi,name in[(0,3,'Holding'),(3,6,'In'),(6,9,'drawer')]:
  vv=v[:,lo:hi];yy=y[:,lo:hi];pp=pred[:,lo:hi];out[name]={'correct':int(((pp==yy)&vv).sum()),'valid':int(vv.sum()),'TP':int((pp&yy&vv).sum()),'FP':int((pp&~yy&vv).sum()),'FN':int((~pp&yy&vv).sum())}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['C','D'],default='C');a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 base=E/'data'if a.split=='C'else E/'data/D';rows=read(base/'public/states.json');pairs=read(base/'public/pairs.json');chains=read(base/'public/chains.json');labels=read(base/'_private/labels.json');outcomes=read(base/'_private/pair_outcomes.json');status={x['pair_id']:x['status']for x in outcomes}
 with np.load(base/'public/inputs.npz')as d:obj=torch.tensor(d['objects'],device='cuda');scene=torch.tensor(d['scene'],device='cuda')
 take=np.array([i for i,p in enumerate(pairs)if a.split=='D'or p['split']=='dev']);bb=np.array([pairs[i]['before']for i in take]);aa=np.array([pairs[i]['after']for i in take]);acts=torch.tensor([pairs[i]['action']for i in take],device='cuda');allacts=torch.tensor([p['action']for p in pairs],device='cuda');states=[i for i,r in enumerate(rows)if (a.split=='D'or r['split']=='dev')and str(i)in labels];result={};errors=[];records=[]
 for kind in['generic','ae','interaction']:
  for seed in[0,1]:
   key=f'{kind}_seed{seed}';m,q,stats=load_query(kind,seed)
   with torch.no_grad():
    z=m.encode(obj,scene);cur=probs(q(normalize(z,stats))).cpu().numpy();pred=m.transition(z[bb],acts);pp=probs(q(normalize(pred,stats))).cpu().numpy();feat=feature_error(m.decode(pred),obj[aa],scene[aa]).cpu().numpy();weighted=feature_error(m.decode(pred),obj[aa],scene[aa],(obj[bb],scene[bb])).cpu().numpy();copy=feature_error((obj[bb],scene[bb]),obj[aa],scene[aa]).cpu().numpy()
    controls={}
    for name in['zero_action','shuffled_action','wrong_target']:
     a2=acts.clone()
     if name=='zero_action':a2[:,:7]=0
     elif name=='shuffled_action':a2[:,:7]=acts[torch.roll(torch.arange(len(acts),device='cuda'),1),:7]
     else:a2[:,4:7]=torch.roll(a2[:,4:7],1,1)
     pred2=m.transition(z[bb],a2);controls[name]={'error':feature_error(m.decode(pred2),obj[aa],scene[aa],(obj[bb],scene[bb])).cpu().numpy(),'prob':probs(q(normalize(pred2,stats))).cpu().numpy()}
   rootlist=[pairs[i]['root_id']for i in take];summary={'all_current_states':score(cur[states],[labels[str(i)]['y']for i in states],[labels[str(i)]['v']for i in states],[rows[i]['root_id']for i in states]),'feature_prediction':{},'semantic_transition':{}}
   for group in['normal','refused','failed','all']:
    ids=[j for j,i in enumerate(take)if group=='all'or status[pairs[i]['id']]=={'normal':'success','refused':'refused','failed':'failed'}[group]]
    if not ids:continue
    roots=[rootlist[j]for j in ids]
    def rm(x):return float(np.mean([x[[k for k,j in enumerate(ids)if roots[k]==r]].mean()for r in sorted(set(roots))]))
    row={'pairs':len(ids),'roots':len(set(roots)),'raw_balanced_MSE':float(feat[ids].mean()),'copy_raw_balanced_MSE':float(copy[ids].mean()),'changed_weighted_MSE':float(weighted[ids].mean()),'group_MSE_object_DINO_geometry_scene_DINO_robot':feat[ids].mean(0).tolist(),'controls':{}}
    for name,cc in controls.items():
     diff=cc['error'][ids].mean(1)-weighted[ids].mean(1);rootdiff=[float(diff[[k for k,r in enumerate(roots)if r==rr]].mean())for rr in sorted(set(roots))];row['controls'][name]={'changed_weighted_MSE':float(cc['error'][ids].mean()),'mean_root_loss_increase':float(np.mean(rootdiff)),'roots_with_positive_loss_increase':sum(d>0 for d in rootdiff),'roots':len(rootdiff)}
    summary['feature_prediction'][group]=row
    sid=[j for j in ids if str(bb[j])in labels and str(aa[j])in labels]
    if sid:
     ys=[labels[str(aa[j])]['y']for j in sid];vs=[labels[str(aa[j])]['v']for j in sid];rs=[rootlist[j]for j in sid]
     summary['semantic_transition'][group]={'actual_current':score(cur[bb[sid]],[labels[str(bb[j])]['y']for j in sid],[labels[str(bb[j])]['v']for j in sid],rs),'actual_successor':score(cur[aa[sid]],ys,vs,rs),'predicted_successor':score(pp[sid],ys,vs,rs),'copy_current_semantics':score(cur[bb[sid]],ys,vs,rs)}
   multistep={}
   for horizon in[1,2,3]:
    starts=[i for i in take if len(chains[i])>=horizon];valid=[i for i in starts if str(pairs[int(chains[i][horizon-1][1:])]['after'])in labels]
    if not valid:continue
    with torch.no_grad():
     zz=z[[pairs[i]['before']for i in valid]]
     for t in range(horizon):ni=[int(chains[i][t][1:])for i in valid];zz=m.transition(zz,allacts[ni])
     probability=probs(q(normalize(zz,stats))).cpu().numpy()
    endpoints=[pairs[int(chains[i][horizon-1][1:])]['after']for i in valid];multistep[str(horizon)]=score(probability,[labels[str(i)]['y']for i in endpoints],[labels[str(i)]['v']for i in endpoints],[pairs[i]['root_id']for i in valid])
   summary['observed_sequence_rollout']=multistep
   with torch.no_grad():zz=z[[i for i,r in enumerate(rows)if a.split=='D'or r['split']=='dev']].reshape(-1,256);sv=torch.linalg.svdvals(zz-zz.mean(0));power=sv.square();pwr=power/power.sum();summary['latent']={'mean_std':float(zz.std(0).mean()),'min_std':float(zz.std(0).min()),'effective_rank':float(torch.exp(-(pwr*pwr.clamp(min=1e-12).log()).sum()))}
   for j,i in enumerate(take):
    if str(aa[j])not in labels:continue
    r={'arm':key,'pair_id':pairs[i]['id'],'root_id':rootlist[j],'before_index':int(bb[j]),'after_index':int(aa[j]),'skill':pairs[i]['skill'],'status':status[pairs[i]['id']],'true_after':labels[str(aa[j])],'current_query':cur[bb[j]].tolist(),'actual_successor_query':cur[aa[j]].tolist(),'predicted_successor_query':pp[j].tolist()};records.append(r)
    yy=np.array(r['true_after']['y']);vv=np.array(r['true_after']['v']);pd=discrete(pp[j:j+1])[0];actual=discrete(cur[aa[j]:aa[j]+1])[0]
    if np.any((pd!=yy)&vv):errors.append(r|{'first_component':'actual_successor_readout'if np.any((actual!=yy)&vv)else'latent_transition_or_off_manifold_query','wrong_facts':[FACTS[k]for k in np.where((pd!=yy)&vv)[0]]})
   result[key]=summary
   np.savez_compressed(E/'results'/f'{a.split}_{key}_predictions.npz',current=cur,predicted=pp,pair_indices=take,feature_error=feat,copy_error=copy,weighted_error=weighted,**{n+'_error':c['error']for n,c in controls.items()})
   print(a.split,key,'current',summary['all_current_states']['all_valid_facts_correct_states'],'/',len(states),'normal',summary['semantic_transition'].get('normal',{}).get('predicted_successor',{}).get('all_valid_facts_correct_states'),flush=True)
 save(E/'results'/f'{a.split}_metrics.json',result);save(E/'results'/f'{a.split}_records.json',records);save(E/'results'/f'{a.split}_errors.json',errors)
if __name__=='__main__':main()
