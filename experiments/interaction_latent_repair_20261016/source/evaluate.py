from common import *
from runtime import *
from model import probs,feature_error
from metrics import score,discrete
import argparse,fcntl
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['C','D2'],required=True);a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);base=E/'data'if a.split=='C'else E/'data/D2';pairs=read(base/'public/pairs.json');states=read(base/'public/states.json');labels=read(base/'_private/labels.json');outcomes=read(base/'_private/outcomes.json');d=dict(np.load(base/'public/inputs.npz'));o=torch.tensor(d['objects'],device='cuda');s=torch.tensor(d['scene'],device='cuda');take=[i for i,r in enumerate(pairs)if a.split=='D2'or r['split']=='dev'];bb=[pairs[i]['before']for i in take];aa=[pairs[i]['after']for i in take];acts=torch.tensor([pairs[i]['action']for i in take],device='cuda');results={};records=[];oldpairs=read(OLD/'data/public/pairs.json')if a.split=='C'else[]
 def color(i):
  p=pairs[i];out=outcomes[p['id']]
  if out.get('target'):return out['target']
  if p['skill'].startswith('lift_'):return p['skill'].split('_')[1]
  if a.split=='C'and i<len(oldpairs)and sum(oldpairs[i]['action'][4:7]):return COLORS[int(np.argmax(oldpairs[i]['action'][4:7]))]
  return 'unspecified'
 for arm in ARMS:
  for seed in[0,1]:
   key=f'{arm}_seed{seed}';m,q,stats=load(seed,arm)
   with torch.no_grad():
    z=m.encode(o,s);cur=probs(q(normalize(z,stats))).cpu().numpy();pred=m.transition(z[bb],acts);pr=probs(q(normalize(pred,stats))).cpu().numpy();err=feature_error(m.decode(pred),o[aa],s[aa],(o[bb],s[bb])).mean(1).cpu().numpy();latent=(pred-z[aa]).square().mean((1,2)).cpu().numpy();copy=feature_error((o[bb],s[bb]),o[aa],s[aa],(o[bb],s[bb])).mean(1).cpu().numpy()
   groups={'all':list(range(len(take)))}
   for status in['success','refused','failed']:groups['outcome_'+status]=[j for j,i in enumerate(take)if outcomes[pairs[i]['id']]['status']==status]
   for col in COLORS+['unspecified']:groups['colour_'+col]=[j for j,i in enumerate(take)if color(i)==col]
   for condition in['held_open','held_closed','empty_open','empty_closed','reference']:groups['condition_'+condition]=[j for j,i in enumerate(take)if outcomes[pairs[i]['id']].get('condition')==condition]
   if a.split=='C':
    for group in['original','matched']:groups['dataset_'+group]=[j for j,i in enumerate(take)if pairs[i]['dataset_group']==group]
    for group in['original','matched']:
     for status in['success','refused','failed']:groups[group+'_'+status]=[j for j,i in enumerate(take)if pairs[i]['dataset_group']==group and outcomes[pairs[i]['id']]['status']==status]
   out={}
   for name,ix in groups.items():
    if not ix:continue
    rr=[pairs[take[j]]['root_id']for j in ix];rootloss={r:float(np.mean([err[j]for j in ix if pairs[take[j]]['root_id']==r]))for r in sorted(set(rr))};dct={'pairs':len(ix),'roots':len(set(rr)),'weighted_observation_MSE_pair_mean':float(err[ix].mean()),'weighted_observation_MSE_root_mean':float(np.mean(list(rootloss.values()))),'root_errors':rootloss,'latent_MSE':float(latent[ix].mean()),'copy_observation_MSE':float(copy[ix].mean())};valid=[j for j in ix if str(bb[j])in labels and str(aa[j])in labels]
    if valid:
     rs=[pairs[take[j]]['root_id']for j in valid];yb=[labels[str(bb[j])]['y']for j in valid];ya=[labels[str(aa[j])]['y']for j in valid];vb=[labels[str(bb[j])]['v']for j in valid];va=[labels[str(aa[j])]['v']for j in valid]
     dct['current']=score(cur[np.array(bb)[valid]],yb,vb,rs);dct['actual_successor']=score(cur[np.array(aa)[valid]],ya,va,rs);dct['predicted_successor']=score(pr[valid],ya,va,rs);dct['all9_valid_pairs']=sum(all(v)for v in va);dct['target_In_false_positive']=sum(bool(pr[j,3+COLORS.index(color(take[j]))]>=.5 and not labels[str(aa[j])]['y'][3+COLORS.index(color(take[j]))])for j in valid if color(take[j])in COLORS and labels[str(aa[j])]['v'][3+COLORS.index(color(take[j]))]);dct['target_In_valid_pairs']=sum(color(take[j])in COLORS and labels[str(aa[j])]['v'][3+COLORS.index(color(take[j]))]for j in valid)
    out[name]=dct
   state_ix=[i for i,r in enumerate(states)if str(i)in labels and(a.split=='D2'or r['split']=='dev')];out['all_current_windows']=score(cur[state_ix],[labels[str(i)]['y']for i in state_ix],[labels[str(i)]['v']for i in state_ix],[states[i]['root_id']for i in state_ix]);results[key]=out
   for j,i in enumerate(take):records.append({'arm':arm,'seed':seed,'pair_id':pairs[i]['id'],'root_id':pairs[i]['root_id'],'skill':pairs[i]['skill'],'color':color(i),'condition':outcomes[pairs[i]['id']].get('condition'),'status':outcomes[pairs[i]['id']]['status'],'before':bb[j],'after':aa[j],'current':cur[bb[j]].tolist(),'actual_successor':cur[aa[j]].tolist(),'predicted_successor':pr[j].tolist(),'true_before':labels.get(str(bb[j])),'true_after':labels.get(str(aa[j])),'observation_error':float(err[j]),'latent_error':float(latent[j])})
   np.savez_compressed(E/'results'/f'{a.split}_{key}.npz',current=cur,predicted=pr,predicted_latent=pred.cpu().numpy(),pair_indices=take,error=err);print('EVAL',a.split,key,'MSE',out['all']['weighted_observation_MSE_root_mean'],flush=True)
 save(E/'results'/f'{a.split}_metrics.json',results);save(E/'results'/f'{a.split}_records.json',records)
if __name__=='__main__':main()
