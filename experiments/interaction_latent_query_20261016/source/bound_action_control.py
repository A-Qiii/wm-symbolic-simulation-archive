"""Parameter intervention restricted to declared lift/place arguments.
Old unbound place calls are retained elsewhere and never retroactively GT-bound.
"""
from common import *
from train_queries import load_stage1
from model import feature_error
import torch,fcntl
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out={}
 for split,base in [('C',E/'data'),('D',E/'data/D')]:
  if not(base/'public/pairs.json').exists():continue
  pairs=read(base/'public/pairs.json');statuses={r['pair_id']:r['status']for r in read(base/'_private/pair_outcomes.json')};ids=[i for i,p in enumerate(pairs)if(split=='D'or p['split']=='dev')and family(p['skill'])in['lift','place']and sum(p['action'][4:7])==1]
  with np.load(base/'public/inputs.npz')as d:o=torch.tensor(d['objects'],device='cuda');s=torch.tensor(d['scene'],device='cuda')
  for kind in ['generic','ae','interaction']:
   for seed in [0,1]:
    m,_=load_stage1(kind,seed);b=[pairs[i]['before']for i in ids];a=[pairs[i]['after']for i in ids];act=torch.tensor([pairs[i]['action']for i in ids],device='cuda');wrong=act.clone();wrong[:,4:7]=torch.roll(wrong[:,4:7],1,1)
    with torch.no_grad():z=m.encode(o[b],s[b]);baseerr=feature_error(m.decode(m.transition(z,act)),o[a],s[a],(o[b],s[b])).mean(1).cpu().numpy();wrongerr=feature_error(m.decode(m.transition(z,wrong)),o[a],s[a],(o[b],s[b])).mean(1).cpu().numpy()
    groups={}
    for g in ['normal','refused','failed','all']:
     ix=[j for j,i in enumerate(ids)if g=='all'or statuses[pairs[i]['id']]=={'normal':'success','refused':'refused','failed':'failed'}[g]]
     if not ix:continue
     roots=[pairs[ids[j]]['root_id']for j in ix];diff=wrongerr[ix]-baseerr[ix];rd=[float(diff[np.array(roots)==r].mean())for r in sorted(set(roots))];groups[g]={'pairs':len(ix),'roots':len(rd),'correct_parameter_MSE':float(baseerr[ix].mean()),'wrong_parameter_MSE':float(wrongerr[ix].mean()),'mean_root_loss_increase':float(np.mean(rd)),'root_increases':rd,'roots_with_positive_increase':sum(x>0 for x in rd)}
    out[f'{split}_{kind}_seed{seed}']=groups
 save(E/'results/bound_action_control.json',out);print('BOUND_ACTION_CONTROL_DONE',flush=True)
if __name__=='__main__':main()
