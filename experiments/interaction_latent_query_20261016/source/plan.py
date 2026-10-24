"""Successors come exclusively from the learned continuous transition.
No hand-coded precondition/effect, physical state, or labels enter this module.
"""
from common import action,COLORS
from model import probs
from train_queries import normalize
import torch
OPTIONS=[('open_drawer',None),('close_drawer',None)]+[(f'lift_{c}_block_table',i)for i,c in enumerate(COLORS)]+[('place_in_drawer',i)for i in range(3)]
@torch.no_grad()
def search(m,q,stats,z,target,horizon=4,beam=32,threshold=.8):
 acts=torch.tensor([action(s,j,96)for s,j in OPTIONS],device=z.device);pr=probs(q(normalize(z,stats)))[0];score=float(torch.minimum(pr[3+target],pr[6]));trace=[]
 if score>=threshold:return {'status':'perceived_goal_satisfied','plan':[],'score':score,'probabilities':pr.cpu().tolist(),'expanded':0}
 states=z;paths=[[]];expanded=0
 for depth in range(1,horizon+1):
  zz=states.repeat_interleave(len(OPTIONS),0);aa=acts.repeat(len(states),1);nxt=m.transition(zz,aa);p=probs(q(normalize(nxt,stats)));values=torch.minimum(p[:,3+target],p[:,6]);expanded+=len(nxt);newpaths=[path+[j]for path in paths for j in range(len(OPTIONS))];best=int(values.argmax());trace.append({'depth':depth,'max_goal_score':float(values[best])})
  if float(values[best])>=threshold:return {'status':'plan_found','plan':[{'skill':OPTIONS[j][0],'target':OPTIONS[j][1]}for j in newpaths[best]],'score':float(values[best]),'probabilities':p[best].cpu().tolist(),'expanded':expanded,'trace':trace}
  keep=values.argsort(descending=True)[:beam];states=nxt[keep];paths=[newpaths[i]for i in keep.tolist()]
 return {'status':'no_plan','plan':[],'score':float(values.max()),'expanded':expanded,'trace':trace}
