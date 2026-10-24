"""Hand-counted metric bookkeeping fixture; no experimental performance."""
from pathlib import Path
import json,torch
from wm_symbolic.core_field_metrics import FieldMetrics
actual=torch.tensor([[[1,1,2,0,0],[6,1,2,2,0]],[[1,1,2,0,0],[1,6,2,0,0]]]);predicted=torch.tensor([[6,1,2,2,0],[2,6,2,2,0]])
objects=torch.full((2,3,8),-20.);drawer=torch.full((2,3),-20.);slider=torch.full((2,3),-20.)
for i in range(2):
 for j in range(3):objects[i,j,predicted[i,j]]=20.
 drawer[i,predicted[i,3]]=20.;slider[i,predicted[i,4]]=20.
pred={'objects':objects,'drawer':drawer,'slider':slider,'lights':torch.full((2,2),-20.)}
b={'cat':actual,'cat_valid':torch.ones_like(actual,dtype=torch.bool),'binary':torch.zeros(2,2,2),'binary_valid':torch.ones(2,2,2,dtype=torch.bool),'skill':torch.zeros(2,1,dtype=torch.long)}
f=FieldMetrics();f.add('prior',pred,b,1,torch.ones(2,dtype=torch.bool),0,['open_drawer']);r=f.result();s=r['state_metrics']['prior'];fields=r['field_metrics']['prior'];names=['red_location','blue_location','pink_location','drawer_state','slider_state']
assert sum(fields[n]['correct'] for n in names)==8 and sum(fields[n]['valid'] for n in names)==10
assert sum(fields[n]['changed_correct'] for n in names)==3 and sum(fields[n]['changed_valid'] for n in names)==3
assert sum(fields[n]['unchanged_correct'] for n in names)==5 and sum(fields[n]['unchanged_valid'] for n in names)==7
assert s['all5_correct']==s['all7_correct']==1 and s['all5_valid']==s['all7_valid']==2 and s['holding_identity_correct']==2
out={'passed':True,'hand_counted_field_accuracy':'8/10','hand_counted_whole_state_accuracy':'1/2','changed':'3/3','unchanged':'5/7','holding_identity':'2/2','scope':'Synthetic CPU metric fixture only; not learned-model or physical accuracy'}
Path('/home/robot/research/projects/wm_symbolic/runs/core1200_v1/field_metrics_regression.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
