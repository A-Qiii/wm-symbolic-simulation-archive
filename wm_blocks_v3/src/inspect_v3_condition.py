"""Frozen conditional-coefficient evidence for the same PlaceLargeA action."""
from paths import ROOT as B, RUNTIME as P
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np,torch
from models import RGBState,Dynamics
from models_v3 import DiagonalStateDynamics
def read(p):return json.load(open(p))
@torch.no_grad()
def main():
 rows=read(B/'cache/v3/confirmation/state_rows.json');features=np.load(B/'cache/v3/confirmation/all_features.npz')['features'];index={(r['episode'],r['boundary']):i for i,r in enumerate(rows)};eps=read(B/'configs/v3_confirmation_manifest.json')['episodes'];predictions=read(B/'reports/v3/confirmation/all_predictions.json');result=[]
 for rec in read(B/'configs/v3_models_frozen.json')['models']:
  est=RGBState(1567);est.load_state_dict(torch.load(rec['state_checkpoint'],map_location='cpu',weights_only=False)['model']);net=DiagonalStateDynamics();net.load_state_dict(torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)['model']);est.eval();net.eval();norm=np.load(rec['normalization']);parts={}
  for recipe in ['B_loaded_reverse','B_empty_reverse']:
   selected=[e for e in eps if e['recipe']==recipe];ii=[index[(e['id'],1)]for e in selected];state=Dynamics.pack(*est(torch.tensor((features[ii]-norm['mean'])/norm['std'])));capture={};hook=net.motion.interaction.register_forward_hook(lambda m,i,o:capture.update(matrix=o.clone().reshape(-1,3,3)));n=len(ii);net(state,torch.ones(n,dtype=torch.long),torch.zeros(n,dtype=torch.long),torch.ones(n,dtype=torch.long),est.sd);hook.remove();diag=capture['matrix'].diagonal(dim1=-2,dim2=-1).numpy();parts[recipe]={'same_action':'Place(large,A)','denominator':n,'coefficients_xyz_mean':diag.mean(0).tolist(),'coefficients_xyz_min':diag.min(0).tolist(),'coefficients_xyz_max':diag.max(0).tolist(),'all_off_diagonal_exact_zero':bool((capture['matrix']-torch.diag_embed(capture['matrix'].diagonal(dim1=-2,dim2=-1))).abs().max()==0)}
  cases=[r for r in predictions if r['version']=='v3'and r['seed']==rec['seed']and r['physical_valid']];total=sum(len(r['truth'])for r in cases);current={'total':total,'region_correct':sum(p['region']==t['region']for r in cases for p,t in zip(r['current'],r['truth'])),'Held_correct':sum(p['held']==t['held']for r in cases for p,t in zip(r['current'],r['truth'])),'joint_correct':sum(p['region']==t['region']and p['held']==t['held']for r in cases for p,t in zip(r['current'],r['truth']))};result.append({'seed':rec['seed'],'conditions':parts,'current':current});print('CONDITION',rec['seed'],parts,'CURRENT',current,flush=True)
 out=B/'reports/v3/analysis';out.mkdir(exist_ok=True,parents=True);(out/'conditional_coefficients.json').write_text(json.dumps({'support_labels_input':False,'state_coefficients_learned_not_specified':True,'models':result},indent=2)+'\n')
if __name__=='__main__':torch.set_num_threads(2);main()
