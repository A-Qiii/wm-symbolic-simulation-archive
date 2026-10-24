"""Independent read-only checks; no model inference, simulation or training."""
from pathlib import Path
import json,hashlib,ast,csv
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/recursive_horizon_diagnostic_20261016';R=P/'reports/recursive_horizon_diagnostic_20261016'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def correct(s,key):
 p=s[key];y=s['truth']['y'];pred=[x>=.5 for x in p[:6]];j=sorted(range(3),key=lambda i:p[6+i],reverse=True)[0];pred += [i==j for i in range(3)];assert len(y)==9 and all(s['truth']['v']);return all(a==b for a,b in zip(pred,y))
def main():
 hashes=read(E/'configs/source_hashes.json')['sha256']
 for p,h in hashes.items():assert sha(p)==h,p
 data=read(E/'results/trajectories.json');met=read(E/'results/metrics.json');ref=data['reference'];logs=data['recorded_action_logs']
 cohorts={'normal_four_step':[r for r in ref if r['four_step_eligible']],'refusal_containing_four_step':[r for r in logs if r['four_step_eligible']and r['contains_refusal_in_first4']],'reference_all_available_variable_N':ref,'logged_all_available_variable_N':logs}
 for name,roots in cohorts.items():
  for h in range(1,5):
   use=[r for r in roots if len(r['steps'])>=h and all(s['observed']for s in r['steps'][:h])];row=met[name][h-1];assert row['N']==len(use)
   assert row['prediction_all9_n']==sum(correct(r['steps'][h-1],'recursive_probability')for r in use)
   assert row['prediction_cumulative_n']==sum(all(correct(s,'recursive_probability')for s in r['steps'][:h])for r in use)
   assert row['actual_readout_all9_n']==sum(correct(r['steps'][h-1],'actual_readout_probability')for r in use)
   for group,lo,hi in[('Holding',0,3),('In',3,6),('drawer',6,9)]:
    if group=='drawer':cnt=sum(max(range(3),key=lambda j:r['steps'][h-1]['recursive_probability'][6+j])!=r['steps'][h-1]['truth']['y'][6:].index(True)for r in use)
    else:cnt=sum(any((r['steps'][h-1]['recursive_probability'][j]>=.5)!=r['steps'][h-1]['truth']['y'][j]for j in range(lo,hi))for r in use)
    assert cnt==row['prediction_'+group+'_wrong_states']
 assert sum(r['observed_steps']==4 for r in ref)==5 and sum(r['observed_steps']==3 for r in ref)==6 and sum(r['observed_steps']==1 for r in ref)==1
 assert sum(r['observed_steps']>=4 for r in logs)==10 and sum(r['observed_steps']==3 for r in logs)==2
 obs=sum(s['observed']for roots in data.values()for r in roots for s in r['steps']);assert obs==85
 csvrows=list(csv.DictReader((R/'per_root_steps.csv').open()));assert len(csvrows)==88
 prov=read(E/'results/inference_provenance.json');assert prov['encoder_calls']==1 and prov['initial_state_encodings']==12 and prov['recursive_transitions']==46 and not prov['intermediate_observation_correction']and prov['training_updates']==0 and prov['new_simulation_calls']==0 and prov['planning_search_calls']==0
 for p in (E/'source').glob('*.py'):ast.parse(p.read_text())
 assert (R/'REPORT.md').is_file()and '<table>'in(R/'index.html').read_text()
 out={'passed':True,'source_hashes_verified':len(hashes),'scored_observed_steps':obs,'CSV_rows_including_three_unreached':len(csvrows),'normal_fixed_N':5,'refusal_fixed_N':10,'independent_recount':True,'training_or_simulation_or_search':False}
 manifest=E/'configs/delivery_manifest.json'
 if manifest.exists():
  files=read(manifest)['sha256']
  for p,h in files.items():assert sha(p)==h,p
  out['sealed_delivery_files_verified']=len(files)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
