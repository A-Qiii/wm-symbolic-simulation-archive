"""Read-only gate audit: transient large lift during small pick and repeats."""
from paths import ROOT as B, RUNTIME as P
import json,gzip,math
from pathlib import Path
G=B/'engineering/gate_06'
def main():
 summary=json.load(open(G/'summary.json'));assert summary['all_passed'];rows=[];repeat=[]
 for folder in sorted(G.glob('g*_r0')):
  one=json.load(open(folder/'result.json'));other=folder.with_name(folder.name[:-1]+'1');two=json.load(open(other/'result.json'));samples=[json.load(gzip.open(x/'physics.json.gz','rt'))for x in [folder,other]]
  for name,r,ss in [(folder.name,one,samples[0]),(other.name,two,samples[1])]:
   maxlift=0.
   for a in r['actions']:
    if a['action'][:2]!=['Pick','small']:continue
    start,end=a['trace_range'];phase=next(x for x in r['phases']if x['phase']=='approach_exposed_side'and x['control']==start);z0=phase['state']['objects']['large']['position'][2];zs=[x['positions']['large'][2]for x in ss if start<=x['control']<end];maxlift=max(maxlift,max([0.]+[z-z0 for z in zs]))
   attachments=all(x['valid']and all(c['force_N']>.05 for c in x['bilateral_contacts'])and not any(c['child']in ['large','small']for c in x['state_before_attach']['constraints'])for x in r['grasp_audits']);passed=maxlift<=.003 and attachments;rows.append({'case':name,'small_pick_max_large_lift_m':maxlift,'attachment_preconditions':attachments,'pass':passed})
  equal_length=len(samples[0])==len(samples[1]);maxpos=max([0.]+[abs(x['positions'][n][j]-y['positions'][n][j])for x,y in zip(*samples)for n in ['large','small']for j in range(3)]);same_symbols=equal_length and all(x['regions']==y['regions']and x['phase']==y['phase']for x,y in zip(*samples));repeat.append({'pair':folder.name[:-1],'equal_physics_steps':equal_length,'max_position_difference_m':maxpos,'all_region_and_phase_sequences_equal':same_symbols,'pass':bool(equal_length and maxpos<=1e-4 and same_symbols)})
 result={'cases':rows,'repeats':repeat,'all_passed':all(r['pass']for r in rows+repeat),'case_denominator':60,'repeat_pair_denominator':30,'gate_m':{'small_pick_large_lift':.003,'repeat_position_difference':.0001},'source':'Saved per-960Hz-step records; no simulation rerun'};(G/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('ENGINEERING_AUDIT',result['all_passed'],max(x['small_pick_max_large_lift_m']for x in rows),max(x['max_position_difference_m']for x in repeat),flush=True);assert result['all_passed'],'Do not collect on a failed engineering audit'
if __name__=='__main__':main()
