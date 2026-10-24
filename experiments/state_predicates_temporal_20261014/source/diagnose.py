from train import *
import gzip
def main():
 a=read(E/'results/audit.json');rows=[]
 for q in a['recovered_samples']:
  rr=[]
  for pp in [q['temporal_inputs']['long_history'][0],q['temporal_inputs']['long_history'][-1]]:
   with np.load(pp)as z:rr.append(z['robot_obs'])
  dz=float((rr[1][2]-rr[0][2])*1000);rows.append({'sample_id':q['sample_id'],'condition':q['condition'],'split':q['split'],'TCP_z_change_mm':dz,'held_truth_for_scoring':q['current_held_for_scoring'],'sign_rule_matches':(dz>0)==q['current_held_for_scoring']})
 source=P/'data/predicate_gap_supplement_20261014/episodes';existing=[]
 for ep in sorted(source.glob('C_*__near_closed_empty')):
  with gzip.open(ep/'trace_private.json.gz','rt')as f:t=json.load(f)
  end=max(i for i,x in enumerate(t)if x['action']['phase']=='far_closed_hold');inds=[end-d for d in [63,42,21,0]];present={j for r in read(ep/'samples_manifest.json')for j in r['history_frame_indices']}
  existing.append({'trajectory':str(ep),'source_trace_sha256':sha(ep/'trace_private.json.gz'),'candidate_anchor':end,'required_long_indices':inds,'missing_exact_RGBD_mask_frames':[j for j in inds if j not in present],'TCP_z_change_mm':float((t[end]['robot_obs'][2]-t[end-63]['robot_obs'][2])*1000),'held_truth':t[end]['labels']['holding_derived'],'status':'Existing root/trajectory, no replay or inference in thisdiagnostic. Would need original-action replay to recover completepublichistory.'})
 save(E/'results/direction_confound.json',{'posthoc_observational_audit':True,'not_a_trained_or_validated_classifier':True,'agreement':sum(x['sign_rule_matches']for x in rows),'windows':len(rows),'rows':rows,'interpretation':'Allheldpositive anchors followupwardTCPmotion; allsupport negatives followdownwardTCPmotion. Longhistoryhasadditionalobservableinformation butthisdatasetcannotseparaterobotmotiondirection/phase fromtrueobjectco-motion.','minimal_next_decision':'Freeze currentlongmodels, recoveronlytheseexistingCemptyclosedtrajectories withoriginalactions/tolerances, once-onlyheldevaluation onupwardemptynegative windows. No new roots/network/training. This is a furtherdevelopmentchallenge, not independentacceptance.','existing_counterexample_candidates':existing})
 ds,_=build('long_history');errors=[]
 for seed in range(3):
  for n in ['old_tune','old_cal','new_dev']:
   p=np.load(E/'results'/f'long_history_seed{seed}_fixed600_{n}.npy');d=ds[n]
   for i,col in zip(*np.where(d['v']&((p>=.5)!=d['y']))):
    errors.append({'seed':seed,'domain':n,'column':int(col),'predicate':'held'if col<3 else'in_drawer'if col<6 else'open_drawer','target_colour':['red','blue','pink'][col%3]if col<6 else None,'truth':bool(d['y'][i,col]),'p':float(p[i,col]),'metadata':d['meta'][i]})
 save(E/'results/remaining_historical_errors.json',errors)
 print(json.dumps({'direction_sign_agreement':sum(x['sign_rule_matches']for x in rows),'n':len(rows),'remaining_long_historical_errors':[{'seed':q['seed'],'root':q['metadata']['root_id'],'predicate':q['predicate'],'truth':q['truth'],'p':q['p'],'category':q['metadata'].get('category'),'paths':q['metadata'].get('public_inputs',q['metadata'].get('public_frames'))[-1]}for q in errors],'existing_counterexample_candidates':len(existing)},ensure_ascii=False))
if __name__=='__main__':main()
