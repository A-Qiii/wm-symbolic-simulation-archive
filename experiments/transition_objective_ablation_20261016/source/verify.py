from config import *
import csv,collections,ast,torch
def main():
 nh=0;prev=0
 for fn in ['protocol_frozen.json','training_frozen.json','choices_frozen.json']:
  for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p;nh+=1
 for old in [SPAN,MECHANISM,DIAG]:
  for p,h in read(old/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p;prev+=1
 cal=read(E/'results/calibration.json');cr=cal['records'];assert len(cr)==64 and sum(r['contains_multistep']for r in cr)==16;assert {(r['schedule_seed'],r['update_index'])for r in cr}=={(s,u)for s in [0,1]for u in range(32)};assert cal['C_examples_used']==cal['D_examples_used']==cal['optimizer_updates']==0;assert cal['initial_model_hash']==cal['final_model_hash'];alpha=float(np.median([r['mixed_parameter_gradient_L2']for r in cr])/np.median([r['rawZ_parameter_gradient_L2']for r in cr]));assert alpha==cal['alpha']==read(E/'configs/training_frozen.json')['alpha']
 rows=read(PACK/'pairs.json');same_initial=[];checkpoints_matched=0;maxdiff=0.;gradient_clips={};chains=read(E/'configs/chains.json');ab=set(chains['AB'])
 for seed in [0,1]:
  sc=dict(np.load(E/'configs'/f'schedule_seed{seed}.npz'));old=dict(np.load(SPAN/'configs'/f'schedule_seed{seed}.npz'));assert np.array_equal(sc['single'],old['single'])and np.array_equal(sc['multi'],old['span3']);assert sc['single'].shape==(1000,64)and sc['multi'].shape==(1000,48,3)
  assert all(rows[i]['split']=='train'for i in np.unique(sc['single']))
  for u in range(1000):
   if(u+1)%4:assert(sc['multi'][u]==-1).all();continue
   seq=sc['multi'][u];assert collections.Counter(rows[ix[0]]['root_id']for ix in seq)=={root:4 for root in ab}
   for ix in seq:assert all(rows[i]['split']=='train'for i in ix)and all(rows[i]['after']==rows[j]['before']for i,j in zip(ix,ix[1:]))
  for arm in ['mixed','latent']:
   name=f'{arm}_seed{seed}';t=read(E/'results'/f'train_{name}.json');same_initial.append(t['initial_model_hash']);assert t['updates']==1000 and t['single_draws']==64000 and t['multi_starts']==12000 and t['alpha']==alpha;assert t['frozen_encoder_decoder_hash']==t['final_frozen_hash']and t['query_hash']==t['final_query_hash'];assert not t['validation_or_test_access'];assert len(t['batch_log'])==1000 and sum(r['contains_multi']for r in t['batch_log'])==250
   for r in t['log']:
    z=r['same_metric_AB_C']['C_select']['recursive_rawZ'];assert abs(r['score']-(.25*z[1]+.5*z[2]+.25*z[3]))<1e-12
   chosen=read(E/'models'/name/'selected.json');best=min(t['log'],key=lambda x:x['score']);assert chosen['C_score']==best['score']and chosen['path'].endswith(f'update{best["update"]:04d}.pt');gradient_clips[name]=sum(x['gradient_norm_before_clip']>5 for x in t['batch_log'])
   if arm=='mixed':
    for update in [0,250,500,750,1000]:
     cp=torch.load(E/'models'/name/f'update{update:04d}.pt',weights_only=True,map_location='cpu')['state'];op=torch.load(SPAN/'models'/f'span3_seed{seed}'/f'update{update:04d}.pt',weights_only=True,map_location='cpu')['state'];diff=max(float((cp[k]-op[k]).abs().max())for k in cp);maxdiff=max(maxdiff,diff);assert diff<1e-6;checkpoints_matched+=1
 assert len(set(same_initial))==1 and same_initial[0]==cal['initial_model_hash']
 pp=read(E/'results/predictions.json');metrics=read(E/'results/metrics.json');labels=read(DIAG/'data/_private/labels.json');assert len(pp)==1680;checked=0
 for r in metrics:
  rr=[p for p in pp if all(p[k]==r[k]for k in ['condition','group','branch','path','global_step'])and(r['color']=='all'or p['color']==r['color'])];assert len(rr)==r['roots'];y=np.array([labels[str(p['after'])]['y']for p in rr]);v=np.array([labels[str(p['after'])]['v']for p in rr]);assert v.all()
  for method in ['actual','one_step','recursive']:
   pr=discrete([p[method+'_probability']for p in rr]);assert int((pr==y).all(1).sum())==r[method]['full9_correct']
   for g,sl in [('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:assert int((pr[:,sl]!=y[:,sl]).any(1).sum())==r[method]['groups'][g]['wrong']
  checked+=1
 roots=read(E/'results/root_trajectories.json');cum=read(E/'results/cumulative.json')
 for r in cum:
  rr=[x for x in roots if all(x[k]==r[k]for k in ['condition','group','branch','method'])and(r['color']=='all'or x['color']==r['color'])];assert len(rr)==r['roots']and sum(x['all4_correct']for x in rr)==r['all4_correct'];assert r['prefix_correct']==[sum(x['prefix_correct'][j]for x in rr)for j in range(4)]
 lookup={(r['condition'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in pp}
 for r in pp:
  if r['condition'].startswith('mixed'):
   other=lookup[(r['condition'].replace('mixed','latent'),r['root_id'],r['branch'],r['path'],r['global_step'])];assert r['actual_probability']==other['actual_probability']and r['actual_logits']==other['actual_logits']
 assert len(list(csv.DictReader((R/'paired_propositions.csv').open())))==15120
 direct_pairs=0
 if(R/'objective_proposition_pairs.csv').exists():
  direct=list(csv.DictReader((R/'objective_proposition_pairs.csv').open()));assert len(direct)==7560
  for row in direct:
   key=(row['root_id'],row['branch'],row['path'],int(row['global_step']));mixed=lookup[(f'mixed_seed{row["seed"]}',*key)];latent=lookup[(f'latent_seed{row["seed"]}',*key)];j=FACTS.index(row['fact']);truth=labels[str(mixed['after'])]['y'][j];assert(row['truth']=='True')==truth
   for name,r,method in [('actual',mixed,'actual'),('mixed_one',mixed,'one_step'),('latent_one',latent,'one_step'),('mixed_recursive',mixed,'recursive'),('latent_recursive',latent,'recursive')]:assert(row[name+'_prediction']=='True')==bool(discrete(r[method+'_probability'])[j])
   direct_pairs+=1
 gate=read(E/'results/gate.json');assert gate['protocol_sha256']==sha(E/'configs/protocol.json');assert not gate['newD_collected']and not gate['formal_adoption'];assert gate['passed']==all(r['passed']for r in gate['seeds'])
 for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
 result={'passed':True,'frozen_hash_entries':nh,'previous_sealed_files_verified':prev,'calibration64ABbatches16multi':True,'fixed_shared_alpha':alpha,'no_C_or_D_calibration':True,'same_3step_schedules_and_initial_model':True,'all_encoder_decoder_query_unchanged':True,'mixed_checkpoints_match_previous_3step_baseline':checkpoints_matched,'mixed_checkpoint_max_parameter_difference':maxdiff,'gradient_clipping_events':gradient_clips,'updates_total':4000,'single_draws_total':256000,'multistep_starts_total':48000,'summary_groups_recounted':checked,'cumulative_counts_recounted':True,'actual_readout_identical':True,'paired_proposition_rows':15120,'direct_objective_proposition_pairs_recounted':direct_pairs,'gate_passed':gate['passed'],'newD_collected':False,'formal_adoption':False}
 if(E/'configs/delivery_manifest.json').exists():
  seal=read(E/'configs/delivery_manifest.json')['sha256']
  for p,h in seal.items():assert sha(p)==h,p
  result['sealed_files_verified']=len(seal)
 print('VERIFICATION_RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
