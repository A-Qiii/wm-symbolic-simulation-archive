from config import *
import csv,ast,collections
def main():
 frozen=read(E/'configs/frozen.json');previous=0
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 for old in [SPAN,DIAG]:
  for p,h in read(old/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p;previous+=1
 d=dict(np.load(E/'data/latent_probes.npz'));h=d['history'];cases=read(E/'data/cases.json');samples=read(E/'configs/samples.json');cp=np.load(E/'data/checkpoints.npz')['values'];cpmeta=read(E/'data/checkpoint_cases.json')
 assert h.shape==(2400,2,6,16)and cp.shape==(9600,14)
 assert all(np.isfinite(v).all()for v in d.values())and np.isfinite(cp).all()
 assert len(samples)==240 and len({s['root_id']for s in samples})==36
 assert collections.Counter(s['split']for s in samples)=={'AB_train':144,'C_development':96}
 assert all(np.array_equal(h[:,0,0],h[:,j,0])for j in [1])
 group_sum=d['gF_groups'].sum(1);gradient_sum_max_error=float(np.abs(group_sum-d['gF']).max());gradient_sum_relative_RMS=float(np.sqrt(np.mean((group_sum-d['gF'])**2))/np.sqrt(np.mean(d['gF']**2)))
 assert gradient_sum_max_error<1e-7 and gradient_sum_relative_RMS<1e-6
 assert np.allclose(d['g025Z'],.5*(d['initial']-d['target'])/256,atol=1e-9)
 shrink=(1-4*.5/256)**32;analytic=d['target']+(d['initial']-d['target'])*shrink;err=float(np.abs(analytic-d['end'][:,1]).max());assert err<1e-5,err
 actual=[i for i,c in enumerate(cases)if c['origin']=='actual'];assert np.array_equal(d['initial'][actual],d['target'][actual]);assert np.array_equal(d['end'][actual,1],d['target'][actual]);assert not np.any(d['g025Z'][actual])
 actual0=[i for i,c in enumerate(cases)if c['model']=='actual_seed0'];actual1=[i for i,c in enumerate(cases)if c['model']=='actual_seed1'];assert np.array_equal(d['end'][actual0],d['end'][actual1])
 assert np.allclose(np.square(d['end']-d['target'][:,None]).mean((2,3)),h[:,:,-1,4],atol=1e-7)
 losses=h[:,0,:,:4].mean(-1)+.25*h[:,0,:,4];assert(np.diff(losses,axis=1)<=1e-6).all()
 labs=read(DIAG/'data/_private/labels.json');ys=np.array([labs[str(s['after'])]['y']for s in samples],bool);y=ys[[c['sample_id']for c in cases]];ok=(discrete_logits(h[...,5:14])==y[:,None,None]).all(-1)
 ends=list(csv.DictReader((R/'latent_endpoints.csv').open()));assert len(ends)==4800
 for r in ends:
  i=int(r['case_id']);j=0 if r['objective']=='mixed'else 1;assert(r['baseline_correct']=='True')==ok[i,j,0];assert(r['final_correct']=='True')==ok[i,j,-1];assert abs(float(r['Z_final'])-float(h[i,j,-1,4]))<1e-9
 summary=read(E/'results/latent_summary.json');primary=0
 for r in summary:
  if r['cohort']!='all':continue
  ii=[i for i,c in enumerate(cases)if c['model']==r['model']and c['origin']==r['origin']and(r['split']=='all'or samples[c['sample_id']]['split']==r['split'])and(r['branch']=='all'or samples[c['sample_id']]['branch']==r['branch'])];j=0 if r['objective']=='mixed'else 1;k=[0,1,4,8,16,32].index(r['step']);assert len(ii)==r['N_boundaries'];assert int(ok[ii,j,k].sum())==r['correct'];assert int((ok[ii,j,0]&~ok[ii,j,k]).sum())==r['correct_to_wrong'];primary+=1
 cpok=(discrete_logits(cp[:,5:14])==ys[[r['sample_id']for r in cpmeta]]).all(1);lookup={(r['model'],r['origin'],r['sample_id'],r['update']):i for i,r in enumerate(cpmeta)};pairs=read(E/'results/checkpoint_pairs.json');assert len(pairs)==7680
 for r in pairs:
  i=lookup[(r['model'],r['origin'],r['sample_id'],r['from_update'])];j=lookup[(r['model'],r['origin'],r['sample_id'],r['to_update'])];assert r['correct_to_wrong']==bool(cpok[i]and not cpok[j]);assert r['F_down']==bool(cp[j,:4].mean()<cp[i,:4].mean()-1e-6);assert r['Z_down']==bool(cp[j,4]<cp[i,4]-1e-6)
 audit=read(E/'results/run_audit.json');assert audit['network_updates']==audit['simulation_calls']==audit['private_truth_reads']==0 and audit['all_model_parameters_frozen']
 for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
 result={'passed':True,'frozen_input_hashes':len(frozen['sha256']),'previous_sealed_files':previous,'all240_boundaries_36roots':True,'all_2400_cases_same_start_for_objectives':True,'all_models_frozen_and_unchanged':True,'no_network_training_or_simulation':True,'no_labels_during_differentiation':True,'latent_only_analytic_max_absolute_error':err,'true_latent_Zonly_stays_exactly_fixed':True,'actual_latent_paths_identical_across_query_seeds':True,'F_gradient_equals_sum_of_group_gradients_with_float32_roundoff':True,'gradient_sum_max_abs_error':gradient_sum_max_error,'gradient_sum_relative_RMS':gradient_sum_relative_RMS,'mixed_loss_nonincreasing_at_all_recorded_steps':True,'endpoint_rows_independently_recounted':len(ends),'primary_summary_groups_recounted':primary,'checkpoint_pairs_recounted':len(pairs),'finite_arrays':True}
 if(E/'configs/delivery_manifest.json').exists():
  seal=read(E/'configs/delivery_manifest.json')['sha256']
  for p,hh in seal.items():assert sha(p)==hh,p
  result['sealed_files_verified']=len(seal)
 print('VERIFICATION_RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
