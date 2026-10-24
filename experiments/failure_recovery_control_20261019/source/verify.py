from common import *
import torch,ast
def main():
 guard('verify');start=time.monotonic();design=verify_freeze();frozen=read(S/'configs/training_frozen.json');fail=[]
 for p,h in frozen['sha256'].items():
  if sha(p)!=h:fail.append(p)
 assert not fail,fail
 for path,info in frozen['append_only_aggregate_inputs'].items():
  assert sha(info['snapshot'])==info['sha256'];before=read(info['snapshot']);after=read(path);key='records'if 'records'in before else'items';old=[r for r in after[key]if r['split']!='evaluation'];assert old==before[key];extra=[r for r in after[key]if r['split']=='evaluation'];assert extra
 for path,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(path)==h
 # Verify prior geometric exclusion source inventory, with only A/B files listed in that inventory.
 inv=read(S/'configs/AB_exclusion_inventory.json')
 for p,h in inv['files_sha256'].items():assert sha(p)==h
 roots=read(S/'data/roots_manifest.json');assert collections.Counter(r['split']for r in roots)=={'train':12,'development':6,'evaluation':6};assert len(set(r['family_id']for r in roots))==24
 selected=read(S/'configs/models_frozen.json');assert selected['utc']<read(S/'results/evaluation_collection_started.json')['utc'];assert read(S/'results/evaluation_collection_started.json')['selection_sha256']==sha(S/'configs/models_frozen.json')
 for p,h in selected['sha256'].items():assert sha(p)==h
 receipts=[read(S/f'results/train_{a}_seed{s}.json')for s in [0,1]for a in ARMS];assert len(receipts)==4 and all(r['updates']==1000 and r['old_samples']==8000 and r['new_samples']==8000 and r['same_terminal_loss_both_arms']for r in receipts)
 for seed in [0,1]:
  a,b=[r for r in receipts if r['seed']==seed];assert a['initial_hash']==b['initial_hash'];assert a['final_hash']!=b['final_hash'];sch=np.load(S/f'configs/schedule_seed{seed}.npz');assert sch['old'].shape==(1000,2,4)and sch['new'].shape==(1000,2,4);assert sch['old'].max()<120 and sch['new'].max()<24
 assert sha(S/'source/losses.py')==sha(T/'source/losses.py');assert read(S/'results/training_complete.json')['updates']==4000
 for phase,n in [('train_dev',324),('evaluation',108)]:
  assert read(S/f'results/audit_{phase}.json')['passed'];assert read(S/f'results/audit_{phase}.json')['transitions']==n;assert read(S/f'results/families_{phase}.json')['passed']
 records=read(S/'data/data_manifest.json')['records'];assert len(records)==432;assert len({r['episode_id']for r in records})==72;assert len(read(S/'data/features/index.json')['items'])==504
 assert not list((S/'results').glob('*FAILED*.json'))
 inference=read(S/'results/inference_complete.json');assert inference['weights_unchanged'];assert all(r['paired_prefault_probabilities_exact']and r['suffix_future_observations_used']==0 and r['samples']==8 for r in inference['reports'])
 arrays=0
 for p in(S/'results').glob('predictions_*.npz'):
  with np.load(p)as d:
   assert np.isfinite(d['prob']).all()and ((d['prob']>=0)&(d['prob']<=1)).all();assert len(d['step'])==396;assert d['cat_valid'][:,:4].all();arrays+=1
 # Independently inspect the suffix body to ensure no future encode/observe function is called.
 tree=ast.parse((S/'source/evaluate.py').read_text());loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.iter,ast.Call)and isinstance(n.iter.func,ast.Name)and n.iter.func.id=='range'and [x.value for x in n.iter.args if isinstance(x,ast.Constant)]==[2,5]];assert len(loops)==1;calls=[n.func.attr for n in ast.walk(loops[0])if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)];assert 'imagine'in calls and 'observe'not in calls and 'encode'not in calls
 # Input whitelist is a code-level check, labels are legitimate offline training/scoring targets only.
 encode_calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='encode'];assert len(encode_calls)==1;assert [n.slice.value for n in encode_calls[0].args]==['tokens','geometry','visible','robot']
 changes=[]
 for seed in [0,1]:
  orig=torch.load(CELL/f'models/joint_seed{seed}/final6000.pt',map_location='cpu',weights_only=False)['model']
  for arm in ARMS:
   chosen=next(r for r in selected['selection']if r['seed']==seed and r['arm']==arm);cp=torch.load(chosen['path'],map_location='cpu',weights_only=False)['model'];groups=collections.defaultdict(list)
   for name,w in orig.items():groups[name.split('.')[0]].append(float((w.double()-cp[name].double()).square().sum()))
   changes.append({'seed':seed,'arm':arm,'selected_update':chosen['update'],'squared_parameter_changes':{k:sum(v)for k,v in groups.items()}});assert sum(groups['posterior'])>0 and sum(groups['gru'])>0 and sum(groups['vision'])>0
  del orig
 assert read(S/'results/report_complete.json')['figures']==8;assert (S/'REPORT.html').stat().st_size>100000;assert read(S/'results/analysis_complete.json')['all_numeric_values_finite'];assert read(S/'results/interpretation.json')['no_new_inference_or_training']
 result={'utc':now(),'passed':True,'design_hashes':design,'training_input_hashes_unchanged':len(frozen['sha256']),'prior_AB_inventory_hashes_unchanged':len(inv['files_sha256']),'new_independent_roots':24,'new_real_transitions':432,'new_evaluation_roots':6,'new_evaluation_transitions':108,'fits':4,'updates':4000,'prediction_arrays':arrays,'model_selection_precedes_evaluation_collection':True,'aggregates_only_append_evaluation':True,'future_observations_in_suffix':0,'forbidden_model_inputs':0,'C_D_samples':0,'uploads':0,'parameter_changes':changes,'seconds':time.monotonic()-start};save(S/'results/verification.json',result)
 files=[p for p in S.rglob('*')if p.is_file()and '__pycache__'not in p.parts and 'cache'not in p.parts and p.name not in ['MANIFEST.json','SEALED.json']and not p.name.endswith('.partial')];manifest={'utc':now(),'files':[{ 'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(files)]};save(S/'MANIFEST.json',manifest);save(S/'SEALED.json',{'utc':now(),'manifest_sha256':sha(S/'MANIFEST.json'),'files':len(files),'bytes':sum(x['bytes']for x in manifest['files']),'verification_passed':True,'no_further_training_or_model_selection':True});print('VERIFIED_AND_SEALED',sha(S/'MANIFEST.json'),len(files),flush=True)
if __name__=='__main__':main()
