from common import *
import argparse
def main(seal):
 sys.addaudithook(guard);count=verify_inputs();frozen=read(S/'configs/heads_frozen.json');selection=read(S/'configs/development_selection.json');comp=read(S/'results/compute_complete.json');pre=read(S/'results/preflight.json');report=read(S/'results/report_complete.json')
 assert comp['fits']==frozen['fits']==6 and comp['updates']==frozen['updates']==9000
 assert all(comp[k]==0 for k in ['encoder_forwards','new_simulations','RSSM_updates','C_D_data_access','uploads'])
 assert frozen['utc']<selection['utc']<comp['utc'];assert read(S/'configs/protocol.json')['utc']<frozen['utc']
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 assert len(list((S/'models').glob('*.pt')))==6 and not(S/'results/FAILED.json').exists()
 for seed in [0,1]:
  raw=features(seed,'train');t=target('train');z=pack(raw)[t['mask'],:1024];stats=np.load(S/f'configs/normalization_seed{seed}.npz');assert np.array_equal(stats['mean'],z.mean(0,dtype=np.float64).astype(np.float32));assert np.array_equal(stats['std'],np.maximum(z.std(0,dtype=np.float64),1e-6).astype(np.float32))
  for kind in ['linear','mlp','geometry']:
   fit=read(S/f'results/fit_{kind}_seed{seed}.json');assert fit['updates']==1500 and fit['curve'][-1]['update']==1500;assert fit['checkpoint_sha256']==sha(S/f'models/{kind}_seed{seed}.pt');assert fit['schedule_sha256']==sha(S/f'configs/schedule_seed{seed}.npy')
  v=target('validation');scores={}
  for kind in ['linear','mlp']:
   q=np.load(S/f'data/visual_{kind}_seed{seed}_validation.npz')['prediction'][...,0];mae,_=root_values(abs(q*WIDTH+LO-v['x'])*1000,v['mask']&v['h0'][:,None],v['root_id']);scores[kind]=float(mae.mean())
  assert scores==selection['selection'][seed]['validation_h0_root_MAE_mm'];assert selection['selection'][seed]['selected']==min(['linear','mlp'],key=lambda k:scores[k])
  for split in ['old_band','gap']:
   for kind in ['linear','mlp']:
    path=S/f'data/visual_{kind}_seed{seed}_{split}.npz';assert path.stat().st_mtime_ns>(S/'configs/development_selection.json').stat().st_mtime_ns
 # Recompute every aggregate from saved predictions, never run a model forward.
 import analyze_report as ar
 metrics=read(S/'results/metrics.json')
 for row in metrics:
  t=target(row['split']);pred,x=ar.prediction(row['seed'],row['split'],row['arm'],t);m=t['mask']&(t['h0'][:,None]if row['cohort']=='h0'else True);acc,ids=root_values(pred==t['cat'],m,t['root_id']);assert np.isclose(acc.mean(),row['root_accuracy'],atol=1e-12);assert len(ids)==row['roots']and int(m.sum())==row['objects'];assert int((m&(pred==t['cat'])).sum())==row['correct']
  if x is not None:
   e=abs(x-t['x'])*1000;mae,_=root_values(e,m,t['root_id']);mse,_=root_values(e**2,m,t['root_id']);margin=np.min(abs(t['x'][...,None]-CUTS),-1)*1000;safe,_=root_values(e<margin,m,t['root_id']);assert np.isclose(mae.mean(),row['MAE_mm'],atol=1e-12)and np.isclose(np.sqrt(mse.mean()),row['RMSE_mm'],atol=1e-12)and np.isclose(safe.mean(),row['margin_sufficient'],atol=1e-12)
 pairs=list(csv.DictReader((S/'results/paired_differences.csv').open()));rootrows=list(csv.DictReader((S/'results/root_statistics.csv').open()))
 for p in pairs:
  a={r['root_id']:float(r['value'])for r in rootrows if (r['split'],r['seed'],r['cohort'],r['arm'],r['metric'])==(p['split'],p['seed'],p['cohort'],p['arm_a'],p['metric'])};b={r['root_id']:float(r['value'])for r in rootrows if (r['split'],r['seed'],r['cohort'],r['arm'],r['metric'])==(p['split'],p['seed'],p['cohort'],p['arm_b'],p['metric'])};assert a.keys()==b.keys();d=np.array([a[r]-b[r]for r in sorted(a)])*(100 if p['metric']=='accuracy'else 1);interval=ar.ci(d,261019+int(p['seed']));assert np.isclose(d.mean(),float(p['difference_a_minus_b']))and np.allclose(interval,[float(p['ci_low']),float(p['ci_high'])])
 cases=read(S/'results/selected_cases.json');assert len(cases)==len({c['root_id']for c in cases})==6
 for c in cases:
  t=target(c['split']);i=c['index'];j=c['object_index'];assert c['root_id']==t['root_id'][i]and c['episode_id']==t['episode_id'][i];assert np.isclose(c['true_x_m'],t['x'][i,j]);assert np.isclose(c['boundary_margin_mm'],np.min(abs(t['x'][i,j]-CUTS))*1000)
  for d in c['details']:
   pred,x=ar.prediction(d['seed'],c['split'],d['arm'],t);assert np.isclose(d['pred_x_m'],x[i,j]);before=bool(np.all(pred[i,t['mask'][i]]==t['cat'][i,t['mask'][i]]));pp=pred[i].copy();pp[j]=t['cat'][i,j];after=bool(np.all(pp[t['mask'][i]]==t['cat'][i,t['mask'][i]]));assert d['offline_one_object_x_substitution']['table_exact_before']==before and d['offline_one_object_x_substitution']['table_exact_after']==after
 assert (S/'REPORT.html').read_text().count('data:image/png;base64,')==report['figures']==11
 assert 'torch'not in sys.modules and 'pybullet'not in sys.modules
 result={'utc':now(),'status':'PASS','input_hashes_unchanged':count,'six_head_hashes_unchanged':True,'fits':6,'total_updates':9000,'prediction_files':len(list((S/'data').glob('*_seed*_*.npz'))),'aggregate_rows_recomputed':len(metrics),'paired_root_CI_rows_recomputed':len(pairs),'independent_case_roots':len(cases),'figures':11,'root_initial_states_unique':70,'identity_records_verified_before_training':3150,'train_only_normalization_recomputed':True,'selection_precedes_evaluation_files':True,'new_encoder_forwards':0,'new_RSSM_updates':0,'new_simulations':0,'C_D_sample_access':0,'uploads':0,'compute_seconds':comp['seconds'],'elapsed_from_protocol_to_verification_seconds':(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(read(S/'configs/protocol.json')['utc'])).total_seconds()}
 if seal:
  assert not(S/'manifest.json').exists(),'No resealing';save(S/'results/final_verification.json',result)
  entries={str(p.relative_to(S)):sha(p)for p in sorted(S.rglob('*'))if p.is_file()and 'cache'not in p.parts and '__pycache__'not in p.parts and p.name!='manifest.json'};save(S/'manifest.json',{'files':len(entries),'sha256':entries});result['sealed_files']=len(entries)
 else:
  manifest=read(S/'manifest.json')
  for name,h in manifest['sha256'].items():assert sha(S/name)==h,name
  result['sealed_files_verified']=manifest['files']
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');args=p.parse_args();main(args.seal)
