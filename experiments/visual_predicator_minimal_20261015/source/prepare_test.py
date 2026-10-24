from common import *
import os,shutil
from omegaconf import OmegaConf
def near(a,b,pad=.0005):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2);return bool(np.all(d<=.012+pad)or(np.all(d<=.020+pad)and np.all(yaw<=.25+pad*10)))
def main():
 assert not B.exists()and not(E/'configs/test_frozen.json').exists();assert read(E/'results/runtime_preflight.json')['passed'];B.mkdir();(B/'episodes').mkdir()
 archive=read(P/'experiments/state_predicates_long_pairs_20261014/configs/root_exclusion.json');sources=dict(archive['sources_sha256']);prior=list(archive['prior_layouts']);ids=set();seeds=set()
 for p,h in sources.items():assert sha(p)==h,p
 def visit(x,src):
  if isinstance(x,list):
   for row in x:visit(row,src)
  elif isinstance(x,dict):
   if isinstance(x.get('root_id'),str):ids.add(x['root_id'])
   if isinstance(x.get('seed'),int):seeds.add(x['seed'])
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
    ar=np.array([x['layout']['block_'+c]for c in COLORS],float)
    if ar.shape==(3,6):prior.append({'layout':ar.tolist(),'source':src})
   for key,val in x.items():
    if key not in ['labels','label','predictions','results']and isinstance(val,(dict,list)):visit(val,src)
 for base in ['data','runs','experiments']:
  for root,dirs,files in os.walk(P/base):
   dirs[:]=[d for d in dirs if d not in ['episodes','features','models','figures','videos','__pycache__']and Path(root)/d not in [E,B]]
   for name in files:
    if name.endswith('.json')and'roots'in name:
     pp=Path(root)/name;sources[str(pp)]=sha(pp);visit(read(pp),str(pp))
 prior=list({np.array(r['layout'],float).tobytes():r for r in prior}.values());layouts=[np.array(r['layout'])for r in prior];lo,hi=np.array(OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml').surfaces.table,float);roots=[]
 for color in COLORS:
  for drawer in ['closed','open']:
   seed=15223000+len(roots);rid=f'D_relquery_{color}_{drawer}_0';assert seed not in seeds and rid not in ids;rng=np.random.default_rng(seed);reject=0
   while True:
    ar=np.zeros((3,6));order=rng.permutation(3)
    for slot,k in enumerate(order):ar[k,:3]=[lo[0]+(hi[0]-lo[0])*(.16+.34*slot)+rng.uniform(-.010,.010),rng.uniform(-.135,-.055),.46];ar[k,5]=rng.uniform(-np.pi,np.pi)
    if not any(near(ar,x)for x in layouts):break
    reject+=1
   j=COLORS.index(color);objects=[j]if drawer=='closed'else[j,(j+1)%3];goal=[[3+k,1]for k in objects]+[[6,0]]
   roots.append({'root_id':rid,'seed':seed,'environment':'D','split':'test','target_color':color,'initial_drawer':drawer,'drawer_reset_m':.22 if drawer=='open'else 0.,'layout':{'block_'+c:ar[k].tolist()for k,c in enumerate(COLORS)},'goal':goal,'goal_text':' AND '.join([f'In(obj_{COLORS[k]},drawer)'for k in objects]+['NOT Open(drawer)']),'new_independent_root':True,'rejected_geometric_layout_candidates':reject});layouts.append(ar)
 save(B/'roots_manifest.json',roots);save(E/'configs/root_exclusion.json',{'prior_unique_layouts':len(prior),'prior_layouts':prior,'sources_sha256':sources,'new_roots':6,'prior_observations_labels_predictions_used':False})
 cp=read(E/'configs/reader_frozen.json');files=[E/'configs/protocol.json',E/'configs/reader_frozen.json',Path(cp['path']),E/'configs/rules_frozen.json',B/'roots_manifest.json',E/'configs/root_exclusion.json',CAND/'reader_seed1.pt',P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/env_adapter.py',P/'scripts/core1200_collect.py',P/'scripts/evaluate_physical_oracle.py',RGB/'source/cache.py',RGB/'source/registry.py']+[E/'source'/n for n in ['common.py','model.py','symbolic.py','worker.py','collect_test.py','validate_test.py']]
 save(E/'configs/test_frozen.json',{'created_utc':now(),'roots':6,'arms':['visual_learned_rules','GT_symbols_learned_rules'],'max_calls':96,'max_calls_per_episode':8,'pilot_root':roots[0]['root_id'],'freeze_before_test_physics':True,'frozen_sha256':{str(p):sha(p)for p in files},'reader':cp,'rule_source':'learnedAB_only','fixed_manual_rules_not_used_for_search':True,'no_D_training_or_reselection':True})
 save(B/'data_manifest.json',{'roots':6,'episodes':[],'version':'query_relations_new_D_v1'});save(E/'results/test_storage_before.json',{'free_bytes':shutil.disk_usage(B).free,'maximum_frames_projection':6*2*(1+96+8*(900+96)),'minimum_free_required_bytes':15*1024**3});assert shutil.disk_usage(B).free>15*1024**3
 for p in [B/'roots_manifest.json',E/'configs/root_exclusion.json',E/'configs/test_frozen.json']:p.chmod(0o444)
 print(json.dumps({'new_roots':6,'prior_layouts':len(prior),'fixed_sources':len(sources),'test_sha256':sha(E/'configs/test_frozen.json')}),flush=True)
if __name__=='__main__':main()
