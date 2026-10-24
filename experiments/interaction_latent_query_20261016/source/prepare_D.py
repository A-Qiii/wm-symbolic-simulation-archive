from common import *
import os
from omegaconf import OmegaConf
def main():
 B=E/'data/D_raw';assert not B.exists();B.mkdir();(B/'episodes').mkdir();layouts=[];seeds=set();sources={}
 def visit(x):
  if isinstance(x,list):
   for r in x:visit(r)
  elif isinstance(x,dict):
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):layouts.append(np.array([x['layout']['block_'+c]for c in COLORS]));seeds.add(x.get('seed'))
   for k,v in x.items():
    if k not in ['labels','results','predictions']and isinstance(v,(dict,list)):visit(v)
 for base in [P/'data',P/'experiments',P/'runs']:
  for path,dirs,files in os.walk(base):
   dirs[:]=[d for d in dirs if d not in['episodes','features','models','__pycache__']and Path(path)/d!=B]
   for name in files:
    if name.endswith('.json')and'roots'in name:
     f=Path(path)/name;sources[str(f)]=sha(f);visit(read(f))
 cfg=OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml');lo,hi=np.array(cfg.surfaces.table);roots=[];prior=len(layouts)
 for color in COLORS:
  for kind,rep in [('chain',i)for i in range(4)]+[('fixtures',i)for i in range(2)]:
   seed=16941000+len(roots);assert seed not in seeds;rng=np.random.default_rng(seed)
   while True:
    arr=np.zeros((3,6));order=rng.permutation(3)
    for slot,j in enumerate(order):arr[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(lo[1]+.015,hi[1]-.015),.46];arr[j,5]=rng.uniform(-np.pi,np.pi)
    if not any(np.all(np.linalg.norm(arr[:,:3]-old[:,:3],axis=1)<.012)for old in layouts):break
   layouts.append(arr);roots.append({'root_id':f'D_interaction_{kind}_{color}_{rep}','environment':'D','split':'test','seed':seed,'kind':kind,'replicate':rep,'color':color,'drawer_reset_m':.22 if kind=='chain'and rep%2 else 0.,'layout':{'block_'+c:arr[j].tolist()for j,c in enumerate(COLORS)},'cavity_local_offsets':{'x_jitter':float(rng.uniform(-.012,.012)),'z_jitter':float(rng.uniform(-.012,.012)),'yaw':float(rng.uniform(-.18,.18))},'closed_loop_matched':kind=='chain'and rep<2})
 save(B/'roots_manifest.json',roots);save(B/'manifest.json',{'episodes':[]});save(E/'configs/D_root_exclusion.json',{'source_sha256':sources,'prior_layout_declarations':prior,'new_unique_roots':len(roots),'old_D_observations_labels_predictions_read':False,'seed_collision':False,'all_new_layouts_separated_by_one_object_at_least12mm':True})
 paths=list((E/'source').glob('*.py'))+list((E/'models').glob('*/selected.json'))+list((E/'models').glob('*/queries/selected.json'))+[B/'roots_manifest.json',E/'data/public/normalization.npz',E/'configs/protocol.json',P/'src/wm_symbolic/physical_oracle.py',P/'scripts/evaluate_physical_oracle.py']
 for d in(E/'models').glob('*/selected.json'):paths.append(Path(read(d)['path']))
 for d in(E/'models').glob('*/queries/selected.json'):paths.append(Path(read(d)['path']))
 save(E/'configs/final_before_D.json',{'created_utc':now(),'sha256':{str(p):sha(p)for p in paths},'test_predictions_opened':False,'model_repair_allowed_after_D':False});print('D_FROZEN',len(roots),flush=True)
if __name__=='__main__':main()
