from common import *
import xml.etree.ElementTree as X,os
from omegaconf import OmegaConf

def main():
 assert not B.exists()and not(E/'configs/protocol.json').exists();B.mkdir();(B/'episodes').mkdir();roots=[];alllayouts=[];sources={}
 # Search only root declaration files; observations/results are not consumed.
 def visit(x,path):
  if isinstance(x,list):
   for v in x:visit(v,path)
  elif isinstance(x,dict):
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
    a=np.array([x['layout']['block_'+c]for c in COLORS]);
    if a.shape==(3,6):alllayouts.append(a)
   for k,v in x.items():
    if k not in ['labels','results','predictions']and isinstance(v,(dict,list)):visit(v,path)
 for base in [P/'data',P/'experiments',P/'runs']:
  for root,dirs,files in os.walk(base):
   dirs[:]=[d for d in dirs if d not in ['episodes','features','models','__pycache__']and Path(root)/d not in [E,B]]
   for name in files:
    if name.endswith('.json')and'roots'in name:
     pp=Path(root)/name;sources[str(pp)]=sha(pp);visit(read(pp),pp)
 def near(a,b):return np.all(np.linalg.norm(a[:,:3]-b[:,:3],axis=1)<.012)
 for env in 'ABC':
  cfg=OmegaConf.load(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml');lo,hi=np.array(cfg.surfaces.table);mesh=P/f'vendor/calvin/calvin_env/data/calvin_table_{env}/meshes/drawer_link.obj';vv=np.array([list(map(float,l.split()[1:4]))for l in mesh.read_text().splitlines()if l.startswith('v ')]);assert all(np.any(np.linalg.norm(vv-corner,axis=1)<1e-6)for corner in [[x,.025,z]for x in [-.225,.225]for z in [-.05,.20]])
  for color in COLORS:
   for kind,rep in [('fixtures',0),('fixtures',1),('chain',0)]:
    seed=15811000+len(roots);rng=np.random.default_rng(seed)
    while True:
     a=np.zeros((3,6));order=rng.permutation(3)
     for slot,j in enumerate(order):a[j,:3]=[lo[0]+(hi[0]-lo[0])*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(lo[1]+.015,hi[1]-.015),.46];a[j,5]=rng.uniform(-np.pi,np.pi)
     if not any(near(a,b)for b in alllayouts):break
    alllayouts.append(a);roots.append({'root_id':f'{env}_semantics_v2_{kind}_{color}_{rep}','environment':env,'split':'train'if env in'AB'else'dev','seed':seed,'kind':kind,'replicate':rep,'color':color,'layout':{'block_'+c:a[j].tolist()for j,c in enumerate(COLORS)},'cavity_local_offsets':{'x_jitter':float(rng.uniform(-.012,.012)),'z_jitter':float(rng.uniform(-.012,.012)),'yaw':float(rng.uniform(-.18,.18))}})
 from labels_v2 import SPEC
 protocol={'created_utc':now(),'scope':'One bounded AB/C semantic repair, full support and geometry evidence, fixed executor. No D authorized until preregistered development gate assessed.','labels':SPEC,'root_groups':27,'fixture_root_groups':18,'chain_root_groups':9,'train_root_groups':18,'dev_root_groups':9,'root_split':'A/B train,C dev. All paired fixture branches retain same root. Old312windows may be reused for Holding supervision ONLY; their In/drawer labels are not silently migrated.','fixtures':['table_open','single_inside','edge_straddle','outside_front','stacked','separated','empty_mid','empty_closed'],'matched_close_conditions':['table_open','single_inside','stacked','separated'],'fixture_hold_controls':127,'post_skill_hold_controls':96,'chain':['open_drawer','lift_target','place_in_drawer','close_drawer'],'capture':'Complete contacts, body poses, drawer link transform, joint position and old/new labels every30Hz control. Public RGB/proprio allowlist separate. 4-frame causal windows offsets63/42/21/0.','diagnostic_windows':'fixture frames63,127; after eachskill+96; intermediate first q<=.14 and first q<=.07; first closed-gripper descend; lower_into_drawer frame63 when available. Probes used for supervised development only.','physical_guard':'No teleport after reset, no changed force, collision, externalforce or dynamics; unchanged PhysicalOracle SHA frozen. Layout changes are explicitly paired reset branches, never a simulated transition.','maximum_skill_calls':108,'maximum_control_frames':140000,'training':{'seed':0,'updates':1200,'batch':32,'checkpoint_updates':[300,600,900,1200],'optimizer':'AdamW lr0.001 weight_decay0.001 clip1','architecture':'Frozen previous object/scene encoders and temporal GRUs; fresh shared query-conditioned head, five relation types Holding/In/Closed/Open/Intermediate; 9 output facts. Drawer outputs normalized jointly with softmax.','sampling':'16 samples from old AB Holding-only root pool,16 from new AB root pool; uniform root then window.','loss':'Equal group weights Holding BCE, In BCE, drawer3wayCE; inverse class weights from valid AB labels only; missing groups excluded.','selection':'New C root-macro group loss plus0.2*old C Holding loss, tie earlier. No test selected.'},'development_gate':{'audit':'All source separation, full evidence and frozen code checks pass. Every requested counterexample family has actual AB/C physical evidence; failed intended fixtures stay recorded.','reader':'All9 accuracy on new C>=90%; Closed false-positive0 on valid Intermediate/Open; In false-positive0 on valid outside/suspended/protruding counterexamples; all classes have Cpositive+negative support. Old C Holding must not regress by more than1 of84windows.','rules':'Explicit Closed goal; no unsupported failed transition forced into success effects. Conflicting identical-state outcomes or incomplete failure coverage must be reported and can block an independent test.','next_step':'Only if audit and perception gates pass, assess one frozen new-root acceptance. Otherwise stop additionalD and produce concrete limitations, no repeated tuning.'},'old_report_correction':'Original7fact6/6 included partiallyopen drawer; strictclosed5/6. Priorreport already separates both.'}
 save(E/'configs/protocol.json',protocol);save(B/'roots_manifest.json',roots);save(E/'configs/root_exclusion.json',{'sources_sha256':sources,'prior_or_new_layouts_count':len(alllayouts),'prior_labels_observations_predictions_read':False});files=[E/'configs/protocol.json',B/'roots_manifest.json',E/'source/common.py',E/'source/labels_v2.py',E/'source/collect.py',P/'src/wm_symbolic/physical_oracle.py',P/'scripts/evaluate_physical_oracle.py'];save(E/'configs/collection_frozen.json',{'created_utc':now(),'sha256':{str(p):sha(p)for p in files}});save(B/'manifest.json',{'episodes':[]});print(json.dumps({'roots':len(roots),'fixture_episodes':144,'chain_episodes':9,'max_calls':108,'source_root_files':len(sources)}))
if __name__=='__main__':main()
