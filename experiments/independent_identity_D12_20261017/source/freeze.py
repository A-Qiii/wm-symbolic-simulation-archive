from config import *
import os,itertools,ast,shutil,subprocess,sys
from omegaconf import OmegaConf
def main():
 assert not E.exists();refs={};prior={}
 def reg(p):p=Path(p);refs[str(p)]=sha(p);return p
 for mf in [COMBO/'configs/delivery_manifest.json',OBJ/'configs/delivery_manifest.json',WHOLE/'configs/delivery_manifest.json',EV/'configs/delivery_manifest.json']:
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
  prior[str(mf)]=sha(mf);reg(mf)
 # Keep all previously frozen controller/assets/representation/transition references.
 for p,h in read(COMBO/'configs/before_collection_frozen.json')['sha256'].items():assert sha(p)==h,p;refs[p]=h
 ids=set();seeds=set();layouts=[];sources={};resetpositions=[];opaque=set()
 def visit(x):
  if isinstance(x,list):
   for r in x:visit(r)
  elif isinstance(x,dict):
   if 'root_id'in x:
    rid=str(x['root_id']);ids.add(rid)
    if isinstance(x.get('seed'),int):seeds.add(x['seed'])
    lay=x.get('layout')
    if isinstance(lay,dict)and all('block_'+c in lay for c in COLORS):
     a=np.array([lay['block_'+c]for c in COLORS],float)
     if a.shape==(3,6):layouts.append((rid,a[:,:3]))
    else:opaque.add(rid)
   for k,v in x.items():
    if k not in ['labels','predictions','results']and isinstance(v,(list,dict)):visit(v)
 scene=OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml');order=list(scene.objects.movable_objects)
 for base in [P/'data',EX,P/'runs']:
  for folder,dirs,files in os.walk(base):
   dirs[:]=[d for d in dirs if d not in ['episodes','features','models','__pycache__']and Path(folder)/d!=E]
   for name in files:
    path=Path(folder)/name
    if name.endswith('.json')and 'roots'in name:reg(path);sources[str(path)]=sha(path);visit(read(path))
    elif name=='reset_signature.json':
     rid=path.parents[2].name
     if not rid.startswith('D_'):continue
     r=read(reg(path));resetpositions.append({'root_id':rid,'environment':'D','xyz':[r['physical']['objects']['block_'+c]['position']for c in COLORS],'source':str(path)})
    elif name=='reset_evaluator_only.npz':
     rid=path.parent.name
     if not rid.startswith('D_'):continue
     z=np.load(reg(path));ss=z['scene_obs'];assert len(ss)==24;byname={name:ss[6+6*i:9+6*i].tolist()for i,name in enumerate(order)};resetpositions.append({'root_id':rid,'environment':'D','xyz':[byname['block_'+c]for c in COLORS],'source':str(path)})
 for r in resetpositions:layouts.append((r['root_id'],np.array(r['xyz'])));ids.add(r['root_id'])
 # Unique declared layouts before any simulator call. Same generation distribution as the validated collector.
 oldN=len(layouts);perms=list(itertools.permutations(BRANCHES));pairs=[('red','blue'),('blue','pink'),('pink','red'),('blue','red'),('pink','blue'),('red','pink')];roots=[]
 for pairidx,(c,d)in enumerate(pairs):
  for rep in range(2):
   n=pairidx*2+rep;rid=f'D_identity_{c}_{d}_{rep:02d}';seed=17076000+n;assert rid not in ids and seed not in seeds;rng=np.random.default_rng(seed)
   for attempt in range(10000):
    arr=np.zeros((3,6));orderobj=rng.permutation(3)
    for slot,j in enumerate(orderobj):arr[j,:3]=[.35*(.14+.35*slot)+rng.uniform(-.012,.012),rng.uniform(-.135,-.045),.46];arr[j,5]=rng.uniform(-np.pi,np.pi)
    if not any(np.all(np.linalg.norm(arr[:,:3]-old,axis=1)<.012)for _,old in layouts):break
   else:raise RuntimeError('Unique precollection layout generation exhausted')
   near=min((float(np.max(np.linalg.norm(arr[:,:3]-old,axis=1))),name)for name,old in layouts);branches={'normal':['open_drawer',f'lift_{c}_block_table','place_in_drawer','close_drawer'],'reverse_then_lift':['open_drawer','close_drawer','open_drawer',f'lift_{d}_block_table'],'place_then_other_lift':['open_drawer',f'lift_{c}_block_table','place_in_drawer',f'lift_{d}_block_table']};root={'root_id':rid,'environment':'D','split':'independent_diagnostic_no_training','seed':seed,'target':c,'other_target':d,'pair_group':'seen'if pairidx<3 else'reverse','replicate':rep,'layout':{'block_'+cc:arr[j].tolist()for j,cc in enumerate(COLORS)},'branches':branches,'branch_actions':{b:[action(s)for s in plan]for b,plan in branches.items()},'branch_order':list(perms[n%6]),'generation_attempt':attempt+1,'nearest_previous_xyz_max_distance':near[0],'nearest_previous_root':near[1]};roots.append(root);layouts.append((rid,arr[:,:3]));assert branches['reverse_then_lift'][-1]==branches['place_then_other_lift'][-1]
 roots=[roots[i]for i in np.random.default_rng(17076100).permutation(12)]
 for i,r in enumerate(roots):r['root_execution_order']=i
 for sub in ['source','configs','results','audit','logs','data/raw','data/reset_references','data/combo/public/features','data/combo/_private']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True);save(E/'configs/roots.json',roots);save(E/'configs/previous_reset_positions.json',resetpositions)
 orig=read(COMBO/'configs/protocol.json');heads={}
 for seed in [0,1]:
  heads[str(seed)]={'original':orig['query_heads'][str(seed)]}
  for arm in ['control','coverage']:
   p=reg(OBJ/'models'/f'{arm}_seed{seed}_update0600.pt');heads[str(seed)]['object_'+arm]={'path':str(p),'sha256':sha(p)}
 for p in [OBJ/'configs/object_normalization.npz',OBJ/'source/model.py']:reg(p)
 cfg={'utc':now(),'authorization':'One independent fixed-candidate D12diagnostic;stop afterthisrun regardlessoutcome. No furthertraining/tuning/adoption/expansion.','models':orig['models'],'query_heads':heads,'thresholds':orig['thresholds'],'roots':12,'seen_pair_roots':6,'reverse_pair_roots':6,'directed_pairs':pairs,'root_seed_range':[17076000,17076011],'root_order_seed':17076100,'branches':'normal:open,liftc,place,close;reverse_then_lift:open,close,open,liftd;place_then_other_lift:open,liftc,place,liftd','budget':{'resets':36,'skills_max':144,'controls_per_skill_max':900,'internal_attempts_max':2,'external_retries':0,'replacement_roots':0,'reset_settle':127,'posthold':96,'wall_minutes':45,'storage_GiB':5},'reset_comparator':'exact everyvalue, only tuple/list representation normalization;same controller and reset as prior validatedrun','failure_rule':'first physical nonsuccess stopsbranch;observed failure remains scored, later slots unreached;next fixed branch allowed. Infrastructure stop preserves partial, no external retry or root replacement.','primary':'For one root,branch2and3terminal: predictIn(c)=false/true respectively AND In(d)=false inboth. Compareobjectcoverage vsobjectcontrol samequeryseed,worldmodel,path. Originalsecondary.','eligibility':'Bothterminalobserved,native In(c/d),Holding(d),Open valid;actualIn(c)false/true,actualIn(d)falseboth,Holding(d)trueboth,Openboth. Ineligible never synthesized;report planned,observed,valid,contextmatched,reasons and n/eligible plus n/planned. All observed failures scored separately.','secondary':'AddHolding(d)correctboth;all3Holdingcorrectboth;all9bothterminal;all9all4boundariesperbranchandpairedbranches;perfactconfusions inclselfheldIn;firsteffectiveerror.','grouping':'all12,seen6,reverse6;rootindependentunit,branch/seednotextraN;pairedgain/lossrootcounts,noconfidenceclaim from repeatedframes','truth':'frozen native evaluator only forlabels/score; no expected-planlabels, no trueHolding masking, no GT dynamics','model_inputs':'currenttwoRGBandrobot15, frozenDINO/encoder and8Dcanonicalaction. No branch/root/groundtruthinputs. FrozenobjectIn uses context-enrichedcorresponding64token only; originalHolding/drawer unchanged.','selection':'no model,threshold,normalization or checkpoint selection;bothseedsallheadsallpathsreported','examples':'After scoring,seed0 latent recursive,firstlexicographic eligible objectcoverage primarysuccess andfirstfailure;prefer reverse within each outcome. If no outcome exists saynone rather than fabricate. Shownegative/positive actualimages,truth and allheads probabilities.','prior_C_retention':'ExistingCdegradation remainsvalid evidence; independentDcannoterase it.','end':'seal and cleanprocesses,stop thisround regardlessresult'}
 save(E/'configs/protocol.json',cfg);save(E/'audit/root_exclusion.json',{'utc':now(),'prior_root_ids':len(ids),'prior_seeds':len(seeds),'prior_layout_records':oldN,'prior_native_reset_records':len(resetpositions),'root_manifest_sources':sources,'opaque_root_ids':sorted(opaque),'new_roots':12,'unique_ids_seeds_and_declared_positions':True,'near_duplicate_rule':'all3samecolorXYZwithin12mm rejected beforefreeze;nativeactualreset checkedbeforefirstskill','source_scopes':['data','experiments','runs'],'actual_reset_check_pending':True})
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
 test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(E/'source'),'-p','test_*.py'],capture_output=True,text=True,check=True);save(E/'audit/precollection_tests.json',{'passed':True,'stdout':test.stdout,'stderr':test.stderr,'skill_calls':0,'optimizer_updates':0})
 for p in E.rglob('*'):
  if p.is_file()and '__pycache__'not in p.parts:refs[str(p)]=sha(p)
 save(E/'configs/before_collection_frozen.json',{'frozen_utc':now(),'first_skill_call_started':False,'sha256':refs,'prior_manifests':prior,'model_updates_allowed':0});print('IDENTITY_D12_FROZEN',len(refs),'refs;12roots;36branches;144maxcalls',flush=True)
if __name__=='__main__':main()
