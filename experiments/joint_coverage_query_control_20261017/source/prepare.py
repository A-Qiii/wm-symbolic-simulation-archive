"""Create matched-label training arms and read-only retention sets, before fitting."""
from config import *
import sys,ast,shutil,torch,collections,gzip
sys.path.insert(0,str(EV/'source'))
from native_evaluator import NativeEvaluator

def dual(y,v):return any(y[j]and v[j]and y[k+3]and v[k+3]for j in range(3)for k in range(3)if j!=k)
def main():
 assert not E.exists();refs={};seals={}
 def reg(p):p=Path(p);refs[str(p)]=sha(p);return p
 for mf in [COMBO/'configs/delivery_manifest.json',PROBE/'configs/delivery_manifest.json',EV/'configs/delivery_manifest.json',FIX/'configs/delivery_manifest.json']:
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
  seals[str(mf)]=sha(mf);reg(mf)
 for sub in ['source','configs','data/train','data/evaluation','results','logs','models','audit']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True)
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
 for p in [REP/'source/model.py',PRIOR/'source/train_query.py',REP/'models/query_seed0.pt']:reg(p)
 split=read(reg(PROBE/'configs/root_split.json'));save(E/'configs/root_split.json',split);stats=torch.load(REP/'models/query_seed0.pt',weights_only=True,map_location='cpu')['normalization'];np.savez_compressed(E/'configs/query_normalization.npz',mean=stats['mean'].numpy(),std=stats['std'].numpy())
 old=dict(np.load(reg(FIX/'data/query_train/empty_open.npz')));native=dict(np.load(reg(PROBE/'audit/current156_native_label_copy.npz')));assert np.array_equal(old['root_ids'],native['root_ids'])and np.array_equal(old['state_ids'],native['state_ids']);assert np.array_equal(old['y'],native['old_y'])and np.array_equal(old['v'],native['old_v']);hist={k:old[k]for k in ['z','state_ids','root_ids']};hist.update(y=native['native_y'],v=native['native_v']);assert len(hist['z'])==156 and len(set(hist['root_ids']))==30;assert not any(dual(y,v)for y,v in zip(hist['y'],hist['v']))
 tr=dict(np.load(reg(PROBE/'data/train/states.npz')));meta=read(reg(PROBE/'data/train/states.json'));ix=[i for i,r in enumerate(meta)if r['dual_relation']];assert len(ix)==6 and {meta[i]['root_id']for i in ix}==set(split['train']);z=((torch.tensor(tr['z'][ix])-stats['mean'])/stats['std']).numpy();new={'z':z,'y':tr['y'][ix].astype(np.float32),'v':np.ones((6,9),np.float32),'state_ids':np.array([f"combo_state:{meta[i]['state_index']}"for i in ix]),'root_ids':np.array([meta[i]['root_id']for i in ix])};assert all(dual(y,v)for y,v in zip(new['y'],new['v']));assert all(new['y'][:,:3].sum(1)==1)and all(new['y'][:,3:6].sum(1)==1)
 assert set(hist['root_ids']).isdisjoint(sum(split.values(),[]));np.savez_compressed(E/'data/train/control.npz',**hist);np.savez_compressed(E/'data/train/coverage.npz',**{k:np.concatenate([hist[k],new[k]])for k in hist});save(E/'configs/added_samples.json',[meta[i]for i in ix]);save(E/'audit/historical_label_changes.json',[{'state_id':str(old['state_ids'][i]),'root_id':str(old['root_ids'][i]),'old_y':old['y'][i],'old_v':old['v'][i],'native_y':hist['y'][i],'native_v':hist['v'][i]}for i in range(156)if not(np.array_equal(old['y'][i],hist['y'][i])and np.array_equal(old['v'][i],hist['v'][i]))])
 # Pool sampling is root-balanced; schedules are identical in their first24 slots.
 schedules=[];oldroots=sorted(set(hist['root_ids']));byroot={r:np.flatnonzero(hist['root_ids']==r)for r in oldroots};pool={a:dict(np.load(E/'data/train'/f'{a}.npz'))for a in ['control','coverage']}
 for seed in [0,1]:
  rng=np.random.default_rng(17074000+seed);sampledroots=rng.integers(0,len(oldroots),size=(600,32));control=np.array([[rng.choice(byroot[oldroots[j]])for j in row]for row in sampledroots]);added=np.concatenate([rng.permutation(6)for _ in range(800)]).reshape(600,8)+156;coverage=control.copy();coverage[:,24:]=added;assert np.array_equal(control[:,:24],coverage[:,:24]);assert np.bincount(added.ravel()-156,minlength=6).tolist()==[800]*6;np.savez_compressed(E/'configs'/f'schedule_seed{seed}.npz',control=control,coverage=coverage)
  for arm,indices in [('control',control),('coverage',coverage)]:
   data=pool[arm];y=data['y'][indices.ravel()].astype(bool);v=data['v'][indices.ravel()].astype(bool);info={'seed':seed,'arm':arm,'updates':600,'batch':32,'sample_draws':19200,'old_draws':int((indices<156).sum()),'new_draws':int((indices>=156).sum()),'distinct_states':len(set(indices.ravel())),'training_roots':len(set(data['root_ids'][indices.ravel()])),'valid_fact_cells':int(v.sum()),'valid_positive_cells':int((v&y).sum()),'valid_negative_cells':int((v&~y).sum()),'per_fact_positive':(v&y).sum(0),'per_fact_negative':(v&~y).sum(0),'per_fact_valid':v.sum(0)};schedules.append(info)
 save(E/'results/sampling_budget.json',schedules)
 # Actual readout evaluation: same normalizedz; all9 labels preserved, no test selection.
 def export(name,z,y,v,rootids,stateids,metas):
  np.savez_compressed(E/'data/evaluation'/f'{name}.npz',z=z,y=y,v=v,root_ids=np.array(rootids),state_ids=np.array(stateids));save(E/'data/evaluation'/f'{name}_meta.json',metas)
 export('historical_AB_training',hist['z'],hist['y'],hist['v'],hist['root_ids'],hist['state_ids'],[{'root_id':str(r),'environment':str(r)[0],'dual_relation':False,'source':'historical_train'}for r in hist['root_ids']])
 for splitname,folder in [('new_AB_train','train'),('new_AB_validation','validation'),('D_seen_diagnostic','evaluator_D')]:
  d=dict(np.load(reg(PROBE/f'data/{folder}/states.npz')));rows=read(reg(PROBE/f'data/{folder}/states.json'));zn=((torch.tensor(d['z'])-stats['mean'])/stats['std']).numpy();export(splitname,zn,d['y'].astype(np.float32),np.ones_like(d['y'],np.float32),[r['root_id']for r in rows],[str(r['state_index'])for r in rows],rows)
 # Retention C: historical selection/validation are now descriptive only; both geometry corrected.
 oldstates=read(reg(REP/'data/public/states.json'));newstates=read(reg(PRIOR/'data/ABC/public/states.json'));newpoints={}
 for ep in read(reg(PRIOR/'results/collection_ABC.json')):
  if ep['environment']!='C':continue
  f=reg(Path(ep['folder'])/'_private/states.json')
  for row in read(f):newpoints[row['public_path']]=row['physical']
 traces={};ev=NativeEvaluator('C');retention_changes=[];partial=[]
 for sourcefolder,name in [('query_train','old_C_selection'),('query_evaluator','old_C_validation'),('query_train','C_selection'),('query_evaluator','C_validation')]:
  d=dict(np.load(reg(FIX/f'data/{sourcefolder}/{name}.npz')));yy=[];vv=[];rows=[]
  for i,sid in enumerate(d['state_ids']):
   prefix,index=str(sid).split(':');index=int(index);root=str(d['root_ids'][i]);assert root.startswith('C_')
   if prefix=='new':state=newstates[index];assert state['root_id']==root;lab=ev.evaluate(newpoints[state['path']]);kind='native_full_snapshot'
   else:
    state=oldstates[index];assert state['root_id']==root
    if state.get('dataset')=='matched_place':
     assert not d['v'][i,3:6].any();lab={'y':d['y'][i].tolist(),'v':d['v'][i].tolist()};kind='old_matched_Holding_drawer_only_In_unknown';partial.append({'cohort':name,'state_id':str(sid),'root_id':root,'reason':'Originalrecordhasnoobjectquaternion; preserveall3Ininvalid;Holding/drawerindependentofobjectdimensions.'})
    else:
     source=Path(state['source']);tracepath=source.parent.parent/'_private/full_physics_trace.json.gz'
     if str(tracepath)not in traces:reg(tracepath);traces[str(tracepath)]={x['frame']:x['physical']for x in json.load(gzip.open(tracepath,'rt'))}
     lab=ev.evaluate(traces[str(tracepath)][int(source.stem)]);kind='native_full_snapshot'
   yy.append(lab['y']);vv.append(lab['v']);rows.append({'root_id':root,'environment':'C','dual_relation':dual(lab['y'],lab['v']),'source':kind})
   if not(np.array_equal(d['y'][i],lab['y'])and np.array_equal(d['v'][i],lab['v'])):retention_changes.append({'cohort':name,'state_id':str(sid),'root_id':root,'old_y':d['y'][i],'old_v':d['v'][i],'native_y':lab['y'],'native_v':lab['v']})
  export(name,d['z'],np.array(yy,np.float32),np.array(vv,np.float32),d['root_ids'],d['state_ids'],rows)
 save(E/'audit/retention_C_label_changes.json',retention_changes);save(E/'audit/retention_C_unknown_In.json',partial)
 origcfg=read(reg(COMBO/'configs/protocol.json'));initial=origcfg['query_heads'];cw=read(reg(FIX/'configs/protocol.json'))['query_training']['class_weights']
 for v in initial.values():reg(v['path'])
 for name in ['mixed_seed0','latent_seed0','mixed_seed1','latent_seed1']:reg(COMBO/'results'/f'latents_{name}.npz')
 for path in [COMBO/'results/predictions.json',COMBO/'data/combo/public/pairs.json',COMBO/'data/combo/_private/labels.json',COMBO/'configs/roots.json']:reg(path)
 cfg={'utc':now(),'question':'Is adding true simultaneous relation coverage sufficient for existingquery architecture, under matched continuation budget?','authorization':'Parent explicitly authorizes bounded2seeds x2arms queryonlytraining,600updates; no newcollection,WMchanges or formalreplacement.','initial_heads':initial,'arms':['control','coverage'],'seeds':[0,1],'updates':600,'batch':32,'optimizer':'AdamW lr0.001 wd0.001,clipglobalnorm1,fresh optimizer eacharm','loss_source':str(PRIOR/'source/train_query.py'),'loss':'unchangedmean ofHoldingmaskedBCE,InmaskedBCE,drawerCE; fixed originalpositive/classweights; all9 supervised','class_weights':cw,'checkpoints':[0,600],'selection':'fixed600,no hypersearch/earlystop/validation/Dselection','root_split':split,'historic_training_roots':oldroots,'new_samples':6,'old_pool':156,'coverage_pool':162,'common_old_slots':24,'replacement_slots':8,'replacement_fraction':.25,'contrast':'control19200old draws;coverage14400old+4800new;sharedfirst24slots exact;eachnewroot800draws;actualvalidpositive/negativecellsreported,informationbudgetsnotclaimedidentical.','geometry_correction':'same native correctedhistoricaly/v botharms;oldsealsunchanged;Cretentionalso correctedwherefullsnapshotavailable,12partialoldCstateskeepInunknown.','evaluation':'actual full9/groups/all9factconfusion/dualpositiveandnegative/newABnondual/oldCretentionfirst;then sameheads readfrozenmixed/latent actual,onestep,recursive z; matchedseed query andtransition; rootpaired6-rootABvalidationand6-rootDdiagnostic; no newgate.','D':'previously observed diagnostic only; no training or selection.','no_automatic_adoption':True}
 save(E/'configs/protocol.json',cfg);save(E/'configs/source_references.json',{'sha256':refs,'prior_manifests':seals});print('CONTROL_DATA_PREPARED',json.dumps({'historical_states':156,'historical_roots':30,'new_dual_states':6,'new_training_roots':6,'C_corrected_states':len(retention_changes),'C_partial_unknown_In_states':len(partial),'sampling':schedules},default=lambda x:x.tolist()),flush=True)
if __name__=='__main__':main()
