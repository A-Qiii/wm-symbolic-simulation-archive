"""AB cache/label alignment and actual shared-query binding tests, no optimization."""
from common import *
import sys,importlib.util,itertools,collections,subprocess,os
import torch
sys.path.insert(0,str(SP/'source'))
from model import QueryRelationReader,parse_query,decode
sys.path.insert(0,str(RGB/'source'))
from registry import extract

def model_load(arm,update=None):
 cp=read(SP/'configs'/arm/'reader_frozen.json')if update is None else next(x for x in read(SP/'results'/arm/'training.json')['checkpoints']if x['update']==update)
 assert sha(cp['path'])==cp['sha256'];net=QueryRelationReader(arm).eval();net.load_state_dict(torch.load(cp['path'],weights_only=False,map_location='cpu')['model']);return net

def main():
 torch.set_num_threads(4);meta=read(BASE/'data/metadata.json');lookup={r['window_id']:i for i,r in enumerate(meta)};wins=read(BASE/'data/windows.json');ab=[w for w in wins if w['environment']in ['A','B']];z=dict(np.load(BASE/'data/inputs.npz'));z.update(dict(np.load(SP/'data/spatial_inputs.npz')));targets=dict(np.load(BASE/'data/targets.npz'));stats=read(CAND/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);diffs={'bbox_from_RGB':0.,'local_from_cache':0.,'grid_from_cache':0.,'bbox_from_cache':0.};visible=collections.Counter()
 for n,w in enumerate(ab):
  i=lookup[w['window_id']];assert targets['y'][i].tolist()==w['targets_v2_1']['y']and targets['v'][i].tolist()==w['targets_v2_1']['v']
  raw=dict(np.load(meta[i]['raw_feature_path']));g=raw['geometry'][-1];im=np.load(w['public_frames'][-1])
  for camid,cam in enumerate(['static','gripper']):
   rgb=im['rgb_'+cam];h,width=rgb.shape[:2];top=(224-h)//2;left=(224-width)//2;masks,_=extract(rgb)
   for j,c in enumerate(COLORS):
    yy,xx=np.where(masks[j]);vis=len(yy)>=4;box=[(xx.min()+left)/224,(yy.min()+top)/224,(xx.max()+left+1)/224,(yy.max()+top+1)/224,1.]if vis else[0.,0.,1/224,1/224,0.];diffs['bbox_from_RGB']=max(diffs['bbox_from_RGB'],float(np.abs(np.array(box,np.float32)-g[camid,j,0,:5]).max()));visible[cam+':'+c]+=int(vis)
  local=(raw['local_tokens'][-1].astype(np.float32)-vm)/vs;local*=g[:,:,0,4][:,:,None,None];grid=(raw['global_grid'][-1].astype(np.float32)-vm)/vs;bb=np.concatenate([g[:,:,0,:5],g[:,:,:,5].mean(2)[...,None]],-1).transpose(1,0,2).reshape(3,12)
  for key,value in [('local',local),('grid',grid),('bbox',bb)]:diffs[key+'_from_cache']=max(diffs[key+'_from_cache'],float(np.abs(value-z[key][i]).max()))
 assert all(v==0 for v in diffs.values()),diffs
 save(E/'results/feature_label_binding.json',{'AB_new_windows_checked':420,'RGB_current_camera_views_checked':840,'object_colour_slots_per_view':3,'order':['red','blue','pink'],'target_columns_match_labels':True,'maximum_absolute_differences':diffs,'visible_counts':dict(visible),'no_backbone_encoding_or_new_download':True})
 ids=np.array([i for i,r in enumerate(meta)if r['environment']in ['A','B']]);assert len(ids)==648;x={k:torch.tensor(v[ids])for k,v in z.items()};newlocal=np.array([meta[i]['pool']=='new_semantics'for i in ids]);y=targets['y'][ids];v=targets['v'][ids];answer={};cp_rows=[]
 errors=read(SP/'results/errors.json');error_ids=sorted({r['window_id']for r in errors if r['model']=='spatial'});assert len(error_ids)==6;errix=[lookup[w]for w in error_ids];cx={k:torch.tensor(val[errix])for k,val in z.items()}
 with torch.inference_mode():
  for arm in ['pooled','spatial']:
   net=model_load(arm);p=net.probabilities(**x).numpy();nodes=net.nodes(**x);perms=[]
   for perm in itertools.permutations(range(3)):
    per=list(perm);xp=dict(x);xp['objects']=x['objects'][:,:,per];xp['local']=x['local'][:,:,per];xp['bbox']=x['bbox'][:,per];pp=net.probabilities(**xp).numpy();expected=p[:,per+[j+3 for j in per]+[6,7,8]];delta=float(abs(pp-expected).max());assert delta<1e-5;assert np.array_equal(decode(pp),decode(expected));perms.append({'permutation':per,'max_probability_difference':delta,'decoded_changes':0})
   query_names=['Holding(item_'+c+')'for c in COLORS]+['In(item_'+c+',drawer)'for c in COLORS]+['Closed(drawer)','Open(drawer)','Intermediate(drawer)'];qs=torch.tensor([parse_query(q,{'item_'+c:i for i,c in enumerate(COLORS)})for q in query_names]);ans=net.query(nodes,qs).numpy();alias_diff=float(abs(ans-p).max());assert alias_diff<1e-5;reorder=[8,5,2,7,4,1,6,3,0];rev=net.query(nodes,qs[reorder]).numpy();assert np.allclose(rev,p[:,reorder],atol=1e-5)
   rejected=[]
   for q in ['In(item_unknown,drawer)','Closed(item_red)','Holding(drawer)','In(item_red,unknown)']:
    try:parse_query(q,{'item_'+c:i for i,c in enumerate(COLORS)})
    except ValueError:rejected.append(q)
   assert len(rejected)==4
   d=decode(p);eligible=newlocal&v[:,3:6].all(1)&(y[:,3:6].sum(1)==1);matrix=np.zeros((3,5),int);mistakes=[]
   for k in np.flatnonzero(eligible):
    truth=int(y[k,3:6].argmax());preds=np.flatnonzero(d[k,3:6]==1);column=int(preds[0])if len(preds)==1 else 3 if len(preds)==0 else 4;matrix[truth,column]+=1
    if column!=truth:mistakes.append({'window_id':meta[ids[k]]['window_id'],'true_object':COLORS[truth],'predicted_In_objects':[COLORS[j]for j in preds]})
   neg=newlocal&v[:,3:6].all(1)&(y[:,3:6].sum(1)==0)
   answer[arm]={'AB_windows_tested':len(ids),'new_AB_windows':420,'legacy_AB_Holding_only':228,'all_six_object_permutations':perms,'entity_alias_max_difference':alias_diff,'query_reorder_max_difference':float(abs(rev-p[:,reorder]).max()),'invalid_typed_queries_rejected':rejected,'exactly_one_In_positive_AB_windows':int(eligible.sum()),'object_identity_confusion_rows':COLORS,'object_identity_confusion_columns':COLORS+['none','multiple'],'object_identity_confusion':matrix.tolist(),'object_identity_errors':mistakes,'all3_In_false_AB_windows':int(neg.sum()),'all3_In_false_AB_windows_any_FP':int((d[neg,3:6]==1).any(1).sum()),'explicit_target_identity_enters_network':False,'note':'Equivariance and aliases test index/name wiring; they do not prove absence of learned colour/background shortcuts.'}
   for update in [300,600,900,1200]:
    ck=model_load(arm,update);cp=ck.probabilities(**cx).numpy();dd=decode(cp)
    for wi,pv,dv in zip(error_ids,cp,dd):cp_rows.append({'variant':arm,'update':update,'window_id':wi,'probabilities':dict(zip(FACTS,map(float,pv))),'decoded':dict(zip(FACTS,map(int,dv))),'purpose':'Explain six previously recorded C error windows only; no checkpoint reselection or C training.'})
 save(E/'results/query_binding.json',answer);save(E/'results/existing_C_checkpoint_trace.json',cp_rows)
 cli=[];wid='C_semantics_v2_fixtures_pink_0__empty_closed__t00127'
 for arm in ['pooled','spatial']:
  reference=None
  for goal in COLORS:
   cmd=[sys.executable,str(SP/'source/query.py'),'--variant',arm,'--window-id',wid,'--query','In(obj_pink,drawer)','--goal-object',goal];p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});o=json.loads(p.stdout)
   if reference is None:reference=o['all_probabilities']
   assert reference==o['all_probabilities'];assert o['GT_loaded_at_inference']==False;assert o['joint_goal_currently_satisfied']==bool(o['decoded_state']['Closed(drawer)']==1 and o['decoded_state'][f'In({goal},drawer)']==1)
   cli.append({'variant':arm,'goal_object':goal,'query':'In(obj_pink,drawer)','output':o})
 save(E/'results/goal_binding_actual_queries.json',{'six_CLI_runs':cli,'changing_goal_did_not_change_probabilities':True,'goal_selected_correct_object_fact':True})
 print('BINDING_AUDIT_COMPLETE',json.dumps(answer),flush=True)
if __name__=='__main__':main()
