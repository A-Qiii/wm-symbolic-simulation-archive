from common import *
import collections
sys.addaudithook(guard)
roots={r['root_id']:r for r in read(S/'data/roots_manifest.json')};rows=read(S/'data/data_manifest.json')['records'];assert len(rows)==720;groups=collections.defaultdict(list);failures=[]
for r in rows:
 b=int(r['episode_id'][-1]);assert r['skill']==roots[r['root_id']]['branches'][b][r['step']];res=read(Path(r['labels_path']).parent/'result_private.json');lab=read(r['labels_path']);actions=read(r['low_level_log']);assert len(actions)<=900 and lab['control_samples_including_start']==len(actions)+1 and lab['success']==res['goal_success'];groups[r['split']].append((res,lab,len(actions)))
 if not res['goal_success']:failures.append({'root_id':r['root_id'],'episode_id':r['episode_id'],'step':r['step'],'skill':r['skill'],'status':res['status'],'reason':res.get('reason'),'controls':len(actions)})
summary={}
for split,items in groups.items():summary[split]={'transitions':len(items),'goal_success':sum(r['goal_success']for r,l,a in items),'status_counts':dict(collections.Counter(r['status']for r,l,a in items)),'zero_control_calls':sum(a==0 for r,l,a in items),'unrequested_device_excursion_gt3mm':sum(r['strict_device_side_effect']for r,l,a in items),'invalid_after_fields_first4':sum(sum(not x for x in l['after']['categorical_valid'][:4])for r,l,a in items),'controls':sum(a for r,l,a in items)}
bytes_used=sum(p.stat().st_size for p in S.rglob('*')if p.is_file());save(S/'results/collection_audit.json',{'utc':now(),'passed':True,'new_roots':60,'new_transitions':720,'split_summary':summary,'failure_records':failures,'bytes_current':bytes_used,'first4_seconds':read(S/'results/collection_train_4_complete.json')['seconds'],'remaining_train_seconds':read(S/'results/collection_train_0_complete.json')['seconds'],'evaluation_seconds':read(S/'results/collection_evaluation_0_complete.json')['seconds'],'h0_roots':len(read(S/'results/h0_audit.json')),'no_post_acceptance_replacement':True})
print(summary)
