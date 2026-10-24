from common import *
import csv,collections
import labels_physical as truth

def force(snap,robot,families,cut):
 ans=[]
 for c in COLORS:
  uid=snap['objects']['block_'+c]['uid'];links=set();external=False
  for x in snap['contacts']:
   if x[9]<=cut:continue
   if x[1]==uid:other,link=x[2],x[4]
   elif x[2]==uid:other,link=x[1],x[3]
   else:continue
   if other==robot:links.add(link)
   else:external=True
  ans.append(bool(all(links&set(f)for f in families)and not external))
 return ans
def count(rows):
 eligible=[r for r in rows if r['eligible']];pos=[r for r in eligible if r['y']];neg=[r for r in eligible if not r['y']]
 return {'queries':len(rows),'eligible_queries':len(eligible),'positive_queries':len(pos),'negative_queries':len(neg),'positive_roots':len({r['root_id']for r in pos}),'negative_roots':len({r['root_id']for r in neg}),'physical_unknown':sum(not r['physical_valid']for r in rows),'unobservable':sum(not r['visually_eligible']for r in rows)}
def csvout(name,rows):
 with(R/name).open('w')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 install_guard([]);protocol=read(E/'configs/protocol.json');rows=read(E/'data/records.json');checks=read(E/'results/integrity.json');roots=read(E/'results/root_summaries.json');registry=read(E/'configs/historical_exclusion.json');assert len(roots)==8 and len(checks)<=64 and set(r['split']for r in rows)=={'engineering_pilot'};assert len({r['record_id']for r in rows})==len(rows)
 for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items():assert sha(p)==h,p
 verify=collections.Counter();rgbids=collections.defaultdict(list);oldrgb=set(registry['old_RGB_array_hashes']);overlap=[]
 for ck in checks:
  d=E/'data/private'/ck['root_id']/f's{ck["sample"]:02d}';m=read(d.parent/'body_link_mapping.json');s=read(d/'physical_truth.json');lab=truth.evaluate(s['physical']);assert lab==s['labels_v2_1'];verify['current_truth_recomputations']+=1;assert read(d/'snapshot_before.json')==read(d/'snapshot_after.json')
  for h in s['last12_controls']:
   for j,cut in enumerate([.005,.01,.015]):assert force(h['physical'],m['robot_uid'],m['finger_families'],cut)==h['holding_at_cutoffs'][j];verify['stability_force_checks']+=1
  valid=[all(all(h['holding_at_cutoffs'][f][j]==lab['y'][j]for f in range(3))and all(sum(z)<=1 for z in h['holding_at_cutoffs'])for h in s['last12_controls'])for j in range(3)];assert valid==s['holding_stability_valid']
  with np.load(ck['public_path'])as pub:
   for cam in ['static','gripper']:
    with np.load(d/f'native_{cam}.npz')as z:
     assert np.array_equal(pub['rgb_'+cam],z['rgb']);seg=z['object_link_mask'];body=seg&((1<<24)-1);link=(seg>>24)-1;fm=[(seg>=0)&(body==m['robot_uid'])&np.isin(link,f)for f in m['finger_families']];fc=[int(f.sum())for f in fm];gb=bbox(fm[0]|fm[1]);rh=hashlib.sha256(z['rgb'].tobytes()).hexdigest();imageid=f'{ck["root_id"]}_s{ck["sample"]:02d}_{cam}';rgbids[rh].append(imageid)
     if rh in oldrgb:overlap.append(imageid)
     for r in [r for r in rows if r['root_id']==ck['root_id']and r['sample']==ck['sample']and r['camera']==cam]:
      j=COLORS.index(r['color']);bm=(seg>=0)&(body==m['object_uids'][r['color']]);assert bbox(bm)==r['object_bbox']and gb==r['gripper_bbox']and fc==r['finger_pixels']and int(bm.sum())==r['object_pixels'];assert r['y']==lab['y'][j]and r['physical_valid']==valid[j];assert r['eligible']==(valid[j]and int(bm.sum())>=4 and min(fc)>=4);verify['query_boxes']+=1
     verify['images']+=1
  verify['states']+=1
 rootrows=[];tracestats=[];initials=[]
 for spec,summary in zip(protocol['roots'],roots):
  rid=spec['root_id'];d=E/'data/private'/rid;trace=[json.loads(l)for l in(d/'control_trace_private.jsonl').read_text().splitlines()];assert len(trace)==summary['controls'];assert [t['control']for t in trace]==list(range(1,len(trace)+1));m=read(d/'body_link_mapping.json');initial=read(d/'s00/physical_truth.json')['physical'];baseline=next(t for t in reversed(trace)if t['stage']=='initial_settle');assert baseline['physical']==initial;lo,hi=np.array(spec['native_table_surface']);initial_nonbase=[]
  for c in COLORS:
   obj=initial['objects']['block_'+c];contacts=[]
   for x in initial['contacts']:
    if x[9]<=.01:continue
    if x[1]==obj['uid']and x[2]==initial['table_uid']:link=x[4]
    elif x[2]==obj['uid']and x[1]==initial['table_uid']:link=x[3]
    else:continue
    if link>=0:contacts.append({'link':link,'force_N':x[9],'gap_m':x[8]})
   initial_nonbase+=contacts;position=np.array(obj['position']);requested=np.array(spec['blocks'][c]['initial_pos']);initials.append({'root_id':rid,'scene':spec['scene'],'color':c,'is_target':c==spec['target_color'],'requested_x_m':requested[0],'requested_y_m':requested[1],'requested_inside_native_surface':bool(np.all((requested[:2]>=lo[:2])&(requested[:2]<=hi[:2]))),'actual_x_m':position[0],'actual_y_m':position[1],'actual_inside_native_surface':bool(np.all((position[:2]>=lo[:2])&(position[:2]<=hi[:2]))),'initial_displacement_m':float(np.linalg.norm(position-requested)),'speed_m_s':float(np.linalg.norm(obj['linear_velocity'])),'non_base_table_contact_count':len(contacts),'max_non_base_force_N':max((x['force_N']for x in contacts),default=0)})
  bystage={}
  for stage in ['initial_settle','frozen_grasp','post_grasp_anchors']:
   tt=[t for t in trace if t['stage']==stage];maxdrift=0.;maxdevice=0.;penetration=0.;peakforce=0.;unrequested_contacts=0
   for t in tt:
    snap=t['physical'];computed=force(snap,m['robot_uid'],m['finger_families'],.01);assert computed==[('block_'+c)in snap['held']for c in COLORS];verify['all_control_Holding_checks']+=1
    maxdrift=max(maxdrift,max(float(np.linalg.norm(np.array(o['position'])-initial['objects'][name]['position']))for name,o in snap['objects'].items()if name!='block_'+spec['target_color']));maxdevice=max(maxdevice,float(np.max(np.abs(np.array(t['scene_obs'])[[0,2,3,4,5]]-np.array(baseline['scene_obs'])[[0,2,3,4,5]]))))
    for x in snap['contacts']:
     if x[9]<=.01 or m['robot_uid']not in [x[1],x[2]]:continue
     penetration=max(penetration,-x[8]);peakforce=max(peakforce,x[9]);other=x[2]if x[1]==m['robot_uid']else x[1]
     if other!=m['object_uids'][spec['target_color']]:unrequested_contacts+=1
   bystage[stage]={'controls':len(tt),'max_unrequested_object_displacement_m':maxdrift,'max_non_drawer_scene_change':maxdevice,'max_robot_penetration_m':penetration,'max_robot_contact_force_N':peakforce,'non_target_robot_contact_samples':unrequested_contacts};tracestats.append({'root_id':rid,'stage':stage,**bystage[stage]})
  run=read(d/'frozen_controller_result.json');ss=bystage['frozen_grasp'];safe=run['status']=='success'and ss['max_unrequested_object_displacement_m']<.01 and ss['max_non_drawer_scene_change']<=.003 and ss['max_robot_penetration_m']<.01;rr=[r for r in rows if r['root_id']==rid];target=[r for r in rr if r['color']==spec['target_color']]
  rootrows.append({'root_id':rid,'scene':spec['scene'],'split':'engineering_pilot','target_color':spec['target_color'],'saved_states':summary['states'],'controller_status':run['status'],'reason':run['reason'],'valid_candidates':sum(x['valid']for x in run['planning_private']),'safety_checked_grasp_success':safe,'official_task_acceptance_claimed':False,'initial_non_base_table_contacts':len(initial_nonbase),'positive_queries_all_colors':sum(r['eligible']and r['y']for r in rr),'target_positive_queries':sum(r['eligible']and r['y']for r in target),'negative_queries_all_colors':sum(r['eligible']and not r['y']for r in rr),'unknown_queries':sum(not r['eligible']for r in rr),'missing_anchors':','.join(summary['missing_anchors'])})
 safeids={r['root_id']for r in rootrows if r['safety_checked_grasp_success']};coverage=[]
 for scope,subset in [('all_actual_outcomes',rows),('safety_checked_successful_grasp_roots',[r for r in rows if r['root_id']in safeids])]:
  for kind,names in [('all',['all']),('color',COLORS),('camera',['static','gripper']),('root',[r['root_id']for r in roots])]:
   for name in names:
    rr=subset if kind=='all'else[r for r in subset if r[{'color':'color','camera':'camera','root':'root_id'}[kind]]==name];coverage.append({'scope':scope,'group_type':kind,'group':name,**count(rr)})
 domain_ok=all(x['requested_inside_native_surface']and x['actual_inside_native_surface']and x['non_base_table_contact_count']==0 for x in initials)
 colors_ok=all(next(x for x in coverage if x['scope']=='safety_checked_successful_grasp_roots'and x['group_type']=='color'and x['group']==c)['positive_roots']>=1 for c in COLORS)
 positive_rows=[r for r in rows if r['root_id']in safeids and r['physical_valid']and r['y']];positive_both=bool(positive_rows)and all(r['visually_eligible']for r in positive_rows);negative_ok=all(next(x for x in coverage if x['scope']=='safety_checked_successful_grasp_roots'and x['group_type']=='color'and x['group']==c)['negative_queries']>0 for c in COLORS)
 decision={'domain_correction_passed':domain_ok,'all_colors_have_safety_checked_positive_root':colors_ok,'all_positive_endpoints_visible_both_cameras':positive_both,'stable_negative_each_color':negative_ok,'integrity_passed':True,'old_RGB_overlap':overlap,'next_independent_collection_design_justified':domain_ok and colors_ok and positive_both and negative_ok and not overlap,'new_collection_authorized_or_started':False,'formal_model_evaluation':False,'original_failed_gate_unchanged':True};assert not overlap
 out={'utc':now(),'verification':dict(verify),'roots':rootrows,'initial_state_audit':initials,'control_trace_statistics':tracestats,'coverage':coverage,'decision':decision,'within_pilot_duplicate_RGB':[{'sha256':h,'images':ids}for h,ids in rgbids.items()if len(ids)>1],'physical_unknown_reasons':dict(collections.Counter(x for r in rows for x in r['unknown_reasons']))};save(E/'results/pilot_audit.json',out);save(R/'pilot_audit.json',out)
 for name,rs in [('pilot_root_outcomes.csv',rootrows),('pilot_initial_states.csv',initials),('pilot_control_safety.csv',tracestats),('pilot_Holding_coverage.csv',coverage)]:csvout(name,rs)
 save(E/'configs/data_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in(E/'data').rglob('*')if p.is_file()},'engineering_only':True});print('PILOT_AUDIT',json.dumps(decision),flush=True)
 for r in rootrows:print(r,flush=True)
 for r in coverage:
  if r['group_type']=='color':print(r,flush=True)
if __name__=='__main__':main()
