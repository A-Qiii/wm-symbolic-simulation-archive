"""Read existing AB snapshots. Do not collect, relabel, or train."""
from common import *
from labels_v2_1 import evaluate
import gzip,collections

def upward_supports(snap,color):
 uid=snap['objects']['block_'+color]['uid'];out=[]
 for c in snap['contacts']:
  if c[1]==uid:other,link,n=c[2],c[4],c[7]
  elif c[2]==uid:other,link,n=c[1],c[3],[-x for x in c[7]]
  else:continue
  if float(c[9])>.01 and float(n[2])>.5 and float(c[8])<=.003:
   out.append({'body':other,'link':link,'force_N':c[9],'table_base':other==snap['table_uid']and link==-1,'drawer':other==snap['table_uid']and link==snap['drawer_link']})
 return out

def row_for(w,snap,root):
 lab=evaluate(snap);stored=w['targets_v2_1'];assert lab['y']==stored['y']and lab['v']==stored['v'],w['window_id']
 rows=[]
 for j,color in enumerate(COLORS):
  g=lab['objects'][color];bottom=min(x[2]for x in g['world_corners']);top=max(x[2]for x in g['world_corners']);held=g['held'];up=upward_supports(snap,color);vel=snap['objects']['block_'+color]['linear_velocity'];closed=bool(lab['v'][6]and lab['y'][6]);valid=bool(lab['v'][3+j]);truth=bool(lab['y'][3+j]);table=any(x['table_base']for x in up)and abs(bottom-.44)<=.005 and not held
  categories={'tabletop_closed':table and closed,'contained_closed':closed and valid and truth,'falling_now':not held and not up and vel[2]<-.02,'fallen_below_table':not held and not g['load_bearing_paths']and top<.30,'protruding_supported':bool(g['outside']and g['load_bearing_paths']),'near_boundary_supported':bool(g['load_bearing_paths']and abs(g['minimum_side_rim_clearance_m'])<.005)}
  rows.append({k:w[k]for k in ['window_id','root_id','episode_id','environment','condition','anchor']}|{'object':color,'intended_object':root['color'],'is_intended':color==root['color'],'In_valid':valid,'In_truth':truth,'Closed_valid':bool(lab['v'][6]),'Closed_truth':bool(lab['y'][6]),'joint_valid':valid and bool(lab['v'][6]),'joint_truth':truth and bool(lab['y'][6]),'categories':categories,'bottom_z_m':bottom,'top_z_m':top,'vertical_velocity_m_s':vel[2],'minimum_clearance_m':g['minimum_side_rim_clearance_m'],'support':up,'load_bearing_paths':g['load_bearing_paths'],'root_seed':root['seed']})
 return rows

def summarise(rr,exposure):
 vv=[r for r in rr if r['In_valid']];pos=[r for r in vv if r['In_truth']];neg=[r for r in vv if not r['In_truth']];jv=[r for r in rr if r['joint_valid']]
 return {'object_windows':len(rr),'unique_windows':len({r['window_id']for r in rr}),'independent_roots':len({r['root_id']for r in rr}),'root_ids':sorted({r['root_id']for r in rr}),'episodes':len({r['episode_id']for r in rr}),'In_positive':len(pos),'In_negative':len(neg),'In_invalid':len(rr)-len(vv),'positive_fraction_valid':len(pos)/len(vv)if vv else None,'roots_with_positive':len({r['root_id']for r in pos}),'roots_with_negative':len({r['root_id']for r in neg}),'joint_positive':sum(r['joint_truth']for r in jv),'joint_negative':sum(not r['joint_truth']for r in jv),'sampler_object_exposures':sum(exposure[r['window_id']]for r in rr),'sampler_unique_windows_seen':sum(exposure[w]>0 for w in {r['window_id']for r in rr}),'sampled_roots':len({r['root_id']for r in rr if exposure[r['window_id']]>0}),'conditions':dict(collections.Counter(r['condition']for r in rr))}

def main():
 roots={r['root_id']:r for r in read(B/'roots_manifest.json')};windows=[w for w in read(BASE/'data/windows.json')if w['environment']in ['A','B']];assert len(windows)==420
 grouped=collections.defaultdict(list)
 for w in windows:grouped[w['episode_id']].append(w)
 out=[];inputhash={};physics_hash=[]
 for n,(eid,ws)in enumerate(grouped.items()):
  path=B/'episodes'/eid/'_private/full_physics_trace.json.gz';physics_hash.append({'path':str(path),'sha256':sha(path)})
  with gzip.open(path,'rt')as f:trace=json.load(f)
  for w in ws:out.extend(row_for(w,trace[w['anchor']]['physical'],roots[w['root_id']]))
  if(n+1)%20==0:print('AB_EPISODES_AUDITED',n+1,flush=True)
 meta=read(BASE/'data/metadata.json');schedule=np.load(SP/'data/train_schedule.npy');exposure=collections.Counter(meta[int(i)]['window_id']for i in schedule.ravel());summary={}
 cats=list(out[0]['categories'])
 for scope in ['intended_target','all_object_queries']:
  rr=[r for r in out if scope=='all_object_queries'or r['is_intended']];ss={}
  for category in ['all']+cats:
   chosen=rr if category=='all'else[r for r in rr if r['categories'][category]]
   ss[category]={'all_colors':summarise(chosen,exposure),'by_color':{c:summarise([r for r in chosen if r['object']==c],exposure)for c in COLORS}}
  # Inspect closed positives/negatives without selecting by In truth.
  closed=[r for r in rr if r['Closed_valid']and r['Closed_truth']];ss['all_closed_states']={'all_colors':summarise(closed,exposure),'by_color':{c:summarise([r for r in closed if r['object']==c],exposure)for c in COLORS}}
  summary[scope]=ss
 abroots=[r for r in roots.values()if r['environment']in ['A','B']];layouts=[hashlib.sha256(json.dumps(r['layout'],sort_keys=True).encode()).hexdigest()for r in abroots];assert len(set(layouts))==18 and len({r['seed']for r in abroots})==18
 table_roots={r['root_id']for r in out if r['categories']['tabletop_closed']};inside_roots={r['root_id']for r in out if r['categories']['contained_closed']};positions=[]
 for r in abroots:
  for c in COLORS:positions.append({'root_id':r['root_id'],'environment':r['environment'],'color':c,'intended':c==r['color'],'x':r['layout']['block_'+c][0],'yaw':r['layout']['block_'+c][5]})
 existing_errors=[]
 for w in read(BASE/'data/windows.json'):
  if w['root_id']=='C_semantics_v2_fixtures_pink_0'and w['condition']=='empty_closed':
   path=B/'episodes'/w['episode_id']/'_private/full_physics_trace.json.gz'
   with gzip.open(path,'rt')as f:trace=json.load(f)
   existing_errors.extend(row_for(w,trace[w['anchor']]['physical'],roots[w['root_id']]))
 save(E/'data/_private/AB_object_state_audit.json',out);save(E/'data/_private/existing_C_joint_error_audit.json',existing_errors);save(E/'results/coverage.json',{'source_AB_windows':420,'AB_roots':18,'AB_unique_root_layouts':18,'AB_unique_seeds':18,'coverage':summary,'closed_table_and_inside_root_overlap':sorted(table_roots&inside_roots),'AB_tabletop_layout_positions':positions,'source_label_recomputed_windows':420,'source_label_y_or_valid_mismatches':0,'existing_C_error_recomputed_windows':2,'limitations':['Root IDs are grouped by initial layout/seed, not adjacent frames or condition reset branches.','Positive and negative root counts overlap when a root contains both states.','These categories are evaluator-only, never fed to model.','falling_now is a current downward unheld unsupported state; fallen_below_table is a separate settled or falling state, not a live trajectory count.','No new observations, training, or C initial-state copying.']});save(E/'configs/physical_sources.json',physics_hash)
 print('AB_COVERAGE_AUDIT_COMPLETE',json.dumps({k:v['all_colors']for k,v in summary['intended_target'].items()}),flush=True)
if __name__=='__main__':main()
