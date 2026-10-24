from common import *
import sys,gzip,collections
sys.path.insert(0,str(RGB/'source'))
from registry import extract
NAMES=['left','right','back','front','top']
def stats(v):
 a=np.array(v,float);return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}if len(a)else{'n':0}
def main():
 wins=read(BASE/'data/windows.json');pred=np.load(BASE/'results/probabilities.npy');allrows=[]
 for i,w in enumerate(wins):
  z=np.load(w['public_frames'][-1]);vis={cam:extract(z['rgb_'+cam])[1]for cam in ['static','gripper']}
  for j,c in enumerate(COLORS):
   g=w['targets_v2_1']['objects'][c];clear=np.array(g['side_rim_clearances_m']);row={k:w[k]for k in ['window_id','root_id','episode_id','condition','split','anchor','public_frames']};row.update({'object':c,'valid_In':bool(w['targets_v2_1']['v'][3+j]),'truth_In':bool(w['targets_v2_1']['y'][3+j]),'initial_probability_In':float(pred[i,3+j]),'clearance_mm':dict(zip(NAMES,(clear*1000).tolist())),'maximum_protrusion_mm':max(0.,-float(clear.min())*1000),'worst_boundary':NAMES[int(clear.argmin())],'above_rim_mm':max(0.,-float(clear[4])*1000),'floor_penetration_mm':max(0.,-g['floor_clearance_m']*1000),'has_support':bool(g['load_bearing_paths']),'indirect_support':any(len(p)>2 for p in g['load_bearing_paths']),'held':g['held'],'outside_supported':g['outside']and bool(g['load_bearing_paths']),'current_RGB_visibility':{cam:{'accepted_pixels':vis[cam][j]['accepted_pixels'],'visible':vis[cam][j]['visible']}for cam in vis}});allrows.append(row)
  if (i+1)%100==0:print('AUDITED_WINDOWS',i+1,flush=True)
 counter=[r for r in allrows if r['split']=='dev'and r['outside_supported']and r['valid_In']and r['initial_probability_In']>=.5];assert len(counter)==41
 bins=[0,2,5,10,20,50,100];by={}
 for key in sorted({r['condition']for r in counter}):
  rows=[r for r in counter if r['condition']==key];by[key]={'object_windows':len(rows),'roots':len({r['root_id']for r in rows}),'protrusion_mm':stats([r['maximum_protrusion_mm']for r in rows]),'above_rim_mm':stats([r['above_rim_mm']for r in rows]),'worst_boundary_counts':dict(collections.Counter(r['worst_boundary']for r in rows))}
 overview={'C_false_positive_counterexamples':41,'independent_roots':len({r['root_id']for r in counter}),'protrusion_mm':stats([r['maximum_protrusion_mm']for r in counter]),'bins_mm':bins,'histogram':np.histogram([r['maximum_protrusion_mm']for r in counter],bins)[0].tolist(),'at_most_tolerance_mm':{str(t):sum(r['maximum_protrusion_mm']<=t for r in counter)for t in [2,5,10,20]},'by_condition':by,'current_visibility':{cam:{'pixels':stats([r['current_RGB_visibility'][cam]['accepted_pixels']for r in counter]),'visible':sum(r['current_RGB_visibility'][cam]['visible']for r in counter)}for cam in ['static','gripper']},'at_least_one_camera_visible':sum(any(v['visible']for v in r['current_RGB_visibility'].values())for r in counter),'limits':'Colour visibility is not visibility of all corners or calibrated metric depth. Tolerance bins are sensitivity analysis, not changed labels or success criteria.'}
 save(E/'results/geometry_visibility.json',overview);save(E/'data/object_window_audit.json',allrows);save(E/'results/C_41_error_details.json',counter)
 # All actual Close calls, with outcome-independent preaction geometry and real contact events.
 calls=[dict(a,origin='prior',data_base=str(BASE_B))for a in read(BASE/'data/calls.json')if a['family']=='close']
 ext=read(B/'manifest.json');assert len(ext['episodes'])==9 and all(r['error']is None for r in ext['episodes'])
 for e in ext['episodes']:
  for a in read(B/'episodes'/e['episode_id']/'_private/actions.json'):
   calls.append(a|{k:e[k]for k in ['root_id','episode_id','split','condition','environment']}|{'before_v2_1':a['before'],'after_v2_1':a['after'],'origin':'extension','data_base':str(B)})
 rows=[]
 for a in calls:
  before=a['before_v2_1'];after=a['after_v2_1'];associated=[c for c,g in before['objects'].items()if g['load_bearing_paths']];obj=[]
  for c in associated:
   b=before['objects'][c];f=after['objects'][c];obj.append({'object':c,'before_above_rim_mm':max(0.,-b['side_rim_clearances_m'][4]*1000),'before_front_protrusion_mm':max(0.,-b['side_rim_clearances_m'][3]*1000),'before_max_protrusion_mm':max(0.,-b['minimum_side_rim_clearance_m']*1000),'before_local_centroid':np.mean(b['drawer_local_corners'],0).tolist(),'after_local_centroid':np.mean(f['drawer_local_corners'],0).tolist(),'before_In':before['y'][3+COLORS.index(c)],'after_In':after['y'][3+COLORS.index(c)],'after_In_valid':after['v'][3+COLORS.index(c)],'after_above_rim_mm':max(0.,-f['side_rim_clearances_m'][4]*1000),'after_support':bool(f['load_bearing_paths']),'world_displacement_mm':float(np.linalg.norm(np.mean(f['world_corners'],0)-np.mean(b['world_corners'],0))*1000)})
  folder=Path(a['data_base'])/'episodes'/a['episode_id']
  with gzip.open(folder/'_private/full_physics_trace.json.gz','rt')as f:trace=json.load(f)
  impacts=[];ztrack=[]
  for rr in trace[a['start']:a['primitive_end']+1]:
   snap=rr['physical'];uid={snap['objects']['block_'+c]['uid']:c for c in associated}
   for co in snap['contacts']:
    c=None;link=None
    if co[1]in uid and co[2]==snap['table_uid']:c=uid[co[1]];link=co[4]
    elif co[2]in uid and co[1]==snap['table_uid']:c=uid[co[2]];link=co[3]
    if c and link!=snap['drawer_link']and co[9]>.01:impacts.append({'frame':rr['frame'],'object':c,'fixed_link':link,'force_N':float(co[9]),'penetration_mm':max(0.,-co[8]*1000),'drawer_q_cm':snap['drawer_q']*100,'world_contact_on_object':co[5]if co[1]in uid else co[6]})
   if associated:ztrack.append({'frame':rr['frame'],'q_cm':snap['drawer_q']*100,'max_above_rim_mm':max(max(0.,-rr['new_labels']['objects'][c]['side_rim_clearances_m'][4]*1000)for c in associated)})
  mx=max((r['before_above_rim_mm']for r in obj),default=0.);y=list(map(int,before['y']));closed=bool(after['y'][6]and after['v'][6]);row={k:a[k]for k in ['root_id','episode_id','split','condition','origin','start','primitive_end','end','native','data_base']};row.update({'before_exact9':y,'associated_objects':associated,'object_geometry':obj,'max_above_rim_mm':mx,'AboveRim_relation':mx>2,'end_q_cm':after['drawer_q_m']*100,'closed':closed,'all_associated_contained_after':all(r['after_In']and r['after_In_valid']for r in obj),'all_associated_supported_after':all(r['after_support']for r in obj),'closed_and_all_initial_drawer_objects_contained':closed and all(r['after_In']and r['after_In_valid']for r in obj),'first_other_fixed_link_contact':impacts[0]if impacts else None,'peak_contact':max(impacts,key=lambda x:x['force_N'])if impacts else None,'contact_samples':len(impacts),'trajectory':ztrack});rows.append(row)
 allgroups=collections.defaultdict(list)
 for r in rows:allgroups[(r['split'],tuple(r['before_exact9']),r['AboveRim_relation'])].append(r)
 conflicts=[{'split':sp,'exact9':y,'AboveRim':above,'outcome_counts':dict(collections.Counter('Closed'if r['closed']else'Mid'for r in rr)),'episodes':[r['episode_id']for r in rr]}for(sp,y,above),rr in allgroups.items()if len({r['closed']for r in rr})>1]
 relstats={}
 for split in ['train','dev']:
  for above in [False,True]:
   rr=[r for r in rows if r['split']==split and r['AboveRim_relation']==above];relstats[split+':AboveRim='+str(above)]={'calls':len(rr),'closed':sum(r['closed']for r in rr),'not_closed':sum(not r['closed']for r in rr),'closed_and_contained':sum(r['closed_and_all_initial_drawer_objects_contained']for r in rr)}
 negcoverage=collections.defaultdict(collections.Counter)
 for a in read(BASE/'data/calls.json'):negcoverage[a['split']+':'+a['family']][a['native']['status']]+=1
 save(E/'results/close_calls.json',rows);save(E/'results/state_sufficiency.json',{'actual_Close_calls':len(rows),'old_Close_calls':len(rows)-9,'new_Close_calls':9,'extra_relation':'GT AboveRim of any drawer-supported object >2mm at action start, diagnostic only','relation_outcomes':relstats,'same_exact9_and_relation_still_conflicting':conflicts,'negative_coverage_old_AB_C':dict(negcoverage),'no_new_deployable_rule':True,'source_guarded_physics':True});print(json.dumps({'geometry':overview,'relation_outcomes':relstats,'remaining_conflicts':len(conflicts)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
