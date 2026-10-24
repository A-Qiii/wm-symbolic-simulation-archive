"""D execution-only evidence, separated from learned prediction and planning."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1';B=P/'data/core1200_v1';COLORS=['red','blue','pink']
def load(p):return json.loads(p.read_text())
def goal(skill,before,after):
 if skill=='open_drawer':return after['categorical'][3]==2
 if skill=='close_drawer':return after['categorical'][3]==0
 if skill.startswith('lift_'):
  j=COLORS.index(skill.split('_')[1]);return after['categorical_valid'][j] and after['categorical'][j]==6
 held=[j for j,x in enumerate(before['categorical'][:3]) if x==6 and before['categorical_valid'][j]]
 return len(held)==1 and after['categorical_valid'][held[0]] and after['categorical'][held[0]]==3 and after['hand_empty_derived'] is True
def measure(d,skill,before,after,result,initial_scene=None):
 with gzip.open(d/'trace_private.json.gz','rt') as f:trace=json.load(f)
 held=[COLORS[j] for j,x in enumerate(before['categorical'][:3]) if x==6];target=skill.split('_')[1] if skill.startswith('lift_') else (held[0] if skill=='place_in_drawer' and len(held)==1 else None)
 allowed={target} if target else set()
 if skill in ['open_drawer','close_drawer']:allowed|={COLORS[j] for j,x in enumerate(before['categorical'][:3]) if x==3}
 displacement={c:0. for c in COLORS};device=np.zeros(6)
 if trace:
  states=np.asarray([x['scene_obs'][:6] for x in trace]);initial=initial_scene if initial_scene is not None else [before['continuous']['slider_m'],before['continuous']['drawer_m'],0.,0.,*before['binary']]
  device=np.abs(states-np.asarray(initial)).max(0)
  for c in COLORS:
   start=np.array(before['continuous']['objects'][c]['position_m']);positions=np.array([x['objects']['block_'+c][:3] for x in trace]);displacement[c]=float(np.linalg.norm(positions-start,axis=1).max())
 return {'skill':skill,'controller_status':result['status'],'controls':result['controls'],'independent_endpoint_goal':bool(goal(skill,before,after)),'slider_peak_displacement_m':float(device[0]),'button_peak_displacement_m':float(device[2]),'switch_peak_displacement_rad':float(device[3]),'light_bits_changed':bool(device[4:].max()>0),'object_peak_displacements_m':displacement,'non_target_peak_displacement_m':max([v for c,v in displacement.items() if c not in allowed] or [0.]),'target':target,'actual_moved_over20mm':[c for c,v in displacement.items() if v>.02]}
def summarize(rows):
 statuses=Counter(x['controller_status'] for x in rows);executed=[x for x in rows if x['controls']>0];by={}
 for skill in sorted({x['skill'] for x in rows}):
  r=[x for x in rows if x['skill']==skill];e=[x for x in r if x['controls']>0];by[skill]={'requests':len(r),'status':dict(Counter(x['controller_status'] for x in r)),'executed':len(e),'actual_endpoint_goal_on_executed':sum(x['independent_endpoint_goal'] for x in e)}
 return {'requests':len(rows),'status':dict(statuses),'executed_requests':len(executed),'actual_endpoint_goal_on_executed':sum(x['independent_endpoint_goal'] for x in executed),'per_skill':by,'slider_over3mm_requests':sum(x['slider_peak_displacement_m']>.003 for x in rows),'slider_max_mm':1000*max([x['slider_peak_displacement_m'] for x in rows] or [0]),'button_over3mm_requests':sum(x['button_peak_displacement_m']>.003 for x in rows),'switch_over003rad_requests':sum(x['switch_peak_displacement_rad']>.003 for x in rows),'light_change_requests':sum(x['light_bits_changed'] for x in rows),'non_target_over10mm_requests':sum(x['non_target_peak_displacement_m']>.01 for x in rows),'non_target_max_mm':1000*max([x['non_target_peak_displacement_m'] for x in rows] or [0])}
def main():
 rows=[];root_setup={};m=load(B/'data_manifest.json')
 for row in m['records']:
  if row['split']!='test':continue
  d=Path(row['obs_before']).parent;x=load(Path(row['labels_path']));res=load(d/'result_private.json');info=load(d/'endpoint_info_private.json')['before']['scene_info'];initial=[info['doors']['base__slide']['current_state'],info['doors']['base__drawer']['current_state'],info['buttons']['base__button']['joint_state'],info['switches']['base__switch']['joint_state'],info['lights']['lightbulb']['logical_state'],info['lights']['led']['logical_state']]
  measured=measure(d,row['skill'],x['before'],x['after'],res,initial);measured.update(root_id=row['root_id'],episode_id=row['episode_id'],step=row['step'],official_success=res['official_success']);rows.append(measured)
  setup=load(d.parent/'setup_private.json');root_setup[row['root_id']]=all(r['status']=='success' for r in setup['results'])
 methods={};detail={}
 for method in ['learned','goal_only','reference','root_label_replacement']:
  rr=[]
  for f in sorted((R/'evaluation/closed_loop'/method).glob('*/execution/evaluator_only/*/summary.json')):
   skill=f.parent.name.split('_',1)[1]
   if skill=='reobserve':continue
   x=load(f);v=measure(f.parent,skill,x['before'],x['after'],x['result']);v['source']=str(f);rr.append(v)
  methods[method]=summarize(rr);detail[method]=rr
 report={'actual_environment_protocol':'Original guide A/B train, C dev, D test preserved in the new six-skill scope. Controller v1 previously validated only onABC; D is new execution-domain evidence, not guaranteed by prior validation.','controller_frozen':True,'controller_retrained_on_D':False,'D_test_roots_setup_success':[sum(root_setup.values()),len(root_setup)],'D_dataset':summarize(rows),'closed_loop_methods':methods,'raw_D_dataset_rows':rows,'raw_closed_loop_rows':detail,'definitions':{'controller_status':'Original unchanged status/official completion; deliberate refusal is separate from executed-skill failure.','actual_endpoint_goal':'Offline categorical target/ownership/held state after a nonzero-control request. In-drawer includes an upward support chain through another block; therefore this can succeed while the unchanged controller direct-contact stop reports failure. It is not a claim of universal control reliability.','side_effects':'Full actual control traces. Slider/button in metres, switch in radians, lights in bits; not combined as one fictitious unit. For opening/closing, existing drawer contents are allowed to move with the drawer. Other non-target object displacement is retained.','service_device_start':'Reference/service button and switch begin at frozen0; collection uses exact logged before values. Tiny numerical switch drift does not affect over0.003rad classification.','same_root_control':'Every closed-loop method checks its actual initial robot+scene state against the identical dataset branch0 root within1e-5; same900-control cap and v1 parameters.'}}
 (R/'evaluation/D_controller_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if not k.startswith('raw_')},indent=2))
if __name__=='__main__':main()
