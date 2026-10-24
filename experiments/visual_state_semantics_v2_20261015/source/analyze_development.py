"""Frozen-model ABC evaluation and action-state sufficiency audit; no D and no training."""
from common import *
from model import QueryRelationReader,decode
from train import metrics
import torch,importlib.util,collections,gzip,copy

def main():
 assert not(E/'results/development_evaluation.json').exists();m=read(E/'data/metadata.json');t=dict(np.load(E/'data/targets.npz'));p=np.load(E/'results/probabilities.npy');raw=dict(np.load(E/'data/inputs.npz'));torch.set_num_threads(4);spec=importlib.util.spec_from_file_location('legacy_7_reader',CAND/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);old=mod.PredicateReader().eval().requires_grad_(False);old.load_state_dict(torch.load(CAND/'reader_seed1.pt',map_location='cpu',weights_only=False)['model'])
 with torch.inference_mode():old7=old.probabilities(**{k:torch.tensor(v)for k,v in raw.items()}).numpy()
 # The known flawed legacy Closed=!Open is evaluated explicitly, not used for new inference.
 old9=np.c_[old7[:,:6],1-old7[:,6],old7[:,6],np.zeros(len(m))];np.save(E/'results/legacy7_probabilities.npy',old7)
 spec=importlib.util.spec_from_file_location('previous_shared_query_definition',TRANSFER/'source/model.py');pm=importlib.util.module_from_spec(spec);spec.loader.exec_module(pm);previous=pm.QueryRelationReader().eval().requires_grad_(False);pcp=read(TRANSFER/'configs/reader_frozen.json');assert sha(pcp['path'])==pcp['sha256'];previous.load_state_dict(torch.load(pcp['path'],map_location='cpu',weights_only=False)['model'])
 with torch.inference_mode():prev7=previous(**{k:torch.tensor(v)for k,v in raw.items()}).sigmoid().numpy()
 np.save(E/'results/previous_query7_probabilities.npy',prev7);prev9=np.c_[prev7[:,:6],1-prev7[:,6],prev7[:,6],np.zeros(len(m))];models={'new_query9':p,'legacy7_with_notOpen_closed':old9,'previous_query7_with_notOpen_closed':prev9};scored={};errors=[]
 for name,pro in models.items():
  dd=decode(pro);correct=(dd==t['y'])&t['v'];scored[name]={}
  for split in ['train','dev']:
   for pool in ['new_semantics','legacy_holding_only']:
    ii=np.array([i for i,r in enumerate(m)if r['split']==split and r['pool']==pool]);out=metrics(pro,t['y'],t['v'],ii);out['independent_root_groups']=len({m[i]['root_id']for i in ii});out['root_all_valid_facts_correct']=sum(bool((((dd[ix]==t['y'][ix])|~t['v'][ix]).all()))for root in {m[i]['root_id']for i in ii}for ix in[np.array([i for i in ii if m[i]['root_id']==root])]);scored[name][split+':'+pool]=out
  for i,r in enumerate(m):
   if r['split']=='dev'and r['pool']=='new_semantics'and not((dd[i]==t['y'][i])|~t['v'][i]).all():errors.append({k:r[k]for k in['window_id','root_id','condition','roles','public_frames']}|{'model':name,'probabilities':pro[i].tolist(),'prediction':dd[i].tolist(),'truth':t['y'][i].tolist(),'valid':t['v'][i].tolist(),'error_columns':np.flatnonzero((dd[i]!=t['y'][i])&t['v'][i]).tolist()})
 newwindows=read(E/'data/windows.json');lookup={r['window_id']:i for i,r in enumerate(m)};counter=[]
 for r in newwindows:
  i=lookup[r['window_id']];lab=r['targets_v2_1'];pred=decode(p[i:i+1])[0]
  for j,c in enumerate(COLORS):
   d=lab['objects'][c];families=[]
   if d['outside']and d['load_bearing_paths']:families.append('outside_supported')
   if d['outside']and any(len(z)>2 for z in d['load_bearing_paths']):families.append('indirect_protruding')
   if d['boundary']:families.append('geometric_boundary')
   if d['held']and 'suspended_lowering'in r['roles']:families.append('suspended_held')
   for family in families:counter.append({'window_id':r['window_id'],'root_id':r['root_id'],'split':r['split'],'object':c,'family':family,'valid':bool(lab['v'][3+j]),'truth_In':bool(lab['y'][3+j]),'prediction_In':int(pred[3+j]),'false_positive':bool(pred[3+j]==1 and not lab['y'][3+j]and lab['v'][3+j])})
 counterstats={}
 for split in ['train','dev']:
  for family in ['outside_supported','indirect_protruding','geometric_boundary','suspended_held']:
   rows=[r for r in counter if r['split']==split and r['family']==family];counterstats[split+':'+family]={'object_windows':len(rows),'independent_roots':len({r['root_id']for r in rows}),'valid':sum(r['valid']for r in rows),'false_positive':sum(r['false_positive']for r in rows)}
 calls=read(E/'data/calls.json');capacity=[];ambiguous=[]
 for root in sorted({a['root_id']for a in calls}):
  aa=[a for a in calls if a['root_id']==root];stack=next((a for a in aa if a['condition']=='stacked'),None);separated=next((a for a in aa if a['condition']=='separated'),None)
  if not(stack and separated):continue
  evidence={}
  for tag,a in [('stacked',stack),('separated',separated)]:
   folder=B/'episodes'/a['episode_id']
   with gzip.open(folder/'_private/full_physics_trace.json.gz','rt')as f:trace=json.load(f)
   contacts=[]
   for row in trace[a['start']:a['primitive_end']+1]:
    snap=row['physical'];uids={o['uid']:name for name,o in snap['objects'].items()}
    for c in snap['contacts']:
     obj=None;link=None
     if c[1]in uids and c[2]==snap['table_uid']:obj=uids[c[1]];link=c[4]
     elif c[2]in uids and c[1]==snap['table_uid']:obj=uids[c[2]];link=c[3]
     if obj and link!=snap['drawer_link']and c[9]>.01:contacts.append({'frame':row['frame'],'object':obj,'fixed_link':link,'force_N':float(c[9]),'drawer_q':snap['drawer_q'],'penetration_m':max(0.,-float(c[8]))})
   # Tabletop third objects have intended base support; report all and isolate actual drawer objects.
   before=a['before_v2_1'];drawerobjs={f'block_{c}'for c,d in before['objects'].items()if d['load_bearing_paths']};impacts=[r for r in contacts if r['object']in drawerobjs];evidence[tag]={'episode_id':a['episode_id'],'native':a['native'],'before_y':before['y'],'before_v':before['v'],'after_y':a['after_v2_1']['y'],'after_v':a['after_v2_1']['v'],'end_q':a['after_v2_1']['drawer_q_m'],'strict_closed':a['after_v2_1']['y'][6]and a['after_v2_1']['v'][6],'drawer_objects_hit_other_fixed_links_samples':len(impacts),'peak_force_on_non_drawer_fixed_link_N':max((r['force_N']for r in impacts),default=0),'first_other_fixed_link_contact':impacts[0]if impacts else None,'peak_other_fixed_link_contact':max(impacts,key=lambda r:r['force_N'])if impacts else None}
  capacity.append({'root_id':root,'split':stack['split'],'matched_reset_conditions_same_objects_controller':True,'branches':evidence,'stack_obstructed_separated_closed':not evidence['stacked']['strict_closed']and evidence['separated']['strict_closed']})
 # Colour permutation-invariant symbolic abstraction for nonparameterized Close.
 groups=collections.defaultdict(list);exactgroups=collections.defaultdict(list)
 for a in calls:
  if a['family']!='close'or not all(a['before_v2_1']['v']):continue
  y=a['before_v2_1']['y'];signature=(sum(y[:3]),sum(y[3:6]),*map(int,y[6:]));outcome='Closed'if a['after_v2_1']['y'][6]and a['after_v2_1']['v'][6]else a['after_v2_1']['drawer_class'];groups[(a['split'],signature)].append({'episode_id':a['episode_id'],'root_id':a['root_id'],'condition':a['condition'],'native_status':a['native']['status'],'outcome':outcome,'end_q':a['after_v2_1']['drawer_q_m']});exactgroups[(a['split'],tuple(map(int,y)))].append(groups[(a['split'],signature)][-1])
 for (split,signature),rows in groups.items():
  if len({r['outcome']for r in rows})>1:ambiguous.append({'split':split,'signature_Holding_count_In_count_Closed_Open_Intermediate':signature,'outcome_counts':dict(collections.Counter(r['outcome']for r in rows)),'examples':rows})
 c=scored['new_query9']['dev:new_semantics'];h=scored['new_query9']['dev:legacy_holding_only'];fpclosed=c['facts']['Closed(drawer)']['FP'];fpin=sum(r['false_positive']for r in counter if r['split']=='dev'and r['family']in ['outside_supported','indirect_protruding','suspended_held']);coverage=all(counterstats['dev:'+family]['object_windows']>0 for family in ['outside_supported','indirect_protruding','geometric_boundary','suspended_held']);classcoverage=all(z['positive']>0 and z['negative']>0 for z in c['facts'].values());rate=c['all9_correct_fully_valid']/max(1,c['fully_valid9']);gates={'new_C_all9_at_least90percent':rate>=.9,'Closed_false_positive_zero':fpclosed==0,'In_counterexample_false_positive_zero':fpin==0,'all_counterexample_families_actual_C_evidence':coverage,'all9_C_classes_positive_and_negative':classcoverage,'legacy_C_Holding_at_least83of84':h['Holding_all3_correct']>=83,'deterministic_Close_state_sufficient':not any(r['split']=='train'for r in ambiguous)};decision=all(gates.values())
 exact_ambiguous=[{'split':split,'exact9':state,'outcome_counts':dict(collections.Counter(r['outcome']for r in rows)),'examples':rows}for(split,state),rows in exactgroups.items()if len({r['outcome']for r in rows})>1]
 save(E/'results/development_evaluation.json',{'metrics':scored,'new_C_full9_rate':rate,'gate_checks':gates,'new_independent_acceptance_permitted_by_frozen_gate':decision,'new_D_runs':0,'counterexample_statistics':counterstats,'limits':'Metrics exclude label-uncertain facts using declared masks; full9 denominator and invalid coverage disclosed. C is development, not independent acceptance. Old Closed=!Open baseline deliberately reproduces prior incorrect semantics.'});save(E/'results/development_errors.json',errors);save(E/'results/counterexample_details.json',counter);save(E/'results/placement_capacity_audit.json',capacity);save(E/'results/close_state_aliasing.json',{'ambiguous_groups':ambiguous,'exact9_ambiguous_groups':exact_ambiguous,'feature_signature':'Holding count, fully-contained In count, Closed/Open/Intermediate; identities permuted, no colour shortcut','conclusion':'If identical symbolic states permit both Closed and Intermediate outcomes, deterministic failure preconditions cannot be exactly learned from these predicates. Do not insert a manual effect or pretend failing Close succeeded. A future obstruction/protrusion relation or controllable placement interface would require separately justified scope.'});print(json.dumps({'gate_checks':gates,'C_all9_rate':rate,'strict_close_matched_cases':sum(r['stack_obstructed_separated_closed']for r in capacity),'paired_root_groups':len(capacity),'acceptance_permitted':decision},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
