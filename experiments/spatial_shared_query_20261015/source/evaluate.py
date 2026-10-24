from common import *
from model import decode
from train import metrics
import collections
def binary_stats(pred,y,valid):
 return {'valid':int(valid.sum()),'positive':int((y&valid).sum()),'negative':int((~y&valid).sum()),'TP':int((pred&y&valid).sum()),'TN':int((~pred&~y&valid).sum()),'FP':int((pred&~y&valid).sum()),'FN':int((~pred&y&valid).sum())}
def main():
 m=read(BASE/'data/metadata.json');t=dict(np.load(BASE/'data/targets.npz'));y=t['y'];v=t['v'];rootinfo={r['root_id']:r for r in read(B/'roots_manifest.json')};wins={r['window_id']:r for r in read(BASE/'data/windows.json')};new=np.array([r['pool']=='new_semantics'for r in m]);train=np.array([r['split']=='train'for r in m]);models={'original_temporal':np.load(BASE/'results/probabilities.npy')}|{a:np.load(E/'results'/a/'probabilities.npy')for a in ['pooled','spatial']};allmetrics={};errors=[];goals=[];decisions={};counter=read(BASE/'results/counterexample_details.json');bywin={r['window_id']:i for i,r in enumerate(m)}
 for name,p in models.items():
  d=decode(p);out={}
  for split,mask in [('AB',train),('C',~train)]:
   for pool,poolmask in [('new',new),('legacy_H',~new)]:
    ii=np.flatnonzero(mask&poolmask);z=metrics(p,y,v,ii);z['independent_roots']=len({m[i]['root_id']for i in ii});z['per_root']={}
    for root in sorted({m[i]['root_id']for i in ii}):
     jj=np.array([i for i in ii if m[i]['root_id']==root]);known=v[jj].all(1);ok=((d[jj]==y[jj])|~v[jj]);z['per_root'][root]={'windows':len(jj),'fully_valid9':int(known.sum()),'full9_correct':int(ok[known].all(1).sum()),'all_valid_facts_correct':bool(ok.all()),'valid_fact_errors':int((~ok).sum())}
    z['root_all_valid_facts_correct']=sum(r['all_valid_facts_correct']for r in z['per_root'].values())
    z['families']={}
    for family,cols in [('Holding',[0,1,2]),('In',[3,4,5]),('drawer3',[6,7,8]),('goal_related_In_and_Closed',[3,4,5,6])]:
     vv=v[ii][:,cols];ok=(d[ii][:,cols]==y[ii][:,cols]);full=vv.all(1);z['families'][family]={'valid_facts':int(vv.sum()),'correct_valid_facts':int((vv&ok).sum()),'eligible_windows':int(full.sum()),'exact_correct_windows':int(ok[full].all(1).sum())}
    if pool=='new':
     py=(d[ii,3:6]==1)&(d[ii,6:7]==1);gy=y[ii,3:6]&y[ii,6:7];gv=v[ii,3:6]&v[ii,6:7];z['joint_goal_all_objects']=binary_stats(py,gy,gv);full=gv.all(1);z['joint_goal_all_objects'].update({'fully_valid_three_goal_windows':int(full.sum()),'all_three_goals_exact':int((py==gy)[full].all(1).sum())});jj=np.array([COLORS.index(rootinfo[m[i]['root_id']]['color'])for i in ii]);idx=np.arange(len(ii));z['joint_goal_intended_object']=binary_stats(py[idx,jj],gy[idx,jj],gv[idx,jj])
     for a,i in enumerate(ii):goals.append({'model':name,'split':split,'window_id':m[i]['window_id'],'root_id':m[i]['root_id'],'object':COLORS[jj[a]],'prediction':bool(py[a,jj[a]]),'truth':bool(gy[a,jj[a]]),'valid':bool(gv[a,jj[a]])})
    out[split+':'+pool]=z
  cats={};union=set()
  for family in ['outside_supported','indirect_protruding','suspended_held','geometric_boundary']:
   cc=[r for r in counter if r['split']=='dev'and r['family']==family];fp=0
   for r in cc:
    i=bywin[r['window_id']];j=3+COLORS.index(r['object']);fp+=int(d[i,j]==1 and not y[i,j]and v[i,j])
    if family!='geometric_boundary'and v[i,j]:union.add((i,j))
   cats[family]={'object_windows':len(cc),'valid':sum(r['valid']for r in cc),'FP':fp,'roots':len({r['root_id']for r in cc})}
  unique_fp=sum(d[i,j]==1 and not y[i,j]for i,j in union);out['C_counterexamples']=cats;out['unique_valid_counterexample_facts']=len(union);out['unique_counterexample_FP']=int(unique_fp)
  for i,r in enumerate(m):
   if r['split']=='dev'and not((d[i]==y[i])|~v[i]).all():errors.append({'model':name,**{k:r[k]for k in ['window_id','root_id','pool','condition','public_frames']},'probabilities':p[i].tolist(),'prediction':d[i].tolist(),'truth':y[i].tolist(),'valid':v[i].tolist(),'error_columns':np.flatnonzero((d[i]!=y[i])&v[i]).tolist()})
  cn=out['C:new'];ch=out['C:legacy_H'];gate={'new_C_all9_at_least90percent':cn['all9_correct_fully_valid']/cn['fully_valid9']>=.9,'Closed_false_positive_zero':cn['facts']['Closed(drawer)']['FP']==0,'counterexample_In_false_positive_zero':unique_fp==0,'all9_C_classes_positive_and_negative':all(f['positive']>0 and f['negative']>0 for f in cn['facts'].values()),'legacy_C_Holding_at_least83of84':ch['Holding_all3_correct']>=83};guards={'new_C_Holding_at_least209of210':cn['Holding_all3_correct']>=209,'drawer3_exact_at_least208of210':cn['families']['drawer3']['exact_correct_windows']>=208,'In_positive_FN_at_most2':sum(cn['facts'][f]['FN']for f in FACTS[3:6])<=2,'joint_goal_FP_zero':cn['joint_goal_all_objects']['FP']==0};out['core_gates']=gate;out['normal_class_and_goal_guards']=guards;out['perception_candidate_passes']=all(gate.values())and all(guards.values());allmetrics[name]=out
  if name!='original_temporal':decisions[name]={'perception_candidate_passes':out['perception_candidate_passes'],'selected_checkpoint':read(E/'configs'/name/'reader_frozen.json'),'failed_checks':[k for k,z in(gate|guards).items()if not z]}
 same_init=read(E/'configs/pooled/initial_checks.json')['initial_model_state_sha256']==read(E/'configs/spatial/initial_checks.json')['initial_model_state_sha256'];same_schedule=np.array_equal(np.load(E/'logs/pooled/train_indices.npy'),np.load(E/'logs/spatial/train_indices.npy'));same_params=read(E/'results/pooled/training.json')['trainable_parameters']==read(E/'results/spatial/training.json')['trainable_parameters'];assert same_init and same_schedule and same_params
 changes={}
 baseline=decode(models['original_temporal']);ii=np.flatnonzero(~train&new)
 for name in ['pooled','spatial']:
  d=decode(models[name]);oldok=(baseline[ii]==y[ii])|~v[ii];newok=(d[ii]==y[ii])|~v[ii];changes[name]={'facts_corrected':int((~oldok&newok).sum()),'facts_regressed':int((oldok&~newok).sum()),'windows_corrected_all_valid':int((~oldok.all(1)&newok.all(1)).sum()),'windows_regressed_all_valid':int((oldok.all(1)&~newok.all(1)).sum()),'regression_details':[{'window_id':m[i]['window_id'],'columns':np.flatnonzero(((baseline[i]==y[i])&v[i])&(d[i]!=y[i])).tolist()}for i in ii if (((baseline[i]==y[i])&v[i])&(d[i]!=y[i])).any()]}
 rules=read(BASE/'configs/learned_rules9.json');support={r['id']:r['supported']for r in rules['operators']};candidates=[a for a,d in decisions.items()if d['perception_candidate_passes']]
 save(E/'results/evaluation.json',{'metrics':allmetrics,'changes_from_original':changes,'controlled_comparison':{'identical_initial_weights':same_init,'identical_batches':same_schedule,'matched_trainable_parameters':same_params,'only_difference_spatial_resolution':True},'old_planning_operator_support':support,'no_planning_execution':True,'new_D':0,'C_is_reused_development':True});save(E/'results/errors.json',errors);save(E/'results/joint_goal_windows.json',goals);save(E/'configs/candidate_decision.json',{'created_utc':now(),'candidates':candidates,'arms':decisions,'independent_test_run':False,'planning_support_still_insufficient':True,'next_boundary':'If perception gates pass, design and freeze independent perception acceptance before any newD; planning needs separate negative coverage and state sufficiency. If fail, preserve errors and stop extra training.'});print(json.dumps({'decisions':{a:{'pass':d['perception_candidate_passes'],'failed':d['failed_checks']}for a,d in decisions.items()},'summary':{n:{'full9':r['C:new']['all9_correct_fully_valid'],'counter_FP':r['unique_counterexample_FP'],'joint_goal':r['C:new']['joint_goal_all_objects']}for n,r in allmetrics.items()}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
