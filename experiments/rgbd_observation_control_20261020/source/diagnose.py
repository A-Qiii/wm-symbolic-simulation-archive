from common import *
import ast
node=next(n for n in ast.parse((O/'source/analyze.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='analyze');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged metrics','exec'))
def main():
 guard();verify();anchors=[];close_readout=[];cases=[];first_summary=[]
 for split in ['train','development','historical_evaluation','diagnosed_confirmation']:
  for seed in [0,1]:
   src=C if split=='diagnosed_confirmation'else O;tag='confirm'if split=='diagnosed_confirmation'else split;z=dict(np.load(src/f'results/predictions_object_seed{seed}_{tag}.npz'));rr,_,_,_=analyze(src/f'results/predictions_object_seed{seed}_{tag}.npz');rr=[r for r in rr if r['kind']=='role'];ok=correct(rel(z['prob']),truth(z['cat']));row={'split':split,'seed':seed,'arm':'old_Object_reference','parameters':2465492,'roles':len(rr),**{k:sum(r[k]for r in rr)/len(rr)for k in ['h0_held','normal_whole','primary_joint3','terminal']}}
   for mode in ['one_step','suffix']:
    ix=np.flatnonzero((z['kind']=='role')&(z['skill']=='close_drawer')&(z['mode']==mode));row['close_'+mode+'_drawer_correct']=float(ok[ix,2].mean())
   anchors.append(row)
   zs={arm:dict(np.load(S/f'results/predictions_{arm}_seed{seed}_{split}.npz'))for arm in ARMS}
   for arm,z in zs.items():
    ok=correct(rel(z['prob']),truth(z['cat']));sel=(z['kind']=='role')&(z['step']==6)
    for mode in ['current','one_step','suffix']:
     ix=np.flatnonzero(sel&(z['mode']==mode));conf=np.zeros((3,3),int)
     for j in ix:conf[z['cat'][j,3],z['prob'][j,24:27].argmax()]+=1
     close_readout.append({'split':split,'seed':seed,'arm':arm,'mode':mode,'n':len(ix),'all_relation_correct':int(ok[ix,3].sum()),'drawer_correct':int(ok[ix,2].sum()),'confusion_true_rows_pred_cols':conf.tolist()})
    _,_,_,errors=analyze(S/f'results/predictions_{arm}_seed{seed}_{split}.npz');counter=collections.Counter()
    for r in errors:
     pp=np.array(r['predicted_relation_probabilities']);yy=np.array(r['truth_relations']);cc=correct(pp[None],yy[None])[0];key='h'+str(r['step'])+':'+','.join(k for k,c in zip(['Held','In','Drawer'],cc[:3])if not c);counter[key]+=1
    first_summary.append({'split':split,'seed':seed,'arm':arm,'from_h0_first_error_counts':dict(counter)})
   b,g=zs['baseline'],zs['geometry'];bo=correct(rel(b['prob']),truth(b['cat']));go=correct(rel(g['prob']),truth(g['cat']));assert np.array_equal(b['root_id'],g['root_id'])
   ids=np.flatnonzero((b['kind']=='role')&(b['mode']=='current')&(b['step']==0)&(b['branch']==1)&(~bo[:,0])&go[:,0])
   for j in ids:
    h=COLORS.index(str(b['held_color'][j]));cases.append({'split':split,'seed':seed,'root_id':str(b['root_id'][j]),'kind':'initial_Held_gain','H':COLORS[h],'baseline_Held_probabilities':rel(b['prob'])[j,:3].tolist(),'geometry_Held_probabilities':rel(g['prob'])[j,:3].tolist(),'true_H_visible_in_old_route':float(b['availability'][j,h])})
 save(S/'results/old_Object_reference.json',anchors);save(S/'results/close_readout_vs_prediction.json',close_readout);save(S/'results/first_error_summary.json',first_summary);save(S/'results/initial_Held_gain_cases.json',cases)
 print(json.dumps({'old_reference_development':[r for r in anchors if r['split']=='development'],'geometry_confirm_close':[r for r in close_readout if r['split']=='diagnosed_confirmation'and r['arm']=='geometry'],'geometry_first_errors':[r for r in first_summary if r['arm']=='geometry'and r['split']!='train'],'example_h0_gains':[r for r in cases if r['split']=='diagnosed_confirmation'][:3]},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
