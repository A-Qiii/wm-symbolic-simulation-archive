from config import *
from model import *
import time
def metrics(prob,y,v,roots):
 pred=prob>=.5;vv=v.astype(bool);yy=y.astype(bool);out={'states':len(y),'roots':len(set(roots)),'all3_valid_states':int(vv.all(1).sum()),'all3_correct':int(((pred==yy).all(1)&vv.all(1)).sum()),'all3_accuracy':float((pred[vv.all(1)]==yy[vv.all(1)]).all(1).mean())};empty=~yy.any(1)&vv.all(1);out.update(empty_states=int(empty.sum()),empty_roots=len(set(roots[empty])),empty_any_FP=int(pred[empty].any(1).sum()),empty_any_FP_rate=float(pred[empty].any(1).mean())if empty.any()else None,colors=[])
 for j,color in enumerate(COLORS):
  valid=vv[:,j];positive=valid&yy[:,j];negative=valid&~yy[:,j];tp=int((pred[:,j]&positive).sum());fn=int((~pred[:,j]&positive).sum());fp=int((pred[:,j]&negative).sum());tn=int((~pred[:,j]&negative).sum());out['colors'].append({'color':color,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'positive_roots':len(set(roots[positive])),'negative_roots':len(set(roots[negative])),'FPR':fp/(fp+tn)if fp+tn else None,'FNR':fn/(fn+tp)if fn+tp else None})
 out['root_accuracy']={r:float((pred[(roots==r)&vv.all(1)]==yy[(roots==r)&vv.all(1)]).all(1).mean())for r in sorted(set(roots))};return out
def passes(m):return m['all3_accuracy']>=.95 and m['empty_any_FP_rate']is not None and m['empty_any_FP_rate']<=.05 and all(c['positive_roots']>=2 and c['FPR']is not None and c['FPR']<=.05 and c['FNR']is not None and c['FNR']<=.10 for c in m['colors'])
def main():
 verify();assert not(E/'results/actual_metrics.json').exists()
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h
 torch.set_num_threads(4);start=time.perf_counter();rows=[];predictions=[]
 for form in FORMS:
  norm=np.load(E/'configs'/f'normalization_{form}.npz')
  for seed in [0,1]:
   q=make(form).eval();q.load_state_dict(torch.load(E/'models'/f'{form}_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state'])
   for path in sorted((E/'data/evaluation').glob('*.npz')):
    if path.stem.startswith('D'):continue
    d=np.load(path);x=((features(d['z'],form)-norm['mean'])/norm['std']).astype(np.float32)
    with torch.inference_mode():prob=q(torch.tensor(x)).squeeze(-1).sigmoid().numpy()
    m=metrics(prob,d['y'],d['v'],d['root_ids']);rows.append({'form':form,'seed':seed,'cohort':path.stem,**m,'gate_pass':passes(m)})
    for i,p in enumerate(prob):predictions.append({'form':form,'seed':seed,'cohort':path.stem,'index':i,'root_id':str(d['root_ids'][i]),'state_id':str(d['state_ids'][i]),'probability':p.tolist()})
 gatecoh=read(E/'configs/protocol.json')['reliability_diagnostic_gate']['cohorts'];gate={form:all(r['gate_pass']for r in rows if r['form']==form and r['cohort']in gatecoh)for form in FORMS};assert all(sum(r['form']==f and r['cohort']in gatecoh for r in rows)==6 for f in FORMS)
 save(E/'results/actual_metrics.json',rows);save(E/'results/actual_predictions.json',predictions);save(E/'results/reliability_decision.json',{'utc':now(),'eligible_forms':gate,'rule':'both seeds pass all3 prespecified AB/C validation cohorts; no checkpoint or threshold selection','evaluation_seconds':time.perf_counter()-start,'if_no_eligible_form':'stop predicted-latent attribution; report inability to establish a reliable independent reader','D_used':False});print('HOLDING_ACTUAL_GATE',json.dumps(gate),flush=True)
 for r in rows:
  if r['cohort']in gatecoh:print(r['form'],r['seed'],r['cohort'],'H3',r['all3_correct'],'/',r['states'],'emptyFP',r['empty_any_FP'],'/',r['empty_states'],'colors',[(c['color'],c['FP'],c['FN'],c['positive_roots'])for c in r['colors']],'pass',r['gate_pass'],flush=True)
if __name__=='__main__':main()
