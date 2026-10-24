from config import *
import importlib.util,csv,collections
spec=importlib.util.spec_from_file_location('fixed_metrics',DIAG/'source/score.py');metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
def main():
    rows=read(E/'results/predictions.json');labels=read(DIAG/'data/_private/labels.json');summary=metrics.summarize(rows,labels,True);save(E/'results/metrics.json',summary);cum=[];rootrows=[];facts=[]
    groups={}
    for r in rows:
        if r['path']=='full4':groups.setdefault((r['condition'],r['group'],r['root_id'],r['color'],r['branch']),[]).append(r)
        lab=labels[str(r['after'])]
        for j,f in enumerate(FACTS):facts.append({k:r[k]for k in['condition','group','root_id','color','branch','path','global_step','local_depth','skill']}|{'fact':f,'truth':lab['y'][j],'valid':lab['v'][j],**{k+'_probability':r[k+'_probability'][j]for k in['actual','one_step','recursive']},**{k+'_prediction':bool(discrete(r[k+'_probability'])[j])for k in['actual','one_step','recursive']},**{k+'_raw_logit':r[k+'_logits'][j]for k in['actual','one_step','recursive']}})
    for (name,g,root,color,branch),rr in groups.items():
        rr.sort(key=lambda r:r['global_step']);assert len(rr)==4
        for method in['actual','one_step','recursive']:
            ok=[]
            for r in rr:
                lab=labels[str(r['after'])];ok.append(bool(all(lab['v'])and np.array_equal(discrete(r[method+'_probability']),lab['y'])))
            rootrows.append({'condition':name,'group':g,'root_id':root,'color':color,'branch':branch,'method':method,'step_correct':ok,'prefix_correct':[all(ok[:h])for h in range(1,5)],'all4_correct':all(ok)})
    for name,g,branch,method in sorted({(r['condition'],r['group'],r['branch'],r['method'])for r in rootrows}):
        for color in['all',*COLORS]:
            rr=[r for r in rootrows if r['condition']==name and r['group']==g and r['branch']==branch and r['method']==method and(color=='all'or r['color']==color)]
            if rr:cum.append({'condition':name,'group':g,'branch':branch,'method':method,'color':color,'roots':len(rr),'prefix_correct':[sum(r['prefix_correct'][h]for r in rr)for h in range(4)],'all4_correct':sum(r['all4_correct']for r in rr)})
    # Query/encoder identity: changing the transition must never change real-state readout.
    lookup={(r['condition'],r['group'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in rows}
    for r in rows:
        if r['condition'].startswith('mixed'):
            other=lookup[(r['condition'].replace('mixed','latent'),r['group'],r['root_id'],r['branch'],r['path'],r['global_step'])];assert r['actual_probability']==other['actual_probability']
    save(E/'results/cumulative.json',cum);save(E/'results/root_trajectories.json',rootrows)
    with(R/'paired_propositions.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(facts[0]));w.writeheader();w.writerows(facts)
    print('SCORED',len(summary),'groups',len(facts),'facts',flush=True)
if __name__=='__main__':main()
