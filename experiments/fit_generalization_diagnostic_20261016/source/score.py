from config import *
import csv,collections

def metric(rr,k,labels):
    y=np.array([labels[str(r['after'])]['y']for r in rr]);v=np.array([labels[str(r['after'])]['v']for r in rr]);p=np.array([r[k+'_probability']for r in rr]);pred=discrete(p);full=v.all(1);m={'N':len(rr),'full9_N':int(full.sum()),'full9_correct':int(((pred==y).all(1)&full).sum()),'feature_mean':float(np.mean([r[k+'_feature_groups']for r in rr])),'raw_MSE':float(np.mean([r[k+'_raw_MSE']for r in rr])),'groups':{}}
    for g,ss in [('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:
        valid=v[:,ss].all(1);ok=(pred[:,ss]==y[:,ss]).all(1)
        if g=='drawer':
            pp=p[:,6:];yy=y[:,6:].argmax(1);wrong=pp.copy();wrong[np.arange(len(rr)),yy]=-np.inf;margin=pp[np.arange(len(rr)),yy]-wrong.max(1)
        else:margin=np.where(y[:,ss],p[:,ss]-.5,.5-p[:,ss]).min(1)
        vals=margin[valid];m['groups'][g]={'N':int(valid.sum()),'correct':int((ok&valid).sum()),'wrong':int((~ok&valid).sum()),'margin_min':float(vals.min())if len(vals)else None,'margin_median':float(np.median(vals))if len(vals)else None,'near_boundary_abs_le_05':int((np.abs(vals)<=.05).sum()),'confident_wrong_margin_le_minus05':int((vals<=-.05).sum())}
    return m

def summarize(rows,labels,colors):
    grouped={}
    for r in rows:
        for color in (['all',r['color']]if colors else['all']):
            key=(r['condition'],r['checkpoint_update'],r['group'],r['branch'],r['path'],r['global_step'],color);grouped.setdefault(key,[]).append(r)
    out=[]
    for (condition,update,group,branch,path,step,color),rr in grouped.items():
        x={'condition':condition,'checkpoint_update':update,'group':group,'branch':branch,'path':path,'global_step':step,'local_depth':rr[0]['local_depth'],'color':color,'roots':len({r['root_id']for r in rr})}
        assert x['roots']==len(rr)
        for k in ['actual','one_step','recursive']:x[k]=metric(rr,k,labels)
        y=np.array([labels[str(r['after'])]['y']for r in rr]);v=np.array([labels[str(r['after'])]['v']for r in rr]);actual=discrete([r['actual_probability']for r in rr]);av=(actual==y).all(1);full=v.all(1)
        for k in ['one_step','recursive']:
            pr=discrete([r[k+'_probability']for r in rr]);ok=(pr==y).all(1);x[k+'_paired']={'actual_right_prediction_wrong':int((av&~ok&full).sum()),'both_wrong':int((~av&~ok&full).sum()),'actual_wrong_prediction_right':int((~av&ok&full).sum()),'both_right':int((av&ok&full).sum())}
        out.append(x)
    return out

def main():
    labels=read(E/'data/_private/labels.json');selected=read(E/'results/predictions_selected.json');curves=read(E/'results/predictions_checkpoints.json');summary=summarize(selected,labels,True);cs=summarize(curves,labels,False);save(E/'results/metrics_selected.json',summary);save(E/'results/metrics_checkpoints.json',cs)
    flat=[]
    for r in summary:
        flat.append({k:r[k]for k in ['condition','group','branch','path','global_step','local_depth','color','roots']}|{k+'_'+f:r[k][f]for k in ['actual','one_step','recursive']for f in ['full9_N','full9_correct','feature_mean','raw_MSE']}|{k+'_'+g+'_'+f:r[k]['groups'][g][f]for k in ['actual','one_step','recursive']for g in ['Holding','In','drawer']for f in ['N','correct','wrong','margin_min','margin_median','near_boundary_abs_le_05','confident_wrong_margin_le_minus05']})
    with(R/'summary_by_color_and_position.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    facts=[];rootrows=[]
    for r in selected:
        lab=labels[str(r['after'])];y=np.array(lab['y']);v=np.array(lab['v']);base={k:r[k]for k in ['condition','group','root_id','color','branch','path','global_step','local_depth','skill']}
        rootrows.append(base|{'full9_valid':bool(v.all()),**{k+'_all9_correct':bool((discrete(r[k+'_probability'])==y).all()and v.all())for k in ['actual','one_step','recursive']},'truth_Holding':','.join(COLORS[j]for j in range(3)if y[j])or'empty','truth_In':','.join(COLORS[j]for j in range(3)if y[j+3])or'none','truth_drawer':['Closed','Open','Intermediate'][int(y[6:].argmax())],**{k+'_errors':','.join(FACTS[j]for j in range(9)if v[j]and discrete(r[k+'_probability'])[j]!=y[j])for k in ['actual','one_step','recursive']}})
        for j,fact in enumerate(FACTS):
            row=base|{'fact':fact,'truth':bool(y[j]),'valid':bool(v[j])}
            for k in ['actual','one_step','recursive']:
                pp=np.array(r[k+'_probability']);margin=(pp[j]-.5)*(1 if y[j]else-1)if j<6 else(pp[j]-max(pp[l]for l in range(6,9)if l!=j))*(1 if y[j]else-1)
                row[k+'_probability']=pp[j];row[k+'_prediction']=bool(discrete(pp)[j]);row[k+'_signed_margin']=margin;row[k+'_feature_error']=float(np.mean(r[k+'_feature_groups']));row[k+'_raw_MSE']=r[k+'_raw_MSE']
            facts.append(row)
    for name,rows in [('paired_propositions_probabilities.csv',facts),('per_root_diagnostics.csv',rootrows)]:
        with(R/name).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Only the latent after step1 is replaced by its real encoding in window1; same later actions/query/model.
    keyed={(r['condition'],r['group'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in selected};replacement=[]
    for r in selected:
        if r['path']!='full4'or r['global_step']not in [3,4]:continue
        w=keyed[(r['condition'],r['group'],r['root_id'],r['branch'],'window1',r['global_step'])];lab=labels[str(r['after'])];y=np.array(lab['y']);v=np.array(lab['v']);a=discrete(r['recursive_probability']);b=discrete(w['recursive_probability']);replacement.append({k:r[k]for k in ['condition','group','root_id','color','branch','global_step']}|{'full9_valid':bool(v.all()),'full4_correct':bool((a==y).all()and v.all()),'window1_correct':bool((b==y).all()and v.all()),'full4_errors':[FACTS[j]for j in range(9)if v[j]and a[j]!=y[j]],'window1_errors':[FACTS[j]for j in range(9)if v[j]and b[j]!=y[j]]})
    save(E/'results/post_open_latent_replacement.json',replacement)
    print('SCORED',len(summary),len(cs),'proposition rows',len(facts),flush=True)
if __name__=='__main__':main()
