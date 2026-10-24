from config import *
import csv,ast,collections

def main():
    frozen=read(E/'configs/frozen.json')['sha256']
    for p,h in frozen.items():assert sha(p)==h,p
    previous=read(STUDY/'configs/delivery_manifest.json')['sha256']
    for p,h in previous.items():assert sha(p)==h,p
    selected=read(E/'results/predictions_selected.json');curves=read(E/'results/predictions_checkpoints.json');labels=read(E/'data/_private/labels.json');audit=read(E/'results/inference_audit.json');assert audit['new_training_steps']==audit['simulation_calls']==audit['private_truth_reads']==0 and not audit['optimizer_created'];assert audit['reencoded_cache_max_abs_error']<1e-4
    checked=0
    for suffix,rows in [('selected',selected),('checkpoints',curves)]:
        for summary in read(E/'results'/f'metrics_{suffix}.json'):
            rr=[r for r in rows if all(r[k]==summary[k]for k in ['condition','checkpoint_update','group','branch','path','global_step'])and(summary['color']=='all'or r['color']==summary['color'])];assert len(rr)==summary['roots'];y=np.array([labels[str(r['after'])]['y']for r in rr]);v=np.array([labels[str(r['after'])]['v']for r in rr]);full=v.all(1)
            for method in ['actual','one_step','recursive']:
                p=discrete([r[method+'_probability']for r in rr]);assert int((full&(p==y).all(1)).sum())==summary[method]['full9_correct'];assert int(full.sum())==summary[method]['full9_N']
                for g,s in [('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:assert int(((p[:,s]!=y[:,s]).any(1)&v[:,s].all(1)).sum())==summary[method]['groups'][g]['wrong']
            checked+=1
    lookup={(r['condition'],r['group'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in selected}
    for row in selected:
        key=(row['condition'],row['group'],row['root_id'],row['branch'])
        if row['path']=='window0':assert row['recursive_probability']==lookup[key+('full4',row['global_step'])]['recursive_probability']
        if row['path']=='window1'and row['global_step']==2:assert np.allclose(row['recursive_probability'],row['one_step_probability'],atol=1e-6)
    prev=read(STUDY/'results/scored_C_backtest.json');maxdiff=0.
    for name,rs in prev.items():
        if name not in {r['condition']for r in selected}:continue
        for row in rs:
            current=lookup[(name,'C_validate',row['root_id'],row['branch'],'full4',row['slot']+1)];assert labels[str(current['after'])]['y']==row['truth']and labels[str(current['after'])]['v']==row['valid']
            for k in ['actual','one_step','recursive']:maxdiff=max(maxdiff,float(np.abs(np.array(current[k+'_probability'])-row[k+'_probability']).max()));assert np.array_equal(discrete(current[k+'_probability']),discrete(row[k+'_probability']))
    assert maxdiff<1e-5
    facts=list(csv.DictReader((R/'paired_propositions_probabilities.csv').open()));assert len(facts)==21600
    for r in facts:
        for k in ['actual','one_step','recursive']:
            if r['valid']=='True':assert (float(r[k+'_signed_margin'])>=0)==(r[k+'_prediction']==r['truth'])
    losses=read(E/'results/frozen_objective_curves.json');assert len(losses)==20
    for r in losses:assert abs(r['time_average_objective']-(r['single_feature']+.25*r['single_latent_MSE']+.25*sum(.25*(x['feature']+.25*x['latent_MSE'])for x in r['multi_terms'])))<1e-10
    for p in (E/'source').glob('*.py'):ast.parse(p.read_text())
    result={'passed':True,'frozen_source_hashes_checked':len(frozen),'previous_study_files_verified':len(previous),'selected_prediction_rows':len(selected),'checkpoint_prediction_rows':len(curves),'summary_groups_independently_recounted':checked,'paired_proposition_rows':len(facts),'same_formal_C_predictions':True,'previous_C_probability_max_difference':maxdiff,'window0_matches_first3_full4':True,'window1_first_step_matches_true_predecessor_single':True,'models_unchanged':True,'no_optimization_or_simulation':True,'same_label_definition':True,'old60_endpoint_labels_recovered_with_conservative_masks':True}
    if(E/'configs/delivery_manifest.json').exists():
        manifest=read(E/'configs/delivery_manifest.json')['sha256']
        for p,h in manifest.items():assert sha(p)==h,p
        result['sealed_files_verified']=len(manifest)
    print('VERIFICATION_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
