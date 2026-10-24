from config import *
import collections,csv,ast
def main():
    hashes=0;previous=0
    for fn in['training_frozen.json','choices_frozen.json']:
        for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p;hashes+=1
    for old in[STUDY,DIAG]:
        for p,h in read(old/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p;previous+=1
    rows=read(PACK/'pairs.json');chains=read(E/'configs/chains.json');rootset=set(chains['AB']);assert len(rootset)==12 and len(chains['C_selection'])==6
    initials=[];times={};weightschecked=0
    for seed in[0,1]:
        sc=dict(np.load(E/'configs'/f'schedule_seed{seed}.npz'));old=dict(np.load(STUDY/'configs'/f'schedule_seed{seed}.npz'));assert np.array_equal(sc['single'],old['single'])and np.array_equal(sc['span3'],old['new_chains']);assert sc['single'].shape==(1000,64)
        for u in range(1000):
            if(u+1)%4:assert np.all(sc['span3'][u]==-1)and np.all(sc['span4'][u]==-1);continue
            totals={}
            for arm,ww in[('span3',[.25,.25]),('span4',[.125,.25,.125])]:
                seq=sc[arm][u];assert seq.shape==(48,3 if arm=='span3'else 4);weight=collections.defaultdict(float);roots=[]
                for ix in seq:
                    root=rows[ix[0]]['root_id'];roots.append(root);assert root in rootset and all(rows[i]['root_id']==root and rows[i]['split']=='train'and rows[i]['branch']=='normal'for i in ix);assert all(rows[i]['after']==rows[j]['before']for i,j in zip(ix,ix[1:]))
                    for d,w in enumerate(ww,1):weight[(root,rows[ix[d]]['slot'])]+=w/48
                assert set(roots)==rootset and set(collections.Counter(roots).values())=={4};assert abs(sum(weight.values())-.5)<1e-12;totals[arm]=weight
            assert totals['span3'].keys()==totals['span4'].keys()
            for k in totals['span3']:assert abs(totals['span3'][k]-totals['span4'][k])<1e-12
            weightschecked+=1
        for arm in['span3','span4']:
            n=f'{arm}_seed{seed}';t=read(E/'results'/f'train_{n}.json');assert t['updates']==1000 and t['single_draws']==64000 and t['multi_starts']==12000;assert t['frozen_encoder_decoder_hash']==t['final_frozen_hash']and t['query_hash']==t['final_query_hash'];assert not t['validation_or_test_access'];initials.append(t['initial_model_hash']);sel=read(E/'models'/n/'selected.json');best=min(t['log'],key=lambda r:r['score']);assert sel['C_score']==best['score']and sel['path'].endswith(f'update{best["update"]:04d}.pt');times[n]=t['elapsed_seconds']
            for r in t['log']:
                assert abs(r['score']-(.25*r['C_recursive_step_feature'][1]+.5*r['C_recursive_step_feature'][2]+.25*r['C_recursive_step_feature'][3]))<1e-9
    assert len(set(initials))==1
    predictions=read(E/'results/predictions.json');labs=read(DIAG/'data/_private/labels.json');summary=read(E/'results/metrics.json');checked=0
    assert len(predictions)==1344
    lookup={(r['condition'],r['group'],r['root_id'],r['branch'],r['path'],r['global_step']):r for r in predictions}
    for r in predictions:
        if r['condition'].startswith('span3'):
            other=lookup[(r['condition'].replace('span3','span4'),r['group'],r['root_id'],r['branch'],r['path'],r['global_step'])]
            assert r['actual_probability']==other['actual_probability']
    for r in summary:
        rr=[x for x in predictions if all(x[k]==r[k]for k in['condition','group','branch','path','global_step'])and(r['color']=='all'or x['color']==r['color'])];assert len(rr)==r['roots'];y=np.array([labs[str(x['after'])]['y']for x in rr]);v=np.array([labs[str(x['after'])]['v']for x in rr]);full=v.all(1)
        for k in['actual','one_step','recursive']:
            p=discrete([x[k+'_probability']for x in rr]);assert int((full&(p==y).all(1)).sum())==r[k]['full9_correct']
            for g,ss in[('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:assert int(((p[:,ss]!=y[:,ss]).any(1)&v[:,ss].all(1)).sum())==r[k]['groups'][g]['wrong']
        checked+=1
    roots=read(E/'results/root_trajectories.json')
    for r in read(E/'results/cumulative.json'):
        rr=[x for x in roots if all(x[k]==r[k]for k in['condition','group','branch','method'])and(r['color']=='all'or x['color']==r['color'])];assert len(rr)==r['roots']and sum(x['all4_correct']for x in rr)==r['all4_correct'];assert r['prefix_correct']==[sum(x['prefix_correct'][h]for x in rr)for h in range(4)]
    assert len(list(csv.DictReader((R/'paired_propositions.csv').open())))==len(predictions)*9
    if(R/'step_summary.csv').exists():assert len(list(csv.DictReader((R/'step_summary.csv').open())))==len(summary)*3
    gate=read(E/'results/gate.json');assert gate['protocol_sha256']==sha(E/'configs/protocol.json')
    if not gate['passed']:assert not(E/'data/D5').exists()
    for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
    result={'passed':True,'frozen_hash_entries':hashes,'prior_sealed_files_verified':previous,'same12ABroots':True,'same_single_data_and_order':True,'matched_root_position_weights_batches_checked':weightschecked,'multi_total_weight_both':.5,'global_position_weights_both':[.125,.25,.125],'same_initial_transition':True,'encoder_decoder_query_unchanged':True,'actual_readout_identical_between_arms':True,'updates_total':4000,'single_draws_total':256000,'multistep_starts_total':48000,'equal_compute':False,'elapsed_training_seconds':times,'summary_groups_recounted':checked,'cumulative_prefix_counts_recounted':True,'gate_passed':gate['passed'],'new_D_collected':(E/'data/D5').exists()}
    if(E/'configs/delivery_manifest.json').exists():
        files=read(E/'configs/delivery_manifest.json')['sha256']
        for p,h in files.items():assert sha(p)==h,p
        result['sealed_files_verified']=len(files)
    print('VERIFICATION_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
