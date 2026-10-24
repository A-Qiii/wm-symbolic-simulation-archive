from config import *
import ast,csv,time,collections

def main():
    start=time.monotonic();entries=0
    for filename in ['training_frozen.json','choices_frozen.json']:
        for path,h in read(E/'configs'/filename)['sha256'].items():assert sha(path)==h,path;entries+=1
    priorfiles=0
    for old in [FIX,PRIOR]:
        manifest=read(old/'configs/delivery_manifest.json')
        for path,h in manifest['sha256'].items():assert sha(path)==h,path;priorfiles+=1
    rows=read(PACK/'pairs.json');matched=read(E/'configs/chain_matching.json');oldroots={r['old_root']for r in matched['root_pairs']};newroots={r['new_root']for r in matched['root_pairs']};assert len(oldroots)==len(newroots)==12 and oldroots.isdisjoint(newroots)
    for pair in matched['root_pairs']:
        for arm in ['old','new']:
            ix=pair[arm+'_pairs'];assert len(ix)==4 and all(rows[i]['root_id']==pair[arm+'_root']and rows[i]['split']=='train'for i in ix)
            assert all(rows[i]['after']==rows[j]['before']for i,j in zip(ix,ix[1:]));assert all(rows[i]['action'][7]==1 for i in ix)
        assert [rows[i]['action']for i in pair['old_pairs']]==[rows[i]['action']for i in pair['new_pairs']]
    allinits=[];frozenhashes=[];selected={}
    for seed in [0,1]:
        sc=dict(np.load(E/'configs'/f'schedule_seed{seed}.npz'));prev=dict(np.load(PRIOR/'configs'/f'transition_schedule_seed{seed}.npz'))
        assert np.array_equal(sc['single'],np.concatenate([prev['base'],prev['new_aux']],axis=1));assert sc['single'].shape==(1000,64)
        assert all(rows[i]['split']=='train'for i in np.unique(sc['single']))
        for arm in ['old_chains','new_chains']:
            multi=sc[arm];assert multi.shape==(1000,48,3)
            for u in range(1000):
                if (u+1)%4:assert np.all(multi[u]==-1);continue
                assert sorted(collections.Counter(rows[ix[0]]['root_id']for ix in multi[u]).values())==[4]*12
                assert len({tuple(ix)for ix in multi[u]})==24
                for ix in multi[u]:
                    assert all(rows[i]['split']=='train'for i in ix);assert len({rows[i]['root_id']for i in ix})==1
                    assert all(rows[i]['after']==rows[j]['before']for i,j in zip(ix,ix[1:]))
            t=read(E/'results'/f'train_{arm}_seed{seed}.json');assert t['updates']==1000 and t['single_draws']==64000 and t['multistep_draws']==12000;assert t['only_transition_optimized']and not t['validation_or_test_access'];assert t['frozen_encoder_decoder_hash']==t['final_frozen_hash']and t['query_hash']==t['final_query_hash'];allinits.append(t['initial_model_hash']);frozenhashes.append(t['frozen_encoder_decoder_hash'])
            sel=read(E/'models'/f'{arm}_seed{seed}'/'selected.json');best=min(t['log'],key=lambda v:v['score']);assert sel['C_score']==best['score']and sel['path'].endswith(f'update{best["update"]:04d}.pt');selected[f'{arm}_seed{seed}']=best['update']
        for u in range(3,1000,4):
            for oi,ni in zip(sc['old_chains'][u],sc['new_chains'][u]):assert [rows[i]['action']for i in oi]==[rows[i]['action']for i in ni]
    assert len(set(allinits))==len(set(frozenhashes))==1
    phases=[];count=0
    for phase in ['C_backtest','D5']:
        if not(E/'results'/f'metrics_{phase}.json').exists():continue
        phases.append(phase);m=read(E/'results'/f'metrics_{phase}.json');scores=read(E/'results'/f'scored_{phase}.json');sp=read(E/'results'/f'true_state_probabilities_{phase}.json');audit=read(E/'results'/f'prediction_audit_{phase}.json')
        assert audit['private_reads']==audit['recursive_corrections']==audit['model_updates']==audit['physics_calls']==0
        for seed in [0,1]:
            for arm in ['old_chains','new_chains']:assert sp['probabilities'][f'initial_seed{seed}']==sp['probabilities'][f'{arm}_seed{seed}']
        for name,allrows in scores.items():
            for branch in ['normal','held_closed']:
                for slot in range(4):
                    rr=[r for r in allrows if r['branch']==branch and r['slot']==slot];mm=m['trajectory_prediction'][name][branch][slot];y=np.array([r['truth']for r in rr]);v=np.array([r['valid']for r in rr]);full=v.all(1);pred={k:discrete([r[k+'_probability']for r in rr])for k in ['actual','one_step','recursive']}
                    assert len(rr)==mm['observed_roots']
                    for k,p in pred.items():
                        assert int(((p==y).all(1)&full).sum())==mm[k]['full9_correct']
                        for group,ss in [('Holding',slice(0,3)),('In',slice(3,6)),('drawer',slice(6,9))]:
                            valid=v[:,ss].all(1);assert int(((p[:,ss]!=y[:,ss]).any(1)&valid).sum())==mm[k]['groups'][group]['wrong_states']
                    for k in ['one_step','recursive']:
                        aa=(pred['actual']==y).all(1);bb=(pred[k]==y).all(1);pair={'actual_right_prediction_wrong':int((aa&~bb&full).sum()),'both_wrong':int((~aa&~bb&full).sum()),'actual_wrong_prediction_right':int((~aa&bb&full).sum()),'both_right':int((aa&bb&full).sum())};assert pair==mm[k+'_paired']['states']
        csvN=len(list(csv.DictReader((R/f'paired_propositions_{phase}.csv').open())));assert csvN==sum(len(rs)*9 for rs in scores.values());count+=csvN
    gate=read(E/'results/gate.json');assert gate['protocol_sha256']==sha(E/'configs/protocol.json')
    if not gate['passed']:assert not(E/'data/D5').exists()and phases==['C_backtest']
    for path in (E/'source').glob('*.py'):ast.parse(path.read_text())
    result={'passed':True,'frozen_hash_entries':entries,'previous_sealed_files_verified':priorfiles,'matched_roots_per_arm':12,'three_skill_windows_per_arm':24,'full_action_and_continuity_matched':True,'same_single_sample_order_and_budget':True,'root_balanced_multistep_draws':True,'query_encoder_decoder_unchanged':True,'initial_model_equal_all_runs':True,'updates_total':4000,'single_draws_total':256000,'multistep_chain_draws_total':48000,'selected_updates':selected,'evaluation_phases':phases,'paired_proposition_rows_recounted':count,'metrics_and_pairs_recounted':True,'gate_passed':gate['passed'],'seconds':time.monotonic()-start}
    if (E/'configs/delivery_manifest.json').exists():
        files=read(E/'configs/delivery_manifest.json')['sha256']
        for path,h in files.items():assert sha(path)==h,path
        result['sealed_files_verified']=len(files)
    print('VERIFICATION_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
