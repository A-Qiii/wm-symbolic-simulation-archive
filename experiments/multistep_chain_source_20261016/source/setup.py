from config import *
import collections

def main():
    assert not(E/'configs/protocol.json').exists()
    for sub in ['source','configs','models','logs','results']:(E/sub).mkdir(parents=True,exist_ok=True)
    R.mkdir(parents=True,exist_ok=True)
    # Audit source seals before constructing the new training schedule. No outcomes or C/D metrics read.
    oldseal=read(PRIOR/'configs/delivery_manifest.json');fixseal=read(FIX/'configs/delivery_manifest.json')
    for manifest in [oldseal,fixseal]:
        for path,h in manifest['sha256'].items():assert sha(path)==h,path
    rows=read(PACK/'pairs.json');states=read(PACK/'states.json');chains=[];roots=[]
    for env in ['A','B']:
        for color in COLORS:
            for rep in [0,1]:
                oldroot=f'{env}_longpair_{color}_closed_{rep}';newroot=f'{env}_intermediate_v3_{color}_{rep}'
                normal=['open_drawer',f'lift_{color}_block_table','place_in_drawer','close_drawer'];paired=[]
                for group,root in [('old',oldroot),('new',newroot)]:
                    ix=[i for i,r in enumerate(rows)if r['root_id']==root and (group=='old' or r['branch']=='normal')]
                    ix.sort(key=lambda i:rows[i].get('step',rows[i].get('slot')))
                    assert len(ix)==4 and [rows[i]['skill']for i in ix]==normal
                    assert all(rows[i]['split']=='train'for i in ix)
                    assert all(rows[a]['after']==rows[b]['before'] for a,b in zip(ix,ix[1:]))
                    assert len({rows[i].get('episode_id',root+'_'+rows[i].get('branch',''))for i in ix})==1
                    assert all(states[rows[i][t]]['root_id']==root for i in ix for t in ['before','after'])
                    assert all(rows[i]['action'][7]==1 for i in ix)
                    paired.append(ix)
                assert all(rows[a]['action']==rows[b]['action']for a,b in zip(*paired))
                roots.append({'environment':env,'target':color,'replicate':rep,'old_root':oldroot,'new_root':newroot,'old_pairs':paired[0],'new_pairs':paired[1]})
                for start in [0,1]:chains.append({'pair_index':len(roots)-1,'start_slot':start,'old':paired[0][start:start+3],'new':paired[1][start:start+3],'skills':normal[start:start+3]})
    assert len(chains)==24 and len(roots)==12
    schedules=[]
    for seed in [0,1]:
        prev=PRIOR/'configs'/f'transition_schedule_seed{seed}.npz';sc=dict(np.load(prev));single=np.concatenate([sc['base'],sc['new_aux']],axis=1)
        assert single.shape==(1000,64) and all(rows[i]['split']=='train'for i in np.unique(single))
        rng=np.random.default_rng(16967000+seed);oldmulti=np.full((1000,48,3),-1,np.int64);newmulti=oldmulti.copy()
        for u in range(3,1000,4):
            ids=rng.permutation(np.tile(np.arange(24),2))
            oldmulti[u]=[chains[i]['old']for i in ids];newmulti[u]=[chains[i]['new']for i in ids]
            for oi,ni in zip(oldmulti[u],newmulti[u]):assert [rows[j]['action']for j in oi]==[rows[j]['action']for j in ni]
        dest=E/'configs'/f'schedule_seed{seed}.npz';np.savez_compressed(dest,single=single,old_chains=oldmulti,new_chains=newmulti);schedules.append(dest)
    heads={str(seed):read(FIX/'models'/f'empty_open_seed{seed}'/'selected.json') for seed in [0,1]}
    audit={'created_utc':now(),'old_source_seal_files':len(oldseal['sha256']),'query_correction_seal_files':len(fixseal['sha256']),'old_roots':12,'new_roots':12,'windows_per_arm':24,'window_length':3,'matched_on':['A/B environment','target color','replicate count','3-skill sequence','full8D action including96-control posthold','2windows per root','root-balanced draws'],'all_within_one_episode':True,'adjacent_after_equals_next_before':True,'old_pair_indices_from_original_pack':True,'new_pair_indices_from_actual_normal_branch':True,'ground_truth_or_outcomes_used_to_select_chains':False,'not_matched':['Exact initial layout, images and measured trajectories differ by data source. This is the intended source treatment; no claim of state-distribution equality.'],'root_pairs':roots,'chain_pairs':chains}
    save(E/'configs/chain_matching.json',audit)
    protocol={'frozen_utc':now(),'scope':'One local, finite, paired multi-step chain-source comparison. No environment expansion, new query fitting, new loss or architecture.','initial_transition':str(REP/'models/canonical_matched_seed0/update1000.pt'),'initial_sha256':sha(REP/'models/canonical_matched_seed0/update1000.pt'),'query_heads':heads,'frozen':'DINO, preprocessing, object encoder, scene encoder, decoder, selected corrected query perseed, query normalization, threshold, controller','arms':['old_chains','new_chains'],'seeds':[0,1],'updates':1000,'single_step_batch':64,'single_step_draws_per_run':64000,'single_step_schedule':'Exact prior new-coverage transition schedule:48common oldAB+12new normal+4new held_closed draws, identical in both arms. No change to samples/order/budget.','multi_step':{'every_updates':4,'batch':48,'distinct_windows_per_arm':24,'root_counts_per_arm':12,'window_length':3,'draws_per_run':12000,'steps_scored':[2,3],'sampling':'Every multi-step batch contains each paired root and each of its two3-skill windows twice; identical paired order across arms, shuffled by fixed seed. No discontinuous or reset edges.','only_treatment':'old12 closed longpair roots vs new12 normal roots; same environment,color,full action,window length,count,frequency and weight.','first_step':'Compute chain first predicted latent from that chain true initial latent. No extra first-step chain loss; this permits identical independent single-step samples in the two arms.','loss':'Existing .25*(mean equal4group changed-weighted decoded feature error+.25 latent MSE) at steps2 and3, exact same formula; no new loss/weight.'},'single_step_loss':'Existing mean equal4group changed-weighted decoded feature error+.25 actual successor latent MSE.','optimizer':'same AdamW lr1e-4 wd1e-4 clip5; fresh optimizer in each arm; only transition parameters.','checkpoints':[0,250,500,750,1000],'selection':'Unchanged minimum .5oldC37-root mean single-step feature error+.5newC-selection6-root same error; earlier tie. No C-validation or D3/D4 reads.','validation':{'roots':'Existing6 C dev_validate roots excluded from this training and checkpoint selection. Historically seen in previous studies: report as held-out development backtest, not fresh independent test.','gate_frozen_before_training':{'C_selection':'Both seeds: selected new-chain score strictly less than old-chain score.','C_backtest_primary':'Both seeds: normal recursive step3 all9 count >= both initial and old-chain count+1; sum recursive step3+step4 >= both reference sums+2.','retention':'Both seeds: normal recursive step1 and step2 and one-step step3 and step4 cannot each lose more than1/6 root vs either reference. Actual-state query predictions must be identical across transition arms.','if_pass':'Freeze choices, then collect exactly12 new D5 roots (4/color),2branches,96calls maximum; same900control+96posthold,internal2/external0, no replacements.','if_fail':'No newD collection, no second tuning round. Report negative/inconclusive result and return for decision.','interpretation':'Operational finite-study gate; not statistical significance or replacement of original acceptance criteria.'}},'evaluation':'Same fixed root cohorts, full9, Holding/In/drawer, paired actual-readout vs one-step vs recursive, per-proposition and physical place truth. Normal primary; refusals separate. D3/D4 only seen diagnostics, not accessed by training/selection.','budget':'4runs x1000updates,256000single draws,48000three-skill chain draws; no other run or hyperparameter change.'}
    save(E/'configs/protocol.json',protocol)
    sources=list((E/'source').glob('*.py'))+schedules+[E/'configs/chain_matching.json',E/'configs/protocol.json',PACK/'inputs.npz',PACK/'pairs.json',PACK/'states.json',REP/'data/public/normalization.npz',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/canonical_matched_seed0/selected.json',REP/'models/query_seed0.pt',PRIOR/'source/score.py',REP/'source/model.py',REP/'source/runtime.py']+[Path(v['path'])for v in heads.values()]
    save(E/'configs/training_frozen.json',{'frozen_utc':now(),'sha256':{str(p):sha(p)for p in sources}})
    print('PROTOCOL_FROZEN matched12roots24windows eacharm;4x1000updates',flush=True)

if __name__=='__main__':main()
