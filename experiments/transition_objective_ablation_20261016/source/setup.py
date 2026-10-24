from config import *
import shutil,ast
assert not E.exists(),E
for name in ['source','configs','models','results','logs']:(E/name).mkdir(parents=True)
R.mkdir(parents=True,exist_ok=False)
for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
n=0
for old in [SPAN,MECHANISM,DIAG]:
 for p,h in read(old/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p;n+=1
chains=read(SPAN/'configs/chains.json');pairs=read(STUDY/'configs/chain_matching.json')['root_pairs'];chains['old_AB']={r['old_root']:r['old_pairs']for r in pairs};save(E/'configs/chains.json',chains)
paths=[]
for seed in [0,1]:
 old=dict(np.load(SPAN/'configs'/f'schedule_seed{seed}.npz'));p=E/'configs'/f'schedule_seed{seed}.npz';np.savez_compressed(p,single=old['single'],multi=old['span3']);paths.append(p)
heads=read(SPAN/'configs/protocol.json')['query_heads']
protocol={'frozen_utc':now(),'scope':'Finite transition objective ablation only. No fresh data, no perception/architecture/query changes, no automatic D or adoption.',
 'arms':['mixed','latent'],'seeds':[0,1],'updates_per_run':1000,'initial_transition':str(REP/'models/canonical_matched_seed0/update1000.pt'),'query_heads':heads,
 'single':'Exact prior1000x64 schedules perseed,48old+12newnormal+4newheldclosed,identical botharms.',
 'multi':'Fixed3step baseline only:every4updates,48starts,12ABnormal roots eachwindow(open,lift,place)/(lift,place,close)twice,localdepth2/3 eachweight.25. No simultaneous span intervention.',
 'objectives':{'mixed':'Each target F+0.25Z, same feature weighting as baseline','latent':'Each target alpha*Z, same target/root/span weights; alpha frozen by initial AB gradient calibration','Z':'raw4x64 latent MSE, no query normalization','forbidden':'no proposition/GT label loss'},
 'calibration':{'data':'First32update indices0..31 of each fixedseed0/1 schedule;64ABbatches,16include multistep; assert every used pair split=train','initial':'same fixed initialtransition, no optimizer updates','gradients':'L2 norm of full transition parameter gradient, before clipping, actual scheduled combined single+multi objective','formula':'one shared alpha=median(norm(grad(F+.25Z)))/median(norm(grad(Z))) over64batches','fixed_after':'freeze alpha and raw calibration records/hashes before training; no adaptation and no C involvement'},
 'optimizer':'AdamW lr1e-4 weight_decay1e-4 clip5, same params/order,seeds and1000updates. Gradient matching does not make Adam moments, direction,clipping,epsilon or later dynamics equivalent.',
 'checkpoints':[0,250,500,750,1000],
 'C_selection':'Shared6newCselect normal full4roots; trueinitial recursive rawZ perh2/h3/h4 weighted.25/.5/.25; minimum andearlier tie. No feature or proposition score selection.',
 'diagnostics':'SameAB12new,12old andC6full4 feature/Z perstep atcheckpoints; exactsingle-schedule frequencyweightedAB F/Z; previousCsingle(.5old37+.5new6)F/Z. Record raw F/Z, never compare scaledobjective magnitudes acrossarms.',
 'gate':{'both_seeds':True,'normal_C_validate':'Eachseed:h3full9>=mixed+1root,h4full9>=mixed+1root,all4boundariescorrect>=mixed+1root (N6).','retention':'Eachseed:Crecursiveh1/h2 and eachCsingle-step may lose atmost1of6; oldABandnewAB all4cumulative may lose atmost1of12 each. Actual-state readout probabilities identical.','C_score':'reported and used for within-armselection, not an extra semantic promotion condition','if_fail':'Stop,noextra tuning or D','if_pass':'Report to parent for review only; do not collect newD until separate decision'},
 'evaluation':'Selectedmodels only onoldAB12normal,newAB12x2,Cselect6x2,Cvalidate6x2;full4andpost-opentrue-state replacement;full9/group/margins/perfact paired andall4cumulative. Existing D not accessed for selection or scoring.',
 'limits':'C repeatedly used development; predicted correctness is not physical task success. Purelatent experiment is candidate only, not adoption.'}
save(E/'configs/protocol.json',protocol)
paths+=list((E/'source').glob('*.py'))+[E/'configs/protocol.json',E/'configs/chains.json',PACK/'inputs.npz',PACK/'pairs.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/canonical_matched_seed0/selected.json',REP/'models/query_seed0.pt',REP/'data/public/normalization.npz',REP/'source/model.py',REP/'source/runtime.py']+[Path(q['path'])for q in heads.values()]
save(E/'configs/protocol_frozen.json',{'frozen_utc':now(),'previous_sealed_files_verified':n,'sha256':{str(p):sha(p)for p in paths}});print('PROTOCOL_FROZEN',n,'prior files',flush=True)
