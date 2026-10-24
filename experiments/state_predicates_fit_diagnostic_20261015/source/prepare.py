from common_diag import *
def main():
 assert not(E/'configs/protocol.json').exists()
 for n in ['configs','results','logs','models','runs','figures']: (E/n).mkdir(exist_ok=True)
 assert sha(PREV/'delivery_manifest.json')=='22c74af96e71b2ea870612478fa7e528ed4502e4391060f5dc912493ec8364e3'
 save(E/'results/immutable_before.json',{'created_utc':now(),'manifests':verify_prior()})
 cohort=read(PREV/'data/training/public/cohorts.json')['open137'];pub=read(PREV/'data/training/public/pairs.json');assert len(cohort)==137 and all(pub[i]['split']=='train'for i in cohort)
 files=[PREV/'delivery_manifest.json',PREV/'configs/protocol.json',PREV/'configs/dataset_frozen.json',PREV/'configs/shared_batch_schedule_seed0.npz',PREV/'results/training_and_leakage_audit.json',PREV/'results/single_replacement_diagnostic.json',PREV/'models/open137_seed0/update1500.pt',OLD/'source/transition.py',OLD/'source/common.py',B/'data_manifest.json']
 files+=list((PREV/'data/training/public').glob('*'))+list((PREV/'data/training/_private').glob('*'))
 files+=list((E/'source').glob('*.py'))
 q={'created_utc':now(),'scope':'Training failure diagnosis only; no candidate, no collection, no C/D inference, no multistep. Preserve all prior results.',
 'seed':0,'checkpoint':str(PREV/'models/open137_seed0/update1500.pt'),'checkpoint_sha256':sha(PREV/'models/open137_seed0/update1500.pt'),
 'reason_for_warm_start':'Zero-last-layer fresh initialization is identity and already appropriate for no-change refusals. Start all diagnostic arms at the same actually failed seed0 weights to test recovery, not trivial identity fit.',
 'initial_optimizer':'fresh AdamW same zero state for every arm; prior optimizer not resumed','joint_global_ids':cohort,'original132_global_ids':cohort[:132],'five_global_ids':cohort[132:],
 'five_pair_ids':[pub[i]['pair_id']for i in cohort[132:]],'training_data':'Only existing AB132 plus the five already selected open samples; all labels/masks/latent targets unchanged; C/D excluded.',
 'arms':[
  {'name':'five_only','n':5,'objective':'Original total loss on five open cases, full batch5.'},
  {'name':'joint_original','n':137,'objective':'Original total loss on all137, full batch137; five:132 share5/137:132/137.'},
  {'name':'joint_equal_groups','n':137,'objective':'0.5*original_total_loss(original132)+0.5*original_total_loss(five); full batch137. Only group weighting differs from joint_original.'}],
 'updates_each':500,'max_total_updates':1500,'no_adaptive_arms':True,'eval_updates':[0,25,100,250,500],
 'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001,'clip_norm':1},'class_weights':read(PREV/'configs/protocol.json')['class_weights_common_to_both_cohorts'],
 'loss':'mean3group weightedBCE + .05*.5*(objectSmoothL1+sceneSmoothL1), same reader, masks and targets','architecture':'unchanged76224 parameter residual LatentTransition',
 'stop':'Run exactly500 updates each unless nonfinite loss/gradient or safety/resource failure; preserve partial logs if failure. Never extend for accuracy, never select a best checkpoint. Final500 is only decision checkpoint.',
 'fit_definition':'All five cases have every valid one of seven facts correct at fixed.5 threshold; multiple-held UNKNOWN counts wrong. Continuous loss/probabilities and original132 regression also reported.',
 'interpretation':{'five_fit':'If five_only recovers5/5, network can fit these training cases from the failed checkpoint within this budget, not proof of general information sufficiency.','joint_failure':'Five-only success but uniform-joint failure suggests interference or relative exposure; fullbatch controls are diagnostic and differ from historical minibatches.','weighting_control':'Recovery under equal-group versus uniform joint supports training objective weighting as one actionable factor; assess old132 regression, never promote as candidate.','failure':'Failure even on five does not prove mathematical incapacity; report finite-budget inconclusive and use verified gradients/loss components to bound explanation.'},
 'gradient_audit':'No optimizer step. Recompute BCE independently, verify labels/masks/raw provenance/actions/slots/batch counts. At original failed seed0 compute per-case and group loss/gradient, finite differences, frozen-reader current/next correctness. Report gradient conflict as local evidence, not AdamW update attribution.',
 'input_audit':'Use existing AB only. Inspect nearest same-action successful input to each failed case; compare target h/global s/other h. Check target output invariance to changing non-target h while own h,s,action fixed. This does not prove global s lacks information.',
 'frozen_sha256':{str(p):sha(p)for p in files}}
 save(E/'configs/protocol.json',q);(E/'configs/protocol.json').chmod(0o444)
 for p in (E/'source').glob('*.py'):p.chmod(0o444)
 print(json.dumps({'protocol_sha256':sha(E/'configs/protocol.json'),'seed':0,'arms':3,'updates_cap':1500,'five_ids':q['five_pair_ids']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
