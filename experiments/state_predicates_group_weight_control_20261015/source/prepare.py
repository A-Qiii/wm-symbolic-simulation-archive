from common_control import *
import ast
def main():
 assert not(E/'configs/protocol.json').exists()
 for n in ['configs','results','logs','models','runs','figures']:(E/n).mkdir(exist_ok=True)
 for f in (E/'source').glob('*.py'):ast.parse(f.read_text())
 save(E/'results/immutable_before.json',{'created_utc':now(),'manifests':verify_prior()})
 cohort=read(PREV/'data/training/public/cohorts.json')['open137'];pub=read(PREV/'data/training/public/pairs.json');assert len(cohort)==137 and all(pub[i]['split']=='train'for i in cohort)
 starts={str(s):{'path':str(PREV/'models'/f'open137_seed{s}'/'update1500.pt'),'sha256':sha(PREV/'models'/f'open137_seed{s}'/'update1500.pt')}for s in [0,1,2]}
 files=list((E/'source').glob('*.py'))+list((PREV/'data/training/public').glob('*'))+list((PREV/'data/training/_private').glob('*'))
 files +=[PREV/'configs/protocol.json',PREV/'configs/dataset_frozen.json',PREV/'results/predictions.npz',PREV/'results/latent_predictions.npz',PREV/'results/evaluation.json',PREV/'delivery_manifest.json',DIAG/'configs/protocol.json',DIAG/'results/training_complete.json',DIAG/'delivery_manifest.json',OLD/'source/common.py',OLD/'source/transition.py',ABC/'results/root_contrasts.json',CB/'roots_manifest.json']+[Path(r['path'])for r in starts.values()]
 upstream=read(PREV/'configs/protocol.json')['frozen_sha256'];hashes={p:h for p,h in upstream.items()if '/candidate/'in p or p.endswith('physical_oracle.py')};hashes.update({str(f):sha(f)for f in files})
 # One exact full batch file is shared by both arms and all seeds.
 schedule=np.tile(np.asarray(cohort,dtype=np.int64),(500,1));np.savez_compressed(E/'configs/shared_fullbatch_schedule.npz',global_indices=schedule);hashes[str(E/'configs/shared_fullbatch_schedule.npz')]=sha(E/'configs/shared_fullbatch_schedule.npz')
 history={'original_training':{'initialization':'fresh per seed residual zero-last-layer initialization','updates':1500,'batch_size':16,'presentations_total_each':24000,'new5_presentations_each':175,'optimizer':'fresh AdamW at start; no reset within run','final_five_fit_each_seed':'0/5'},'previous_diagnostic_joint_arms':{'initialization':'warm start from original open137_seed0 final1500 weights','additional_updates_each':500,'weight_trajectory_updates':2000,'batch_size':137,'presentations_additional_each':68500,'new5_additional_presentations_each':500,'new5_cumulative_presentations_each':675,'optimizer':'reset AdamW zero state, no old momentum','original_weight_final_five_fit':'5/5','equal_group_final_five_fit':'5/5','difference_from_original':'More optimization and exposure, full batches, warm start, reset optimizer; cannot attribute improvement to weighting alone.'},'this_control':{'start':'original open137_seed0/1/2 final1500 checkpoints, not prior diagnostic final500 models','paired_same_start_optimizer_batches_budget':True,'new_updates_total':3000,'additional_updates_each':500,'trajectory_updates_each':2000,'full_batch':137,'new5_additional_presentations_each':500,'optimizer':'same reset AdamW zero state for every run','sole_between_arm_change':'old132/five group weights132:5 versus1:1'}}
 save(E/'configs/history_comparison.json',history)
 q={'created_utc':now(),'scope':'Three-seed paired loss-group-weight control, development evaluation only; no D model forward, no new data, no multistep, no candidate promotion.',
 'seeds':[0,1,2],'arms':['uniform','equal_groups'],'starts':starts,'updates_each':500,'total_update_cap':3000,'batch_size':137,'train_global_ids':cohort,'new_open_global_ids':cohort[132:],'original132_global_ids':cohort[:132],
 'schedule':str(E/'configs/shared_fullbatch_schedule.npz'),'schedule_sha256':sha(E/'configs/shared_fullbatch_schedule.npz'),'presentations_each':68500,'presentations_per_example_each':500,
 'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.0001,'clip_norm':1,'handling':'fresh zero state for both arms, prior optimizer not loaded'},
 'loss':'same mean3class-group weightedBCE + .025*objectSmoothL1 + .025*sceneSmoothL1',
 'class_weights':read(PREV/'configs/protocol.json')['class_weights_common_to_both_cohorts'],
 'arm_weights':{'uniform':{'old132':132/137,'new5':5/137},'equal_groups':{'old132':.5,'new5':.5}},
 'implementation':'Both arms use the same per-group loss computation and weighted sum; only these two scalar group weights differ. All137 masks valid, so uniform is equivalent to the original whole137 loss, verified before first update.',
 'architecture':'unchanged76224 parameter LatentTransition; all image encoders, GRUs, normalization, reader heads, thresholds frozen',
 'selection':{'comparison_checkpoint':'final500 only','representative_per_arm':'seed0 preselected','all3seeds_reported':True,'C_usage':'Only after all3000 optimizer updates finished, no C selection, no extension','training_observation_steps':[0,25,100,250,500]},
 'stop':'Exactly500 updates per run; stop and retain logs only for nonfinite loss/gradient or resource/safety failure. Never add steps, seeds, arms, data, or choose a best checkpoint to pass.',
 'development_sets':{'C_old':24,'C_state':18,'C_drawer':12,'C_combined':54},'paired_sets':{'C_state':6,'C_drawer':6},
 'criteria':{'stable_weight_advantage':'All3 equal_groups seeds improve C_drawer both-sides-full7 root count strictly over same-seed uniform, with no Cold or Cstate all7 decrease versus uniform or starting model.',
 'complete_restoration_per_arm':'All3 seeds: originalAB132/132, new_open5/5, Cold24/24,Cstate18/18,Cdrawer12/12, bothstate anddrawer paired6/6.',
 'optimization_alone':'If uniform improves against starting model, extra optimization under fixed fullbatch/reset protocol is sufficient for that observed improvement; do not credit weighting for shared gains. This does not isolate extra steps from fullbatch/reset changes versus the historical training.',
 'interpretation':'Report all paired differences, ties and regressions. Latent metrics separate, no inference of full world-model repair. Reused C is development, not independent generalization.',
 'future_validation':'Only propose, never collect. If one or both arms completely restore across3seeds, a fresh independent-test proposal may be justified. If tied, uniform seed0 is the prespecified simplest proposal, no claim of weight advantage. If equal_groups alone restores, use preselected equal_groups seed0 in a proposal. If neither restores, no independent-test proposal.'},
 'readout_rule':'All7 valid facts, fixed.5 thresholds, multiplepositiveheld=>UNKNOWN countswrong; keep known reader errors in main denominator and report clean subsets separately.',
 'latent_metrics':['object_MAE','scene_MAE','all384_MAE','raw_SmoothL1','abs_gt1','abs_gt1p1','abs_gt2','max_abs'],
 'first_error':'Preselected equal_groups_seed0 first C error; if any, saved actual-next latent/readout single replacement only; no corrective updates.',
 'known_limits':'Five posthoc matched AB preparation pairs, colour2/2/1,A/B3/2; same previously inspected C; warm start from trained weights. Fullbatch500 not equivalent in exposure to historical minibatch500.',
 'frozen_sha256':hashes}
 save(E/'configs/protocol.json',q)
 for p in list((E/'source').glob('*.py'))+list((E/'configs').glob('*')):p.chmod(0o444)
 (E/'HISTORY_COMPARISON.md').write_text('''# 原训练、上一轮诊断与本轮对照的区别

| 项目 | 原1500步训练 | 上轮联合诊断 | 本轮三种子配对对照 |
| --- | --- | --- | --- |
| 起始参数 | 每种子全新初始化 | 已训练open137_seed0第1500步 | 各自open137_seed0/1/2第1500步；同种子两组完全相同 |
| 新增梯度更新 | 1500 | 每组500 | 每组500，共6组3000 |
| 权重经历累计更新 | 1500 | 1500+500=2000 | 1500+500=2000，不接上轮诊断最终模型 |
| 批次 | batch16小批次 | 全批137 | 同一全批137、同一顺序，重复500次 |
| 每条新增样本展示 | 175次 | 额外500次，累计675 | 额外500次，累计675 |
| 每运行样本展示总量 | 24000 | 额外68500 | 额外68500 |
| 优化器 | 开始时新建，运行中未重置 | 从旧权重开始，但AdamW状态重置 | 两组均重置AdamW，不载入旧动量 |
| 组权重 | 原132:5 | 原132:5或1:1 | 仅此项在两组间不同 |

上轮原权重再优化500步也达到五条5/5。延长优化、每条曝光增加、全批次和优化器重置均区别于原训练，因此不能把相对原1500步的改善直接归因组等权。本轮两组共享这些改变，只比较组权重的额外效果；若两组都改善，共同收益首先归于这套额外优化过程。不能进一步声称已独立证明只有“步数”导致改善。

本轮预算在训练前固定为各500步，最终步比较，3种子全部报告。全部3000步完成后才评估旧C24、状态C18、抽屉C12，不重测D、不增加样本。该文在本轮训练前生成。
''')
 (E/'HISTORY_COMPARISON.md').chmod(0o444)
 print(json.dumps({'protocol_sha256':sha(E/'configs/protocol.json'),'starts':starts,'updates':3000,'batch':137,'five_exposure_each':500},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
