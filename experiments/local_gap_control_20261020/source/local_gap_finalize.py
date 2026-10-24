from common import *
def main():
 guard();verify();checks=[]
 for root in [O,C,P/'reports/relation_interface_diagnostic_20261019',OLD,P/'reports/rgbd_close_diagnostic_20261020']:
  m=read(root/'MANIFEST.json');assert sha(root/'MANIFEST.json')==read(root/'SEALED.json')['manifest_sha256']
  bad=[r['path']for r in m['files']if not(root/r['path']).exists()or(root/r['path']).stat().st_size!=r['bytes']or sha(root/r['path'])!=r['sha256']]
  assert not bad,(root,bad);checks.append({'root':str(root),'files_verified':len(m['files']),'manifest_sha256':sha(root/'MANIFEST.json'),'unchanged':True})
 fits=[read(S/f'results/fit_{arm}_seed{seed}.json')for seed in [0,1]for arm in ARMS];assert all(r['updates']==1500 and r['parameters']==2490068 for r in fits)
 for seed in [0,1]:
  f=[r for r in fits if r['seed']==seed];assert f[0]['initial_state_sha256']==f[1]['initial_state_sha256'];assert sha(S/f'configs/schedule_seed{seed}.npz')==sha(O/f'configs/schedule_seed{seed}.npz')
 for p,h in read(S/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h
 assert not read(O/'results/analysis.json')['continuation_gate']
 assert not read(C/'results/analysis.json')['relation_confirmation_gate']
 assert not read(OLD/'results/analysis.json')['supports_fresh_test_under_frozen_rule']
 a=read(S/'results/analysis.json');fresh=(S/'results/fresh_evaluation_complete.json').exists();new_transitions=read(S/'fresh_test/data/data_manifest.json')['real_transitions']if(S/'fresh_test/data/data_manifest.json').exists()else 0
 assert(S/'results/report_complete.json').exists()
 assert read(S/'results/scoring_execution_receipt.json')['all_analysis_values_equal_except_timestamp']
 assert len(list((S/'results').glob('predictions_*.npz')))==(12 if fresh else 8)
 save(S/'results/final_verification.json',{'utc':now(),'old_sources':checks,'fits':fits,'same_parameters_and_initialization_within_seed':True,'schedules_equal_original':True,'design_hashes_unchanged':True,'four_checkpoint_hashes_unchanged_after_evaluation':True,'coarse_tensors_identical_to_previous_geometry':all(r['identical']for r in read(S/'results/coarse_reproduction.json')['checks']),'fresh_independent_evaluation_completed':fresh,'new_physical_transitions':new_transitions,'training_updates':6000,'CD_accesses':0,'uploads':0,'old_training_gate':False,'old_confirmation_gate':False,'previous_RGBD_gate':False,'full_guide_acceptance_claimed':False})
 (S/'README.md').write_text('''# 可见局部边界与粗网格输入对照

主报告：reports/REPORT.html。图表：comparison.png/svg、confusion.png/svg、visibility_projection.png。每图的颜色、坐标、方向、对象和分母在主报告解释。

两组同为Object结构，2,490,068参数，同原训练数据、批次顺序及每种子1500更新。只有32维场景输入从双相机粗网格改为可见对象周围通用边界距离/缺失描述；原16维对象几何和原冻结特征完全相同。粗网格两种子最终55个模型张量逐字节复现上一轮geometry模型。

冻结设计：configs/protocol.json、design_frozen.json；训练终点：models_frozen.json。原开发已用于假设选择，所有开发结果为探索证据。训练前完成可见性审计；上层对象外部前向边界可见，下层常被容器遮挡，不以GT填补或声称完整间隙可见。

评分：results/analysis.json、group_scores.json、close_case_metrics.csv、root_metrics.csv、retention_opportunities.csv、Held_retention.json、first_errors.json、paired_family_intervals.json。完整失败保留，没有第二轮调参。后续独立测试仅能按冻结筛选触发；是否发生以analysis.json及SEALED.json为准。

旧实验与原指南门槛均不修改；本轮不等于完整指南或规划验收。验证：results/final_verification.json；清单：MANIFEST.json、SEALED.json。所有源码/配置/缓存/检查点/日志按目录保留，封存后只读，后续实验须建新目录。
''')
 files=[{'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(S.rglob('*'))if p.is_file()and p.name not in ['MANIFEST.json','SEALED.json']]
 save(S/'MANIFEST.json',{'utc':now(),'root':str(S),'files':files});elapsed=(time.time()-read(S/'configs/launch.json')['start_epoch'])/60
 seal={'utc':now(),'status':'FIXED_COMPARISON_COMPLETE_WITH_FRESH_TEST'if fresh else'FIXED_COMPARISON_COMPLETE_NO_FRESH_TEST','manifest_sha256':sha(S/'MANIFEST.json'),'manifested_files':len(files),'manifested_bytes':sum(r['bytes']for r in files),'elapsed_minutes':elapsed,'within40_minutes':elapsed<=40,'fresh_test_completed':fresh,'development_trigger_passed':a['supports_independent_test'],'training_updates':6000,'new_physical_transitions':new_transitions,'old_gates_unchanged_and_false':True,'full_guide_acceptance':False}
 save(S/'SEALED.json',seal)
 for p in S.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in sorted([p for p in S.rglob('*')if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 S.chmod(0o555)
 receipt={'directory':str(S),'report':str(S/'reports/REPORT.html'),**seal};save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/local_gap_control_delivery.json'),receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
