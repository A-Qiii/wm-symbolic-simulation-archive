from common import *
import stat
def main():
 guard();verify();checks=[]
 for root in [O,C,P/'reports/relation_interface_diagnostic_20261019']:
  m=read(root/'MANIFEST.json');assert sha(root/'MANIFEST.json')==read(root/'SEALED.json')['manifest_sha256'];bad=[r['path']for r in m['files']if not(root/r['path']).exists()or(root/r['path']).stat().st_size!=r['bytes']or sha(root/r['path'])!=r['sha256']];assert not bad,(root,bad);checks.append({'root':str(root),'files_verified':len(m['files']),'unchanged':True})
 fits=[read(S/f'results/fit_{arm}_seed{seed}.json')for seed in [0,1]for arm in ARMS];assert all(x['updates']==1500 and x['parameters']==2490068 for x in fits)
 for seed in [0,1]:
  q=[r for r in fits if r['seed']==seed];assert q[0]['initial_state_sha256']==q[1]['initial_state_sha256'];assert sha(S/f'configs/schedule_seed{seed}.npz')==sha(O/f'configs/schedule_seed{seed}.npz')
 for p,h in read(S/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h
 for p,h in read(S/'configs/scoring_frozen.json')['sha256'].items():assert sha(p)==h
 a=read(S/'results/analysis.json');fresh=(S/'results/fresh_evaluation_complete.json').exists();new_transitions=read(S/'fresh_test/data/data_manifest.json')['real_transitions']if(S/'fresh_test/data/data_manifest.json').exists()else 0
 save(S/'results/final_verification.json',{'utc':now(),'old_sources':checks,'fits':fits,'all_four_fixed1500_verified':True,'identical_parameter_counts':True,'identical_initialization_within_seed':True,'old_data_membership_and_schedules_unchanged':True,'four_checkpoint_hashes_unchanged_after_evaluation':True,'fresh_independent_evaluation_completed':fresh,'new_physical_transitions':new_transitions,'new_training_updates':6000,'CD_accesses':0,'uploads':0,'old_training_fit_gate':False,'old_relation_confirmation_gate':False})
 (S/'README.md').write_text('''# RGB-D 可观测几何输入对照

主报告在 reports/REPORT.html；图表 reports/comparison.png、comparison.svg。固定设计 configs/protocol.json、design_frozen.json，固定评分 configs/scoring_frozen.json。

两组 Object 模型各 2,490,068 参数、1500 更新、两种子；同种子初始化及数据顺序相同。baseline新增投影接收已有冻结特征，geometry接收公开RGB-D几何。原15维机器人状态已经输入前端，新支路不重复加入原始机器人状态。仿真实例掩码辅助定位条件保留；不使用私有物体姿态、接触、Held/In、执行结果或未来观测作为模型特征。

原训练/开发划分不变。历史与上轮已诊断确认集只作探索和开发证据，未加入训练。所有原封存文件与门槛不变；本轮不等于完整指南或规划验收通过。

机器可读结果：results/analysis.json、summary.csv、family_metrics.csv、close_metrics.csv、retention_opportunities.csv、Held_retention.csv、changed_cases.json。输入来源与缺失分类：input_source_audit.json、geometry_missing_*.json、depth_visibility_breakdown_*.json。执行及复核：logs/、results/final_verification.json。

源码、配置、缓存、固定检查点、完整失败结果均保留。目录封存后只读；后续实验须建新目录，不能覆写本轮或旧实验。所有图表颜色、坐标、条件和分母在HTML中说明。
''')
 files=[{'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(S.rglob('*'))if p.is_file()and p.name not in ['MANIFEST.json','SEALED.json']]
 save(S/'MANIFEST.json',{'utc':now(),'root':str(S),'files':files});elapsed=(time.time()-read(S/'configs/launch.json')['start_epoch'])/60
 seal={'utc':now(),'status':'BOUNDED_COMPARISON_COMPLETE_WITH_FRESH_TEST'if fresh else'BOUNDED_COMPARISON_COMPLETE_NO_FRESH_TEST','manifest_sha256':sha(S/'MANIFEST.json'),'manifested_files':len(files),'manifested_bytes':sum(x['bytes']for x in files),'elapsed_minutes':elapsed,'within45_minutes':elapsed<=45,'fresh_test_completed':fresh,'fresh_test_development_trigger_passed':a['supports_fresh_test_under_frozen_rule'],'training_updates':6000,'new_physical_transitions':new_transitions,'old_gates_unchanged_and_false':True};save(S/'SEALED.json',seal)
 for p in S.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in sorted([p for p in S.rglob('*')if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 S.chmod(0o555);print(json.dumps({'directory':str(S),**seal},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
