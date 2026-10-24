from common import *
def main():
 guard();verify();checks=[]
 for root in [O,C,OLD,P/'experiments/local_gap_control_20261020',P/'reports/relation_interface_diagnostic_20261019',P/'reports/rgbd_close_diagnostic_20261020']:
  m=read(root/'MANIFEST.json');assert sha(root/'MANIFEST.json')==read(root/'SEALED.json')['manifest_sha256'];bad=[r['path']for r in m['files']if not(root/r['path']).exists()or(root/r['path']).stat().st_size!=r['bytes']or sha(root/r['path'])!=r['sha256']];assert not bad,(root,bad);checks.append({'root':str(root),'files_verified':len(m['files']),'unchanged':True})
 for p,h in read(S/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h
 assert read(S/'results/evaluation_complete.json')['no_training'];assert(S/'results/report_complete.json').exists();collection=read(S/'results/collection_complete.json');assert collection['passed']
 rows=read(S/'data/data_manifest.json')['records'];physical=sum(read(Path(r['labels_path']).parent/'result_private.json')['control_samples_including_start']>1 for r in rows);setupcalls=sum(len(r['setup_successes'])for r in read(S/'results/episode_audits.json'))
 save(S/'results/final_verification.json',{'utc':now(),'old_sources':checks,'fixed_model_hashes_unchanged':True,'frozen_design_unchanged':True,'training_updates':0,'task_calls':len(rows),'calls_with_actual_controls':physical,'setup_calls_separately_logged':setupcalls,'additional_hold_controls':0,'CD_accesses':0,'uploads':0,'old_failed_gates_unchanged':True,'no_full_guide_acceptance_claim':True})
 (S/'README.md').write_text('''# 冻结模型关系泛化检验

主报告 reports/REPORT.html；图表 whole_trajectory.png/svg 和 families.png/svg。方案、动作接口/训练顺序审计在 configs/protocol.json 与 results/action_sequence_audit.json。

无训练，引用已封存RGB-D粗网格Object和同容量无新增显式几何Object，两个种子。4个全新A/B布局族、3组H/T、两分支，已见六步恢复链与相同动作集合/初态的已见及未见四步顺序分开评分。全部失败与拒绝保留，不根据模型成绩替换。

模型无hold动作词。Held连续正样本仅来自未修改执行器抓取完成阶段的控制4/8/12；没有添加控制。每窗口独立从同一抓取先验加入观测，不能称为独立保持技能或长时间状态转移测试。主轨迹与probe分开。

所有逐步、整轨、布局族、Held/In保持分母、恢复语义、关闭结果及首次错误均保留在results。图表解释颜色/轴/条件/实际含义，全部源数据、配置和日志可复核。旧门槛不变，本轮不代表原完整指南或规划验收通过。
''')
 files=[{'path':str(p.relative_to(S)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(S.rglob('*'))if p.is_file()and p.name not in ['MANIFEST.json','SEALED.json']];save(S/'MANIFEST.json',{'utc':now(),'root':str(S),'files':files});elapsed=(time.time()-read(S/'configs/launch.json')['start_epoch'])/60
 seal={'utc':now(),'status':'FROZEN_RELATION_GENERALIZATION_COMPLETE','manifest_sha256':sha(S/'MANIFEST.json'),'manifested_files':len(files),'manifested_bytes':sum(r['bytes']for r in files),'elapsed_minutes':elapsed,'within45_minutes':elapsed<=45,'training_updates':0,'episodes':collection['episodes'],'task_calls':len(rows),'calls_with_actual_controls':physical,'setup_calls':setupcalls,'probe_windows':collection['probe_windows'],'additional_hold_controls':0,'old_gates_unchanged_and_false':True}
 save(S/'SEALED.json',seal)
 for p in S.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in sorted([p for p in S.rglob('*')if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 S.chmod(0o555);receipt={'directory':str(S),'report':str(S/'reports/REPORT.html'),**seal};save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/relation_generalization_delivery.json'),receipt);print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
