from pathlib import Path
import sys,difflib,json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/interaction_latent_repair_20261016';A=P/'experiments/interaction_latent_repair_followup_audit_20261016';R=P/'reports/interaction_latent_repair_followup_audit_20261016'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)
def main():
 R.mkdir(exist_ok=True);old=(E/'source/train.py').read_text();start=old.index(' access=[]\n');end=old.index(' if not a.preflight:',start);new=old[:start]+' access=[]\n from access_guard import install_training_guard\n install_training_guard(P,E,a.arm,a.seed,access)\n'+old[end:];patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/source/train.py',tofile='b/source/train.py'));guard=(A/'source/access_guard.py').read_text();patch+=''.join(difflib.unified_diff([],guard.splitlines(True),fromfile='/dev/null',tofile='b/source/access_guard.py'));(A/'patches/future_training_guard.patch').write_text(patch);save(A/'patches/patch_provenance.json',{'baseline_training_source':str(E/'source/train.py'),'baseline_sha256':sha(E/'source/train.py'),'applied_to_sealed_experiment':False,'patch_sha256':sha(A/'patches/future_training_guard.patch'),'tests':read(A/'results/guard_tests.json')['cases']})
 # Confirm that the generated future source compiles, without executing it.
 compile(new,'future_train.py','exec');compile(guard,'access_guard.py','exec')
 records=read(E/'results/D2_records.json');sem=[]
 for arm in['canonical_control','canonical_matched']:
  for seed in[0,1]:
   rows=[r for r in records if r['arm']==arm and r['seed']==seed and r['condition']=='held_closed']
   for mode in['current','actual_successor','predicted_successor']:
    p=np.array([r[mode]for r in rows]);y=np.array([r['true_before'if mode=='current'else'true_after']['y']for r in rows]);sem.append({'arm':arm,'seed':seed,'readout':mode,'states':len(rows),'holding_all_correct':int(np.all((p[:,:3]>=.5)==y[:,:3],axis=1).sum()),'drawer_class_correct':int((p[:,6:].argmax(1)==y[:,6:].argmax(1)).sum()),'mean_probability_closed':float(p[:,6].mean())})
 save(A/'results/held_closed_breakdown.json',sem)
 refs=read(A/'results/reference_diagnosis.json');drift=[]
 for arm in['canonical_control','canonical_matched']:
  rows=[r for r in refs if r['arm']==arm and r['reference_physically_completed']];drift.append({'arm':arm,'physically_completed_reference_roots':len(rows),'full_reference_score_below_08':sum(r['reference_predicted_final_score']<.8 for r in rows),'final_recursive_latent_MSE':float(np.mean([r['reference_steps'][-1]['recursive_latent_MSE']for r in rows])),'final_single_step_from_real_before_latent_MSE':float(np.mean([r['reference_steps'][-1]['single_step_latent_MSE']for r in rows]))})
 save(A/'results/multistep_summary.json',drift)
 sr=[[r['arm'],r['seed'],r['readout'],f"{r['holding_all_correct']}/12",f"{r['drawer_class_correct']}/12",f"{r['mean_probability_closed']:.4f}"]for r in sem]
 rr=[]
 for r in refs:
  if r['arm']!='canonical_matched':continue
  chosen='→'.join(x['skill'].replace('lift_','lift ').replace('_block_table','').replace('_drawer','').replace('_in','')for x in r['chosen_search']['plan']);rr.append([r['root_id'].replace('D2_place_repair_',''),'是'if r['current_all_valid_correct']else'否',chosen,f"{r['chosen_search']['score']:.4f}",len(r['reference_sequence']),f"{r['reference_predicted_final_score']:.4f}",'成功'if r['reference_physically_completed']else'open失败，后续未到达'])
 text=f'''# 封存后审计补充：关键断点与未来访问防护

本文件补充而不改写已封存的本轮实验。新增工作仅为已存D2的推理诊断、访问策略测试与校验包装器；新增训练0、新物理调用0、新测试集0，模型与0.8阈值未改。原训练源、`training_frozen.json`、`before_D2_frozen.json`及1041文件交付清单全部保留原哈希。

## 关键结论

连续后果误差约减半是真实局部改善；它没有消除错误语义，且完整闭环仍两臂3/12、关闭初态0/6。**正确参考序列的最终评分并不低**：实际完成的11条参考链在两臂中的预测终点均超过0.8。关键断点是错误短序列也被判为高分，搜索在较浅层第一次达阈值便返回，未走到需要更多步骤的正确参考链。

## 持物关闭place：连续改善与错误语义同时存在

此处12个实际调用均因抽屉关闭而拒绝，真实后继仍应持物、抽屉关闭。表中每种读出分别来自实际前态、实际后态、预测后继；持物一栏要求三个Holding同时正确，抽屉一栏要求三类正确，分母均12。概率均值不是准确率。

{table(['模型','种子','读出来源','Holding全对','抽屉分类正确','平均P(关闭)'],sr)}

匹配臂的目标In假阳性由对照12/12降到seed0的6/12、seed1的8/12；持物正确也稍有改善。但预测后继的抽屉正确仅0/12与1/12，seed1对照原有7/12反而下降。因此完整九命题仍0/12。真实后继自身的抽屉读出只有3/12、7/12，说明冻结表示/查询也不可靠；预测后继还在此基础上进一步偏离。不能因为连续观测MSE下降就认定拒绝的符号效果已学会。

## 当前读出 → 预测后继 → 计划选择

以下是匹配臂seed0、每根初始状态的存档搜索。参考链起点与闭环起点的双RGB及本体数组逐项完全相同；复用已缓存D2输入，当前概率重算差小于1e-4。参考链来自预声明评估，仅在这份事后诊断里评分，没有回灌原搜索。

{table(['根','当前全部命题正确','实际选中计划','选中分数','参考链步数','参考终点分数','实际参考执行'],rr)}

具体例子red_0：当前读出正确；搜索在深度3选择lift→place→close，评分0.9970，却省略了打开抽屉。正确open→lift→place→close的完整预测分数0.9975，并不低，但需要深度4。搜索已经在深度3返回；该正确链在深度3的未完成前缀排名117，也排不进32宽beam。blue_0/blue_2/red_2/pink_0有相同早停现象。pink_2参考open本身真实执行失败，不能把那条链称为实证可执行成功。

blue_3初始读出全对，错误place→place在深度2得0.9399，正确3步链得0.9976，却未被等到。pink_1/3还叠加当前读出错误，单步place分别得0.8398/0.9870，搜索在深度1就停止。由此应区分：错误后继使坏短计划获得高分；当前终点分数用于前缀排序和首次达阈值停止又使需要更多步骤的链失去机会。这里没有改beam、阈值或停止规则，也没有验证任何搜索补丁能提高完成率。

原单项替换诊断仍成立：匹配臂9个失败段中7段，在定位的首个有影响错误处当前全命题是正确的；替换该步为实际观测后继隐状态后，保持后续动作和query固定，原计划不再达标。剩余失败不能全部归给冻结query，也不能据此唯一隔离所有组件。

## 多步漂移

{table(['模型','实际完成参考根','参考终点低于0.8数','递归推演最终latent MSE','从真实前态预测最后一步MSE'],[[r['arm'],r['physically_completed_reference_roots'],r['full_reference_score_below_08'],f"{r['final_recursive_latent_MSE']:.4f}",f"{r['final_single_step_from_real_before_latent_MSE']:.4f}"]for r in drift])}

只统计11条实际完成参考链，失败且未执行的pink_2后续不伪造真值。两臂递归终点误差均约为从真实前态只推最后一步的2.4倍，支持多步误差累积；它并非每一步必然单调增加。匹配臂终点平均递归误差0.3168高于对照0.2969。因此本轮只改善特定拒绝条件，不能宣称多步稳定性提高。该数值是同一seed0固定latent坐标下的MSE，不是物理成功率。

## 训练隔离守卫的明确限制与未来补丁

原守卫只按路径子串阻止`/_private/`、`/data/D2/`、`/data/D_raw/`和`/data/closed_loop/`，没有覆盖本轮`/data/D2_raw/`或首版旧D的某些公共路径。原报告“拒绝测试路径”的表述应理解为**部分路径过滤**，不应称为完备隔离。本轮源代码、配对数据、时间冻结和独立审计未发现实际泄漏；这一事实与守卫存在缺口必须分开。

后续补丁单独位于`patches/future_training_guard.patch`，从原训练源生成，但**没有应用到本轮源**。它采用显式AB/C训练文件允许清单，对项目数据、实验和运行目录中其他读取默认拒绝；解析真实路径并检查原路径，阻止`..`和软链接逃逸；限制可写模型/报告位置；记录项目内允许访问。D2根声明仅作为允许的冻结配置元数据，不允许D2观测。

19项测试已通过，包括D2_raw、归一化D2、旧D公共文件、私有标签、字节路径、目录穿越、软链接越界、覆盖输入、跨臂写入与真实`sys.addaudithook`拦截。生成的未来训练源码和守卫模块编译通过。未为测试启动新的训练；该补丁在未来使用前仍需在新目录做兼容性预检。Python审计钩子不是OS安全沙箱，原生I/O、继承描述符和事先复制到未保护目录的文件需更强的进程级隔离；不作完备安全保证。

## 视频、路径与最终校验

84段视频均逐段解码核对，共67,140真实控制帧，30fps；全部241次原生调用的控制预算、96步保持及无参数place接口通过检查。代表成功与失败视频、全部24段自主闭环链接、图中颜色/轴/条件说明见[主图集](../interaction_latent_repair_20261016/index.html)。没有只保留成功片段。

原交付后的`source/verify.py`漏了`import sys`，在校验新文件及冻结清单之后、调用旧交付验证器时发生NameError。独立`source/verify_sealed.py`仅向该模块注入sys后调用原校验逻辑，未改原文件。包装校验通过：新1041文件229,149,376字节、此前复核51文件及首版49,861文件均保持封存哈希。原始错误没有掩盖或伪称首次校验成功。

运行入口：
```
source {P}/env.sh
python {A}/source/verify_sealed.py
python {A}/source/test_access_guard.py
```

本轮已完成，无活跃训练或采集任务；不再启动新训练、新测试集或D2调参。原HULC完整指南仍未完成。
'''
 (R/'AUDIT_ADDENDUM.md').write_text(text)
 (R/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>最终交付与审计补充</title><style>body{font-family:system-ui;max-width:900px;margin:48px auto;line-height:1.8;color:#163047}a{color:#1b67a1}</style><h1>最小转移修复已封存</h1><p>连续后果误差约减半；闭环两臂仍3/12，关闭起点均0/6。正常动作语义预测退化，本轮未修好主任务。</p><p><a href="AUDIT_ADDENDUM.md">最终审计补充：正确参考链评分、多步漂移与守卫限制</a></p><p><a href="../interaction_latent_repair_20261016/REPORT.md">封存实验完整报告与分母</a></p><p><a href="../interaction_latent_repair_20261016/index.html">真实连续视频与图表</a></p><p><a href="../../experiments/interaction_latent_repair_followup_audit_20261016/patches/future_training_guard.patch">单独的未来防护补丁（未应用于本轮）</a></p><p>新训练0，新物理执行0，新测试集0；所有原冻结文件保持原哈希。</p></html>')
 files=[p for b in[A,R]for p in b.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.name!='audit_delivery_manifest.json'];save(A/'audit_delivery_manifest.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sealed_experiment_manifest_sha256':sha(E/'configs/delivery_manifest.json'),'sha256':{str(p):sha(p)for p in sorted(files)},'original_frozen_sources_modified':False});print('AUDIT_ADDENDUM_COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
