from experiment import *
import numpy as np,importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
FP=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');CN={'red':'红','blue':'蓝','pink':'粉'}
def main():
 q=frozen();out=read(E/'results/blocked_outcome.json');gate=read(E/'results/training_collection_gate.json');ms=gate['matched_pairs'];calls=read(E/'results/collection_call_audit.json');failure=read(E/'results/matching_failure_audit.json')[0];rid=failure['root_id'];traj=next(x for x in read(E/'results/preparation_trajectory_differences.json')if x['root_id']==rid);a=traj['anchor_frame'];t=np.array([(x['frame']-a)/30 for x in traj['rows']]);fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
 axes[0].plot(t,[x['objects']['red']['position_mm']for x in traj['rows']],color='#c62828',label='红块位置差（原持物）');axes[0].plot(t,[x['robot_TCP_position_mm']for x in traj['rows']],color='#7e22ce',label='机器人TCP位置差');axes[0].axhline(1,color='#475569',ls='--',label='冻结位置容差1毫米');axes[0].set_ylabel('同根开/闭两侧位置差（毫米）',fontproperties=FP)
 axes[1].plot(t,[x['objects']['blue']['orientation_deg']for x in traj['rows']],color='#2474b5',label='蓝块姿态差（非目标物）');axes[1].plot(t,[x['objects']['red']['orientation_deg']for x in traj['rows']],color='#c62828',label='红块姿态差（原持物）');axes[1].axhline(.5,color='#475569',ls='--',label='冻结姿态容差0.5度');axes[1].set_ylabel('同根开/闭两侧四元数夹角（度）',fontproperties=FP);axes[1].set_xlabel('相对计分调用的真实模拟时间（秒）；0为拒绝调用',fontproperties=FP)
 for ax in axes:
  ax.axvspan(-2.1,0,color='#16a34a',alpha=.12,label='模型输入的2.1秒历史');ax.axvline(0,color='#172b4d',lw=.8)
  for dt in [-2.1,-1.4,-.7,0]:ax.axvline(dt,color='#16a34a',alpha=.4,ls=':')
  ax.legend(prop=FP,fontsize=9,loc='upper left');ax.grid(alpha=.15)
 fig.suptitle('B_coverage_pink_1：合法准备后的非抽屉姿态未匹配',fontproperties=FP,fontsize=16);fig.text(.5,.015,'红/蓝线按真实物体颜色命名；紫线为机器人。绿色区域及4条竖虚线=实际四帧输入历史；水平虚线=采集前冻结容差。曲线来自私有计分审计，未输入模型。',ha='center',fontproperties=FP,fontsize=10);fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(E/'figures/matching_failure_trajectory.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[m['root_id'].replace('_coverage_','\n')for m in ms];xx=np.arange(6)
 for ax,key,limit,title,unit in [(axes[0],'max_object_position_mm',1,'四帧历史：三物体最大位置差','毫米'),(axes[1],'max_object_orientation_deg',.5,'四帧历史：三物体最大姿态差','度')]:
  yy=[m[key]for m in ms];ax.bar(xx,yy,color=['#16a34a'if m['strictly_matched']else'#dc2626'for m in ms]);ax.axhline(limit,color='#475569',ls='--',label=f'冻结容差{limit}{unit}');ax.set_xticks(xx,labels,fontsize=9);ax.set_ylabel(unit,fontproperties=FP);ax.set_title(title,fontproperties=FP);ax.set_ylim(0,max(max(yy),limit)*1.2)
  for i,y in enumerate(yy):ax.text(i,y+.02*max(max(yy),limit),f'{y:.4f}',ha='center',fontsize=9)
  ax.legend(prop=FP);ax.grid(axis='y',alpha=.15)
 fig.suptitle('6个A/B配对根：5个通过，1个未匹配',fontproperties=FP,fontsize=16);fig.text(.5,.015,'每根含关闭/打开两个相关分支。绿=整套匹配规则通过，红=失败；水平虚线为预先冻结容差。没有删除失败根，也没有追加替补。',ha='center',fontproperties=FP,fontsize=10);fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(E/'figures/matching_all_roots.png',dpi=160);plt.close(fig)
 spec=importlib.util.spec_from_file_location('frozen_render_helper',P/'experiments/state_predicates_drawer_factor_C_20261015/source/render.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);windows={w['window_id']:w for w in read(B/'data_manifest.json')['windows']};figs=[]
 for mm in ms:
  rs=[x for x in calls if x['root_id']==mm['root_id']];im=Image.new('RGB',(1700,1060),'white');helper.text(im,(30,15),f'{mm["root_id"]}：采集审计｜原132冻结模型',32);helper.text(im,(30,67),f'已持{CN[rs[0]["held_color"]]}→请求抓{CN[rs[0]["target_color"]]}。本图不是两组138训练后的结果；本轮训练未开始。',24)
  for drawer,y in [('closed',120),('open',500)]:
   row=next(x for x in rs if x['drawer']==drawer);row=row|{'actual_all7_unchanged':row['unchanged'],'primitive_status':row['primitive']['status'],'p_before':row['current_probability'],'p_after':row['after_probability'],'p_prediction':row['original132_probability'],'prediction':row['original132_prediction'],'candidate_all7_correct':row['original132_prediction_correct'],'before_readout_correct':row['current_readout_correct']};helper.drawrow(im,row,y,windows)
  helper.text(im,(30,880),f'严格匹配：'+('通过'if mm['strictly_matched']else'失败，不能作为严格单因素训练配对')+f'。物体最大位置差{mm["max_object_position_mm"]:.6f}mm / 姿态差{mm["max_object_orientation_deg"]:.6f}°',23)
  helper.text(im,(30,916),'冻结门槛：物体/机器人位置≤1毫米、姿态≤0.5度；全部4帧必须满足。真值和姿态仅用于审计。',23);helper.text(im,(30,953),'p=事实为真的概率；阈值0.5；?=多持物冲突等产生的UNKNOWN；红色预测项表示错或UNKNOWN。',21);helper.text(im,(30,990),'蓝/橙标题表示抽屉闭/开，不是物体类别。保留所有调用，不使用本图证明新增训练条件有效。',21);fp=E/'figures'/(mm['root_id']+'.png');im.save(fp);figs.append(str(fp))
 report=['# 条件覆盖对照实验：配对入口未通过，训练未启动\n',
 '**本轮完成6个全新A/B根、12次合法准备和计分调用，但只有5/6根满足预先冻结的姿态匹配。两组138对训练没有启动，实际优化器更新为0；因此没有三组C的训练效果、潜变量改善或条件覆盖因果收益结论。**\n',
 '阻塞根为`B_coverage_pink_1`（已持红→请求抓粉）：四帧历史中红块位置差最大1.005241毫米，超过1毫米；蓝块姿态差最大0.955904度，超过0.5度。不是仅有0.005毫米的边界问题。机器人和目标粉块仍满足要求，七个事实也一致，但这不足以替代全部物体姿态匹配。\n',
 '## 原定对照与实际执行\n',
 '|项目|采集前固定计划|本轮实际|\n|---|---|---|',
 '|新增根|A/B各3根；三种既有持有→目标组合各2根|6根全部尝试；与545旧布局排重，种子15203000–15203005|',
 '|新增调用|每根关闭/打开，共最多12次；不补根|12/12完成，无重试；准备均成功，计分均拒绝且96步后七事实不变|',
 '|匹配|初始非抽屉状态≤1e-5；4帧历史位置≤1mm、姿态≤0.5°等|5/6根通过；第6根保留为严格因果未覆盖|',
 '|训练组|原132+6关闭 vs 原132+6打开；各138对|未释放训练入口，未组装为最终训练输入|',
 '|随机性与预算|每组seed0/1/2；同初始化、同对应批索引；各1500更新|未生成批计划、未调用优化器；0/9000更新|',
 '|固定项|结构、类别权重、损失、LR、编码器、读出、阈值不变|没有修改；无参数更新|',
 '|模型选择|仅final1500；展示seed0，报告全部3种子；不按C选优|无新模型可选择|',
 '|开发评估|旧C24、状态依赖C18、既有抽屉C12分别报告再合并|未执行新模型C前向；最后一组原本就是用于定位的开发材料|',
 '|独立测试与闭环|不新增C/D，不回测旧D，不闭环、不上传|均未执行；旧D68/84、配对20/36未通过保持不变|\n',
 f'方案SHA256：`{sha(E/"configs/protocol.json")}`。入口政策在采集前写入[protocol.json](configs/protocol.json)的`collection_failure_policy`：必须有6个完成、严格匹配且监督有效的配对，否则停止训练，不能缩减样本、替换失败根或放宽门槛。本轮正是触发该政策，不是训练算法失败或C分数触发提前停止。\n',
 '## 六根匹配结果\n',
 '|根|初始非抽屉最大差|物体位置最大差mm|物体姿态最大差°|TCP位置最大差mm|匹配|\n|---|---:|---:|---:|---:|---|']
 for x in ms:report.append(f'|{x["root_id"]}|{x["initial_non_drawer_max_abs"]:.9f}|{x["max_object_position_mm"]:.6f}|{x["max_object_orientation_deg"]:.6f}|{x["max_robot_TCP_position_mm"]:.6f}|'+('通过'if x['strictly_matched']else'失败')+'|')
 report+=['\n![全根匹配](figures/matching_all_roots.png)\n','横轴是六个布局根，每根含开/闭相关分支；两幅图分别显示四帧历史内三物体最大位置差和姿态差。绿色柱表示整套规则通过，红柱表示失败；水平虚线是采集前冻结容差。四个根完全相同，A粉根小差异在容差内，B粉根两项越限。\n',
 '## 最早出现的有效差异\n',
 '根据已保存真实轨迹，B粉根的蓝块姿态差首次超过0.5度发生于第140帧（复位后4.667秒），此时合法准备处于`above_named_block`，差值0.679780度；差异一直保留到计分历史。红块位置差首次超过1毫米在第346帧（11.533秒、准备后保持开始），当时1.402831毫米。用于预测的四帧为377、398、419、440；不能只看最终一帧或只看持有事实。\n',
 '|计分相对帧|红块位置差mm|蓝块姿态差°|TCP位置差mm|六个非抽屉事实|\n|---:|---:|---:|---:|---|']
 mm=next(x for x in ms if x['root_id']==rid)
 for h in mm['history']:report.append(f'|{h["offset_controls"]}|{h["objects"]["red"]["position_m"]*1000:.6f}|{h["objects"]["blue"]["orientation_deg"]:.6f}|{h["robot"]["TCP_position_m"]*1000:.6f}|一致|')
 report+=['\n![差异轨迹](figures/matching_failure_trajectory.png)\n',
 '横轴是真实模拟时间，相对于计分调用时刻；0秒为拒绝调用。上图红线是红块位置差、紫线是TCP位置差；下图蓝线是蓝块姿态差、红线是红块姿态差。绿色阴影及四条竖虚线对应实际2.1秒四帧输入，横虚线是冻结容差。曲线来自计分审计，未输入模型。\n',
 '两分支合法准备均用248步且一次成功；初始非抽屉差9.2861e-6通过复位要求，但后续物理路径没有保持在配对容差内。保存的逐控制帧接触记录没有发现蓝块与机器人直接接触，因此不能把差异断言为机器人碰撞蓝块；现有数据只证明非抽屉姿态确实随准备过程分离，尚未唯一确定微观原因。没有重放、额外仿真、瞬移、强制复位中间状态或修控制器来消除差异。\n',
 '## 保留的数据与读出\n',
 '所有12次当前读出和12次真实后续读出均正确，全部84个事实在前后两端有效；没有因读出错误筛除样本。公开记录6600帧、419171134字节，24个真实2.1秒窗口；12个采集用原132冻结模型的预测均先于动作写入。采集审计图展示的是**原132模型**，不是两组138训练后的模型。\n']
 for p in figs:report.append(f'- [{Path(p).stem} 双相机及七事实采集审计](figures/{Path(p).name})')
 report+=['\n各图左侧为调用前/96步后静态和腕部相机；右侧为真实标签、当前读出、原132后继预测、后续读出。蓝/橙标题=抽屉闭/开；红色预测数值=错误或UNKNOWN；p为事实概率，0.5阈值固定。图像肉眼难辨的毫米/角度差仍按冻结数值规则判断。详见[逐调用审计](results/collection_call_audit.json)、[逐帧差异](results/preparation_trajectory_differences.json)、[失败定位](results/matching_failure_audit.json)。\n',
 '## 本轮结论与下一步边界\n',
 '本轮不能回答“只补打开条件是否修复C预测”，因为训练尚未发生；也没有训练后潜变量误差/越界对比。既有抽屉C诊断所证明的条件依赖继续有效，但不等于补数据已经有效。当前证据不足以提出新的独立D测试执行。\n',
 '建议保留本轮为入口失败版本。若继续，应另行明确如何获得符合既定匹配要求的合法A/B准备，再预登记新的有限配对试验与预算；不要把本轮容差改宽或删除失败根。若选择继续使用这对不匹配数据，只能另行授权并标记为探索性训练，不能回写本协议或声称严格单因素因果收益。本轮不自动增加采集或训练。\n',
 '源码、根、配置和原始帧都已保留；`source/train.py`、`source/evaluate.py`是准备好但未执行的冻结实现。新模型目录及训练运行目录为空，未生成数据冻结清单、批计划或选型清单。旧D、原训练版本、上一轮C诊断的全部文件在执行前后逐文件哈希和只读权限验证未变。复现路径与日志见[HANDOFF.md](HANDOFF.md)。\n']
 (E/'REPORT.md').write_text('\n'.join(report));save(E/'NEXT_STEP_PROPOSAL.json',{'status':'blocked_scope_decision_required_not_executed','current_gate_failure':{'root':rid,'position_mm':mm['max_object_position_mm'],'orientation_deg':mm['max_object_orientation_deg'],'limits':[1,.5]},'optimizer_updates':0,'next_independent_test_proposed':False,'recommendation':'Keep this failed-entry version sealed. Before a new matched training trial, separately define a bounded legal-preparation reproducibility/collection plan that preserves strict matching; no automatic resampling or threshold changes.','alternative_if_explicitly_authorized':'An explicitly exploratory study may retain the unmatched pair and disclaim strict single-factor attribution, in a new version; never rewrite this frozen protocol.','no_new_collection_or_training_this_round':True})
 handoff=f'''# 本轮交接：训练入口真实阻塞\n\n实验目录：{E}\n数据目录：{B}\n原定6新AB根、12调用、两组138、每组3种子1500更新。实际12准备成功、12refused_nochange，24真实2.1秒窗口，6600公开帧419171134字节；当前/后续读出均12/12正确。\n\n冻结protocol SHA256：{sha(E/'configs/protocol.json')}。新根排除545旧布局/336元数据来源，seed15203000..15203005。\n\n阻塞：B_coverage_pink_1严格匹配失败，其余5根通过。四帧内红块位置1.005241mm>1mm，蓝块姿态0.955904deg>0.5deg；四帧全越限。蓝块姿态首次越限frame140/4.667s/above_named_block；红块位置首次越限frame346/11.533s/preparation_boundary_hold。不是仅0.005mm边界。没有记录蓝块与机器人直接接触，不能擅自归因碰撞；无需重放即可证明当前配对未隔离其他姿态差异。\n\n执行：prepare.py -> worker.py --preflight -> collect.py --phase pilot -> validate_data.py --phase pilot -> collect.py --phase remaining -> validate_data.py --phase full 均完成。extract.py保存training_collection_gate.json后按冻结collection_failure_policy抛AssertionError退出（logs/extract.log完整保留）；没有source修补，也没有数据冻结/组装或优化器调用。随后failure_audit.py与blocked_report.py只读保存材料生成报告。\n\n训练0/9000；没有批索引文件、checkpoint、model_selection_frozen.json；runs/与models/为空。source/train.py和evaluate.py为未执行冻结代码。没有新的C54评估，旧C诊断和D没有重算。无法给训练收益、退化或latent改善结论。原D68/84及20/36未通过不变。\n\n报告REPORT.md包含根表、四帧差异、逐帧曲线和6对双相机审计（仅原132模型），不是训练后的效果图。后续不能直接在封存目录补跑；需要新的明确范围：重新设计合法匹配准备/有限采集，或者另行接受不匹配数据的探索性训练并放弃严格因果声称。本轮不自动扩额或降阈值。\n''';(E/'HANDOFF.md').write_text(handoff);save(E/'results/figure_manifest.json',{'files':figs+[str(E/'figures/matching_all_roots.png'),str(E/'figures/matching_failure_trajectory.png')],'training_effect_figures':0,'labels_explicit_original132_collection_audit':True});print('BLOCKED_REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()
