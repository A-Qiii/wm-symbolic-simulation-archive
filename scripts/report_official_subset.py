from pathlib import Path
import json,collections,statistics,datetime,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/robot/research/projects/wm_symbolic');run=P/'runs/official_standard_subset_20261014'
d=json.loads((run/'results.json').read_text());protocol=json.loads((run/'protocol.json').read_text());assert d['status']=='complete' and len(d['sequences'])==20
videos=json.loads((run/'video_integrity.json').read_text());assert videos['passed']
by=collections.defaultdict(lambda:{'attempted':0,'success':0,'success_controls':[]});prefix=[];attempts=[]
for s in d['sequences']:
 prefix.append(s['successful_prefix'])
 for t in s['subtasks']:
  attempts.append(t);v=by[t['skill']];v['attempted']+=1;v['success']+=int(t['success'])
  if t['success']:v['success_controls'].append(t['controls'])
steps=[t['controls'] for t in attempts if t['success']]
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':d['scope'],'sequences':20,'attempted_subtasks':len(attempts),'successful_subtasks':len(steps),'attempted_subtask_success_fraction':len(steps)/len(attempts),'first_task_success_count':sum(x>=1 for x in prefix),'first_task_success_fraction':sum(x>=1 for x in prefix)/20,'at_least_k_successes':[sum(x>=k for x in prefix) for k in range(1,6)],'average_successful_prefix':statistics.mean(prefix),'success_controls':steps,'success_control_median':statistics.median(steps),'success_control_range':[min(steps),max(steps)],'failure_controls':sorted(set(t['controls'] for t in attempts if not t['success'])),'by_skill':dict(by),'unattempted_planned_subtasks':100-len(attempts),'dual_camera_videos':videos['videos'],'official_real_state_oracle_stopping':True,'main_data_oracle_stopping':False,'main_training_resumed':False,'main_manifest_unchanged':d['main_manifest_sha256_before']==d['main_manifest_sha256_after'],'checkpoint_unchanged':d['checkpoint']['checkpoint_sha256']==d['checkpoint_sha256_after'],'runtime_trainable_HULC_parameters':d['model_trainable_parameter_count'],'main_pause_preserved':d['main_pause_preserved'],'limitations':['Small preselected subset, single seed; not1000-sequence benchmark','Conditional29 attempted subtasks are not independent unbiased skill samples','Red and blue tabletop lift were not reached in this subset; it cannot settle all three user-observed failures','Original release dependency stack was not fully reproduced; loader/pyhash/EGL compatibility is disclosed','Official success stopping is simulator-ground-truth-assisted and cannot be silently adopted in the main method'],'protocol_version':'official_oracle_subset_diag_v1'}
(run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
fig,axs=plt.subplots(1,2,figsize=(12.5,4.3),layout='constrained');x=np.arange(20);a=axs[0];success=np.array(prefix);failed=(success<5).astype(int);untested=5-success-failed
a.bar(x,success,color='#168575',label='Succeeded');a.bar(x,failed,bottom=success,color='#c87636',label='Attempted, failed at360');a.bar(x,untested,bottom=success+failed,color='#cbd2dc',label='Not attempted after failure');a.set_xticks(x);a.set_ylim(0,6.4);a.set_yticks(range(6));a.set_xlabel('Official sequence index (fixed first20)');a.set_ylabel('Number of subtasks in five-skill sequence');a.set_title('Actual sequence progress');a.legend(fontsize=7.5,ncol=1,loc='upper right')
a=axs[1];a.barh(range(len(steps)),steps,color='#168575');a.set_yticks(range(len(steps)),[t['skill'].replace('_',' ') for t in attempts if t['success']],fontsize=8);a.invert_yaxis();a.set_xlim(0,390);a.set_xlabel('Controls until official success (30 Hz)');a.set_title('All nine successful attempts');a.axvline(360,linestyle='--',color='#8a5960',label='360-control upper limit')
for i,v in enumerate(steps):a.text(v+5,i,str(v),va='center',fontsize=8)
a.legend(fontsize=8);fig.savefig(P/'reports/figures/official_standard_subset_20261014.png',dpi=155);plt.close(fig)
names={'rotate_blue_block_right':'蓝块右转','turn_off_led':'关闭LED','lift_pink_block_slider':'从滑门取粉块','open_drawer':'打开抽屉','lift_pink_block_table':'从桌面抬粉块','place_in_slider':'放入滑门区','turn_on_lightbulb':'开启灯泡','push_blue_block_left':'向左推蓝块','rotate_red_block_right':'红块右转','push_pink_block_right':'向右推粉块','push_red_block_right':'向右推红块','push_red_block_left':'向左推红块','rotate_blue_block_left':'蓝块左转','move_slider_left':'滑门向左','turn_on_led':'开启LED','close_drawer':'关闭抽屉','lift_red_block_slider':'从滑门取红块','move_slider_right':'滑门向右'}
rows=''.join(f"<tr><td>{names.get(k,k)}</td><td>{v['success']}/{v['attempted']}</td><td>{', '.join(map(str,v['success_controls'])) or '—'}</td></tr>" for k,v in by.items())
allrows=[]
for s in d['sequences']:
 for j,t in enumerate(s['subtasks']):
  folder=Path(t['run']);links=' · '.join(f'<a href="../{folder.relative_to(P)}/{cam}.mp4">{label}</a>' for cam,label in [('rgb_static','静态相机'),('rgb_gripper','手部相机')])
  allrows.append(f"<tr><td>{s['index']:02d}/{j+1}</td><td>{names.get(t['skill'],t['skill'])}</td><td>{'成功' if t['success'] else '超时'}</td><td>{t['controls']}</td><td>{links}</td></tr>")
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HULC 官方标准子集：20条序列</title><style>body{{max-width:1150px;margin:32px auto;padding:0 24px;background:#f7f9fc;color:#18273b;font:16px/1.7 system-ui,sans-serif}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:28px}}table{{border-collapse:collapse;width:100%;background:#fff}}td,th{{padding:9px 12px;border-bottom:1px solid #dfe5ed;text-align:left}}img{{width:100%}}.box{{padding:16px;border-left:5px solid #367c96;background:#eaf4f8}}small{{color:#526276}}a{{color:#176496}}code{{font-size:14px}}</style>
<h1>官方标准初态的20条序列健康检查</h1><p class="box">已完成20条预先固定序列，实际尝试29个子任务，9个成功（31.0%）。首任务成功6/20（30.0%）；平均连续成功0.45个，完整五任务链0/20。原HULC没有微调，正式采集与世界模型训练保持暂停。</p>
<p>这里使用官方1000条生成序列中的固定前20条，随机种子0，按顺序运行并保留所有失败。它是隔离的D环境诊断子集，不是1000条完整benchmark，不用于主D测试调参。链中一个任务失败，后续任务不再执行；因此29次尝试是有条件产生的样本，不应作为无偏的技能成功率。</p>
<img src="figures/official_standard_subset_20261014.png" alt="20条序列完成情况及全部成功动作的控制步数"><p>左图横轴为预先选定的官方序列编号，纵轴为每条五技能链中的技能数量。绿色是成功，橙色是实际尝试后超时，灰色是因前序失败而未尝试；灰色不能算作执行失败。右图纵轴列出全部9次成功技能，横轴为达到官方成功所用控制步数，绿色条表示实际步数，虚线表示360步上限。相机控制频率30Hz，成功步数51—184，中位数107，即约1.7—6.13秒。两个图说明本次真实运行进度和停止时机，不是与先前C场景实验的因果对照。</p>
<h2>本次到底用了什么停止规则</h2><table><tr><th>协议</th><th>结束规则</th><th>数据用途</th></tr><tr><td>原主方法 v1</td><td>按技能族冻结固定预算：抽屉240，抬取/放置360，其他120；真值只作离线标签</td><td>已保留4215条主数据</td></tr><tr><td>此前视觉停止候选 v2</td><td>可见RGB-D等判断完成或抓错；存在已记录误报，未正式采用</td><td>单独C诊断，不混入主数据</td></tr><tr><td>本次官方诊断</td><td>每技能最多360步；官方真值评分成功即结束。每链复位环境，每技能重置HULC</td><td>仅本次20条标准序列健康检查</td></tr></table>
<p>官方真实状态辅助停止只能用于这个隔离诊断，不能悄悄接入主方法。改变停止机制会改变同一技能的实际终态分布，也就改变世界模型要学的转移。指南第142行要求为改变后的执行器收集新转移；第454行要求对照共用执行条件。旧数据保留原版本，不与新停止协议冒充同一执行器的数据。</p>
<h2>各技能的真实尝试</h2><table><tr><th>技能</th><th>成功/实际尝试</th><th>所有成功所用控制步数</th></tr>{rows}</table><p>20个失败子任务均用满360步。抽屉4/4成功；粉块桌面抬取出现1次并成功，随后放入滑门区成功；滑门内抬取0/4：四条失败日志均未出现目标块与机器人正接触，目标最大上升不足0.13毫米，属于未建立抓取，不能解释成拿起后掉落。这个子集没有运行到红块、蓝块的桌面抬取，不能声称已解释或解决用户看到的全部抓取问题。</p>
<h2>没有微调HULC：此前训练属于另一个模块</h2><p>已完成的6000次试跑更新的是新建世界模型：视觉特征投影、空间Transformer、动作条件编码、RSSM及共享符号、重建和事件头，共11,515,199参数；其训练器根本没有加载HULC。DINOv2只生成冻结特征。HULC当前文件哈希与原下载包及视频记录一致，本次运行可训练参数数为0。</p>
<p>这份检查点的感知编码器只拼接静态和手部图像的64+64维特征。预处理虽然生成8维robot_obs，但proprio为空，它没有作为感知特征送进网络。真实、未归一化的15维robot_obs_raw单独传给解码器，用姿态第3—5索引把TCP动作转成世界坐标动作，转换只做一次；本次每帧断言原始向量传递一致。每30步重新采样计划并清解码器隐状态，按当前官方源码执行。</p>
<h2>对用户看到的三种失败分别说明</h2><p>红块视频（C根1）没有任何方块正接触，TCP距红块中心最近约31.2厘米，支持“未到达目标”；不能直接定位为颜色识别内部模块错误。蓝块视频（C根4）有11帧目标接触、同时也接触红块，目标最大上升约2毫米，支持“接近并尝试夹取，但未形成有效抬取”，未证明曾稳定拿起后掉落。粉紫块视频（C根5）先接触蓝块、随后接触粉块、又接触蓝块；粉块最高上升约2.35厘米，从未达到官方抬取成功。它包含错误对象交互和后续目标扰动，但没有证据保证延长时间即可成功。</p>
<h2>官方入口与本机兼容边界</h2><p>本地HULC版本7fdb09f的评估入口实际导入CALVIN版本fa03f01中的evaluate_policy；本次直接复用其中未修改的rollout函数及标准初态/序列生成器，加载检查点自身配置。控制模型仍用此前严格加载并通过对拍的兼容加载器，没有启动世界模型或视觉停止器。没有声称原封不动复现历史发布环境。</p>
<p>已披露的兼容事项：本机Torch2.9.1+CUDA12.8，对应仓库依赖写的是Torch1.13.1；旧变换路径别名、Resize antialias=False和不参与推理的旧LayerNorm结构按检查点恢复。缺失pyhash时，仅本诊断加载按官方0.9.3源码实现的FNV32兼容层，406个官方测试向量、字符串UTF16/零种子和192初态计算检查均通过。EGL自动探测程序缺失，使用此前验证的CUDA0/EGL0映射；启动失败日志保留。</p>
<p>每条链都保存了预期复位向量、实际复位向量、完整序列化状态和随机数状态。当前官方reset会步进一次物理仿真，且不恢复最后夹爪命令；因此部分记录末位由预期+1保留为-1。这一末位不参与该检查点的感知编码或TCP姿态转换，未据此改写官方复位行为。其他复位差异完整保留供复核。</p>
<h2>全部29次动作的双相机视频</h2><p>58个视频均验证帧数=控制步数+初态1帧，30fps，无删帧或结果筛选。静态相机显示全局目标与机械臂；手部相机显示夹爪附近接触。序列/步骤列给出链中的位置，成功/超时取官方诊断规则。</p><table><tr><th>序列/步骤</th><th>技能</th><th>结果</th><th>控制步</th><th>完整视频</th></tr>{''.join(allrows)}</table>
<p><a href="executor_diagnosis.html">此前C场景配对与停止误报诊断</a> · <a href="progress.html">主实验暂停状态</a></p><small>证据目录：runs/official_standard_subset_20261014；训练与视频只读核验：runs/hulc_readonly_review_20261014.json。统计生成于{summary['utc']}。本轮不决定换模型或微调，不恢复正式训练。</small></html>'''
(P/'reports/hulc_official_baseline_20261014.html').write_text(page)
print(json.dumps(summary,ensure_ascii=False,indent=2))
