from pathlib import Path
import json,hashlib,base64,datetime,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/predicate_gap_supplement_20261014';R=P/'experiments/predicate_gap_supplement_20261014';C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def table(h,rows):return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def image(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 m=json.loads((B/'data_manifest.json').read_text());qa=json.loads((R/'results/final_qa.json').read_text());a=json.loads((R/'results/final_anchors.json').read_text());roots=json.loads((B/'roots_manifest.json').read_text());freeze=json.loads((R/'results/collector_freeze_before_remaining_and_test.json').read_text());assert qa['technical_pass']
 for f,h in freeze['source_sha256'].items():assert sha(f)==h
 assert sha(R/'protocol.json')==freeze['protocol_sha256']
 # Audit actual settled roots, including orientation; the reset layout can differ slightly after settling.
 settled={};near=[]
 for root in roots:
  cfg=OmegaConf.load(P/f"vendor/calvin/calvin_env/conf/scene/calvin_scene_{root['environment']}.yaml");names=list(cfg.objects.movable_objects)
  with np.load(B/'roots'/root['root_id']/'reference_settled_root_private.npz') as z:q=dict(zip(names,z['scene_obs'][6:].reshape(3,6)))
  settled[root['root_id']]=np.stack([q['block_'+c] for c in C])
 for i,x in enumerate(roots):
  for y in roots[i+1:]:
   dx=np.linalg.norm(settled[x['root_id']][:,:3]-settled[y['root_id']][:,:3],axis=1);dy=np.abs((settled[x['root_id']][:,5]-settled[y['root_id']][:,5]+np.pi/2)%np.pi-np.pi/2)
   if np.all(dx<=.012) or np.all(dx<=.02) and np.all(dy<=.25):near.append({'left':x['root_id'],'right':y['root_id'],'same_environment':x['environment']==y['environment'],'splits':[x['split'],y['split']],'drawer_initial_m':[x['drawer_reset_m'],y['drawer_reset_m']],'object_position_distances_mm':(dx*1000).tolist(),'same_complete_initial_state':x['environment']==y['environment'] and x['drawer_reset_m']==y['drawer_reset_m']})
 save(R/'results/settled_root_duplicate_audit.json',{'root_count':len(roots),'rule':'Frozen collection rule excludes near layouts within the same native environment. Additional audit compares positions across all environments: all three named object positions <=12mm, or all<=20mm and modulo-pi yaw differences<=.25rad. Report cross-environment matches without changing frozen selection.','near_pairs':near,'cross_split_position_near_pairs':sum(x['splits'][0]!=x['splits'][1] for x in near),'same_native_environment_near_pairs':sum(x['same_environment'] for x in near),'same_complete_initial_state_pairs':sum(x['same_complete_initial_state'] for x in near),'root_replay_max_error':max(max(t['root_replay'].values()) for t in m['trajectories'])})
 assert not any(x['same_environment'] for x in near)
 # Training-ready index is target-only. Raw labels for every entity remain in private audit files.
 target_view=[]
 for r in m['records']:
  lab=json.loads(Path(r['labels_private']).read_text());z=lab['frames'][-1];j=C.index(r['target_color']);truth=z['labels'];target_view.append({'sample_id':r['sample_id'],'root_id':r['root_id'],'trajectory_id':r['trajectory_id'],'split':r['split'],'public_frames':r['public_frames'],'entity':r['target_color'],'supervised_facts':[{'predicate':'held','arguments':[r['target_color']],'value':bool(truth['holding_derived'][j]),'valid':bool(truth['categorical_valid'][j]),'negative_target_visible_all_past_frames':all(any(n>=4 for n in t['objects_audit'][r['target_color']]['visible_pixels'].values()) for t in lab['frames'])},{'predicate':'open','arguments':['drawer'],'value':truth['categorical'][3]==2,'valid':truth['categorical_valid'][3]},{'predicate':'closed','arguments':['drawer'],'value':truth['categorical'][3]==0,'valid':truth['categorical_valid'][3]}],'private_labels':r['labels_private'],'input_forbidden':['sample/root/trajectory ID','split','condition/probe type','true object XYZ','contacts','labels','future video frames','controller success']})
 save(B/'target_supervision_manifest.json',{'role':'Supervision index only; caller must route public frames to model and labels to loss/evaluator separately','other_entity_missing_rule':'No automatic held=false labels for unobserved or nontarget entities in this index. Original simulator truth remains separately preserved.','records':target_view})
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 cats=['held_positive','near_closed_negative','far_closed_negative','near_open_negative','contact_supported_close_command_negative'];labels=['真实拿住','靠近闭合未拿住','远离闭合未拿住','靠近张开未拿住','受支撑＋闭合指令']
 fig,axes=plt.subplots(1,2,figsize=(14,5.2),layout='constrained');x=np.arange(5)
 for i,(s,title,color) in enumerate([('train','训练','#397FB4'),('dev','开发','#DB9146'),('test','测试','#6D9F70')]):
  v=[sum(r['split']==s and r['condition']==c for r in m['records']) for c in cats];axes[0].bar(x+(i-1)*.24,v,.23,label=title,color=color)
  for xx,vv in zip(x+(i-1)*.24,v):axes[0].text(xx,vv+.15,str(vv),ha='center',fontsize=9)
 axes[0].set_xticks(x,labels,rotation=18);axes[0].set_ylabel('独立初始场景数');axes[0].set_ylim(0,14);axes[0].set_title('主要条件覆盖：每个条件每根最多一个窗口');axes[0].legend();axes[0].grid(axis='y',alpha=.2)
 for held,color,label in [(False,'#B87932','未拿住'),(True,'#288657','真实拿住')]:
  for drawer,marker in [(0,'o'),(2,'^')]:
   rows=[r for r in a if r['held']==held and r['drawer_category']==drawer];axes[1].scatter([r['tcp_distance_mm'] for r in rows],[r['width_mm'] for r in rows],c=color,marker=marker,s=45,alpha=.65,label=label+('／抽屉关' if drawer==0 else '／抽屉开'))
 axes[1].axvline(75,color='#687786',ls='--',lw=1,label='靠近阈值75mm');axes[1].set_xlabel('目标物体与夹爪TCP距离（毫米）');axes[1].set_ylabel('实际夹爪开度（毫米）');axes[1].set_title('输入覆盖：靠近或闭合都不能直接当 held');axes[1].legend(fontsize=8,loc='upper right');axes[1].grid(alpha=.2)
 fig.savefig(R/'figures/coverage_and_confounds.png',dpi=155);plt.close(fig)
 # Same-root, nearly same gripper width: external support distinguishes the two labels.
 rootid='A_predicate_gap_blue_closed';chosen=[next(r for r in m['records'] if r['root_id']==rootid and r['condition']==c) for c in ['contact_supported_close_command_negative','held_positive']]
 fig,axes=plt.subplots(2,3,figsize=(12,6.7),gridspec_kw={'width_ratios':[1,1,1.35]},layout='constrained')
 for i,r in enumerate(chosen):
  z=np.load(r['public_frames'][-1]);v=next(x for x in a if x['sample_id']==r['sample_id'])
  for j,cam in enumerate(['static','gripper']):axes[i,j].imshow(z['rgb_'+cam]);axes[i,j].axis('off');axes[i,j].set_title('静态相机' if j==0 else '夹爪相机')
  axes[i,2].axis('off');axes[i,2].text(0,.96,'仍受桌面支撑 → 未拿住' if i==0 else '离开桌面并跟随运动 → 拿住',va='top',fontsize=12,fontweight='bold',color='#B87932' if i==0 else '#288657');axes[i,2].text(0,.75,f"同一初始场景、同一蓝块\nheld = {v['held']}\n夹爪开度 {v['width_mm']:.2f} mm\n距TCP {v['tcp_distance_mm']:.2f} mm\n双侧手指受力 {v['finger_bilateral']}\n外部支撑接触数 {v['support_contacts']}\n抽屉位置 {v['drawer_m']:.4f} m",va='top',fontsize=12,linespacing=1.7)
 fig.suptitle('标签核验实例：几乎相同的闭合开度和很近的距离，held 真值仍然不同',fontsize=14);fig.savefig(R/'figures/support_vs_held.png',dpi=145);plt.close(fig)
 stats=[]
 for s in ['train','dev','test']:
  rr=[r for r in m['records'] if r['split']==s];tr=[t for t in m['trajectories'] if t['split']==s];stats.append([s,len({r['root_id'] for r in rr}),len(tr),len(rr),sum(r['actual_held'] for r in rr),sum(not r['actual_held'] for r in rr)])
 combinations=[]
 for s in ['train','dev','test']:
  rr=[r for r in m['records'] if r['split']==s];combinations.append([s]+[sum(r['actual_held']==h and r['actual_drawer_category']==d for r in rr) for h,d in [(True,0),(True,2),(False,0),(False,2)]])
 percolor=[]
 for c in C:
  percolor.append([c]+[sum(r['target_color']==c and r['condition']==cond for r in m['records']) for cond in ['held_positive','near_closed_negative','contact_supported_close_command_negative']])
 pos=[t['motion_audit']['held_follow_audit'] for t in m['trajectories'] if t['branch']=='grasp'];neg=[t['motion_audit'] for t in m['trajectories'] if t['branch']=='near_closed_empty' and t['motion_audit']['verified']];bad=[t for t in m['trajectories'] if t['branch']=='near_closed_empty' and not t['motion_audit']['verified']];assert len(bad)==1
 motionstats={'held_roots':len(pos),'held_tcp_movement_range_mm':[min(x['tcp_displacement_m'] for x in pos)*1000,max(x['tcp_displacement_m'] for x in pos)*1000],'held_object_follow_relative_drift_max_mm':max(x['relative_drift_max_m'] for x in pos)*1000,'accepted_negative_roots':len(neg),'negative_tcp_movement_range_mm':[min(x['tcp_displacement_m'] for x in neg)*1000,max(x['tcp_displacement_m'] for x in neg)*1000],'accepted_negative_object_motion_max_mm':max(x['object_max_displacement_m'] for x in neg)*1000,'failed_negative_probe':{'trajectory':bad[0]['trajectory_id'],'object_motion_mm':bad[0]['motion_audit']['object_max_displacement_m']*1000,'fixed_quality_limit_mm':3.0}}
 save(R/'results/motion_evidence_summary.json',motionstats)
 allbytes=sum(f.stat().st_size for f in B.rglob('*') if f.is_file());fullframes=sum(t['frames'] for t in m['trajectories']);controls=sum(t['controls'] for t in m['trajectories']);drawerpositions=[x['drawer_m'] for x in a]
 blocks=['<h1>状态读取缺口补采与质量验证</h1><p class="sub">2026-10-14 · 本机物理仿真 · 仅补采，不训练、不规划、不上传</p>']
 blocks.append('<div class="callout"><b>已执行24个独立初始场景、48条轨迹，保存148个四帧窗口。</b><p>“拿住物体且抽屉关闭”新增12个独立根；“闭合夹爪靠近但未拿住”新增23个独立根。数据可读性、分组、可见性和标签证据核验通过；这不代表模型性能已提升。</p></div>')
 blocks.append('<h2>1. 固定范围与划分</h2>'+table(['分区','独立根','完整轨迹','四帧窗口','held正例窗口','held负例窗口'],stats))
 blocks.append('<p>每根配两条轨迹：原有冻结抓取技能，以及固定的空夹爪靠近探测。训练来自A/B，开发来自C，测试来自D；根、位置、颜色、开关状态、姿态、采样条件在执行前固定。同根的两个分支从相同初始状态重放，始终属于同一分区。先完成4个训练根、8条轨迹的试采与目视核验，再按不变规则补齐。没有根据测试模型结果改采集——本轮根本没有运行识别模型。</p>')
 blocks.append(table(['执行元素','具体内容','为什么保留'],[['初始reset','设置三个桌面物体的位置/朝向，以及抽屉0或0.22米；随后24次真实电机控制稳定','reset来源明确；没有把物体瞬移到夹爪中。'],['真实抓取','调用原样 PhysicalOracle.run(lift_COLOR_block_table)','使用冻结技能及其已有重试；未修改控制参数或修复单例。'],['负例探测','调用未修改的 move/hold 等电机原语：闭合后移到物体前55毫米、上25毫米，再横移15毫米','这是预先固定的数据测量轨迹，不是新规划技能，不冒充抓取成功。'],['执行保护','轨迹执行时禁止物体reset、动力学/碰撞修改、附着约束、外力，只允许机器人电机','排除非物理抓取和标签伪造。'],['公开输入','双相机RGB、深度、可见掩码、相机标定、本体感知；四帧全在当前锚点或之前','接触、真值坐标、来源类别和未来视频帧不进入模型。']]))
 blocks.append('<h2>2. 这次填补了哪些组合</h2>'+table(['分区','held＋抽屉关','held＋抽屉开','未held＋抽屉关','未held＋抽屉开'],combinations)+'<p>表内是目标物体的窗口数，不是独立试验数。每个 held 正例来自不同根；同根的多个负例相关。两种抽屉状态都配真实 held、近距离未 held 和远离未 held，避免把抽屉开关或动作来源直接当作 held 标签。</p>'+table(['物体','真实held：独立根','靠近闭合未held：独立根','接触受支撑负例：独立根'],percolor))
 blocks.append(image(R/'figures/coverage_and_confounds.png')+'<p class="caption">左图：横轴是采集条件，纵轴是具备该条件的独立根数；蓝/橙/绿分别表示训练/开发/测试。右图：每点一个窗口，横轴为物体与夹爪TCP的真实距离，纵轴为夹爪实际开度；绿色是真实held、棕色是未held，圆点/三角分别为抽屉关/开。虚线是75毫米近距离定义。右图仅审计数据覆盖，坐标不作为学习输入；同根点不独立。</p>')
 blocks.append('<h2>3. held 标签如何确认</h2><p>原有定义保持不变：两侧手指及其固定接触垫都存在超过0.01牛的受力接触，且物体没有非机器人外部接触。距离近、发送闭合命令、甚至手指夹到物体，都不自动等于 held。逐控制步保存接触与支撑事实，并核验真实运动。</p>'+table(['核验','结果','解释'],[['真实held的跟随运动',f"24/24根通过；夹爪位移{motionstats['held_tcp_movement_range_mm'][0]:.1f}–{motionstats['held_tcp_movement_range_mm'][1]:.1f}毫米；物体相对夹爪漂移最大{motionstats['held_object_follow_relative_drift_max_mm']:.2f}毫米",'物体确实随夹爪抬升，而非靠近或仅闭合。'],['近距离负例的运动',f"23/24根通过；夹爪横移{motionstats['negative_tcp_movement_range_mm'][0]:.1f}–{motionstats['negative_tcp_movement_range_mm'][1]:.1f}毫米；物体最大位移{motionstats['accepted_negative_object_motion_max_mm']:.2f}毫米",'目标持续有外部支撑且未形成held接触；只移动空夹爪。'],['可见性','所有入选负例的目标在四帧中均至少被一个相机看到','未把物体缺失或不可见直接标成未held。'],['标签一致性','原标签与原始受力接触审计未发现冲突','无标签标准修改；原始全实体真值独立保存。']]))
 blocks.append(image(R/'figures/support_vs_held.png')+'<p class="caption">同一根、同一蓝块、同一抓取轨迹。上下两例夹爪开度均约40毫米且TCP很近，但上例仍有桌面支撑，held=False；下例已离开桌面并随夹爪运动，held=True。左/中为真实静态/夹爪相机，右为离线接触核验。颜色表示标签类别，不是模型置信度。</p>')
 blocks.append('<h2>4. 真实双视角样例</h2>'+image(R/'figures/final_contact_sheet.png')+'<p class="caption">三色块各配真实拿住与闭合靠近负例。绿色标题为held正例，棕色为负例；每行左为静态相机，中为夹爪相机，右为独立真值、受力、支撑、距离与开度。图片直接来自保存观测，无合成场景，无模型预测。</p>')
 blocks.append('<h2>5. 未收录与仍缺失的情况</h2><ul><li><b>B环境蓝块、抽屉关闭的负例探测：</b>夹爪运动过程中物体移动3.88毫米，超过冻结的3毫米静止对照阈值；两个近距离窗口未入选。原始视频、控制和标签全部保留。它没有被偷偷换成另一个根，也没有调控制器补齐。</li><li><b>接触后仍受桌面支撑：</b>24条抓取轨迹中只有6条出现稳定四帧窗口；其余18条记为可选条件缺口。三色均有此类样本，但数量少、形状/接触状态覆盖仍窄。</li><li><b>尚未覆盖：</b>更多极近距离抓空、单侧接触/滑落、低高度持物、抽屉半开或阈值附近、长时间遮挡与新物体。当前支持接触负例可削弱“夹爪有开度就是拿住”的捷径，但不能证明所有捷径已消除。</li></ul><p>抽屉使用原几何定义：关闭&lt;0.025米，打开&gt;0.16米，阈值附近±0.005米标为不确定。本轮仅主动覆盖明确关/开两端；保存精确位置、边界距离及可见像素，不伪称已补齐中间状态。</p>')
 blocks.append('<h2>6. 数据质量与文件</h2>'+table(['检查','结果'],[['分组与去重','24个根、48条轨迹无跨分区；148个窗口RGB哈希唯一；592个引用帧中590个RGB帧哈希唯一，两次重复均留在同组，不增加独立样本数。'],['物理初始状态近重复','按采集前冻结的同原生环境规则，无近重复；旧根排除规则已应用。额外跨环境位置检查找到1对C/D近似布局，详见下文。'],['视频完整性',f"96段原始双视角视频全部完整解码，帧数逐条匹配；两相机各{fullframes}帧，对应{controls}次真实控制。"],['只读保护','冻结控制器、标签源码、旧数据清单、旧世界模型与上一轮读取模型哈希一致；采集脚本与筛选规则在测试前后哈希一致。'],['存储',f"数据约{allbytes/1024**2:.1f} MiB；未生成DINO缓存或训练新checkpoint。"],['模型结论','本轮未训练、未推理、未运行规划；质量通过不等于识别性能提高。']]))
 blocks.append('<p><b>跨环境位置重叠的限制：</b>C_predicate_gap_pink_closed 与 D_predicate_gap_red_open 的三物体位置接近。它们是不同原生环境、不同物体几何，抽屉初始状态分别为0和0.22米，也不是同一轨迹或相邻帧；因此不是相同完整初始状态。但不能宣称所有测试位置都与开发位置充分分离。原始记录已标注，未依据这一事后检查改采集或挑换测试根；若下一轮要求严格的跨环境位置外推，应单列这对场景。</p>')
 blocks.append(table(['本机路径','内容'],[[str(B/'data_manifest.json'),'148个窗口与48条完整轨迹；来源、分组、质量缺口'],[str(B/'target_supervision_manifest.json'),'目标held与抽屉命题监督索引；公开输入和私有标签分路，其他不可见实体不自动作负例'],[str(B/'roots_manifest.json'),'采集前冻结的24根及train/dev/test划分'],[str(B/'episodes'),'每轨迹双相机MP4、逐控制步私有标签/接触、reset来源、四帧公开NPZ'],[str(R/'results/final_qa.json'),'质量计数、去重、视频解码、哈希保护和缺口'],[str(R/'source'),'本次采集、核验与报告代码'],[str(R/'figures'),'独立PNG图；本报告已内嵌图像']]))
 blocks.append('<p class="sub">交付已准备在本机。未上传任何新增文件；下一步训练或扩大采集均未启动。</p>')
 css='body{margin:0;background:#f2f5f8;color:#1b2b3b;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1240px;margin:auto;background:white;padding:42px}h1{font-size:29px}h2{margin-top:38px;font-size:23px;border-bottom:2px solid #dde5ec;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{border:1px solid #d7e0e8;padding:10px;vertical-align:top;text-align:left;overflow-wrap:anywhere}th{background:#eef3f7}tr:nth-child(even){background:#fafbfd}img{max-width:100%;display:block;margin:20px auto}.sub,.caption{font-size:14px;color:#536579}.callout{padding:20px;background:#edf6ef;border-left:5px solid #388359}li{margin:12px 0}@media(max-width:700px){main{padding:16px}table{font-size:12px}td,th{padding:6px}}'
 (R/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态读取缺口补采</title><style>'+css+'</style><main>'+''.join(blocks)+'</main></html>')
 summary={'status':'collection_and_QA_complete_no_training','roots':24,'trajectories':48,'windows':148,'history_length':4,'root_split_counts':{'train':12,'dev':6,'test':6},'window_split_counts':{'train':73,'dev':37,'test':38},'held_positive_windows':24,'held_negative_windows':124,'held_closed_roots':12,'held_open_roots':12,'near_closed_negative_roots':23,'near_closed_negative_windows_including_motion_followup':46,'contact_supported_close_command_negative_roots':6,'core_gap_coverage':'Every split has all three colors in held+closed, held+open and near-closed-negative conditions; B-blue-closed native root lacks accepted near-negative clips, while train A-blue-closed covers that color/drawer cell.','label_standard_changed':False,'controller_changed':False,'model_training_or_evaluation':False,'planner_execution':False,'uploaded':False,'report':str(R/'report.html'),'data_manifest':str(B/'data_manifest.json'),'target_supervision_manifest':str(B/'target_supervision_manifest.json'),'retained_gap_trajectory':bad[0]['trajectory_id'],'remaining_execution_blockers':[],'data_limits':['One negative probe exceeded3mm object-stillness threshold','Only6 stable supported-contact closed-command windows','One cross-environment dev/test near-position pair, different native environment/drawer state; no full-state duplicate','Half-open/boundary/long-occlusion/unseen-object coverage absent'],'dataset_bytes':allbytes}
 save(R/'SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
