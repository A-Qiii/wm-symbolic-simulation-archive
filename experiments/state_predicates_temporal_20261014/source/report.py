from pathlib import Path
import json,base64,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from wm_symbolic.vision_cache import VisibleRegistry
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_temporal_20261014'
ARMS=['current_only','short_history','long_history'];NAMES=['当前帧重复4次','原四帧：0.10秒','稀疏四帧：2.10秒'];COLORS=['#8b94a0','#397fb5','#258b72']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table>'
def img(name,cap):return '<figure><img alt="'+html.escape(cap)+'" src="data:image/png;base64,'+base64.b64encode((E/'figures'/name).read_bytes()).decode()+'"><figcaption>'+html.escape(cap)+'</figcaption></figure>'
def main():
 A=read(E/'results/audit.json');D=read(E/'results/evaluation.json');cases=read(E/'results/fresh_cases.json');remain=read(E/'results/remaining_historical_errors.json');conf=read(E/'results/direction_confound.json')
 font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,3,figsize=(14.2,4.7))
 for i,(arm,name,c)in enumerate(zip(ARMS,NAMES,COLORS)):
  vals=[r['modes']['fixed600']['fresh']['held']for r in D[arm]];x=np.arange(3)+(i-1)*.23
  for ax,values in [(axs[0],[v['FP']+v['FN']for v in vals]),(axs[2],[v['known_true_positive']for v in vals])]:
   bars=ax.bar(x,values,.22,color=c,label=name)
   for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.06,str(v),ha='center',fontsize=10)
  axs[1].plot(range(3),[v['BCE']for v in vals],marker='o',color=c,label=name,lw=2)
 for ax in axs:ax.set(xticks=range(3),xticklabels=['种子0','种子1','种子2']);ax.grid(axis='y',alpha=.15)
 axs[0].set(title='held原始错误：18窗',ylabel='错误窗口数（越低越好）',ylim=(0,3.1),yticks=range(4));axs[1].set(title='held概率误差：18窗',ylabel='交叉熵BCE（对数轴，越低越好）',yscale='log');axs[2].set(title='真正held的有效已知输出',ylabel='已知正确数 / 6个正例根',ylim=(0,7),yticks=range(7));fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.5,-.015));fig.suptitle('同一目标时刻、同一模型容量：较长历史在本批新C开发根中更稳定',fontsize=16);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(E/'figures/temporal_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
 # Current public observations, with no private contact/pose information in measurements.
 selected=['C_causal_dev_pink_closed__held_positive','C_causal_dev_red_closed__hard_negative'];labels=['粉块真正held','红块仍有支撑'];examples=[]
 for sid,label in zip(selected,labels):
  q=next(q for q in A['recovered_samples']if q['sample_id']==sid);anchor=q['anchor'];folder=Path(q['temporal_inputs']['long_history'][-1]).parent;reg=VisibleRegistry();rows=[]
  for k in range(anchor-63,anchor+1):
   with np.load(folder/f'{k:04d}.npz')as z:f={n:z[n]for n in z.files}
   reg.update([f]);uid=reg.colour_ids[['red','blue','pink'].index(q['target_color'])];mask=VisibleRegistry.grouped(f['seg_static'])==uid;y,x=np.where(mask);rows.append({'frame':k,'time_seconds':(k-anchor)/30,'tcp_z_m':float(f['robot_obs'][2]),'static_centroid_y_px':float(y.mean()),'public_file':str(folder/f'{k:04d}.npz')})
  examples.append({'sample_id':sid,'label':label,'measurements':rows})
 save(E/'results/public_motion_examples.json',examples)
 fig,axs=plt.subplots(1,2,figsize=(12.5,4.7))
 for example,c in zip(examples,['#258b72','#d77e29']):
  rows=example['measurements'];xx=np.array([r['time_seconds']for r in rows]);zz=np.array([r['tcp_z_m']for r in rows]);yy=np.array([r['static_centroid_y_px']for r in rows]);values=[(zz-zz[0])*1000,yy[0]-yy]
  for ax,v in zip(axs,values):
   ax.plot(xx,v,color=c,label=example['label'],lw=2);ax.scatter(xx[[0,21,42,63]],v[[0,21,42,63]],marker='o',facecolors='white',edgecolors=c,s=50,zorder=4);ax.scatter(xx[-4:],v[-4:],marker='s',color=c,s=20,zorder=5)
 for ax in axs:ax.set(xlabel='距目标时刻的模拟时间（秒；0为当前帧）');ax.grid(alpha=.15);ax.axvspan(-.1,0,color='#397fb5',alpha=.08)
 axs[0].set(title='公共本体状态：末端高度变化',ylabel='相对最早帧的高度变化（毫米，向上为正）');axs[1].set(title='静态相机：目标可见掩码重心变化',ylabel='相对最早帧的图像位移（像素，向上为正）')
 handles=axs[0].get_legend_handles_labels()[0]+[Line2D([0],[0],marker='o',color='gray',markerfacecolor='white',lw=0,label='长窗实际输入4帧'),Line2D([0],[0],marker='s',color='gray',lw=0,label='短窗输入4帧')];fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.015));fig.suptitle('实际可观测运动：末端向上与物体图像上移同时出现，但阶段也与标签混杂',fontsize=14);fig.tight_layout(rect=(0,.09,1,1));fig.savefig(E/'figures/public_motion.png',dpi=160,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(4,5,figsize=(14,11.3));offsets=[63,42,21,3,0]
 for si,(sid,label)in enumerate(zip(selected,labels)):
  q=next(q for q in A['recovered_samples']if q['sample_id']==sid);anchor=q['anchor'];folder=Path(q['temporal_inputs']['long_history'][-1]).parent
  for col,offset in enumerate(offsets):
   with np.load(folder/f'{anchor-offset:04d}.npz')as z:
    for ri,cam in enumerate(['static','gripper']):
     ax=axs[2*si+ri,col];ax.imshow(z['rgb_'+cam],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
     if ri==0:ax.set_title(f't−{offset/30:.2f}秒'if offset else'当前t')
     if col==0:ax.set_ylabel(label+'\n'+('静态相机'if cam=='static'else'腕部相机'),fontsize=12)
 fig.suptitle('同一目标时刻的真实图像：前三列与最后一列构成长窗，倒数第二列说明短窗起点',fontsize=14);fig.tight_layout();fig.savefig(E/'figures/temporal_frames.png',dpi=150,bbox_inches='tight');plt.close(fig)
 unique={}
 for q in remain:
  m=q['metadata'];paths=m.get('public_inputs',m.get('public_frames'));key=(paths[-1],q['predicate'],q['target_colour']);unique.setdefault(key,[]).append(q)
 fig,axs=plt.subplots(len(unique),3,figsize=(12.5,3.6*len(unique)),gridspec_kw={'width_ratios':[1,1,1.3]})
 for row,((path,pred,colour),items)in zip(axs,unique.items()):
  with np.load(path)as z:
   for ax,cam in zip(row[:2],['static','gripper']):ax.imshow(z['rgb_'+cam],interpolation='nearest');ax.axis('off');ax.set_title('静态相机'if cam=='static'else'腕部相机')
  row[2].axis('off');q=items[0];text=[q['metadata']['root_id'],f'{pred}({colour}) 真值={q["truth"]}','旧四帧回归输入（未补长历史）']+[f'种子{x["seed"]}：p={x["p"]:.4f}'for x in items];row[2].text(0,.95,'\n'.join(text),va='top',fontsize=12)
 fig.suptitle('尚未解决的3个旧C场景：保留错误，不把新C改善等同于全接口通过',fontsize=15);fig.tight_layout();fig.savefig(E/'figures/remaining_old_C.png',dpi=150,bbox_inches='tight');plt.close(fig)
 results=[]
 for arm,name in zip(ARMS,NAMES):
  for r in D[arm]:
   q=r['modes']['fixed600'];h=q['historical'];f=q['fresh'];co=q['conditions'];results.append([name,r['seed'],co['hard_negative']['held']['FP'],co['ordinary_negative']['held']['FP'],co['held_positive']['held']['FN'],f"{f['held']['BCE']:.6f}",f['held']['known_true_positive'],f"{h['held']['FP']}/{h['held']['FN']}",f"{h['in_drawer']['FP']}/{h['in_drawer']['FN']}"])
 case_rows=[]
 for sid in ['C_causal_dev_pink_closed__held_positive','C_causal_dev_red_closed__ordinary_negative','C_causal_dev_red_closed__hard_negative']:
  q=next(q for q in cases if q['metadata']['sample_id']==sid);j=q['metadata']['target_index'];case_rows.append([sid.replace('C_causal_dev_',''),bool(q['truth'][j])]+[' / '.join(f"{q['predictions'][f'{arm}_seed{s}_fixed600']['p'][j]:.4f}"for s in range(3))for arm in ARMS])
 oldknown=[r['modes']['fixed600']['historical']for r in D['long_history']]
 parts=['<h1>时间证据诊断：短窗缺少运动，但长窗仍存在阶段混杂</h1><p class="muted">本机 · 2026-10-14 · 既有12个A/B训练根与6个新C开发根 · 3组×3种子×600更新 · 无新增根/独立测试/世界模型/规划</p>',
 '<p class="lead"><b>本轮得到明确方向：</b>原四帧覆盖0.10秒，大多只看到了抓取后的稳定停留；覆盖2.10秒的稀疏四帧补入可观测抬升过程，在本批新C开发根上比当前帧和原四帧更稳定。长窗组3个种子均正确识别全部18窗，held概率误差也逐种子下降。<b>但它还不能证明模型学会了持有关系：</b>54个训练/开发窗口中，末端上移与held真、末端下移与held假完全重合。下一步应打破这个混杂，停止继续加网络或试更多窗口长度。</p>',
 '<h2>1. 时间与重复帧审计</h2>',table(['项目','实测/源码证据','实际含义'],[['物理步进与记录','240Hz物理步进，每个动作执行8子步，每步动作后记录一次：30Hz。','时间为模拟动作时间，不是渲染或训练耗时。'],['原四帧','[t−3,t−2,t−1,t]，首尾相隔3/30=0.10秒。','四张图不等于四个明显不同的动作阶段。'],['恢复轨迹','18条轨迹、5330帧；相邻双相机RGB完全相同的对数为0。','画面不逐位相同，并不保证有足够的任务运动信息。'],['历史1310窗','1309窗有4个不同的双相机RGB组合，1窗有2个；没有全窗相同的情况。','原采集先稳定24步再存根历史，240个根快照也是实际连续帧，未触发补帧逻辑。'],['真实held的运动（训练12根）','0.10秒末端位移中位数0.438mm，静态图像目标重心位移0.103px；2.10秒为83.130mm、20.194px。','短窗主要覆盖稳定停留；长窗包含明显抬升和物体图像运动。'],['历史可用性','54个既定锚点均有至少64步历史，目标在这64帧均至少一个相机可见。','三组使用相同根、相同目标时刻与标签；没有因长窗缺历史而选择性删样本。']]),
 '<p>初版审计曾把“根快照”标记为“填充历史”；检查采集源码中的24步稳定过程后已纠正。初版保留在logs/audit_initial_root_snapshot_misclassification.json，正式审计在协议冻结之前重算，最终结果为零合成填充。</p>',
 '<h2>2. 公平比较的边界与输入</h2>',table(['组别','处理后的4个输入时刻','保持不变的原因'],[['当前帧','[t,t,t,t]','重复同一当前帧仅为对齐GRU展开次数，不增加信息。'],['原四帧','[t−3,t−2,t−1,t]，0.10秒','沿用原输入。'],['长历史','[t−63,t−42,t−21,t]，2.10秒','仍只有4帧，不增网络容量或输入帧数；不是64帧密集模型。']]),
 '<p>恢复的36个训练窗口（12根×困难负例/普通负例/真正held）和18个新C窗口接受上述处理。<b>旧900＋73个训练窗口在三组均使用相同原四帧</b>，作为共同的既有任务训练数据；旧C回归输入也一致。它们的完整RGBD与掩码中间帧未导出，不能伪造长历史。恢复部分没有in正例，直接共同排除全部旧数据又会失去in正例监督。因此本轮是恢复窗口上的时间证据干预，并非全语料单帧与长窗比较。</p>',
 table(['控制项','固定设置','避免的混淆'],[['编码器/模型','DINOv2冻结；原归一化；原675,395参数读出器；每组输入形状完全一致。','更大的模型、更多帧或重新训练视觉骨干。'],['当前帧/标签','54个锚点的末帧特征三组逐项一致；held物理定义和标签未改。','把当前画面或标签变化误当作历史贡献。'],['数据/预算','1009窗/组；相同标签、有效掩码；3种子×600更新×60=36,000呈现/次。','负例数量、正例数量、训练预算不同。'],['随机性/优化','逐种子复用相同批次索引流；同初始化、AdamW与原固定损失权重。','另一个随机抽样过程产生的差异。'],['主要结果','统一第600次更新；原联合开发选模与校准另报。','不同检查点选择造成的比较偏差。'],['停止规则','仅3组×3种子，现已全部完成；不试第二个长窗、不补新根、不修单个失败。','在同一小开发集上无限寻找有效配置。']]),
 table(['已保存且可构造的输入','本轮用途','限制'],[['双相机RGB、可见掩码、像素框','同一冻结DINO和ROI/全局特征；可选任一过去时刻。','注册来自可见像素，接触和仿真物体位置不进入编码器。'],['机器人15维本体状态','同一时间点的末端位置、姿态、夹宽与关节等。','能够提供运动方向，因此必须检查动作阶段捷径。'],['公共深度与相机标定','已保存，可用于以后预先声明的几何核验。','本轮仍不加入读出器；没有把增加几何变量混入时间对照。'],['接触、外部支撑、未来跟随、成功信息','只用于当前标签与离线解释。','全部禁止作为训练或推理输入。']]),
 '<h2>3. 相同更新预算下的结果</h2>',img('temporal_comparison.png','横轴是配对种子。灰色只提供当前帧，蓝色覆盖0.10秒，绿色覆盖2.10秒。左图是18窗held原始错误数，中图是held概率交叉熵BCE（对数轴，越低越好），右图是6个真正held根中校准后仍能给出正确已知断言的数量。UNKNOWN不计为正确。三个种子重复使用同6个开发根，不是新增18个独立场景。'),
 table(['组别','种子','困难FP/6','普通FP/6','held FN/6','新C held BCE','已知held TP/6','旧C held FP/FN','旧C in FP/FN'],results),
 '<p>新C两类负例共12窗、真正held6窗；long组全部正确，原四帧组错误数为[2,0,0]，当前帧组为[1,1,1]。长窗相对短窗的held BCE差值依种子为−0.641983、−0.000914、−0.012032，三个种子方向一致。差异不是只由不同选模更新造成：主要比较使用统一600更新。次要联合选模分别选当前帧[100,600,100]、短窗[100,600,600]、长窗[600,600,600]，长窗改善方向保持。</p>',
 '<p>校准后的新C真正held已知正确数为：当前帧[0,0,0]/6，短窗[0,6,6]/6，长窗[6,6,6]/6。当前帧三个种子与短窗种子0没有合格held阈值；1.000001是失败哨兵，不是部署参数。原联合开发门槛通过数是0/3、1/3、3/3，但本报告不靠这一数目把已有状态识别全部判为失败或宣称整体完成。</p>',
 '<h2>4. 具体图像与可观测运动</h2>',img('temporal_frames.png','上两行是粉块真正held，下两行是仍受支撑的红块；每例依次展示静态与腕部相机。列为距当前2.10、1.40、0.70、0.10和0秒。前三列加最后一列才是长窗实际输入；0.10秒列用于说明短窗起点。颜色是仿真中真实物体颜色，不代表预测。'),img('public_motion.png','绿线对应粉块真正held，橙线对应红块支撑负例。左图从公共机器人观测读取末端高度变化；右图从静态相机可见掩码计算目标图像重心上移，像素位移不是物体世界坐标或接触真值。空心圆标出长窗4输入时刻，方点标出短窗4输入时刻，右端淡蓝区为原0.10秒窗。两图共同展示长窗增加的信息，也暴露上下动作方向与标签的混杂。'),
 table(['样本','当前held真值','当前帧p：种子0/1/2','短窗p：种子0/1/2','长窗p：种子0/1/2'],case_rows),
 '<p>当前帧组种子0将粉块真正held给为0.0220；长窗三个种子都接近1。红块支撑负例在当前帧组种子1/2和短窗组种子0出现误报，长窗为0.0035/0.0044/0.0087。红块普通负例与支撑负例的当前锚点相邻，仍受支撑的事实没有改变；区别首先出现在状态读出，尚无世界模型或规划参与。</p>',
 '<h2>5. 改善证明了什么，仍未隔离什么</h2><p><b>支持的解释：</b>原0.10秒窗对这批held完成时刻几乎没有明显运动；增加真实过去时刻后，在同一当前帧、同一监督和预算下得到一致的新C改善。它提供了比继续盲目补相同短窗更明确的方向。</p><p><b>不能成立的更强结论：</b>“模型已学会物体随夹爪共同运动或真实持有关系”。后验公共观测审计发现，所有held正例2.10秒内TCP上移80.47～84.32mm，所有支撑负例TCP下移117.61～126.27mm。仅用高度变化正负，就与54/54个当前标签一致。这个规则是数据混杂诊断，未作为模型、验收器或学习转移使用。长窗还改变了过去画面、相机姿态及夹爪外观，现有对照不能拆开它们的贡献。</p>',
 '<h2>6. 至今可识别的范围与少数未解决场景</h2>',table(['状态范围','现有证据','仍需保留的边界'],[['抽屉open','本轮9个模型在历史C337窗（243正/94负）和新C18窗（9正/9负）均无原始错误。','是这些开发场景内的稳定表现，不等于任意遮挡或新环境验收。'],['物体in抽屉','长窗组历史C124正/813负，种子1/2全对；种子0漏掉1个红块in，未新增in误报。','新C全部是in负例，不能单凭新C证明in正例召回。'],['常见held与非held','长窗组历史C97正/840负，三个种子分别正确935/937、937/937、936/937个事实。','跨物体混淆和少数held漏判仍存在；已知断言/UNKNOWN另报。'],['闭爪但空、靠近但未持有','历史补充C37窗含远处/近处开爪、闭爪、近处空爪运动、真实held及1个接触支撑负例；长窗组在其原四帧回归上3种子均无held错误。','尚未把这些空爪轨迹恢复成同样2.10秒输入，不能据此排除长窗上移捷径。'],['短暂双侧接触但仍受支撑','本批新C6个困难根，长窗3种子均无误报。','与动作下降阶段混杂；只有6个开发根，没有独立复核。'],['真实held完成时刻','本批新C6个正根，长窗3种子均正确且已知。','held长期静止、下降移动以及失败抓取后空爪上移尚未完成同窗对照。']]),
 img('remaining_old_C.png','长窗组仍有3个不同的旧C失败场景：红块in漏判、粉块held漏判、抬粉块时把红块也判断为held。每行展示原始末帧与失败种子的概率。输入仍是共同保留的原0.10秒历史；这些错误不会因新C长窗全部正确而消失。'),
 '<p>C_core1200_06中的红块in在种子0概率0.4720；C_core1200_07中的粉块held在种子0概率0.0368；C_core1200_14抬粉块时红块held在种子0/2概率0.5945/0.9998。最后一种会触发多物体held冲突或不确定输出。全量结果保留真假、置信度和UNKNOWN，未挑掉这些记录。</p>',
 '<h2>7. 停止点与下一项最小决策</h2><p>本轮有限诊断已结束，没有新增根、没有D推理/验收，也没有晋升模型或开始世界模型/规划。结果支持继续检查真实时间证据，不支持继续试更多网络或窗口长度。</p><p><b>下一项最小方案：</b>仅恢复6条既有C的near_closed_empty轨迹，沿原动作与冻结复现容差保存缺失公共长窗，然后对现有冻结长窗模型做一次held评估，完全不训练。其far_closed_hold既定锚点已由原日志确认：2.10秒内末端上移43.57～48.28mm，但所有物体held均为假；每条缺少长窗中的前三个精确RGBD/掩码帧。它直接打破“末端上移=held”的混杂，无需新增根或更换控制器。该方案尚未执行，需要下一步范围决策；它仍属于开发挑战，不能冒充独立验收。若失败，先定位模型是否忽略物体相对运动，避免直接启动新模型。</p>',
 '<p class="muted">协议：configs/protocol.json；审计：results/audit.json；全部预测/指标：results/evaluation.json；动作方向混杂与既有反例候选：results/direction_confound.json；残余旧C错误：results/remaining_historical_errors.json；冻结候选：configs/candidates_frozen.json；完整性：results/delivery_validation.json与delivery_manifest.json。原始数据与此前实验保持只读。深度未加入模型，GT接触/物体位置/未来/成功未进入输入。</p>']
 css='body{font:16px/1.65 "Noto Sans CJK SC",sans-serif;background:#f2f5f8;color:#253745;margin:0}main{max-width:1240px;margin:auto;padding:40px;background:white}h1{font-size:30px}h2{font-size:23px;margin-top:38px;color:#245775}.lead{border-left:4px solid #258b72;background:#edf7f4;padding:18px}.muted{font-size:14px;color:#607586}table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0}th,td{border:1px solid #dce4ea;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#fbfcfd}figure{margin:26px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#607586}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>时间证据诊断</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
 save(E/'SUMMARY.json',{'status':'finite_temporal_diagnostic_completed','current_span_s':0,'short_span_s':.1,'long_span_s':2.1,'input_frames_each_arm':4,'arms':3,'seeds':3,'updates_per_run':600,'model_parameters':675395,'train_pool_per_arm':1009,'treated_train_windows':36,'treated_C_dev_windows':18,'common_unchanged_historical_train_windows':973,'fixed600_fresh_held_error_counts':{a:[r['modes']['fixed600']['fresh']['held']['FP']+r['modes']['fixed600']['fresh']['held']['FN']for r in D[a]]for a in ARMS},'long_fresh_held_known_TP':[6,6,6],'long_minus_short_held_BCE':[q['long_minus_short_held_BCE']for q in D['paired_fixed600']],'robot_z_direction_label_agreement':{'correct':54,'total':54,'posthoc_confound_audit_only':True},'remaining_historical_distinct_error_scenarios':3,'next_minimal_decision':'Recover6existingCemptyclosedupwardnegative longwindows; frozenmodels, onceonlyevaluation,no training. Not executed.','new_roots_collected':0,'D_inference_or_acceptance':False,'model_promoted':False,'world_model_or_planner_run':False})
 print('TEMPORAL_REPORT_READY',E/'report.html')
if __name__=='__main__':main()
