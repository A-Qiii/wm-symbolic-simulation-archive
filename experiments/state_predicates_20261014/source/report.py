from pathlib import Path
import json,base64,html,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from model import predicate_api

P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_20261014'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'matched_eight_class':'#3577B9','direct_predicates':'#DC7535','old':'#83919C'}
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def img(path):return '<img alt="实验结果图" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def main():
 d=json.loads((E/'results/evaluation.json').read_text());diag=json.loads((E/'results/failure_diagnosis.json').read_text());meta=json.loads((E/'data/metadata.json').read_text());training=json.loads((E/'results/training_complete.json').read_text());p=np.load(E/'results/direct_predicates_seed0_probabilities.npy');old=np.load(E/'results/frozen_old_native_history_probabilities.npy');baseline=np.load(E/'results/matched_eight_class_seed0_probabilities.npy');a=d['three_seed_summary'];names=['held','in_drawer','open_drawer'];cn=['拿住物体 held','位于抽屉内 in','抽屉打开 open'];x=np.arange(3)
 fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
 for mode,delta,label in [('matched_eight_class',-.19,'同输入八类聚合'),('direct_predicates',.19,'直接谓词头')]:
  for ax,key,title,ylim in zip(axes,['balanced_accuracy','false_positive_rate','coverage'],['正负例均衡准确率（越高越好）','假阳性率：FP / 负例（越低越好）','已知覆盖率：非 UNKNOWN / 有效例'],[(0,105),(0,30),(0,105)]):
   vals=[a[mode][n][key]['mean']*100 for n in names];errs=[a[mode][n][key]['std']*100 for n in names];ax.bar(x+delta,vals,.36,yerr=errs,capsize=3,label=label,color=COLORS[mode]);ax.set_ylim(*ylim);ax.set_xticks(x,cn,rotation=12);ax.set_ylabel('%');ax.set_title(title,fontsize=12);ax.grid(axis='y',alpha=.2)
   for xx,v in zip(x+delta,vals):ax.text(xx,v+1,f'{v:.1f}',ha='center',fontsize=9)
 axes[0].legend(loc='lower left');fig.suptitle('同输入、同根级划分、同训练预算；三个随机种子分别训练，误差线为种子标准差',fontsize=13)
 fig.savefig(E/'figures/comparison.png',dpi=160);plt.close(fig)
 cont=diag['training_state_contingencies'];labels=['空手＋关闭','空手＋打开','拿住＋关闭','拿住＋打开'];spl=['train','tune','calibration','diagnostic_D'];matrix=np.array([[cont[s][k] for k in ['empty_closed','empty_open','held_closed','held_open']] for s in spl])
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained');im=ax.imshow(matrix,cmap='Blues',aspect='auto');ax.set_xticks(range(4),labels);ax.set_yticks(range(4),['AB训练：60根','C开发：12根','C校准：8根','D诊断：20根']);ax.set_title('状态组合覆盖：训练与校准缺少“拿住物体且抽屉关闭”');
 for i in range(4):
  for j in range(4):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>180 else '#1A2330',fontsize=15,fontweight='bold')
 fig.colorbar(im,ax=ax,label='状态边界记录数（同根记录相关，不能视为独立样本）');fig.savefig(E/'figures/state_coverage.png',dpi=160);plt.close(fig)
 # Real dual-camera examples, selected deterministically and saved for audit.
 test=np.array([r['split']=='diagnostic_D' for r in meta]);cat=np.array([r['category'] for r in meta]);truth=np.c_[cat[:,:3]==6,cat[:,:3]==3,cat[:,3]==2]
 success=int(np.flatnonzero(test&(cat[:,0]==6)&(cat[:,3]==2)&(p[:,0]>.5))[0]);inside=int(np.flatnonzero(test&(cat[:,0]==3)&(cat[:,3]==0)&(p[:,3]>.5))[0])
 cases=[('正确：拿住红块',success,0),('正确：红块在关闭的抽屉内',inside,3),('失败：靠近红块但未拿住',1283,0),('失败：漏读拿住粉块',1277,2),('失败：抽屉关闭误报打开',1201,6),('未见组合：拿住红块＋抽屉关闭',1337,6)]
 fig,axes=plt.subplots(6,3,figsize=(13,18),gridspec_kw={'width_ratios':[1,1,1.45]},layout='constrained')
 selected=[]
 for row,(title,i,j) in enumerate(cases):
  r=meta[i]
  with np.load(r['public_inputs'][-1]) as f:
   for c,cam in enumerate(['static','gripper']):axes[row,c].imshow(f['rgb_'+cam]);axes[row,c].axis('off');axes[row,c].set_title('静态相机' if c==0 else '夹爪相机',fontsize=11)
  ax=axes[row,2];ax.axis('off');bad=bool((p[i,j]>=.5)!=truth[i,j]);color='#B3392D' if bad else '#24764D'
  ax.text(0,.96,title,fontsize=13,color=color,fontweight='bold',va='top');atom=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'][j]
  txt=f"{r['episode_id']}\n边界 {r['step']} · {atom}\n\n模拟器离线真值：{bool(truth[i,j])}\n直接头 P(True)：{p[i,j]:.6f}\n直接头判定：{bool(p[i,j]>=.5)}\n同输入八类 P(True)：{baseline[i,j]:.6f}\n旧模型参考 P(True)：{old[i,j]:.6f}"
  ax.text(0,.78,txt,fontsize=11,va='top',linespacing=1.55);selected.append({'title':title,'index':i,'predicate_column':j,'source_image':r['public_inputs'][-1],'truth':bool(truth[i,j]),'direct_probability':float(p[i,j]),'matched_probability':float(baseline[i,j])})
 fig.suptitle('真实观测与当前状态预测：绿色为正确，红色为错误；使用预定主展示种子0',fontsize=15);fig.savefig(E/'figures/actual_examples.png',dpi=140);plt.close(fig)
 (E/'results/example_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
 # Diagnostic sweep only. It does not replace C-calibrated thresholds.
 sweep=[]
 for t in [.5,.95,.99,.995]:
  pred=p[test]>=.5;yy=truth[test];known=(p[test]>=t)|(p[test]<=1-t)
  rr={'threshold':t}
  for name,cols in [('held',[0,1,2]),('open',[6])]:
   k=known[:,cols];pv=pred[:,cols];yv=yy[:,cols];rr[name]={'known':int(k.sum()),'n':k.size,'wrong_known':int(((pv!=yv)&k).sum()),'false_positive_known':int((pv&~yv&k).sum())}
  sweep.append(rr)
 (E/'results/threshold_sensitivity_diagnostic_only.json').write_text(json.dumps(sweep,indent=2))
 # Ready-to-adapt structured symbolic interface example, with real probabilities.
 api={'source':'Direct predicate head seed0, real existing D diagnostic boundary','index':success,'thresholds':d['models']['direct_predicates_seed0']['thresholds'],'entities':['red','blue','pink'],'facts':predicate_api(p[success],d['models']['direct_predicates_seed0']['thresholds'],['red','blue','pink']),'planner_integration_status':'Offline readout only; not connected to LTL executor or future transition search'}
 (E/'results/interface_example.json').write_text(json.dumps(api,ensure_ascii=False,indent=2))
 rows=[]
 for mode,title in [('matched_eight_class','同输入八类聚合'),('direct_predicates','直接谓词头')]:
  for name,label in zip(names,cn):
   q=a[mode][name];rows.append([title,label,f"{q['accuracy']['mean']*100:.2f} ± {q['accuracy']['std']*100:.2f}%",f"{q['balanced_accuracy']['mean']*100:.2f}%",f"{q['false_positive_rate']['mean']*100:.2f}%",f"{100-q['coverage']['mean']*100:.2f}%",f"{q['known_error']['mean']*100:.2f}%"])
 detail=[]
 for name,title in [('matched_eight_class_seed0','同输入八类／种子0'),('direct_predicates_seed0','直接头／种子0'),('frozen_old_native_history','旧八类头／历史参考')]:
  for pred,label in zip(names,cn):
   q=d['models'][name]['subsets']['diagnostic_D']['predicates'][pred];detail.append([title,label,f"{q['positive']} / {q['negative']}",f"{q['FP']} / {q['FN']}",pct(q['accuracy']),pct(q['coverage']),pct(q['known_error'])])
 ext=d['models']['direct_predicates_leave_pink_labels_out_seed0']['atoms'];exrows=[]
 for atom in ['held(pink)','in(pink,drawer)']:
  q=ext[atom];exrows.append([atom,f"{q['positive']} / {q['negative']}",f"{q['TP']} / {q['FN']} / {q['FP']}",pct(q['accuracy']),pct(q['balanced_accuracy'])])
 caption1='蓝色为重新训练的八类状态头，再聚合成命题；橙色为直接预测命题。两者使用相同四帧 DINOv2、可见几何与本体感知输入，根级划分、预算相同。纵轴均为百分比；误差线只表示三次训练的差异，不是样本置信区间。假阳性率指实际为假却预测真的比例。'
 blocks=[]
 blocks.append('<h1>连续视觉特征 → 逻辑命题：小规模读取实验</h1><p class="sub">本机 RTX 5080 Laptop · 2026-10-14 · 当前状态诊断 · 控制器与旧模型保持不变</p>')
 blocks.append('<div class="callout"><strong>结论：可以学到这三个当前状态命题，但尚不能作为可靠规划事实直接使用。</strong><p>直接头的 held、抽屉内准确率很高；抽屉开关误报和近距离抓空混淆未解决。直接头相对公平八类对照的宏平均均衡准确率只增加 <b>0.49 个百分点</b>，未达到预先设定的优势标准。整体验收未通过，本轮没有再训练修补。</p></div>')
 blocks.append('<h2>1. 到底学了什么，为什么这样做</h2>'+table(['元素','实现','原因与边界'],[
 ['输入','冻结 DINOv2 的双相机特征＋可见框/掩码＋本体感知，四帧均不晚于当前边界','不输入物体真值坐标、接触、技能名、成功结果、未来帧。现有掩码来自模拟器可见分割；不是纯 RGB 检测系统。'],
 ['目标','held(object)、in(object, drawer)、open(drawer)','避免为 LTL 无关的桌面左中右耗费输出容量；真值仅生成训练标签与独立评分。'],
 ['共享实体头','约67.5万参数；每个物体使用同一个特征编码器、GRU 和 held/in 输出头','没有红/蓝/粉专属可学习输出槽；实体名称与谓词名称分离。可变物体数量与置换等变的代码检查通过。'],
 ['公平对照','相同输入/主干的八类物体头＋三类抽屉头，独立从头训练','将其类别概率聚合为同样的三个命题。不能把“新模型胜过旧世界模型”直接归因于改成二元输出。'],
 ['旧模型参考','旧冻结世界模型的当前后验八类聚合，保留原有因果历史','它还使用更长的边界/动作历史、结构和训练目标不同，只作历史参考，不属于公平因果对照。'],
 ['UNKNOWN','每个命题单独保留概率；按 C 校准阈值拒答，多 held 冲突保留 UNKNOWN','本次 C 过于容易，轻量头阈值都选到0.5，几乎不拒答；这不意味着测试可靠。']]))
 blocks.append('<h2>2. 数据与冻结标准</h2>'+table(['分区','独立初始状态','边界记录','用途'],[['AB','60','900','训练；720条转移的起终边界，含同根相关分支'],['C开发','12','180','选 checkpoint；每5轮检查'],['C校准','8','120','选择逐谓词拒答阈值'],['D诊断','20','300','既有、已多次观察的D；非全新盲测']]))
 blocks.append('<p>全部轨迹随初始状态分组，分区间没有相同图像内容；未随机拆相邻帧。计划、根名单和标准在训练前冻结。每个主模型训练100轮、三个种子；另有一个留出粉块标签模型。旧控制器、旧 checkpoint、旧清单哈希复核一致。未重新采集、未重训世界模型、未执行新控制。</p><p>D 的300个边界中，三物体 XYZ 拼接向量到最近训练边界的欧氏距离均超过2厘米；这是位置组合不同的证据，不等于每个物体都超出训练空间范围。159条边界的细类别组合训练未见，24条的粗逻辑状态组合训练未见（来自6个初始状态）。坐标只用于这些评估分组，不进入网络。</p><p>预设绝对标准：三个命题均衡准确率≥90%、held 假阳性率≤2%、已知判定错误率≤5%、已知覆盖率≥70%。预设相对标准：宏平均均衡准确率提高至少2个百分点，或在低错误率下覆盖率提高10个百分点。<b>两个主方法均未通过全部绝对标准，直接头也未通过相对优势标准。</b></p>')
 blocks.append('<h2>3. 公平比较：整体数值不错，但不能隐藏失败</h2>'+img(E/'figures/comparison.png')+'<p class="caption">'+caption1+'</p>'+table(['方法','命题','准确率：均值±种子标准差','均衡准确率','假阳性率','UNKNOWN比例','已知判定错误率'],rows))
 blocks.append('<p>均衡准确率是正例召回率与负例正确率的平均，避免大量“未拿住/不在抽屉”掩盖错误。三个种子独立评估，不是集成。下表给出主展示种子0的实际计数；完整逐物体结果和逐条预测保存在 evaluation.json 与 D_predictions.csv。</p>'+table(['方法','命题','正例 / 负例','FP / FN','准确率','已知覆盖','已知错误'],detail))
 blocks.append('<p>held 的三次假阳性分别为 2、4、2 / 801 个负例。主展示种子0按20个初始状态重采样的 held 假阳性率95%区间为0–0.75%；这些状态记录相关，这个区间也不能补足缺失的抓空场景。按根配对自举，直接头与八类头的均衡准确率差异区间包含0，不能宣称稳定优越。</p>')
 blocks.append('<h2>4. 失败先在哪里出现</h2>'+img(E/'figures/state_coverage.png')+'<p class="caption">列为物体是否被拿住与抽屉开关的组合；行为数据分区。数字为状态边界数量，蓝色越深表示越多；零不是缺失值。训练、开发和校准均没有“拿住＋关闭”，D 有16条。</p>')
 blocks.append('<ul><li><b>抽屉开关：</b>种子0的19次误报集中在D的两个初始状态。16个“拿住＋关闭”全部误报为打开；另外3个误报发生在失败的开抽屉尝试之后。D中24条训练未见的粗状态组合上，抽屉准确率只有33.3%，关闭负例16/16全错。</li><li><b>held：</b>种子0两次假阳性正是D仅有的两个“距夹爪很近、但未拿住”记录；它们属于同一初始状态、同一轨迹的相邻边界，不能算两次独立成功或失败试验。训练集此类负例为0。因此99.7%左右总体准确率不能证明已区分靠近、抓空与拿住。</li><li><b>校准：</b>C阈值选择几乎全放行，D仍发生高置信错误。仅把显示阈值提高到0.995，种子0仍保留16个“抽屉打开”错误和2个 held 假阳性。该阈值扫描仅用于解释，没有用D重新选择部署阈值。</li></ul>')
 blocks.append('<p>对首个抽屉误报和首个 held 误报，仅用同轨迹上一边界的本体感知替换当前四帧本体感知，图像和参数保持不变：概率分别从0.99313变为0.99941、从0.999904变为0.999903，错误均未消失。这是离线输入敏感性检查，输入组合不一定物理一致；它不支持“只因夹爪本体信息走捷径”的结论。数据组合缺失有直接计数证据，但尚不能单独证明所有错误的因果来源。</p>')
 blocks.append('<h2>5. 看真实图像与概率</h2>'+img(E/'figures/actual_examples.png')+'<p class="caption">每行是一条真实边界观测，左为静态相机，中为夹爪相机，右为独立真值与模型概率。绿色标题表示所展示命题判对，红色表示判错；方块本身的红/蓝/粉是实际对象颜色，不是图表编码。P(True)是网络输出，不是经验证的正确率。所有图像均直接读取已保存观测，没有生成或修饰场景。</p>')
 blocks.append('<h2>6. “支持扩展”具体到什么程度</h2>'+table(['扩展类型','本轮证据','仍然不能声称'],[
 ['新任务/状态组合','输出为实体参数化原子命题，可供不同LTL公式复用；D有24条训练未见粗状态组合','抽屉在这些组合上失败；本轮没有新LTL规划成功率或未来转移证据。'],
 ['已知物体的标签迁移','辅助模型删除所有粉块 held/in 训练、选模、校准标签，仍用共享头读取粉块','训练图像中仍存在粉块，颜色注册也认识粉色；这不是从未见过的新物体。'],
 ['新物体数量/身份','网络支持可变 K；物体顺序置换误差为0；没有每色专属学习参数','现有注册器只识别红/蓝/粉；新外观、形状、材质、遮挡、更多实例仍需验证，未实现可靠零样本新物体泛化。'],
 ['新谓词或新容器','接口把谓词名与实体参数分开，源码易添加共享输出','目前 in 的第二参数仅固定抽屉；新关系、容器或谓词需要定义标签并训练/校准，不是输入名称即可理解。']])+table(['留出粉块标签后的测试','正例 / 负例','TP / FN / FP','准确率','均衡准确率'],exrows))
 blocks.append('<p>粉块 held 漏掉4/32个正例（12.5%），因此虽然具有参数共享与迁移的初步证据，仍不能把接口可扩展等同于泛化已解决。遮挡子集上本轮种子0判对68个 held/in原子记录，但这些记录来自当前数据模式；没有证明长时间遮挡、身份丢失或新场景下也能保持状态。</p>')
 blocks.append('<h2>7. 下一步与交付</h2><p>本轮先停止扩大训练。若继续，优先补少量固定控制下的“抓空/靠近但未抓住”和“拿住但抽屉关闭”独立状态，并覆盖抽屉中间开度；再用独立初始状态复核。当前没有证据表明必须换检测器、引入 Slot Attention 或更大的模型。识别当前状态通过也不等于未来状态预测或 LTL 搜索通过。</p>')
 blocks.append(table(['本机文件','用途'],[['report.html','本文件；图像已内嵌，可独立打开'],['configs/protocol.json','训练前固定计划、根名单、预算、通过标准'],['data/audit.json / metadata.json','输入来源、独立分组、真实图像索引；坐标仅供评价'],['source/model.py','共享实体谓词头与独立概率/UNKNOWN接口'],['models/*.pt','两种主方法×3种子＋留出粉块标签模型，共7个轻量checkpoint'],['results/evaluation.json / D_predictions.csv','完整计数、逐物体结果、逐条概率与判定'],['results/failure_diagnosis.json','首错单输入替换诊断与状态组合计数'],['results/interface_example.json','真实样本的结构化命题输出'],['results/training_complete.json / logs/','训练预算、选模、耗时与日志']]))
 blocks.append('<p class="sub">所有新增文件仅保存在本机独立实验目录，未上传 Library。旧项目数据和运行结果没有覆盖。技术限制是识别与分布覆盖尚未达标，文件交付无阻塞。</p>')
 css='body{font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif;color:#172433;background:#f5f7fa;margin:0}main{max-width:1240px;margin:auto;background:white;padding:44px}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:42px;border-bottom:2px solid #dce3ea;padding-bottom:8px}p{margin:16px 0}.sub,.caption{color:#536475;font-size:14px}.callout{background:#fff1e4;border-left:5px solid #d87931;padding:20px}table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px}th,td{padding:10px 12px;border:1px solid #d9e1e9;vertical-align:top;text-align:left}th{background:#edf2f7}tr:nth-child(even){background:#fafbfd}img{display:block;width:100%;height:auto;margin:18px 0}li{margin:12px 0}code{overflow-wrap:anywhere}@media(max-width:700px){main{padding:16px}table{font-size:12px}th,td{padding:6px}}'
 htmlout='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续视觉状态到逻辑命题：读取实验</title><style>'+css+'</style><main>'+''.join(blocks)+'</main></html>'
 (E/'report.html').write_text(htmlout)
 summary={'status':'completed_diagnostic_not_passed','main_result':'Direct predicate recognition is feasible on existing easy states but not reliable on missing state combinations or near-gripper negatives; no clear advantage over matched eight-class aggregation.','direct_three_seed':a['direct_predicates'],'gates':d['preregistered_gates'],'extension_pink':{k:ext[k] for k in ['held(pink)','in(pink,drawer)']},'report':str(E/'report.html'),'upload_status':'not_uploaded_new_permission_pending_parent','no_new_control_or_world_model_training':True,'remaining_blockers':[],'model_limitations':['Historical D is diagnostic, not blind','Near-negative training=0; test2 records in same trajectory both false held','16 held+closed states absent from train/C, all false open','No unseen object/predicate generalization established']}
 (E/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('REPORT_READY',E/'report.html',len(htmlout),flush=True)
if __name__=='__main__':main()
