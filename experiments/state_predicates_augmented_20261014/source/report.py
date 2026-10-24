from pathlib import Path
from report_regression_addendum import build_addendum
import json,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'experiments/state_predicates_20261014';E=P/'experiments/state_predicates_augmented_20261014'
def table(h,rows):return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
def img(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 e=json.loads((E/'results/evaluation.json').read_text());d=json.loads((E/'results/failure_diagnosis.json').read_text());protocol=json.loads((E/'configs/protocol.json').read_text());freeze=json.loads((E/'results/model_threshold_freeze.json').read_text());m=e['models'];nm=json.loads((E/'data/metadata_test.json').read_text());om=[r for r in json.loads((O/'data/metadata.json').read_text()) if r['split']=='diagnostic_D'];nidx={r['sample_id']:i for i,r in enumerate(nm)};oidx={r['index']:i for i,r in enumerate(om)}
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 blue='#397DB5';orange='#DB7737';x=np.arange(3)
 panels=[('new_test','held','FP','新增测试：held 误判／32负例'),('new_test','open_drawer','FP','新增测试：打开误判／19关闭例'),('new_test','in_drawer','FP','新增测试：抽屉内误判／38负例'),('old_D_regression','held','FN','旧D回归：held 漏判／99正例'),('old_D_regression','held','FP','旧D回归：held 误判／801负例'),('old_D_regression','open_drawer','FP','旧D回归：打开误判／96关闭例')]
 fig,axes=plt.subplots(2,3,figsize=(15,8.3),layout='constrained')
 for ax,(cohort,pred,key,title) in zip(axes.flat,panels):
  for mode,delta,color,label in [('old_original',-.19,blue,'补数据前'),('augmented',.19,orange,'补数据后（开发集选模）')]:
   values=[m[f'{mode}_seed{s}'][cohort][pred][key] for s in [0,1,2]];ax.bar(x+delta,values,.36,color=color,label=label)
   for xx,v in zip(x+delta,values):ax.text(xx,v+.12,str(v),ha='center',fontsize=11)
  maxv=max(m[f'{mode}_seed{s}'][cohort][pred][key] for mode in ['old_original','augmented'] for s in [0,1,2]);ax.set_ylim(0,maxv+max(1.5,maxv*.2));ax.set_xticks(x,['种子0','种子1','种子2']);ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.set_title(title,fontsize=12);ax.set_ylabel('错误记录数（越少越好）');ax.grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=9);fig.suptitle('同一图像序列、同一真值；图中错误按原始概率0.5判定，不依赖 UNKNOWN 阈值',fontsize=14);fig.savefig(E/'figures/error_counts.png',dpi=155);plt.close(fig)
 def probabilities(mode,seed,cohort):return np.load(E/'results'/f'{mode}_seed{seed}_{cohort}_probabilities.npy')
 # Predeclared six examples; no best-case choice after scoring.
 cases=[]
 for sid in protocol['image_cases_predeclared']:
  if sid.startswith('old_index'):
   idx=1283 if '1283' in sid else 1277;j=0 if idx==1283 else 2;i=oidx[idx];r=om[i];cases.append({'cohort':'old_D_regression','i':i,'col':j,'seed':0,'path':r['public_inputs'][-1],'title':'旧D原始靠近误判' if idx==1283 else '旧D原始 held 漏判','id':r['episode_id']+' 边界'+str(r['step']),'truth':r['category'][j]==6})
  else:
   i=nidx[sid];r=nm[i];j=6 if r['condition']=='held_positive' else r['target_index'];title='新测试：拿住且抽屉关闭' if j==6 else '新测试：靠近且夹爪闭合' if r['condition']=='near_closed_negative' else '新测试：接触但仍受支撑';cases.append({'cohort':'new_test','i':i,'col':j,'seed':0,'path':r['public_frames'][-1],'title':title,'id':r['root_id'],'truth':r['category'][3]==2 if j==6 else r['category'][j]==6})
 def gallery(cases,path,title):
  fig,axes=plt.subplots(len(cases),3,figsize=(13,3.05*len(cases)),gridspec_kw={'width_ratios':[1,1,1.45]},layout='constrained',squeeze=False)
  for row,c in enumerate(cases):
   oldp=probabilities('old',c['seed'],c['cohort'])[c['i'],c['col']];newp=probabilities('augmented',c['seed'],c['cohort'])[c['i'],c['col']];j=c['col'];atom=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'][j]
   with np.load(c['path']) as z:
    for k,cam in enumerate(['static','gripper']):axes[row,k].imshow(z['rgb_'+cam]);axes[row,k].axis('off');axes[row,k].set_title('静态相机' if k==0 else '夹爪相机',fontsize=11)
   ax=axes[row,2];ax.axis('off');correct=(newp>=.5)==c['truth'];ax.text(0,.96,c['title'],va='top',fontsize=12,fontweight='bold',color='#287D50' if correct else '#B43B31');text=f"{c['id']}\n种子{c['seed']} · {atom}\n\n独立真值：{c['truth']}\n补数据前 P(True)：{oldp:.6f}\n补数据后 P(True)：{newp:.6f}\n原始0.5判定：{bool(oldp>=.5)} → {bool(newp>=.5)}"
   if c['col']==6 and '拿住' in c['title']:
    jp=nm[c['i']]['target_index'];hp=probabilities('augmented',c['seed'],c['cohort'])[c['i'],jp];text+=f"\n同帧补后 held 概率：{hp:.6f}"
   ax.text(0,.78,text,va='top',fontsize=11,linespacing=1.6)
  fig.suptitle(title,fontsize=14);fig.savefig(path,dpi=135);plt.close(fig)
 gallery(cases,E/'figures/paired_examples.png','预先指定的六例：前后读取完全相同的四帧历史；绿色/红色表示补后判对/判错')
 f=d['first_regression_for_visualization'];i=oidx[f['index']];r=om[i];reg=[{'cohort':'old_D_regression','i':i,'col':f['predicate_column'],'seed':0,'path':r['public_inputs'][-1],'title':'新增漏判：原来拿住识别正确','id':r['episode_id']+' 边界'+str(r['step']),'truth':True}]
 f=d['first_in_error_for_visualization'];i=nidx[f['sample_id']];r=nm[i];reg.append({'cohort':'new_test','i':i,'col':f['predicate_column'],'seed':2,'path':r['public_frames'][-1],'title':'新增误判：桌上物体被读成抽屉内','id':r['root_id'],'truth':False})
 gallery(reg,E/'figures/regression_examples.png','失败诊断补充：按索引选首个新增错误；不是用于估计总体表现的随机样本')
 def seq(mode,cohort,pred,key):return '、'.join(str(m[f'{mode}_seed{s}'][cohort][pred][key]) for s in [0,1,2])
 nrows=[];orows=[]
 for pred,label in [('held','held（目标物体）'),('in_drawer','in（目标物体,抽屉）'),('open_drawer','open（抽屉）')]:
  for cohort,rows in [('new_test',nrows),('old_D_regression',orows)]:
   q=m['old_original_seed0'][cohort][pred];rows.append([label,f"{q['positive']} / {q['negative']}",seq('old_original',cohort,pred,'FP')+' → '+seq('augmented',cohort,pred,'FP'),seq('old_original',cohort,pred,'FN')+' → '+seq('augmented',cohort,pred,'FN'),seq('old_original',cohort,pred,'unknown')+' → '+seq('augmented',cohort,pred,'unknown')])
 buckets=[]
 for key,label,kind in [('true_held','真实拿住','FN'),('near_closed_negative','闭合空夹爪靠近','FP'),('near_closed_motion_negative','上述夹爪移动后，同一6根','FP'),('contact_supported_notheld','接触但仍受桌面支撑','FP'),('near_open_notheld','张开夹爪靠近','FP'),('far_notheld','远离且未拿住','FP'),('held_and_drawer_closed','拿住且抽屉关闭：open误报','FP')]:
  q=m['old_original_seed0']['new_test_buckets'][key];old='、'.join(str(m[f'old_original_seed{s}']['new_test_buckets'][key]['metric'][kind]) for s in [0,1,2]);new='、'.join(str(m[f'augmented_seed{s}']['new_test_buckets'][key]['metric'][kind]) for s in [0,1,2]);buckets.append([label,q['roots'],q['metric']['n'],old+' → '+new])
 finalrows=[]
 for s in [0,1,2]:
  q=m[f'final_seed{s}'];finalrows.append([s,1500,q['new_test']['held']['FP']+q['new_test']['held']['FN'],q['new_test']['open_drawer']['FP']+q['new_test']['open_drawer']['FN'],q['new_test']['in_drawer']['FP'],q['new_test']['held']['unknown'],q['old_D_regression']['held']['FN'],q['old_D_near_not_held']['FP']])
 budgetrows=[[s,x['selected_update'],x['selected_presentations'],x['selected_supplement_presentations'],x['original_presentations'],x['supplement_presentations']] for s,x in d['budget_ledger'].items()]
 blocks=['<h1>补充困难样本后：逻辑命题读取前后对照</h1><p class="sub">2026-10-14 · 本机RTX 5080 Laptop · 相同模型结构与冻结DINOv2 · 三个随机种子</p>']
 blocks.append('<div class="callout"><b>定向补数据有效，但还不能视为整体可靠性提高。</b><p>新增测试上，三个选定模型都读对全部 held 和抽屉开关，保住6/6真实held，UNKNOWN为0。旧D的抽屉误报也从每种子19次降到0次。但旧D held漏判从1、0、0次增加到7、7、4次；种子2还新增4次“物体在抽屉内”的误报。本轮在此停止，没有继续修补或扩大实验。</p></div>')
 blocks.append('<h2>1. 比较方式与预算</h2>'+table(['项目','本轮固定做法','影响与边界'],[['模型','原model.py逐字节复用，675,395个参数；DINOv2与控制器不变','没有引入检测器、世界模型训练或控制优化。'],['训练数据','原900个训练状态＋补采train的73窗口／12根，自然配比900:73','约7.5%的窗口曝光被补采替换；补采只监督目标held和open，未给in或其他物体加标签。'],['训练预算','每种子从头初始化，1500次更新×60窗口＝90,000次呈现','与旧实验100轮×15批×60相同。不是继续训练旧模型，也没有额外增加每种子更新数。'],['选模与阈值','只用新增dev的37窗口／6根；模型和阈值先冻结，之后才缓存test特征','开发集既选checkpoint又选阈值，样本少，不能当统计校准保证。原C与旧D不参与本轮选模。'],['测试','新增38窗口／6根；旧D300边界／20根仅作回归','新增集做过采集质量审计；此前没有模型预测。旧D已观察，不能当未见测试。'],['对照','同一种子旧/新模型看同一四帧输入；原始0.5判定和UNKNOWN分别统计','额外给出旧模型在新dev重选阈值的对照，以及所有新模型第1500步的固定预算对照。']]))
 blocks.append('<p>下表列出实际训练账本。训练总预算一样，但开发集选出的最佳checkpoint可能较早；这部分选择效应不能冒充纯数据效应，所以另列最终1500步对照。补采有2个监督事实，旧状态有最多7个；按有效事实平均损失，补采在事实级损失中的占比比7.5%窗口占比更小。</p>'+table(['种子','选中更新步','选中模型已看窗口次数','其中补采次数','完整预算旧窗口次数','完整预算补采次数'],budgetrows))
 blocks.append('<h2>2. 新增测试：哪些错误被修正</h2><p>表中每组三个计数依次对应种子0、1、2；它们重复测试同一批样本，不能相加为更多独立试验。FP是实际为假却判真；FN是实际为真却判假。</p>'+table(['命题','正例 / 负例','FP：补前 → 补后','FN：补前 → 补后','UNKNOWN：补前 → 补后'],nrows))
 blocks.append('<p>新测试的in没有正例，因此不能评价“在抽屉内”的召回率。种子2的4/38次误报必须保留，不能被held/open的零错误掩盖。</p>'+table(['实际状态子集','独立根','记录分母','错误数：补前 → 补后'],buckets))
 blocks.append('<p><b>闭合空夹爪的偏移负例原模型已经0/6误判，因此这一行只是保持正确，不是修复证据。</b>真正发生改善的是张开靠近、仍受桌面支撑，以及拿住但抽屉关闭这些子集。每个子集最多6根，接触支撑仅2例，仍不足以证明所有抓空外观都能识别。</p>'+img(E/'figures/error_counts.png')+'<p class="caption">蓝色为旧模型，橙色为补数据后的开发集选定模型。横轴为三个种子；纵轴为错误记录个数，每个小图标题给出分母。上排是新增测试，下排是旧D回归。全部错误由原始概率0.5判定，UNKNOWN不会把错误从图中删除；六个测试根和二十个回归根才是独立分组。</p>')
 blocks.append('<h2>3. 旧分布回归：held出现了真实退步</h2>'+table(['命题','正例 / 负例','FP：补前 → 补后','FN：补前 → 补后','UNKNOWN：补前 → 补后'],orows))
 blocks.append('<p>旧D held正例召回从98/99、99/99、99/99下降为92/99、92/99、95/99。总准确率只下降约0.44–0.56个百分点，是因为有801个负例；不能据此说“没有退步”。新增漏判分别集中在2、3、2个根，部分是同根无动作边界的重复状态，不是7次独立失败。</p><p>原先那两个“靠近但未拿住”的旧D记录（同一根、同一轨迹），三个种子的误判从2、2、2变为0、2、2：只在种子0修正了这两个记录，尚未跨种子解决。旧D的16个held＋closed状态中，open误报从16、16、16降为0、0、0；这一项改善更一致。</p>')
 blocks.append('<h2>4. 是否只是多拒答，或把held都判成false</h2><p>不是。选定模型在新增测试中三个种子都输出6个held正判，正好覆盖6个真held；32个负例均判假，held/open UNKNOWN均为0。旧D上虽然没有全判假，但正判减少并出现新增漏判，已经单独报告。</p><p>held的开发集阈值分别是0.5、0.975、0.5，open都是0.5；in沿用0.5，没有重新校准。若只给旧模型在同一新dev上重选阈值，held和open都找不到满足预设错误约束的阈值，退化为全UNKNOWN。这与补数据后在新测试上仍保持全部已知输出的结果不同。旧模型许多错误的P(True)&gt;0.99；种子2的新in误报中也有3次P(True)&gt;0.998，因此高概率本身不是可靠性证据。</p>')
 blocks.append('<h2>5. 完整1500步对照：相同预算下也存在取舍</h2>'+table(['种子','固定更新数','新held错误/38','新open错误/38','新in误报/38','新held UNKNOWN/38','旧D held漏判/99','旧D原近距离误报/2'],finalrows))
 blocks.append('<p>三个最终checkpoint的原始held/open错误仍为0，支持定向补数据有实际作用。但新in误报为1、3、4次，旧D held仍漏判4次，且旧D原近距离误判均保留2次。最终种子1在开发集校准失败，held全部UNKNOWN；它不能作为可部署成功结果。这里仅按训练前规定列作诊断，没有根据测试替换开发集选定checkpoint。</p>')
 blocks.append('<h2>6. 同一图像序列的前后预测</h2>'+img(E/'figures/paired_examples.png')+'<p class="caption">六例在测试推理前指定，主展示均为种子0。左右模型读取完全相同的四帧，图中显示最后一帧：左为静态相机，中为夹爪相机，右为真值与前后P(True)。绿色/红色标题表示补后对/错，不表示物体身份；画面中的红、蓝、粉才是物体颜色。P(True)是模型输出，不是经验证的正确概率。整体结论来自上方全量计数。</p>')
 blocks.append('<h2>7. 新错误与原因证据</h2>'+img(E/'figures/regression_examples.png')+'<p class="caption">上行是按索引找到的首个“旧正确、新漏判”held记录，使用种子0；下行是种子2首个新in误报。两者是失败诊断案例，不是随机样本，也不用于估计总体错误率。</p><ul><li><b>held回归：</b>首个新增漏判的红块明明被拿住，概率从0.99996降到0.19124。新增漏判对象都可见、原标签有效，不能归因于“物体消失便当作未held”。局部判定边界或共享表示变化，以及只用新开发集选模，都是可能因素，未做独立因果消融，不能唯一归因。</li><li><b>跨谓词影响：</b>补采只监督held/open，共享编码器却会改变in的输入表示。种子2在四个新增状态上把桌面物体误读为抽屉内，三个错误接近概率1。原D的in指标未退步，也不能保证新状态上的其他谓词稳定。原数据标签未改变。</li><li><b>样本分布仍窄：</b>新偏移空夹爪负例与旧真实抓空失败外观不完全相同；新增测试全部正确不能替代旧失败的回归验证。</li></ul>')
 blocks.append(build_addendum(E,table,img))
 blocks.append('<h2>8. 限制、预设标准与下一步</h2><p>预先设定的新增主条件和“旧D总体准确率下降不超过2个百分点”检查均通过，但后者被大量负例稀释，不能抵消held召回下降和in新误报。报告保留原标准，不事后改成失败或成功门槛；实际结论是部分问题修正、有明确回归，尚不满足整体可靠的任务规划事实接口。</p><p>开发C_pink_closed与测试D_red_open有一对跨环境近位置布局；原生场景、物体几何、抽屉状态不同。按训练前固定的敏感性检查排除该D根后，5根32窗口上held/open改善仍保持，种子2的4次in误报也仍存在。根级自举区间保存在evaluation.json，但6个测试根不足以给出广泛泛化保证。</p><p>若继续，优先在有限预算内同时约束原分布的held召回，并考虑把补采中可审计的in负例纳入监督；需要另设独立开发依据，不使用本轮测试错误来挑超参数。本轮没有执行这些调整，也没有新增采集、换模型或启动LTL规划。当前状态读取的进步不等于未来状态预测或LTL规划通过。</p>')
 blocks.append('<h2>9. 本机交付</h2>'+table(['文件','用途'],[['configs/protocol.json','训练前冻结的数据配比、预算、根划分、阈值规则与评价标准'],['results/model_threshold_freeze.json','所有模型和阈值在测试前冻结的记录与哈希'],['models/augmented_seed{0,1,2}_selected.pt','三个开发集选定模型；结构与旧模型完全相同'],['models/augmented_seed{0,1,2}_final.pt','三个固定1500步对照；不替换主结果'],['results/evaluation.json / predictions.csv','逐谓词、逐子集、逐样本原始概率与UNKNOWN'],['results/failure_diagnosis.json','原分布新增漏判、in误报、根数量与训练曝光账本'],['data/','新增冻结特征、输入数组和独立监督掩码'],['source/ / logs/','完整代码、训练曲线、每一步的采样索引'],['report.html','本报告；四幅图已内嵌，独立可打开']]))
 blocks.append('<p class="sub">文件仅保存在本机；旧模型、原数据、补采清单、控制器保持不变。未上传新增文件。没有未完成的执行步骤，但科学结论仍存在上述回归与泛化限制。</p>')
 css='body{margin:0;background:#f3f6f9;color:#1a2c3c;font:16px/1.72 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1260px;margin:auto;padding:40px;background:white}h1{font-size:30px;line-height:1.3}h2{font-size:23px;margin-top:40px;padding-bottom:8px;border-bottom:2px solid #dce5ec}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{padding:10px;border:1px solid #d6e0e8;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3f7}tr:nth-child(even){background:#fafcfd}.sub,.caption{font-size:14px;color:#536676}.callout{padding:20px;background:#fff1e3;border-left:5px solid #d17c35}img{display:block;max-width:100%;height:auto;margin:18px auto}li{margin:12px 0}@media(max-width:700px){main{padding:16px}table{font-size:12px}th,td{padding:6px}}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>补样本后的命题读取对照</title><style>'+css+'</style><main>'+''.join(blocks)+'</main></html>')
 summary={'status':'completed_partial_improvement_with_regressions','new_test_roots':6,'new_test_windows':38,'old_D_regression_roots':20,'old_D_boundaries':300,'seeds':[0,1,2],'new_test_held_FP_old':[6,8,6],'new_test_held_FP_new':[0,0,0],'new_test_held_FN_new':[0,0,0],'new_test_open_FP_old':[8,9,8],'new_test_open_FP_new':[0,0,0],'new_test_in_FP_new':[0,0,4],'new_test_held_unknown_new':[0,0,0],'old_D_held_FN_old':[1,0,0],'old_D_held_FN_new':[7,7,4],'old_D_held_FP_old':[2,4,2],'old_D_held_FP_new':[0,2,2],'old_D_open_FP_old':[19,19,19],'old_D_open_FP_new':[0,0,0],'old_D_original_near_failure_FP_old':[2,2,2],'old_D_original_near_failure_FP_new':[0,2,2],'selected_updates':[300,150,1500],'fixed_budget_updates_per_seed':1500,'new_total_presentations_per_seed':90000,'model_structure_unchanged':True,'no_new_collection_or_control_or_world_model_or_planning':True,'report':str(E/'report.html'),'upload_status':'not_uploaded_new_authorization_not_given','remaining_execution_blockers':[],'recommendation':'Keep old frozen models as current reproducible baseline; augmented models are candidates only, not a deployment reliability claim.','regression_state_analysis':str(E/'results/regression_state_analysis.json'),'interpretation':'Targeted data repairs new held/open distribution and old drawer errors, but original held recall drops and one seed develops new in errors. No overall reliability success claim.'}
 (E/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('REPORT_READY',E/'report.html',flush=True)
if __name__=='__main__':main()
