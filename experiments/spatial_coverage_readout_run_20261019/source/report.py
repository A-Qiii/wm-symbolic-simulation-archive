from common import *
import csv,html,re,base64,collections,ast
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |'for r in rows])
def pct(x):return '不适用'if x is None else f'{x*100:.1f}%'
def signed(x):return '不适用'if x is None else f'{x*100:+.1f}'
def csvread(f):return list(csv.DictReader((S/'results'/f).open()))
ns={'re':re,'html':html};node=next(n for n in ast.parse((O/'source/report.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='render_markdown');exec(compile(ast.Module(body=[node],type_ignores=[]),'<renderer>','exec'),ns)
def main():
 sys.addaudithook(guard);analysis=read(S/'results/analysis.json');rows=analysis['summary'];rm={r['root_id']:r for r in read(S/'data/roots_manifest.json')};fair=read(S/'results/admission_fairness_audit.json');pairaudit=read(S/'results/training_pair_audit.json');prior=read(S/'results/prior_audit.json');rawcounts=csvread('counts.csv');execrows=csvread('paired_training_execution.csv');cov=csvread('paired_h0_coverage.csv')
 def get(split,seed,metric,cohort='h0'):return next(r for r in rows if(r['split'],r['seed'],r['metric'],r['cohort'])==(split,seed,metric,cohort))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
 colors={0:'#8855a5',1:'#dd8719',2:'#008e95'};fig,ax=plt.subplots(figsize=(11,5),layout='constrained');rng=np.random.default_rng(4)
 for e,base in [('A',3),('B',1)]:
  for arm,offset in [('control',0),('coverage',-.45)]:
   pts=[r for r in cov if r['stratum']=='all40'and r['scene']==e and r['arm']==arm and int(r['class'])<3]
   for k in range(3):
    q=[r for r in pts if int(r['class'])==k];ax.scatter([float(r['x_m'])for r in q],base+offset+rng.uniform(-.07,.07,len(q)),c=colors[k],s=22,alpha=.8,label=['Left','Middle','Right'][k]if e=='A'and arm=='control'else None)
 for cut in[-.1466666667,.1466666667]:ax.axvline(cut,c='black',ls='--',lw=1)
 ax.set(xlim=(-.38,.38),yticks=[3,2.55,1,.55],yticklabels=['A control','A coverage','B control','B coverage'],xlabel='Actual h0 table-local x (metres)',title='Training h0 support: one point per table-supported object; held objects excluded');ax.legend(ncol=3);fig.savefig(S/'figures/training_coverage.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,metric,title in zip(axes,['table_macro_recall','exact4'],['h0 table macro recall','h0 all-four-field accuracy']):
  for si,split in enumerate(['old_band','gap','externalC']):
   for seed in[0,1]:
    r=get(split,seed,metric);x=si+(-.14 if seed==0 else.14);v=r['delta']*100;lo=r['lo95']*100;hi=r['hi95']*100;c=['#2166ac','#e08120'][seed];ax.vlines(x,lo,hi,color=c,lw=1.5);ax.plot([x-.035,x+.035],[lo,lo],c=c);ax.plot([x-.035,x+.035],[hi,hi],c=c);ax.plot(x,v,'o',c=c,label=f'Encoder {seed}'if si==0 else None)
  ax.axhline(0,c='black',lw=1);ax.set(xticks=range(3),xticklabels=['New A/B old-band\n10 roots','New A/B gap\n10 roots','Existing C exploratory\n20 roots'],ylabel='Coverage minus control (percentage points)',title=title);ax.grid(axis='y',alpha=.2);ax.legend()
 fig.suptitle('Paired root bootstrap: dots are observed differences; lines are 95% intervals');fig.savefig(S/'figures/paired_h0.png');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');conf=csvread('confusions.csv');labels=['left','middle','right','drawer','slideL','slideR','held','other']
 for seed in[0,1]:
  for col,arm in enumerate(['control','coverage']):
   a=np.zeros((3,8),int)
   for r in conf:
    if r['split']=='gap'and int(r['seed'])==seed and r['cohort']=='h0'and r['arm']==arm and int(r['truth'])<3:a[int(r['truth']),int(r['prediction'])]+=int(r['count'])
   norm=a/np.maximum(a.sum(1,keepdims=True),1);ax=axes[seed,col];im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='auto')
   for i in range(3):
    for j in range(8):
     if a[i,j]:ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white'if norm[i,j]>.5 else'black')
   ax.set(xticks=range(8),xticklabels=labels,yticks=range(3),yticklabels=labels[:3],xlabel='Predicted object location',ylabel='True table class',title=f'Gap h0 | encoder {seed} | {arm}');ax.tick_params(axis='x',rotation=45)
 fig.colorbar(im,ax=axes,label='Recall fraction within true class',shrink=.8);fig.savefig(S/'figures/gap_confusion.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
 for seed in[0,1]:
  for arm in['control','coverage']:
   r=read(O/f'results/train_object_seed{seed}.json')if arm=='control'else read(S/f'results/train_seed{seed}.json');curve=r['curve'];ax.plot([x['update']for x in curve],[x.get('total',x.get('loss'))for x in curve],c=['#2166ac','#e08120'][seed],ls='--'if arm=='control'else'-',label=f'{arm}, encoder {seed}')
 ax.set(xlabel='Head optimizer update',ylabel='Four-field training cross-entropy',yscale='log',title='Same initialization, schedule and all 600 four-field label vectors');ax.legend(ncol=2);fig.savefig(S/'figures/training_loss.png');plt.close(fig)
 # Deterministic first error/gain/regression examples; no tuning after viewing them.
 arrays={(s,k,a):dict(np.load(S/f'data/predictions_{a}_seed{k}_{s}.npz'))for s in['old_band','gap','externalC']for k in[0,1]for a in['control','coverage']};selected=[]
 for split in['gap','old_band']:
  for seed in[0,1]:
   a=arrays[split,seed,'control'];b=arrays[split,seed,'coverage'];pred=lambda q:np.stack([q['prob'][:,j*8:j*8+8].argmax(1)for j in range(3)]+[q['prob'][:,24:].argmax(1)],1);pa,pb=pred(a),pred(b);h0=(a['step']==0)&np.char.endswith(a['episode_id'],'_b0')&a['valid'].all(1);ca=(pa==a['cat']).all(1);cb=(pb==b['cat']).all(1)
   for cond,reason in [(h0&~cb,'first remaining full-state error'),(h0&~ca&cb,'first corrected full-state error'),(h0&ca&~cb,'first full-state regression')]:
    ii=np.where(cond)[0]
    if len(ii):
     i=sorted(ii,key=lambda j:str(a['root_id'][j]))[0];eid=str(a['episode_id'][i]);item=(split,eid,reason+f', encoder {seed}')
     if not any(x[:2]==item[:2]for x in selected):selected.append(item)
 selected.append(('externalC','C_core1200_00_b1','previously audited C right-to-middle case'))
 index={(x['episode_id'],x['step']):x for x in read(S/'data/features/index.json')['items']+read(P/'data/core1200_v1/features/index.json')['items']};rowlookup={(r['episode_id'],r['step']):r for r in read(S/'data/data_manifest.json')['records']+read(P/'data/core1200_v1/data_manifest.json')['records']};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);cases=[]
 for ci,(split,eid,reason)in enumerate(selected):
  with np.load(index[eid,0]['public_inputs'][-1])as z:static=Image.fromarray(z['rgb_static']);wrist=Image.fromarray(z['rgb_gripper'])
  canvas=Image.new('RGB',(1100,480),'white');draw=ImageDraw.Draw(canvas);draw.text((10,10),eid+' | '+reason,font=font,fill='black');canvas.paste(static.resize((360,360),Image.Resampling.NEAREST),(10,65));canvas.paste(wrist.resize((252,252),Image.Resampling.NEAREST),(385,65));draw.text((10,40),'Static camera - saved RGB',font=font,fill='black');draw.text((385,40),'Gripper camera - saved RGB',font=font,fill='black');outputs=[]
  for seed in[0,1]:
   for j,arm in enumerate(['control','coverage']):
    q=arrays[split,seed,arm];i=np.where((q['episode_id']==eid)&(q['step']==0))[0].item();p=q['prob'][i];gt=q['cat'][i];guess=[int(p[z*8:z*8+8].argmax())for z in range(3)]+[int(p[24:].argmax())];text=f'Encoder {seed} / {arm}\nR/B/P: '+', '.join(labels[k]for k in guess[:3])+f'\nDrawer: {guess[3]}'
    draw.multiline_text((650,65+seed*160+j*75),text,font=font,fill='#666666'if arm=='control'else'#008e95',spacing=4);wrong_fields=[j for j in range(4)if guess[j]!=int(gt[j])];replaced=list(guess);diagnostic=None
    if wrong_fields and wrong_fields[0]<3:
     field=wrong_fields[0];truth_label=read(rowlookup[eid,0]['labels_path'])['before'];x=truth_label['continuous']['objects'][COLORS[field]]['position_m'][0];diagnostic={'color':COLORS[field],'truth_class':int(gt[field]),'predicted_class':guess[field],'GT_x_m':x,'table_boundary_margin_m':min(abs(x+.14666666666666667),abs(x-.14666666666666672))if int(gt[field])<3 else None,'public_roi_availability':float(q['availability'][i,field]),'GT_label_valid':bool(truth_label['categorical_valid'][field])}
    if wrong_fields:replaced[wrong_fields[0]]=int(gt[wrong_fields[0]])
    outputs.append({'seed':seed,'arm':arm,'truth':gt.tolist(),'prediction':guess,'probabilities':p.tolist(),'first_wrong_field':(COLORS+['drawer'])[wrong_fields[0]]if wrong_fields else None,'first_wrong_diagnostic':diagnostic,'single_field_GT_scoring_substitution':replaced,'exact4_after_single_substitution':replaced==gt.tolist(),'remaining_wrong_fields':[(COLORS+['drawer'])[j]for j in range(4)if replaced[j]!=int(gt[j])]})
  draw.text((10,440),'GT R/B/P: '+', '.join(labels[int(k)]for k in gt[:3])+f'; drawer {gt[3]} (0 closed, 1 partial, 2 open)',font=font,fill='black');path=f'case_{ci:02d}.png';canvas.save(S/'figures'/path);cases.append({'split':split,'episode_id':eid,'reason':reason,'figure':path,'outputs':outputs})
 save(S/'results/error_cases.json',cases)
 held_rows=[];target_rows=[]
 for split in ['old_band','gap']:
  for seed in [0,1]:
   a=arrays[split,seed,'control'];b=arrays[split,seed,'coverage']
   for i in np.where((a['step']==0)&np.char.endswith(a['episode_id'],'_b0'))[0]:
    r=rm[str(a['root_id'][i])]
    if r['initial_state']=='open_held':
     truth=','.join(c for j,c in enumerate(COLORS)if a['cat'][i,j]==6)or'none';preds=[','.join(c for j,c in enumerate(COLORS)if q['prob'][i,6+8*j]>=.5)or'none'for q in[a,b]];held_rows.append([split,r['root_id'],seed,r['held_color'],truth,*preds])
   if split=='gap':
    for k in [1,2]:
     ix=[(i,COLORS.index(rm[str(a['root_id'][i])]['gap_target_color']))for i in np.where((a['step']==0)&np.char.endswith(a['episode_id'],'_b0'))[0]if rm[str(a['root_id'][i])]['gap_target_class']==k];valid=[(i,j)for i,j in ix if a['valid'][i,j]];correct=[sum(int(q['prob'][i,j*8:j*8+8].argmax()==a['cat'][i,j])for i,j in valid)for q in[a,b]];target_rows.append({'seed':seed,'planned_target_class':k,'planned_roots':len(ix),'valid_roots':len(valid),'actual_class_matches_plan':sum(int(a['cat'][i,j]==k)for i,j in valid),'control_correct':correct[0],'coverage_correct':correct[1]})
 save(S/'results/target_class_metrics.json',target_rows)
 # Tables and interpretation.
 improve=all(get('gap',seed,'table_macro_recall')['delta']>0 for seed in[0,1]);robust=improve and all(get('gap',seed,'table_macro_recall')['lo95']>0 for seed in[0,1]);conclusion='新A/B空白区收益不均衡：两个E种子的h0桌面宏召回分别变化'+signed(get('gap',0,'table_macro_recall')['delta'])+'、'+signed(get('gap',1,'table_macro_recall')['delta'])+'个百分点；第二个种子的95%区间跨0。预定目标召回仅从7/10到8/10、9/10保持9/10，不能认为空间泛化已稳定解决。'
 lines=['# 固定对象读出后的空间覆盖对照','', '**本轮有限实验已完成：40个新训练根、20个新A/B评价根，共720条真实转移；两个新读出头各1500步。** '+conclusion,'','保留能力与关系也存在代价：E种子1的全边界持物身份全对率在旧带内从91.3%降至82.6%，空白区从88.9%降至82.2%。C初态四字段全对仍仅1/20和0/20；全边界右区召回为13.7%和28.1%。这不支持“只扩大读出监督覆盖就能解决C泛化”的判断，也不能仅凭本轮把余下错误归因于编码器或外观。','','这是当前观测读出实验，不是RSSM自由预测或完整指南验收。编码器曾在旧窄带40根上训练，本轮全部冻结；新增监督仅训练对象读出。C20根曾被查看，仅为探索性复查。','','## 改变与保持的内容','',table(['元素','本轮实际做法与原因'],[['训练对照','旧窄带40根对新覆盖40根；A20/B20，逐根复制颜色类别、y、yaw、初态、持物颜色和3×4技能模板，只改变x'],['采样的实际含义','33点可行域筛查后，联合分配每场景×颜色×类别的分位名额，围绕代表点±1mm连续抖动并复检。不是整个物理类内均匀采样，也不是无条件扩大区间'],['准入','完整旋转底面≤2mm网格射线、物体间距≥10mm、场景穿透≤1mm、五个接近高度的IK/桌面检查、24次电机保持后的复位漂移≤3mm。有限代理，不证明任意动态路径成功'],['控制器','同一V1物理执行器、900控制步预算；接受根之后的真实失败保留，无表现筛选'],['模型','两个已有冻结E，公开像素严格多数颜色路由、对象池化与309165参数头均相同；不使用GT位姿或接触作为模型输入'],['优化','原初始化、根×分支×边界采样表与归一化，1500×64、AdamW lr0.0003/wd0.0001，无类别权重；不按结果追加更新'],['新评价','旧带内10根+空白区10根，A/B各半；坐标在训练前冻结，两个头冻结后才正式采集评价轨迹'],['时间隔离的边界','物理几何预检曾查询候选布局/复位，未评分模型或保存评价轨迹；不是完全未接触几何的盲测。C仅冻结头后复用特征和评分。']]),'','## 新准入与旧数据的可比性限制','',f"旧40根有{fair['old40_pass']}根通过新几何代理，{fair['old40_fail']}根未通过（10个接近姿态、1个底面）。对被标记颜色、接近原位置/朝向且具备真实前提的旧抓取，查到{fair['matching_actual_lift_attempts_on_geometrically_flagged_colors']}次，其中{fair['matching_actual_success']}次成功。这说明新检查偏保守，不能把旧11根称为无效数据。",'', '本轮估计的是“空间覆盖与几何准入共同定义的x采样政策”的总效应。本次600个训练边界四字段标签实际逐一相同，因此已观察标签先验的混淆没有发生；仍不能把保守准入、实际轨迹/视角变化排除后声称纯几何机制。以下29根分层只是同一40根训练的描述性敏感性分析，不能冒充另训的公平29根对照。所有11根均保留，未重加权。','']
 gr=[]
 for name,v in pairaudit['strata'].items():
  armcounts={arm:[r for r in execrows if r['stratum']==name and r['arm']==arm]for arm in['control','coverage']};gr.append([name,v['roots'],f"{v['h0_exact_pairs']}/{v['roots']}",f"{v['all_boundary_exact_pairs']}/{v['boundaries']}",sum(int(r['goal_success'])for r in armcounts['control']),sum(int(r['goal_success'])for r in armcounts['coverage'])])
 lines +=[table(['训练根分层','根数','h0类别完全匹配','全部边界标签匹配','旧组技能成功','覆盖组技能成功'],gr),'','![训练空间覆盖](figures/training_coverage.png)','图1：横轴为实际h0桌面局部x（米）；纵向四行是A/B场景各自的旧组和覆盖组。每点是一个由桌面支撑的对象，持物对象不画；紫/橙/青分别是真值左/中/右类，不表示物体颜色。黑虚线为统一类别边界。图示实际覆盖变化，不能据此推断未采区域也安全或可识别。','','## 新根评价：主结果','']
 mt=[]
 for split,label in [('old_band','新A/B旧带内10根'),('gap','新A/B空白区10根'),('externalC','旧C20根（探索）')]:
  for seed in[0,1]:
   for metric,mname in [('table_macro_recall','h0桌面宏召回'),('exact4','h0四字段全对')]+([('target_recall','h0预定目标召回')]if split=='gap'else[]):
    r=get(split,seed,metric);mt.append([label,seed,mname,pct(r['control']),pct(r['coverage']),signed(r['delta']),f"[{signed(r['lo95'])}, {signed(r['hi95'])}]",r['valid_bootstrap']])
 lines +=[table(['根集','E种子','指标','旧窄带头','覆盖头','差/百分点','95%配对区间/百分点','有效重采样'],mt),'','每套只含10个独立新根，三分支和多边界不增加独立样本数。2000次按根配对bootstrap，种子分开，区间未做多指标/多种子比较校正；每个完整根集先固定其有真值支持的桌面类别（A/B通常三类，C通常中/右两类）；重采样缺失任一固定类别时，该次宏召回不定义，表中披露有效次数。h0持物仅2根，不把其小样本区间当稳定泛化证据。','','![初态配对差异](figures/paired_h0.png)','图2：横轴是共同评价根集，纵轴是覆盖头减旧头的百分点差。蓝点/橙点分别是两个冻结编码器种子；点为实测差，竖线为按根配对95%区间，黑水平线为无差异。左图桌面宏召回，右图四字段全对。C在图中明确为既有探索性根集。','','![空白区混淆](figures/gap_confusion.png)','图3：仅新A/B空白区h0的真值桌面对象。行是真值左/中/右，列是预测位置；每行颜色从浅到深表示该真值类的比例，格内数字为对象数。上下是E种子，左右是旧/覆盖头。对角线上的深色代表正确位置读出，非对角线显示首次状态识别错误的方向。','','## 预定目标与初始持物逐例核查','',table(['E种子','计划目标类','计划/有效根','实际类仍匹配计划','旧头正确','覆盖头正确'],[[r['seed'],['左','中','右'][r['planned_target_class']],str(r['planned_roots'])+'/'+str(r['valid_roots']),r['actual_class_matches_plan'],r['control_correct'],r['coverage_correct']]for r in target_rows]),'',table(['根集','持物根','E种子','计划持物','实际GT持物','旧头Holding集合','覆盖头Holding集合'],held_rows),'','每套新根只有两例计划持物初态。表中Holding集合按0.5阈值列出具体身份，不把这两例当稳定关系泛化证据。目标类为采集前指定的A三右/两中、B五中；表中也检查实际准备动作后是否仍属计划类别。','','## 关系与后续边界，防止只看位置改善','']
 rt=[]
 for split in['old_band','gap','externalC']:
  for seed in[0,1]:
   for metric,name in [('Holding_recall','Holding正例召回'),('Holding_FPR','Holding误报率'),('holding_identity_positive','正例持物身份全对'),('InDrawer_recall','InDrawer正例召回'),('InDrawer_FPR','InDrawer误报率'),('drawer_accuracy','抽屉准确率'),('exact4','四字段全对')]:
    r=get(split,seed,metric,'all_boundaries');rt.append([split,seed,name,pct(r['control']),pct(r['coverage']),signed(r['delta'])])
 lines +=[table(['根集','E种子','全边界次级指标','旧头','覆盖头','差/百分点'],rt),'','类别采用argmax，Holding/InDrawer关系按各自类别概率≥0.5；两种决策可能不同。无正例的指标不定义。原始分子/分母、无效标签数量和逐根统计保存在counts.csv、root_statistics.csv中，不能用多帧样本量掩盖根级不确定性。','','## 训练监督与数值复现','',table(['字段','600边界标签先验总变差距离'],[[r['field'],f"{r['total_variation']:.4f}"]for r in prior['fields']]),'','h0逐根匹配；后续600边界中完全相同标签序列的边界数为'+str(prior['paired_boundary_label_exact'])+'。未调整损失权重、补成功样本或删除失败。按场景×初态×颜色/字段×类别的样本数与根数见training_priors.csv；全部40、原通过29和未通过11的配对统计另存。','','![训练损失](figures/training_loss.png)','图4：横轴为读出头更新步数，纵轴为四字段交叉熵的对数刻度。蓝/橙对应E种子；虚线旧头、实线覆盖头。两组对应相同采样索引和本次实际完全相同的四字段标签；视觉特征不同，曲线差仍不能单独证明表示机制。没有延长训练或选择最佳步。','','旧头复用经过初始化、权重、采样表、显式600索引与归一化检查；当前环境对每个E的20个旧边界特征、每个旧头600边界预测逐位复现。新增训练仅两个头，共3000更新；RSSM更新0。','','## 首个有效错误与图像证据','', '所有评分使用当前四帧，因此本实验的首个有影响错误是当前状态读出，不涉及后续动力学或搜索。下面按根ID固定顺序选择首个剩余错误、首个修正和首个退化，展示共同原始双相机帧及两个种子的预测；这些案例未用于再训练、调阈值或改采样。error_cases.json另保存单项替换诊断：仅把首个错误字段替换为GT，检查四字段是否恢复全对及仍剩哪些错误；这是评分层反事实，不是模型重跑或闭环成功，也不能区分编码器与读出头的内部成因。公开ROI可用比例指四帧×双相机中32个对象ROI token的有效比例，不是分类置信度；各图中的原始对象颜色保持不变。']
 for c in cases:
  explanations=[]
  for out in c['outputs']:
   if out['arm']!='coverage':continue
   diag=out['first_wrong_diagnostic']
   if diag:
    margin=''if diag['table_boundary_margin_m']is None else f"，距物理分界{diag['table_boundary_margin_m']*1000:.1f}mm"
    explanations.append(f"E{out['seed']}首个错误为{diag['color']}对象：真值{labels[diag['truth_class']]}→预测{labels[diag['predicted_class']]}，GT x={diag['GT_x_m']:.4f}m{margin}；公开ROI可用比例{diag['public_roi_availability']:.3f}。只替换该字段后四字段全对={out['exact4_after_single_substitution']}，余错字段={','.join(out['remaining_wrong_fields'])or'无'}。")
   elif out['first_wrong_field']:explanations.append(f"E{out['seed']}首错是抽屉字段；单项替换后全对={out['exact4_after_single_substitution']}。")
   else:explanations.append(f"E{out['seed']}本帧四字段全对。")
  lines +=['',f"![{c['episode_id']}](figures/{c['figure']})",f"案例 {c['episode_id']}：{c['reason']}。左为原始静态相机、中央为夹爪相机，均仅最近邻放大；灰字旧头、青字覆盖头，R/B/P为红/蓝/粉对象，底部是真值位置和抽屉类别。图像用于解释实际可见对象，不是合成数据。",' '.join(explanations)]
 collection=read(S/'results/collection_audit.json');compute_train=read(S/'results/compute_train_complete.json');compute_eval=read(S/'results/compute_evaluation_complete.json')
 lines +=['','## 实际预算、物理失败与工程记录','',table(['数据段','转移','成功','拒绝（未满足前提等）','实际执行后失败','无效四字段数','未请求设备>3mm'],[[split,v['transitions'],v['goal_success'],v['status_counts'].get('refused',0),v['status_counts'].get('failed',0),v['invalid_after_fields_first4'],v['unrequested_device_excursion_gt3mm']]for split,v in collection['split_summary'].items()]),'',f"正式物理采集合计约{(collection['first4_seconds']+collection['remaining_train_seconds']+collection['evaluation_seconds'])/60:.1f}分钟；训练缓存/数值核验/冻结特征/两头拟合共{compute_train['seconds']:.1f}秒，评价计算{compute_eval['seconds']:.1f}秒。当前目录约{collection['bytes_current']/1e9:.2f}GB（后续报告与校验文件略增）。122次拒绝和14次实际执行后失败全部保留；12次是目标未达且重试前提消失，2次为抬升净空受阻。",'', '预检初始化和独立分层的失败历史、物理可行子域与联合分配过程均保留；它们发生在正式根清单冻结和学习前，未使用模型表现选根。首次缓存调用因训练确定性开关触发缺编译器的路径而停止，当时新增拟合为0；之后仅恢复历史缓存运行设置，旧缓存所有字段逐位复现，训练与提取仍用原确定性设置。见engineering_attempt1、geometry_attempt1/2、engineering_cache_attempt1和logs。']
 lines +=['','## 完成范围与复核入口','','720条新增转移、900个新增边界缓存、2个1500步头、20根共同新评价与C20探索性复查均已完成。未上传文件，旧476个已封存文件在最终核验中逐个比对。正式完整指南的CALVIN/HULC、完整RSSM自由预测、学习后继TS/LTLf搜索与闭环验收仍不能由本轮读出实验替代。','','主要证据：configs/roots_frozen.json、configs/heads_frozen.json、results/numerical_control_preflight.json、results/training_pair_audit.json、results/paired_metrics.csv、results/error_cases.json。详细技术脚本、公开观测、私有离线标签、控制视频和失败记录分目录保存；复核见source/verify.py。']
 md='\n'.join(lines)+'\n';(S/'REPORT.md').write_text(md);body=ns['render_markdown'](md)
 for p in(S/'figures').glob('*.png'):body=body.replace('figures/'+p.name,'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>空间覆盖读出对照</title><style>body{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.7 sans-serif;color:#17222e}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ced7de;padding:7px;text-align:left}th{background:#edf4f5}img{max-width:100%;height:auto}strong{color:#006a75}</style>'+body+'</html>')
 save(S/'results/report_complete.json',{'utc':now(),'conclusion':conclusion,'images':len(list((S/'figures').glob('*.png'))),'cases':len(cases)});print(conclusion,flush=True)
if __name__=='__main__':main()
