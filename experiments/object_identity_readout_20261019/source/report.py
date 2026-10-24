from common import *
import csv,base64,html,collections,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
def csvread(name):return list(csv.DictReader((S/'results'/name).open()))
def num(x):return float(x)if x not in ['',None]else None
def pct(x):return '无正例/不适用'if x is None else f'{100*x:.1f}%'
def ci(r):return '不适用'if r['lo95']==''else f"[{100*float(r['lo95']):+.1f}, {100*float(r['hi95']):+.1f}]"
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |'for row in rows])
def render_markdown(md):
 # This report uses a small controlled subset; no external Markdown dependency.
 def inline(s):return re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',html.escape(s))
 out=[];lines=md.splitlines();i=0
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines)and lines[i].startswith('|'):
    rows.append([v.strip()for v in lines[i].strip('|').split('|')]);i+=1
   out.append('<table><thead><tr>'+''.join('<th>'+inline(v)+'</th>'for v in rows[0])+'</tr></thead><tbody>')
   out.extend('<tr>'+''.join('<td>'+inline(v)+'</td>'for v in row)+'</tr>'for row in rows[2:]);out.append('</tbody></table>');continue
  if line.startswith('- '):
   out.append('<ul>')
   while i<len(lines)and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
   out.append('</ul>');continue
  m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
  if m:out.append('<figure><img alt="'+html.escape(m[1],quote=True)+'" src="'+html.escape(m[2],quote=True)+'"></figure>')
  elif line.startswith('#'):
   n=len(line)-len(line.lstrip('#'));out.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
  else:out.append('<p>'+inline(line)+'</p>')
  i+=1
 return '\n'.join(out)
def main():
 sys.addaudithook(audit_writes);metrics=csvread('metrics.csv');pairs=csvread('paired_differences.csv');counts=csvread('counts.csv');conf=csvread('confusions.csv');compute=read(S/'results/compute_complete.json')
 def metric(split,seed,arm,name,cohort='all_boundaries',stratum='all'):
  return next(r for r in metrics if (r['split'],r['seed'],r['arm'],r['metric'],r['cohort'],r['stratum'])==(split,str(seed),arm,name,cohort,stratum))
 def count(split,seed,arm,cohort='all_boundaries',stratum='all'):
  return next(r for r in counts if (r['split'],r['seed'],r['arm'],r['cohort'],r['stratum'])==(split,str(seed),arm,cohort,stratum))
 def pair(split,seed,name,cohort='all_boundaries',stratum='all'):
  return next(r for r in pairs if (r['split'],r['seed'],r['metric'],r['cohort'],r['stratum'])==(split,str(seed),name,cohort,stratum))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
 names=['loc_macro_recall','holding_identity_positive','InDrawer_recall','exact4'];labels=['Location\nmacro recall','Held-object identity\n(positive states)','InDrawer\npositive recall','All 4 fields\ncorrect'];colors={'global':'#6B7280','object':'#008C95'}
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for seed in [0,1]:
  for col,split in enumerate(['test','externalC']):
   ax=axes[seed,col];xx=np.arange(4)
   for arm,shift in [('global',-.18),('object',.18)]:
    vals=[num(metric(split,seed,arm,k)['value'])*100 for k in names];bars=ax.bar(xx+shift,vals,.34,label=arm,color=colors[arm]);ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
   ax.set(xticks=xx,xticklabels=labels,ylim=(0,108),ylabel='Percent (higher is better)',title=f"{'A/B test: 10 roots, 150 boundaries'if split=='test'else'C: 20 roots, 300 boundaries'} | encoder {seed}");ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2);fig.suptitle('Frozen readout: global vs visual identity routing + object pooling',fontsize=14)
 fig.savefig(S/'figures/primary.png');fig.savefig(S/'figures/primary.pdf');plt.close(fig)
 classlabels=['left','middle','right','drawer','slide L','slide R','held','other']
 for split in ['test','externalC']:
  fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained');cmap=plt.cm.Blues.copy();cmap.set_bad('#e5e7eb')
  for seed in [0,1]:
   for col,arm in enumerate(ARMS):
    rows=[r for r in conf if r['split']==split and r['seed']==str(seed)and r['arm']==arm and r['cohort']=='all_boundaries'and r['stratum']=='all'];mat=np.zeros((8,8),int)
    for r in rows:mat[int(r['truth']),int(r['prediction'])]+=int(r['count'])
    den=mat.sum(1,keepdims=True);norm=np.divide(mat,den,out=np.full((8,8),np.nan),where=den>0);ax=axes[seed,col];im=ax.imshow(norm,vmin=0,vmax=1,cmap=cmap)
    for i in range(8):
     for j in range(8):
      if mat[i,j]:ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=9,color='white'if norm[i,j]>.55 else'black')
    ax.set(xticks=range(8),yticks=range(8),xticklabels=classlabels,yticklabels=classlabels,xlabel='Predicted location',ylabel='True location',title=f'{split}, encoder {seed}, {arm}');ax.tick_params(axis='x',rotation=45)
  fig.colorbar(im,ax=axes,label='Fraction within true class (gray = no true examples)',shrink=.8);fig.suptitle('Object location confusion: numbers are object-record counts',fontsize=14);fig.savefig(S/f'figures/confusion_{split}.png');plt.close(fig)
 # Predeclared illustrative cases, then the first h0 gain/loss in sorted order. No model selection.
 selected=[('externalC','C_core1200_00_b1',0,'previously audited right-to-middle case'),('externalC','C_core1200_02_b0',0,'previously audited held-identity case'),('test','B_core1200_26_b0',0,'previously audited held-object control')]+[('externalC',f'C_core1200_{r}_b0',0,'remaining initial-held C root')for r in ['06','12','16']]
 arrays={}
 for split in ['test','externalC']:
  for seed in [0,1]:
   for arm in ARMS:arrays[split,seed,arm]=dict(np.load(S/f'data/predictions_{arm}_seed{seed}_{split}.npz'))
   a=arrays[split,seed,'global'];b=arrays[split,seed,'object'];p=lambda x:np.stack([x['prob'][:,j*8:j*8+8].argmax(1)for j in range(3)]+[x['prob'][:,24:27].argmax(1)],1)
   ca=(p(a)==a['cat']).all(1);cb=(p(b)==b['cat']).all(1);h0=(a['step']==0)&np.char.endswith(a['episode_id'],'_b0')
   for mask,reason in [(h0&~ca&cb,f'first h0 full-state gain, encoder {seed}'),(h0&ca&~cb,f'first h0 full-state regression, encoder {seed}')]:
    ix=np.where(mask)[0]
    if len(ix):
     i=sorted(ix,key=lambda i:str(a['episode_id'][i]))[0];entry=(split,str(a['episode_id'][i]),0,reason)
     if not any(e[:3]==entry[:3]for e in selected):selected.append(entry)
 index={(x['episode_id'],x['step']):x for x in read(P/'data/core1200_v1/features/index.json')['items']};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);cases=[]
 for k,(split,eid,h,reason)in enumerate(selected):
  with np.load(index[eid,h]['public_inputs'][-1])as z:static=Image.fromarray(z['rgb_static']);gripper=Image.fromarray(z['rgb_gripper'])
  canvas=Image.new('RGB',(1100,480),'white');d=ImageDraw.Draw(canvas);d.text((12,8),eid+' | '+reason,font=font,fill='black');canvas.paste(static.resize((360,360),Image.Resampling.NEAREST),(12,58));canvas.paste(gripper.resize((252,252),Image.Resampling.NEAREST),(388,58));d.text((12,35),'Static RGB (original saved frame)',font=small,fill='black');d.text((388,35),'Gripper RGB',font=small,fill='black');details=[]
  for seed in [0,1]:
   for j,arm in enumerate(ARMS):
    a=arrays[split,seed,arm];i=np.where((a['episode_id']==eid)&(a['step']==h))[0].item();p=a['prob'][i];gt=a['cat'][i];guess=[int(p[z*8:z*8+8].argmax())for z in range(3)];drawer=int(p[24:].argmax());H=p[[6,14,22]]>=.5
    textline=f'Encoder {seed} | {arm}\nR/B/P: '+', '.join(classlabels[x]for x in guess)+f'\nDrawer: {drawer}; Holding: '+(','.join(c for c,v in zip(['R','B','P'],H)if v)or'none')
    d.multiline_text((660,65+seed*160+j*75),textline,font=small,fill=colors[arm],spacing=4);details.append({'seed':seed,'arm':arm,'truth':gt.tolist(),'prediction':guess+[drawer],'holding_probabilities':p[[6,14,22]].tolist()})
  d.text((12,435),'GT R/B/P: '+', '.join(classlabels[int(x)]for x in gt[:3])+f'; drawer {gt[3]} (0 closed / 2 open)',font=font,fill='black');file=f'case_{k:02d}.png';canvas.save(S/'figures'/file);cases.append({'split':split,'episode_id':eid,'step':h,'selection':reason,'figure':file,'outputs':details})
 save(S/'results/illustrative_cases.json',cases)
 lines=['# 保留对象对应是否改善当前状态读出', '', '**结论：保留公开视觉对象对应有明确价值，但没有解决C场景的完整位置泛化。** 两个固定编码器种子的对象组，在A/B测试和C集的位置宏平均、持物身份与入抽屉正例召回都优于容量匹配的全局组；按根配对区间支持这些差异。C的桌面右侧召回仍只有13.5%/20.6%，初始四字段全对仍为0/20。不能把本轮读出改进写成世界模型自由预测已修复。', '', '这是当前观测识别实验：每个边界使用自己的四帧观测，未修改或运行RSSM自由递推。本轮validation曾用于训练前路由预检；全部四个头冻结后才开始test/C路由、特征提取和预测（before_heldout_read在此仅指test/C）；这些根历史上已被查看，因此结果是探索性复用，不是全新盲测。', '', '## 对照具体改变了什么', '', table(['项目','固定条件/差异'],[['冻结部分','同一DINO缓存、原共同20根归一化、两个既有roots40 joint编码器；旧动力学权重不动'],['汇总与投影','同一次冻结空间Transformer后的256维token，分别全局/按颜色对象池化，再经同一冻结LayerNorm/Linear投影到512维'],['相同输入与容量','相同全局上下文、颜色查询、可见比例；无额外几何或颜色置信度。每头309,165参数，共享对象分类MLP；抽屉头独立裁剪且两组最终概率逐位相同'],['唯一主要处理差异','全局组用相同全局向量回答每种颜色；对象组选择当前四帧内公开视觉注册的对应ROI token。两组缺失对象均零focal并保留相同缺失信息'],['监督及预算','全部40训练根、600边界；每头固定1500步×64，AdamW lr=0.0003，四头共6000更新。按根均匀采样、配对相同初始化与顺序'],['解释限制','结构化视觉身份路由与对象内容选择一同变化；不能把改善唯一归因于池化。空间token已经过全局注意力；不是新训练的对象编码器。旧共同20根线性probe不是本轮匹配基线。']]), '', '## 路由预检与一次修订', '', '第一版完整候选歧义检查发现A场景棕色木纹满足旧宽松红色均值条件，导致289/571个可见红块记录被置空。预检据此停止，当时尚无特征提取、头训练或test/C读取。唯一修订是复用项目既有逐像素colour_masks，并要求一个可见掩码内严格多数像素符合该颜色。仍有多个候选就固定缺失，不靠GT选择。第一次代码和失败记录保留；原协议SHA与修订时间均已保存。', '', '修订后全部split可见对象的路由正确率为100%，无可见对象未解决、无错误身份；真正不可见对象不删样本。GT的物理UID只在独立审计中核对，未进入路由函数或读出输入。', '']
 rt=[]
 for split in SPLITS:
  r=read(S/f'results/routing_{split}.json');rt.append([split,r['records'],r['independent_roots'],sum(v['visible']for v in r['stats'].values()),sum(v['zero_fallback']for v in r['stats'].values()),len(r['wrong_assignments'])])
 lines +=[table(['集合','边界数','根数','可见对象记录','不可见/零fallback','错绑'],rt),'','## 全边界结果与根配对区间','','A/B test为150边界/10根；C为300边界/20根。每根15边界，三条分支不是独立根。位置宏平均对当前集合有真实支持的类别等权：A/B有左/中/右/入抽屉/抓持，C无桌面左侧正例，因此两集合宏平均不能直接视为同一类别构成的难度比较。','']
 pr=[]
 for split in ['test','externalC']:
  for seed in [0,1]:
   for key,label in [('loc_macro_recall','位置宏召回'),('table_macro_recall','仅桌面位置宏召回'),('holding_identity_positive','真持物状态：持有谁全对'),('InDrawer_recall','入抽屉正例召回'),('exact4','四字段全对')]:
    r=pair(split,seed,key);pr.append([split,seed,label,pct(num(r['global_value'])),pct(num(r['object_value'])),f"{100*float(r['difference']):+.1f}",ci(r)])
 lines +=[table(['集合','编码器种子','指标','全局','对象','差值/百分点','95%根配对区间'],pr),'','![主要结果](figures/primary.png)','','图示：灰色=全局组，青色=对象组；纵轴是百分比，横轴是四种不同指标，每格固定数据集和编码器种子，柱上为点估计。图不把不同指标合成一个分数；区间见上表。实际意义是对象对应提升了可读性，但C完整状态仍失败。','','## 正例、误报和初始持物切片','','四字段为红/蓝/粉各自位置和抽屉状态，均取类别argmax。Holding与InDrawer分别按对应类别概率≥0.5独立判断，因此可能与argmax统计不同；九关系是三个Holding、三个InDrawer及三个抽屉one-hot全部正确。召回分母是真实正例，精确率分母是预测正例，误报率分母是真实负例。','']
 pos=[]
 for split in ['test','externalC']:
  for seed in [0,1]:
   for arm in ARMS:
    c=count(split,seed,arm);tp,fn,fp,tn=[int(float(c['Holding_'+k]))for k in ['TP','FN','FP','TN']];itp,ifn,ifp,itn=[int(float(c['InDrawer_'+k]))for k in ['TP','FN','FP','TN']]
    pos.append([split,seed,arm,f'{tp}/{tp+fn}',f'{tp}/{tp+fp}'if tp+fp else'无预测正例',f'{fp}/{fp+tn}',f'{itp}/{itp+ifn}',f'{ifp}/{ifp+itn}',f"{int(float(c['holding_multiple']))}/{int(float(c['n']))}"])
 lines +=[table(['集合','种子','组','Holding召回','Holding精确率','Holding误报/负例','InDrawer召回','InDrawer误报/负例','多持物冲突'],pos),'','C初始持物身份在全局组两种子均为0/4，对象组为3/4和2/4：同一冻结编码器中，显式选择对应物体内容能减少‘知道有物体被抓住却答错颜色’或完全漏报的情况。这支持联合处理‘视觉身份路由+对象汇聚’的可读性价值，不能唯一归因于mean pooling，也不意味着四根足以保证总体泛化。h0每个物理根只取b0初始记录。以下持物切片的独立根很少：A/B仅2根、C仅4根，不能把比例外推为稳定总体保证。','']
 hrows=[]
 for split in ['test','externalC']:
  for seed in [0,1]:
   for arm in ARMS:
    c=count(split,seed,arm,'h0','initial_held');tp,fn,fp=[int(float(c['Holding_'+k]))for k in ['TP','FN','FP']];hrows.append([split,seed,arm,c['roots'],f'{tp}/{tp+fn}',f'{tp}/{tp+fp}'if tp+fp else'无预测正例',int(float(c['holding_identity_correct_positive'])),int(float(c['holding_identity_wrong_positive'])),int(float(c['holding_missed_positive'])),int(float(c['holding_multiple']))])
 lines +=[table(['集合','种子','组','根','召回','精确率','身份全对','身份错','漏报','多持物'],hrows),'','四个C持物根逐例：C02粉块，两个对象组均改为正确；C06红块，仅对象seed0正确，seed1仍漏报；C12红块，两个对象组均正确；C16蓝块，两组两种子均漏报。四根都展示在下方图片，不能只展示改善案例。初始非持物切片、全边界初始持物/非持物切片均在metrics.csv与counts.csv单列，没有只保留成功样本。小切片出现[0,0]或[1,1]等退化bootstrap区间只说明重采样无法产生新结果，不代表总体零不确定性；本报告对四根只逐案例解释。','','## 独立初始根与完整状态限制','']
 hr=[]
 for split in ['test','externalC']:
  for seed in [0,1]:
   for arm in ARMS:
    c=count(split,seed,arm,'h0');hr.append([split,seed,arm,f"{int(float(c['exact4']))}/{int(float(c['n']))}",f"{int(float(c['nine']))}/{int(float(c['n']))}",pct(num(metric(split,seed,arm,'drawer_accuracy','h0')['value']))])
 lines +=[table(['集合','种子','组','h0四字段全对','h0九关系全对','h0抽屉正确率'],hr),'','位置有部分改善：C仅桌面位置宏召回从34.5%/36.9%升到53.3%/58.5%，但提升主要来自中区；右侧seed0从3.2%升到13.5%，seed1却从25.3%降到20.6%，因此右侧未稳定改善。C的关系改善不能掩盖完整位置错误：对象组全边界右侧召回为63/466和96/466（13.5%、20.6%），h0完整状态两种子仍0/20。灰/青两组抽屉分支在每个种子内逐位相同，因此抽屉差异不是对象处理造成；全边界C抽屉正确率分别91%和98%，也是完整状态分数的限制。关闭/打开逐类召回见CSV；中间态无正例。对象seed0的286个错误状态中283个含桌面位置错、281个含右侧错；seed1的283个错误状态中279个含桌面位置错、276个含右侧错（可重叠，不能相加）。对象字段错误分别451和396次，其中右→中占378和337次。仅抽屉错而三个对象全对的状态仅1和0个，说明低完整分的主要直接来源仍是桌面位置，而非抽屉。该分解是事后描述，不是新的因果干预。','','![AB位置混淆](figures/confusion_test.png)','','![C位置混淆](figures/confusion_externalC.png)','','混淆图：纵轴真实位置、横轴预测位置；每个数字为对象记录数，蓝色深浅表示该真实类别中落入该预测的比例，灰行表示没有该真实类别样本。对角线正确，非对角线是具体混淆。两个种子和两组分别展示，未合并重复样本当作独立根。','','## 典型案例与仍然失败的地方','']
 for c in cases:lines +=[f"### {c['episode_id']}",f"选例规则：{c['selection']}。",f"![案例](figures/{c['figure']})",'图中左侧为原静态相机、中央为原夹爪相机，右侧灰/青文字分别给出全局/对象组的类别与Holding判断，底部是评分GT。颜色文字只用于区分条件；RGB未做增强。类别按红/蓝/粉顺序，drawer 0=关、2=开。','']
 lines +=['## 如何解释与下一步','','本轮支持保留公开对象对应对这些冻结特征的状态读出有价值；两个种子、两个评价集合方向一致，Holding误报同时下降，InDrawer没有以明显更高误报换取召回。它没有证明RSSM递推已经改善，也没有唯一定位到全局池化，因为路由选择与表示汇总共同变化。','','C的第一个主要未解决错误仍是桌面位置类别；本轮不再调整头或路由。最小下一步是保持对象组冻结，先按根检查公开静态相机ROI中心对中/右的可分性；若有分辨信息，再单独预注册相同预算的几何辅助读出对照，并用打乱几何的对照检查是否真正利用位置。GT仍只作标签/评分，本轮结果封存；当前没有执行这项后续试验。若以后接入动态，需要新的对象状态递推对照及完整自由预测验收，不能把这里每步重新看观测的成绩替代自由递推。','','## 复现、校验和文件说明','',f"核心计算耗时{compute['wall_seconds']:.2f}秒；峰值GPU已分配{compute['gpu_peak_allocated_GiB']:.3f}GiB。四头固定6000次总更新，无新RSSM训练、无采集、无上传。初次失败预检耗时约17秒另计，文档与人工检查另计。",'','- configs/protocol.json和routing_amendment.json：原预注册与训练前一次修订。','- configs/schedule_seed*.npy、models/head_*.pt、results/train_*.json：完整采样顺序、最终权重、损失与初始化哈希。','- data/features_seed*和predictions_*：冻结表示与逐样本概率；相同全局表示逐位复现原编码器。','- results/metrics.csv：逐集合/种子/组/口径/初态分层指标。counts.csv：原始分母和混淆计数。','- results/paired_differences.csv、root_statistics.csv：根配对差和2000次根簇重采样。两个编码器种子不合并；极少根的切片区间不稳定。宏平均的类别集合固定，缺少必需类的bootstrap抽样不计入该指标，CSV记录有效重复次数；未作多重比较校正。','- results/confusions.csv、per_example.csv：颜色分开的全量混淆与逐样本结果。','- results/permutation.json：同时重排对象向量、身份查询与可见性后，输出正确随之重排；这是索引/共享头等变性检查，不是证明语义正确或通过图像换色干预。','- source/common.py、run.py、analyze.py、report.py和logs/run.log：可复现实现和运行日志。run_attempt1.py保留首次路由失败实现。','','重跑需新输出目录，保持本目录与旧实验只读；按source中的固定配置执行路由、冻结特征提取、训练、统一评估和统计。不能直接覆盖本轮结果。','']
 md='\n'.join(lines);(S/'REPORT.md').write_text(md)
 rendered=render_markdown(md)
 for p in(S/'figures').glob('*.png'):rendered=rendered.replace('figures/'+p.name,'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>对象对应读出实验</title><style>body{max-width:1200px;margin:40px auto;padding:0 28px;font:17px/1.65 sans-serif;color:#18212b}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}th,td{border:1px solid #d0d7de;padding:7px;text-align:left}th{background:#edf4f5}img{max-width:100%;height:auto}h1,h2,h3{line-height:1.3;margin-top:1.7em}strong{color:#006d77}</style>'+rendered+'</html>')
 save(S/'results/report_receipt.json',{'utc':now(),'figures':len(list((S/'figures').glob('*.png'))),'cases':len(cases),'report_html_self_contained':True,'report_md_sha256':sha(S/'REPORT.md')})
 print('REPORT_COMPLETE',str(S/'REPORT.html'),flush=True)
if __name__=='__main__':main()
