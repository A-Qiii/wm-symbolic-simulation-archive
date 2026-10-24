from common import *
import csv,re,unicodedata
os.environ['MPLCONFIGDIR']=str(B/'results/matplotlib')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.backends.backend_pdf import PdfPages
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','pdf.fonttype':3,'axes.unicode_minus':False,'font.size':10})
COLORS={'joint':'#2166AC','route_current_only':'#D6604D'}
NAMES={'joint':'未来梯度进入编码器','route_current_only':'编码器仅当前损失'}
def wrap(t,width=92):
 out=[]
 for line in t.split('\n'):
  cur='';n=0
  for tok in re.findall(r'[A-Za-z0-9_.:/+−–×-]+|.',line):
   w=sum(2 if unicodedata.east_asian_width(c)in'WF'else 1 for c in tok)
   if n+w>width:out.append(cur);cur='';n=0
   cur+=tok;n+=w
  out.append(cur)
 return '\n'.join(out)
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |'for row in rows)+'\n'
def fmt(x,n=3):return '无分母'if x is None or x==''else f'{float(x):.{n}f}'
def interval(r):return f"{fmt(r['value'])} [{fmt(r['lo95'])}, {fmt(r['hi95'])}]"
def count(r):return f"{round(float(r['micro'])*int(r['records']))}/{r['records']}"

def main():
 guard('report');assert read(B/'results/compute_complete.json')['main_fits']==8
 rows=list(csv.DictReader((REPORT/'all_metrics.csv').open()));pairs=list(csv.DictReader((REPORT/'paired_contrasts.csv').open()));relations=list(csv.DictReader((REPORT/'relation_confusions.csv').open()));design=read(B/'configs/fixed_design.json');coverage=read(B/'results/coverage_review.json');scores=read(B/'results/scores.json')
 def m(size,seed,arm,split,mode,fact,metric='accuracy',stratum='all',change='all'):
  key={'size':str(size),'seed':str(seed),'arm':arm,'split':split,'mode':mode,'fact':fact,'metric':metric,'stratum':stratum,'change':change}
  return next(r for r in rows if all(r[k]==v for k,v in key.items()))
 def pair(seed,arm,split,mode,fact,metric):
  key={'seed':str(seed),'contrast':'size40_minus20_'+('joint'if arm=='joint'else'control'),'split':split,'mode':mode,'fact':fact,'metric':metric,'stratum':'all','change':'all'}
  return next(r for r in pairs if all(r[k]==v for k,v in key.items()))
 primary=[]
 for size in [20,40]:
  for seed in [0,1]:
   for arm in ARMS:
    primary.append([size,seed,NAMES[arm],fmt(m(size,seed,arm,'test','probe','primary4','NLL')['value']),count(m(size,seed,arm,'test','probe','relation9_exact')),fmt(m(size,seed,arm,'test','free3','primary4','NLL')['value']),count(m(size,seed,arm,'test','free3','relation9_exact')),count(m(size,seed,arm,'test','free3','all5_exact'))])
 contrast=[];statements=[]
 for seed in [0,1]:
  for arm in ARMS:
   for mode in ['probe','free3']:
    n=pair(seed,arm,'test',mode,'primary4','NLL');a=pair(seed,arm,'test',mode,'relation9_exact','accuracy');contrast.append([seed,NAMES[arm],mode,interval(n),interval(a)])
 for arm in ARMS:
  for mode in ['probe','free3']:
   rr=[pair(seed,arm,'test',mode,'primary4','NLL')for seed in [0,1]];direction='两种子点估计均降低'if all(float(r['value'])<0 for r in rr)else'两种子点估计均上升'if all(float(r['value'])>0 for r in rr)else'两种子方向不一致'
   bounded='两种子的根重采样区间均低于0'if all(float(r['hi95'])<0 for r in rr)else'至少一个种子的根重采样区间包含0或位于0以上'
   statements.append(f"{NAMES[arm]}的{mode}：20→40根NLL{direction}（{fmt(rr[0]['value'])}、{fmt(rr[1]['value'])}）；{bounded}。")
 fig,axs=plt.subplots(2,2,figsize=(11.8,9.2))
 for col,mode in enumerate(['probe','free3']):
  for row,(fact,metric)in enumerate([('primary4','NLL'),('relation9_exact','accuracy')]):
   ax=axs[row,col]
   for seed in [0,1]:
    for arm in ARMS:
     vv=[m(size,seed,arm,'test',mode,fact,metric)for size in [20,40]];v=np.array([float(r['value'])for r in vv]);lo=np.array([float(r['lo95'])for r in vv]);hi=np.array([float(r['hi95'])for r in vv]);x=np.array([20,40])+(seed-.5)*.6
     ax.errorbar(x,v,yerr=[v-lo,hi-v],color=COLORS[arm],marker='o'if seed==0 else's',linestyle='-'if seed==0 else'--',capsize=3,label=NAMES[arm]+f' / 种子{seed}')
   ax.set_xticks([20,40]);ax.set_xlim(17,43);ax.set_xlabel('参与主训练的独立物理根数');ax.set_ylabel('四字段平均NLL（低为好）'if row==0 else'九关系同时正确比例（高为好）');ax.grid(alpha=.2)
   ax.set_title(('冻结E线性查询 · 150边界'if mode=='probe'else'原RSSM三请求终点 · 30终点')+' / A/B固定测试10根')
   if row==1:ax.set_ylim(-.04,1.08)
   if row==0 and col==0:ax.legend(fontsize=7)
 cap='颜色：蓝=未来梯度进入编码器，橙=编码器仅当前损失。实线圆点=种子0，虚线方点=种子1；两点连线仅连接实测20与40根，不表示规模定律。误差线为2000次整根重采样95%区间，同根分支一起重采样。上排是平均类别概率误差，下排是九项关系同时正确；两种口径都保留。固定共同20根预处理及300边界probe监督，所有模型/查询头冻结后评分。三请求指标仅评价终点，不代表整个路径正确。'
 fig.suptitle('相同6000次更新下，20→40根的共同测试结果',fontsize=15);fig.tight_layout(rect=(0,.17,1,.95));fig.text(.05,.025,wrap(cap,150),fontsize=9,va='bottom');fig.savefig(REPORT/'figures/AB_heldout_scaling.png',dpi=170,bbox_inches='tight');fig.savefig(REPORT/'figures/AB_heldout_scaling.pdf',bbox_inches='tight')
 strata=[];gaps=[];refusals=[];changed=[]
 for size in [20,40]:
  for seed in [0,1]:
   for arm in ARMS:
    for split in ['test','externalC']:
     strata.append([size,seed,NAMES[arm],split,count(m(size,seed,arm,split,'free3','relation9_exact',stratum='initial_not_held')),count(m(size,seed,arm,split,'free3','relation9_exact',stratum='initial_open_held')),count(m(size,seed,arm,split,'free3','all5_exact'))])
     refusals.append([size,seed,NAMES[arm],split]+[fmt(m(size,seed,arm,split,'one_step','primary4','NLL',st)['value'])+' / '+m(size,seed,arm,split,'one_step','primary4','NLL',st)['records']for st in ['zero_control','nonzero_control','executed_failure']])
    for mode in ['probe','free3']:
     gaps.append([size,seed,NAMES[arm],mode]+[fmt(m(size,seed,arm,sp,mode,'primary4','NLL')['value'])for sp in ['train','common20','validation','test','externalC']])
    for fact in CAT_LOCAL:
     changed.append([size,seed,NAMES[arm],fact]+[fmt(m(size,seed,arm,'test','one_step',fact,change=ch)['value'])+' / '+m(size,seed,arm,'test','one_step',fact,change=ch)['records']for ch in ['changed','unchanged']])
 runtime=[]
 for size in [20,40]:
  for seed in [0,1]:
   for arm in ARMS:
    q=read(B/f'cells/roots{size}/results/train_{arm}_seed{seed}.json');runtime.append(q)
 mainmins=sum(r['elapsed_seconds_this_segment']for r in runtime)/60;peak=max(r['peak_GPU_bytes']for r in runtime)/1024**3
 cvtable=table(['划分','根','分支/转移','关空/开空/开持','非零/零控制'],[[c['group'],c['roots'],f"{c['episodes']}/{c['transitions']}",'/'.join(str(c['initial_states'][k])for k in ['closed_empty','open_empty','open_held']),f"{c['nonzero']}/{c['zero_control']}"]for c in coverage['coverage']])
 md=f'''# RSSM经验规模与新根泛化诊断：固定20/40根、成对梯度路由

{now()}。本轮8条主训练各6000更新、8个冻结编码器线性查询头各600更新已完成。新采集0，D样本0，长链拼接0，局部Holding修补0；旧实验和设计文件保持原位。这里只完成本次获准的规模诊断，不代表原《核心预实验方案》全流程完成。

## 先看共同A/B测试结果

{' '.join(statements)}

这些是这套固定嵌套根、两个训练种子和同一10根测试集上的观察。完整关系和位置结果必须与NLL一起看；下表和图保留全部条件，不能以单一有利指标替代结论。

{table(['训练根','种子','梯度条件','查询NLL','查询九关系全对/150','三步NLL','三步九关系全对/30','三步五字段全对/30'],primary)}

![固定A/B测试结果](figures/AB_heldout_scaling.png)

{cap}

20→40根配对变化如下：NLL负值表示改善，准确率正值表示改善；括号为2000次整根重采样95%区间。每种子单列，不把同根分支当作独立样本。所有效应值和两种梯度路由的“规模变化之差”见paired_contrasts.csv。

{table(['种子','条件','路径','NLL：40减20 [95%区间]','九关系准确率：40减20 [95%区间]'],contrast)}

## 指标定义与有效分母

查询=冻结512维E，固定600步训练单层Linear512→32；主NLL=三物体各8类位置/归属和抽屉3类的四字段无类别权重负对数似然。概率截断下限1e−9，先字段平均，再根内记录平均，最后根等权。训练损失包含类别权重与其他项，不能与这个评价NLL混为一谈。

三步=只从真实根观测初始化，沿三次真实请求自由递推，不读取中途观测；8条潜轨迹概率先平均再取NLL。A/B测试10根×3分支=30终点、120有效字段；查询10×3×5=150边界、600字段。C20根分别60终点/240字段与300边界/1200字段。三次请求可能包括零控制拒绝，全部保留；该指标只证明终点正确与否。没有free2保存结果，因此不计算或声称整段路径全对；没有5步数据，不拼接。

五字段全对=三个8分类物体字段、抽屉、滑门均argmax正确；七字段再含两灯0.5二值分类。九关系全对=三色Holding概率≥0.5、三色In概率≥0.5，以及抽屉三类argmax对应事实全部正确；九个事实不是独立标签，也不是开放世界全部状态。同物体Holding和In共享softmax概率预算；这种结构约束不能当作学会物理规律。多物体同时Holding另列冲突率。

一步“改变”按真实前后标签不同；三步“改变”是终点对根的净改变，不能解释成中途从未变化。零控制/非零/执行失败由原执行记录划分；执行失败不意味着全部状态字段为假。

## 根划分与覆盖审查

{cvtable}

沿用原设计的一次SHA256环境×初态分层排序，20根严格嵌套40根，验证/测试与训练不交叉；同根三分支不拆分，未用验证或测试表现挑选根。主训练两档A/B各半，三初态50/30/20%。共同20根覆盖六技能、三色持物与抽屉开闭，包含40条零控制、200条非零请求。其物体类别4/5/7、抽屉中间类缺失，滑门无动态、两灯均固定；不能声称学会这些无支持能力。测试初始持物仅蓝/粉两根，执行失败仅2条、滑门无变化；小分母结果按原样报告，不重抽根。

所有旧数据均曾在历史工作中被观察，本轮heldout只相对当前优化器隔离，不是全新盲测。旧60根模型见过这批A/B测试根，没有被加入新测试对照。

## 控制了什么，仍不能回答什么

两档从配对随机初值开始，未加载旧训练权重。每个规模/种子内，两路使用相同已物化6000×2×8抽样、潜变量随机起点、目标和优化器，E/R分别clip1；两路都计算相同前向和反向分支，只改变进入E的梯度。两档间初始化和随机种子配对，但样本内容不同。原5000单步＋1000三步训练节奏不变，最终6000权重唯一用于评分，没有早停或挑检查点。

**共同20根的归一化、类别权重和300个probe标注边界固定到所有条件。** 每个模型的probe特征标准化也只拟合这同300边界，抽样和初始化配对。因而这是控制预处理和查询监督预算后的主模型经验规模比较，不是普通全流程扩容。主模型40根仍比20根使用更多符号监督经验；它不是无命题标签交互学习。

8个主模型与8个查询头全部冻结之后才统一预测A/B验证、测试与C。验证仅作诊断，没有选择权重/阈值/规模。每档6000更新抽96,000分支，20/40根约1600/800次等效重复暴露。这是等计算对照，不能推出等暴露结论；只有20→40一个规模区间，不能拟合规模定律或外推6000转移效果。根bootstrap不覆盖更换训练子集和全部训练随机性，多项次要指标也没有作多重检验校正。

## 训练、验证与外部C

{table(['训练根','种子','条件','路径','本档训练NLL','共同20根NLL','验证10根NLL','测试10根NLL','C20根NLL'],gaps)}

C同时改变环境与初始持物→后续对象顺序：A/B为红→蓝、蓝→粉、粉→红，C为反向。纯动作字符串虽覆盖，带初始关系的组合并不相同。因此C只作外部分布诊断；initial_state_standardized.csv同时提供按50/30/20%初态权重标准化分数，仍不能把C−A/B当作纯场景因果效应。

三步终点九关系分解如下：A/B非持物8根/24终点、持物2根/6终点；C非持物16根/48终点、持物4根/12终点。持物组样本少，区间与全部种子结果见all_metrics.csv。

{table(['训练根','种子','条件','评估集','非持物初态九关系全对','持物初态九关系全对','五字段全对'],strata)}

## 改变、不变、拒绝与失败

下表是A/B测试一步字段根平均准确率/有效字段记录数；不是微平均，也不把零变化字段漏掉。滑门改变无分母，不报告成0%或100%。

{table(['训练根','种子','条件','字段','改变：准确率/N','不变：准确率/N'],changed)}

以下为一步四字段NLL/记录数。零控制请求原样保留；执行失败为非零控制子集，A/B测试只有2条、C只有5条。

{table(['训练根','种子','条件','评估集','零控制NLL/N','非零控制NLL/N','执行失败NLL/N'],refusals)}

每种关系/颜色、完整五/七字段、九关系、多持物冲突、当前/一步/自由1/3/4步、错误动作与持续状态基线均保存在all_metrics.csv；Holding/In混淆计数、精确率、召回率、平衡准确率及整根区间在relation_confusions.csv。所有根级值在root_metrics.csv，以便复算，不仅保留均值。

## 实际运行与核验

主训练总计{mainmins:.1f}分钟，峰值分配显存{peak:.2f}GiB；单卡串行运行，共48,000主更新＋4,800查询头更新，预检优化更新为0。进程与首个真实更新见results/launch.json、first_actual_update.json和每条jsonl日志。没有因为显存空闲追加batch或试验。

训练读取防护对A/B测试、验证、C、D样本的拒绝检查通过；FP32与BF16在同批次/权重/RNG下与原损失及完整R梯度一致，控制E梯度等于当前损失梯度，分组裁剪不受R幅值干扰；未来观测污染检查差为0。计分器用旧封存C预测复现原主NLL及九关系计数，未将旧模型在新A/B测试子集上重新称为heldout。

最终封存核验结果在results/final_verification.json；文件清单与SHA256在configs/delivery_manifest.json。设计、旧模型和旧报告的封存复验单独记录；训练和评估源码、固定配置、初始权重、最终权重、恢复点、预测、统计和图表均留在本机。原指南更长链规划与独立测试验收仍未由本实验完成。
'''
 (REPORT/'REPORT.md').write_text(md)
 page1='\n\n'.join(['本轮执行已完成：8条主训练各6000更新＋8个冻结E查询头各600更新；不新增数据、不使用D、不拼接长链、不修补局部谓词。','固定A/B60根分为训练40、验证10、测试10；小训练20嵌套40。主评估共同测试10根，C20仅作已见外部分布诊断。所有旧数据曾被观察，heldout只相对本轮优化器隔离。',*statements,'四字段NLL衡量平均真实类别概率；九关系全对衡量关系组合；五字段全对还包含台面粗位置及滑门。完整表与根级区间见后图及REPORT.md。','本轮共同20根归一化、类别权重和300个probe标注边界固定到所有条件。这是控制预处理与查询监督预算的规模实验，不是普通全流程扩容，也不是无命题标签学习。'])
 page3='\n\n'.join(['解释范围：两个训练种子、一套嵌套训练根、10个A/B测试根；整根bootstrap区间不包含全部训练子集随机性。仅20→40一个区间，不声称规模定律或外推。','终点与路径：三步读数只看根观测后三次真实请求的终点；零动作拒绝占请求，不能算作三个有效动作。中间free2未保存，不声称全路径正确，也没有5步数据。','C含联合变化：环境改变，初始持物颜色→后续对象顺序也从A/B的循环改为反向；不能把C差距全归于场景或经验不足。原始与50/30/20%初态标准化分数均保留。','覆盖限制：共同20根覆盖六技能、三色持物、抽屉开闭与40条零控制请求。无滑门变化、灯变化、物体类别4/5/7及抽屉中间类别支持。A/B测试持物仅2根、执行失败2条，不挑选其他根替换。',f'主训练累计{mainmins:.1f}分钟、峰值分配显存{peak:.2f}GiB；所有主模型和查询头冻结后才评分测试与C。两个规模6000次更新相同，但分支重复暴露约1600/800次；不代表等暴露对照。','交付包括完整中文报告、明确颜色/坐标/分母的图、根级CSV、每条件预测、固定协议、训练日志、初始/最终/恢复权重与哈希核验。旧实验保护和最终核验见final_verification.json。','本轮结论限于当前固定技能、输出字段与数据结构，尚不代表原核心指南、规划闭环或独立D测试全部完成。'])
 with PdfPages(REPORT/'RSSM规模与泛化诊断_20261019.pdf')as pdf:
  for title,body in [('固定规模实验：结果与解释',page1),('figure',None),('分母、控制条件与局限',page3)]:
   if title=='figure':pdf.savefig(fig);continue
   page=plt.figure(figsize=(8.27,11.69));txt=wrap(body,88);page.text(.07,.95,title,fontsize=18,va='top');page.text(.07,.89,txt,fontsize=min(11,680/(len(txt.splitlines())*1.65)),va='top',linespacing=1.65);pdf.savefig(page);plt.close(page)
 plt.close(fig)
 save(B/'results/report_receipt.json',{'utc':now(),'main_minutes':mainmins,'peak_GPU_GiB':peak,'PDF_pages':3,'statements':statements,'scope':'fixedcompute20to40 roots; fixedcommon20 preprocessing/probelabelpool; exploratory historicallyseen roots','training_complete':True})
 print('REPORT_COMPLETE',flush=True)

CAT_LOCAL=['red_location','blue_location','pink_location','drawer','slider']
if __name__=='__main__':main()
