from common import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
from mid_report import table,find,ratio,NAMES
def main():
 guard();p=read(S/'results/same_boundary_initialization_paths.json');rr=p['summary'];display=[]
 for r in rr:
  if r['subset']!='initial_In_positive':continue
  display.append({'model':NAMES[r['arm']],'seed':r['seed'],'which':'同一边界读出'if r['comparison']=='initial'else'同一下一动作预测','n':r['boundaries'],'cold_In':ratio(r['cold_In_correct'],r['boundaries']),'history_In':ratio(r['history_In_correct'],r['boundaries']),'cold_all':ratio(r['cold_all_correct'],r['boundaries']),'history_all':ratio(r['history_all_correct'],r['boundaries']),'lost':r['history_right_cold_wrong_all'],'gained':r['cold_right_history_wrong_all']})
 fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for row,comparison in enumerate(['initial','same_next_action']):
  for seed in [0,1]:
   ax=axs[row,seed]
   for i,a in enumerate(ARMS):
    r=find(rr,arm=a,seed=seed,comparison=comparison,subset='initial_In_positive')
    for off,path,color,label in [(-.16,'cold','#dd7e20','空递归记忆初始化'),(.16,'history','#2877b8','公开历史初始化')]:ax.bar(i+off,100*r[path+'_all_correct']/r['boundaries'],.28,color=color,label=label if i==0 else None)
   ax.set(xticks=range(3),xticklabels=['完整轨迹训练','裁剪＋历史训练','裁剪＋冷启动训练'],ylim=(0,105),ylabel='全部 Held / In / drawer 正确率 / %',title=('同一边界读出'if row==0 else'同一下一动作预测')+f'；种子{seed}');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('同一物理边界、同一公开观测：只比较获得内部状态的路径（起点已有In子集）')
 for ext in ['png','svg']:fig.savefig(S/f'reports/initialization_paths.{ext}',dpi=170)
 plt.close(fig)
 text='''<h2>同一物理状态为何换成初态会失效</h2><p>这里不是“物理状态没见过”，也不是“从这个状态出发的后续变化没训练过”。原训练已包含658个不同的、已有In的非终止中途起点；此前模型通过公开动作与观测历史走到这些位置。需要检验的是另一条路径：从空递归记忆仅吸收当前可得窗口，能否形成与后续转移函数相容的内部表示。四帧窗口保留短时视觉信息，但不等同于递归状态中的完整技能历史。</p><p>以下对同一开发边界逐一配对：两个条件索引完全相同的冻结观测特征、机器人信息和标签；历史条件从此前公开经历递推到此处，冷启动条件清零递归先验。再输入同一下一动作，在不读取下一观测的情况下比较动作后果预测。此次补充由用户在训练期间提出，只分析现有数组，没有另训或改变预设主要评分。</p>'''
 text+=table(display,[('model','训练条件'),('seed','种子'),('which','比较'),('n','同一边界数'),('cold_In','空记忆In全对象正确'),('history_In','历史In全对象正确'),('cold_all','空记忆所有符号'),('history_all','历史所有符号'),('lost','历史对/空记忆错'),('gained','空记忆对/历史错')]);text+='<img src="initialization_paths.png"><p>蓝柱是保留公开历史的评估初始化路径，橙柱是空递归记忆的评估初始化路径；这里的颜色表示评估路径，而非其他图的训练条件。横轴是三个训练条件，纵轴是全部符号正确率。上下行分别为当前状态读出与同一下一动作预测，左右列分别为种子0/1。两条路径使用同一物理边界，因此不能把差距解释为换了物理状态或动作内容。</p><p><a href="../results/same_boundary_initialization_paths.json">逐边界、逐模型概率和配对计数</a>；<a href="../results/support_and_leakage_audit.json">658个原训练中途In起点及曝光/泄漏审计</a>。</p>'
 f=S/'reports/REPORT.html';body=f.read_text();body=body.replace('<h2>独立开发：冷启动符号与后续转移</h2>',text+'<h2>独立开发：冷启动符号与后续转移</h2>');f.write_text(body)
 initial=read(S/'results/initial.json');whole=read(S/'results/whole.json');lines=['本轮完成三组同架构同参数训练，每组两种子，每模型1500次更新，共9000次；未新增训练数据。','原状态及后续转移已经作为中间过程训练过；本轮检验空记忆初始化表示与历史形成状态的衔接，不是把它笼统称为未见物理状态。','两裁剪组的起点、动作和监督完全匹配；它们比完整组少约25%的转移曝光。完整组两个种子逐张量复现原RGB-D粗网格候选。','开发中途起点初始In正例召回（种子0/1）：']
 for a in ARMS:
  z=[find(initial,source='development',arm=a,seed=seed,startgroup='middle',template='ALL',branch='all')for seed in [0,1]];lines.append(NAMES[a]+'：'+'；'.join(ratio(x['In_TP'],x['In_positive'])for x in z))
 lines.append('原完整轨迹逐步观测整轨正确（种子0/1）：')
 for a in ARMS:
  z=[find(whole,source='development',arm=a,seed=seed,startgroup='full',template='ALL',branch='all',mode='current')for seed in [0,1]];lines.append(NAMES[a]+'：'+'；'.join(ratio(x['all_correct'],x['trajectories'])for x in z))
 if any(x['source']=='independent'for x in initial):
  lines.append('新独立验证24条、4个布局族，均正常分支；两顺序各12条。结果必须与探索四族分开。')
  for a in ARMS:
   for template in ['order_known4','order_novel4']:
    z=[find(whole,source='independent',arm=a,seed=seed,startgroup='full',template=template,branch='all',mode='current')for seed in [0,1]];lines.append(NAMES[a]+' '+template+' 观测整轨：'+'；'.join(ratio(x['all_correct'],x['trajectories'])for x in z))
 lines.extend(['旧非目标In错误全部集中于初始In漏读的四步上下文，未观察到此前正确再遗忘；正常六步与外部落空扰动首错分别统计。','世界模型无GT输入；固定低层控制器有GT辅助，按授权保留。没有上传、没有C/D访问，旧门槛与成绩保持原样。','本轮不是原指南整体完成或规划验收。Held探针只验证原抓取完成阶段短时观测，没有新增hold动作。']);(S/'reports/SUMMARY.md').write_text('\n\n'.join(lines)+'\n');save(S/'results/paired_paths_report_complete.json',{'utc':now(),'same_physical_states_and_actions':True,'new_training':0,'figure':'initialization_paths.png'});print('PAIRED_INTERPRETATION_ADDED',flush=True)
if __name__=='__main__':main()
