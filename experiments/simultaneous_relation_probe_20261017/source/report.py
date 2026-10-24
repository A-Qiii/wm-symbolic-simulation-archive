from config import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
LABELS={'query_seed0':'原query0','query_seed1':'原query1','majority_train':'多数类','colour_prior_train':'颜色先验','object_linear':'对象/线性','object_mlp32':'对象/MLP','context_linear':'对象+共享/线性','context_mlp32':'对象+共享/MLP'}
def main():
 font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
 metrics=read(E/'results/metrics.json');joint=read(E/'results/joint_metrics.json');names=['majority_train','query_seed0','query_seed1','object_linear','object_mlp32','context_linear','context_mlp32'];splits=['train','validation','diagnostic_D_seen'];sl={'train':'AB训练6根','validation':'AB验证6根','diagnostic_D_seen':'已观察D诊断6根'}
 def get(split,name,cond):return next(r for r in metrics if r['split']==split and r['model']==name and r['group']==('all'if cond=='all'else'condition')and r['value']==cond)
 def pc(x):return '—'if x is None else f'{100*x:.1f}%'
 fig,axes=plt.subplots(2,3,figsize=(17,8),sharey=True)
 for ci,split in enumerate(splits):
  for ri,condition in enumerate(['all','dual']):
   ax=axes[ri,ci];rr=[get(split,n,condition)for n in names];x=np.arange(len(rr));ax.bar(x-.18,[100*r['recall']if r['recall']is not None else 0 for r in rr],width=.34,color='#176a9c',label='In真：召回率');ax.bar(x+.18,[100*r['FPR']if r['FPR']is not None else 0 for r in rr],width=.34,color='#cf6a35',label='In假：误报率');ax.set_xticks(x,[LABELS[n]for n in names],rotation=35,ha='right');ax.set_ylim(0,115);ax.set_title(f"{sl[split]} | {'全部边界'if condition=='all'else'双关系成立'}\n正例{rr[0]['positive_N']} / 负例{rr[0]['negative_N']}");ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
   if ci==0:ax.set_ylabel('百分比 (%)')
   if ri==0 and ci==0:ax.legend(fontsize=9)
 fig.suptitle('冻结实际观测潜特征的In读取诊断；蓝越高越好，橙越低越好；各面板均为6根',fontsize=15);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(R/'probe_recall_false_positive.png',dpi=150);plt.close(fig)
 lines=['# 同时关系读取的有界离线诊断','','本报告中的probe是诊断工具，不替换正式query。编码器、转移和原query均冻结；只用已经保存的当前观测z。D已被观察，仅作诊断，不是全新独立验证。','','## 结构与数据证据','','原query的Holding/In使用独立sigmoid，只有抽屉分类使用softmax。共存概率、对象置换、逐事实掩码梯度、RGB颜色绑定和300行历史样本z/标签绑定测试通过；未发现确定的新代码错误。旧144/156状态池各30根，双关系条件均为0状态、0根，说明训练覆盖存在明确缺口。已知旧几何错误只在新诊断副本并列更正。','','每组6根、54个去重技能边界、162个对象In标签，18正/144负；每组6个实际双关系观测。每个存在正例的(对象颜色,Held颜色)组合也有同身份负例。所有同根帧和分支在同一划分，三个对象全部保留；不按成功与否筛样。','','|输入|包含内容|线性/MLP的共同条件|','|---|---|---|','|对象token|对应对象当前64维z|相同162个训练对象标签、同根划分|','|对象+共享|对象64+场景64+三对象平均64，共192维|同为固定500步，无搜索/早停|','','模型不接收颜色ID、Held身份、动作计划、历史真值或仿真坐标；这些仅用来分层评价。线性和32隐藏单元SiLU MLP各两种输入，共4次拟合，CPU训练；标准化只用训练根，损失按根和根内正负类等权。阈值固定0.5，最终500步权重在验证/D评分前封存。没有使用历史样本训练probe，因此与原query的训练数据覆盖不同。','','## 结果与正负例分母','','|划分|模型|全部In TP/FN/FP/TN|真类召回|假类召回|平衡准确率|双关系 In TP/FN/FP/TN|双关系三In全对|','|---|---|---|---|---|---|---|---|']
 for split in splits:
  for n in names:
   a=get(split,n,'all');b=get(split,n,'dual');j=next(r for r in joint if r['split']==split and r['model']==n and r['condition']=='dual');lines.append(f"|{sl[split]}|{LABELS[n]}|{a['TP']}/{a['FN']}/{a['FP']}/{a['TN']}|{pc(a['recall'])}|{pc(1-a['FPR']if a['FPR']is not None else None)}|{pc(a['balanced_accuracy'])}|{b['TP']}/{b['FN']}/{b['FP']}/{b['TN']}|{j['three_In_correct']}/{j['state_N']}|")
 lines+=['','![召回与误报](probe_recall_false_positive.png)','','列区分训练、AB验证、已观察D；上排为全部边界，下排为真实同时关系。横轴区分多数类基线、两个冻结原query和四个probe；蓝柱是In真类召回，橙柱是In假类误报。面板标题标明实际正负例分母，颜色不表示物体颜色。所有帧属于6个根，不能按162条对象标签当独立样本。','','## 限制与解释边界','','线性/小MLP成功只能证明对应固定特征在本样本与划分下包含可读取的信息；不能证明原query失败只有覆盖原因，也不说明任何新模型已满足项目验收。若失败，不能据此证明信息完全不存在。验证仅6根，颜色和Held身份的分层结果每格更少；当前循环换目标规则没有测试全部可能物体搭配，因此不能宣称排除所有捷径或一般关系泛化。','','多数类基线按训练标签确定。由于18/162为正，单看整体准确率可能掩盖完全漏报；报告因此主列真/假类召回、误报和双关系同时正确。颜色先验基线与逐颜色/Held/根分层完整保存在CSV中。','','[逐对象预测与真值](per_object_predictions.csv) · [全部分层混淆](metrics.csv) · [三个In联合状态与根平均](joint_metrics.csv)', '',f'[结构与掩码审计]({E}/results/structure_audit.json) · [同身份正负例覆盖]({E}/results/matched_identity_support.json) · [训练根划分]({E}/configs/root_split.json) · [冻结协议]({E}/configs/protocol.json)']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print('PROBE_REPORT_WRITTEN',R,flush=True)
if __name__=='__main__':main()
