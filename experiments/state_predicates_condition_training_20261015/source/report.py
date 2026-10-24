from core import *
import csv,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
FAM={'close':'关抽屉','lift':'提起指定物','place':'放入抽屉'};CN={'red':'红','blue':'蓝','pink':'粉'};FACT=['持红','持蓝','持粉','红入屉','蓝入屉','粉入屉','抽屉开']
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table></div>'
def ratio(a,b):return f'{a}/{b}'
def csvsave(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 frozen();assert read(E/'results/final_validation.json')['passed'];ev=read(E/'results/evaluation.json');sel=read(E/'configs/model_selection_frozen.json');meta=read(E/'data/_private/evaluation_metadata.json');t=dict(np.load(E/'data/_private/targets.npz'));details={(r['model'],r['pair_id']):r for r in ev['details']};figdir=E/'figures';aug=sel['representatives']['aug132']+'_selected';old=sel['representatives']['old96']+'_selected';cnew=ev['contrast_summary'];palette={'old96':'#d68940','aug132':'#247a71'}
 # Six complete training curves; all runs finished1500 regardless of selected checkpoint.
 fig,ax=plt.subplots(2,3,figsize=(15,8.6),sharex=True)
 for seed in [0,1,2]:
  for cohort in ['old96','aug132']:
   run=next(r for r in sel['runs']if r['cohort']==cohort and r['seed']==seed);curve=[json.loads(x)for x in (E/'runs'/run['run']/'curve.jsonl').read_text().splitlines()];yy=np.array([r['total_loss']for r in curve]);sm=np.convolve(yy,np.ones(100)/100,mode='valid');label='旧96对'if cohort=='old96'else'旧新132对';ax[0,seed].plot(np.arange(100,1501),sm,color=palette[cohort],label=label,lw=1.7)
   xx=[c['update']for c in run['checkpoints']];scores=[c['tune_root_macro_BCE']for c in run['checkpoints']];ax[1,seed].plot(xx,scores,'o-',color=palette[cohort],label=label);cp=run['selected'];ax[1,seed].scatter(cp['update'],cp['tune_root_macro_BCE'],marker='*',s=160,color=palette[cohort],edgecolor='#263747',zorder=4)
  ax[0,seed].set_title('种子 '+str(seed));ax[0,seed].set_yscale('log');ax[0,seed].set_ylabel('训练总损失（100更新滑动平均，对数轴）');ax[1,seed].set_ylabel('C tune 按根平均加权BCE');ax[1,seed].set_xlabel('优化器更新次数');ax[0,seed].legend();ax[1,seed].set_xticks([0,300,600,900,1200,1500])
 for a in ax.flat:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
 for a in ax[0]:a.set_ylim(1e-4,1)
 for a in ax[1]:a.set_ylim(0,3.5)
 fig.suptitle('相同1500更新预算：橙色旧96对，绿色旧新132对',fontsize=16);fig.text(.03,.025,'上排曲线为训练损失，不是成功率；下排圆点为预定检查点，星号为按最低C tune损失选中的点。C calibration不用于选择。所有运行均训练到1500。',fontsize=10);fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(figdir/'training_curves.png',dpi=160);plt.close(fig)
 methods=[('old96_seed0_final1500','旧96对神经模型（3种子同值）','#d68940'),('aug132_seed0_final1500','旧新132对神经模型（3种子同值）','#247a71'),('tree_old96','固定浅树·旧96对','#b7a2c9'),('tree_aug132','固定浅树·旧新132对','#81659e'),('copy_effect','原冻结复制＋效果','#ba9b35'),('persistence','保持当前状态','#6c9591'),('real_next_observation','真实后观测读出（参照）','#a2aeb8')]
 fig,axes=plt.subplots(1,3,figsize=(16,6.5),gridspec_kw={'width_ratios':[1.3,1,1]})
 for j,(sn,title,total)in enumerate([('C_old','旧成功C：七事实全部正确',24),('C_new','新条件C：七事实全部正确',18),('contrast','新C两侧七事实同时正确',6)]):
  a=axes[j]
  for i,(name,label,color)in enumerate(methods):
   n=ev['results'][name][sn]['all']['all7']if sn!='contrast'else cnew[name]['C_new']['both_all7_correct'];a.barh(i,n,color=color,hatch='///'if name=='real_next_observation'else None,height=.65);a.text(n+total*.02,i,ratio(n,total),va='center',fontsize=9)
  a.set_yticks(range(len(methods)),labels=[m[1]for m in methods]if j==0 else[]);a.invert_yaxis();a.set_xlim(0,total*1.18);a.set_xticks(np.arange(0,total+1,4 if total==24 else 3 if total==18 else 1));a.set_title(title);a.set_xlabel('转移次数（来自6个根）'if j<2 else'独立根数（每根b0/b1）');a.grid(axis='x',alpha=.2);a.set_axisbelow(True);a.spines[['top','right']].set_visible(False)
 fig.suptitle('等预算比较：补入条件数据后改善，旧C未退化',fontsize=16);fig.text(.02,.025,'各颜色对应左侧方法；神经行展示全部三个种子的共同计数，使用同为1500更新的检查点。斜线灰色使用真实未来，仅作读出参照。UNKNOWN不计正确。',fontsize=10);fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(figdir/'equal_budget_C.png',dpi=160);plt.close(fig)
 # Latent discrepancy shown alongside symbolic success, not hidden in an appendix.
 names=[f'{cohort}_seed{s}_final1500'for cohort in ['old96','aug132']for s in [0,1,2]];xx=np.arange(6);fig,axes=plt.subplots(1,3,figsize=(14.5,5.4));keys=['object_MAE','scene_MAE','outside_GRU_range'];titles=['物体后继编码误差','场景后继编码误差','超出GRU编码范围的坐标']
 for j,(key,title)in enumerate(zip(keys,titles)):
  vv=[ev['latent_metrics'][n]['C_new'][key]*(100 if j==2 else 1)for n in names];axes[j].bar(xx,vv,color=[palette['old96']]*3+[palette['aug132']]*3)
  for i,v in enumerate(vv):axes[j].text(i,v+max(vv)*.03,f'{v:.2f}'+('%'if j==2 else''),ha='center',fontsize=9)
  axes[j].set_xticks(xx,labels=['旧0','旧1','旧2','新0','新1','新2']);axes[j].set_ylim(0,max(vv)*1.2);axes[j].set_title(title);axes[j].set_xlabel('训练数据组＋种子，均1500更新');axes[j].set_ylabel('平均绝对误差（隐状态单位）'if j<2 else'绝对值>1的坐标占比（%）');axes[j].grid(axis='y',alpha=.2);axes[j].set_axisbelow(True)
 fig.suptitle('新C符号正确，并不意味着预测潜状态接近真实后继',fontsize=15);fig.text(.02,.02,'橙=旧96对，绿=旧新132对。误差相对同一冻结编码器的真实后继计算；单位不是米。GRU真实输出范围为[-1,1]，未对预测做裁剪或修正。',fontsize=10);fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(figdir/'latent_error.png',dpi=160);plt.close(fig)
 # Same predeclared cases as preceding diagnosis, now with new learned models.
 newpairs={r['episode_id']:r for r in read(NEW/'data/public/pairs_full.json')};windows={r['window_id']:r for r in read(NEW/'data/public/windows_full.json')};index={r['pair_id']:i for i,r in enumerate(meta)};cases=read(NEW/'results/figure_cases.json')
 for case in cases:
  rid=case['root_id'];fig=plt.figure(figsize=(14,11.8));outer=fig.add_gridspec(1,2,left=.10,right=.98,bottom=.11,top=.88,wspace=.33)
  for b in [0,1]:
   p=newpairs[rid+f'__b{b}'];i=index[p['pair_id']];gs=outer[0,b].subgridspec(3,2,height_ratios=[1,1,2.05],hspace=.30,wspace=.08)
   for row,end in enumerate(['before','after']):
    w=windows[p[end+'_window_id']];a=np.load(w['public_frames'][-1])
    for col,cam in enumerate(['rgb_static','rgb_gripper']):
     axis=fig.add_subplot(gs[row,col]);axis.imshow(a[cam],interpolation='nearest');axis.axis('off');axis.set_title(('前'if row==0 else'后')+('·外相机'if col==0 else'·腕相机')+f' {w["simulation_times"][-1]:.2f}s',fontsize=10)
   values=[t['before_y'][i].astype(int).tolist(),details[('persistence',p['pair_id'])]['prediction'],t['after_y'][i].astype(int).tolist()]+[details[(name,p['pair_id'])]['prediction']for name in ['old96_seed0_final1500','aug132_seed0_final1500','tree_aug132','copy_effect','real_next_observation']];c=np.zeros((8,7),int)
   for row in [1,3,4,5,6,7]:
    truth=values[0]if row==1 else values[2]
    for j,val in enumerate(values[row]):c[row,j]=3 if val=='UNKNOWN'else 1 if val==truth[j]else 2
   a=fig.add_subplot(gs[2,:]);a.imshow(c,cmap=ListedColormap(['#e0eaf2','#c2e1d8','#f2b78d','#c5c8cc']),vmin=0,vmax=3,aspect='auto');a.set_xticks(range(7),labels=FACT,fontsize=9);a.set_yticks(range(8),labels=['真实前事实','当前读出','真实后事实','旧96神经·0','新132神经·0','新132浅树','冻结复制效果','真实后读出'],fontsize=9)
   for row in range(8):
    for j,val in enumerate(values[row]):a.text(j,row,'?'if val=='UNKNOWN'else str(val),ha='center',va='center',fontsize=10)
   fig.text(.30 if b==0 else .77,.915,'b0 可执行参考：实际成功改变'if b==0 else'b1 关键条件对照：控制器拒绝不变',ha='center',fontsize=11)
  fig.suptitle(FAM[case['family']]+'｜相同新根、相同动作，两种调用前状态\n'+rid,fontsize=14,y=.98);fig.text(.03,.045,'蓝灰行=真实标签；绿=与对应时刻标签一致，橙=错误，灰=UNKNOWN。1真、0假、?未知。神经模型同为1500更新。\n两相机图片来自同一批原始实测，显示每端锚点帧；模型实际使用每端2.1秒四帧。拒绝后96真实保持不变，是固定控制器接口语义。',fontsize=10);fig.savefig(figdir/('case_'+case['family']+'.png'),dpi=160);plt.close(fig)
 # All C roots are visible, and every root of all splits is available as CSV.
 rootrows=[]
 for name,sets in ev['results'].items():
  for sn in ['AB_old','AB_new','C_old','C_new']:
   for rd in sets[sn]['all']['root_details']:rootrows.append({'model':name,'set':sn,'root_id':rd['root_id'],'pairs':rd['pairs'],'all7':rd['all7'],'all_correct':rd['all_correct'],'changed_correct':rd['facts']['changed']['strict_correct'],'changed_facts':rd['facts']['changed']['n'],'unchanged_correct':rd['facts']['unchanged']['strict_correct'],'unchanged_facts':rd['facts']['unchanged']['n'],'UNKNOWN':rd['facts']['all7']['UNKNOWN']})
 csvsave(E/'results/all_root_metrics.csv',rootrows);croot=sorted({r['root_id']for r in meta if r['split']=='dev'},key=lambda rid:(not'longpair'in rid,rid));cols=['old96_seed0_final1500','aug132_seed0_final1500','tree_aug132','copy_effect'];matrix=[];annot=[]
 for rid in croot:
  rr=[r for r in meta if r['root_id']==rid];counts=[sum(details[(n,r['pair_id'])]['all7_correct']for r in rr)for n in cols];matrix.append(np.array(counts)/len(rr));annot.append([ratio(c,len(rr))for c in counts])
 fig,ax=plt.subplots(figsize=(11,7.5));im=ax.imshow(matrix,cmap=ListedColormap(['#fae8d9','#edc399','#90b7af','#23675d']),vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(len(croot)),labels=croot,fontsize=9);ax.set_xticks(range(4),labels=['旧96神经·0','新132神经·0','新132浅树','冻结复制效果']);ax.axhline(5.5,color='#273949',lw=2)
 for i in range(len(croot)):
  for j in range(4):ax.text(j,i,annot[i][j],ha='center',va='center',color='white'if matrix[i][j]==1 else'#273949')
 fig.colorbar(im,ax=ax,label='根内七事实全对比例（0—1）',shrink=.7);ax.set_title('全部12个C根：上方旧成功数据，下方新条件数据');ax.set_ylabel('独立根（每根旧4次／新3次）');ax.set_xlabel('神经模型同为1500更新；格内是正确次数／根内次数');fig.tight_layout();fig.savefig(figdir/'C_roots.png',dpi=160);plt.close(fig)
 # Export learned tree structure in plain text. No hand-authored effect rules.
 features=NAMES+['action_open','action_lift','action_place','action_close','target_red','target_blue','target_pink','declared_budget/900','declared_posthold/96'];lines=['从AB当前观测读出的七事实与动作拟合；以下全为学习结果，不是手写规则。','UNKNOWN输入=-1，真=1，假=0；叶概率为加权正例比例。']
 for j,tree in enumerate(read(E/'models/tree_aug132.json')['trees']):
  lines.append('\n输出 '+NAMES[j])
  def render(node,indent=''):
   if 'feature'in node:lines.append(indent+f'if {features[node["feature"]]} <= {node["threshold"]}:');render(node['left'],indent+'  ');lines.append(indent+'else:');render(node['right'],indent+'  ')
   else:lines.append(indent+f'p(true)={node["weighted_positive_probability"]:.6f}; AB support={node["n"]}')
  render(tree)
 (E/'results/tree_aug132_readable.txt').write_text('\n'.join(lines)+'\n')
 body=[];add=body.append
 add('<div class="eyebrow">本机有限训练 · '+now()[:10]+'</div><h1>新条件的符号预测已修正，潜状态仍明显失配</h1><p class="lead">使用旧新132对AB训练后，三个种子均达到旧C 24/24、新C 18/18七事实全对，六组新C条件对照两侧全部正确。相同1500更新预算只训练旧96对，仍为新C 4/18。收益与补入条件数据一致，不能仅归因于多训练。</p><div class="note">候选通过本轮预先冻结的单步符号开发标准，已经保存。但场景潜状态误差增大，不能据此认定世界模型动力学或多步规划可靠。没有采D、没有闭环、没有上传、没有改变控制器或继续调参。</div>')
 add('<h2>同预算比较证明了什么</h2><figure><img src="figures/equal_budget_C.png"><figcaption>橙色神经模型只使用旧96对，绿色使用旧新132对；三个种子的计数一致，均展示1500更新时结果。左、中横轴是正确转移数，右横轴是两侧七事实同时正确的独立根数。灰色斜线为真实后观测读出，使用未来数据，仅是测量参照。各图分母和样本单位不同，不把根内窗口当独立场景。</figcaption></figure>')
 rows=[]
 for n,label,_ in methods:
  a=ev['results'][n];f=a['C_new']['all']['facts'];rows.append([label,ratio(a['C_old']['all']['all7'],24),ratio(a['C_new']['all']['all7'],18),ratio(a['C_all']['all']['all7'],42),f'{100*a["C_all"]["all"]["root_macro_all7"]:.2f}%',ratio(cnew[n]['C_new']['both_all7_correct'],6),ratio(f['changed']['strict_correct'],8),ratio(f['unchanged']['strict_correct'],118),str(f['all7']['UNKNOWN'])])
 add(table(['方法（神经均1500更新）','旧C全对','新C全对','合并C全对次数','合并C按根平均全对率','新C两侧全对根','新C改变事实','新C未变事实','新C UNKNOWN项'],rows))
 add('<p>合并按根平均：先计算每根内的七事实全对比例，再对12个C根等权平均。旧根4次、新根3次，因此该列与直接用42次作分母的比例可能不同。42次不是42个独立场景。</p>')
 add('<p>新C只有8个改变事实、118个未变事实，因此只看整体事实准确率容易掩盖错误。保持当前状态得到12/18，但六次成功改变全部失败。原复制效果在改变事实上8/8正确，却只有10/18转移全对、16个UNKNOWN，不能把成功效果无条件套到拒绝情形。</p>')
 add('<h2>训练前固定的控制与数据范围</h2>')
 add(table(['项目','固定设置','作用'],[['根与数据来源','旧AB96对／24根＋新AB36对／12根；旧C24对／6根＋新C18对／6根','C不进入梯度；旧每根4次、新每根3次，所有尝试保留'],['同预算神经对照','原完整动作条件潜转移，76,224参数；两组各3种子×1500更新，总9000','同种子初始化相同，每运行24,000次样本呈现'],['损失与采样','每组三符号加权BCE＋0.05×物体／场景SmoothL1均值；批量16、完整随机排列流','两组共享仅由132对AB计算的类别权重；不另给成功／拒绝加权，防止损失权重变化混淆'],['优化器','AdamW，lr=0.001，weight_decay=0.0001，梯度裁剪1','沿用原方法，没有C驱动的调参'],['检查点选择','300／600／900／1200／1500；按合并C tune六根的平均BCE选点','旧新根各等权；C calibration不用于选择；同时报告固定1500结果'],['冻结项','DINO、RGB v1注册、归一化、状态编码器／GRU、共享held/in/open头、0.5阈值、控制器、原复制规则','只训练转移模块；输入仍是当前潜状态＋声明动作'],['浅层符号对照','七棵独立CART，深度≤4、叶至少2例、加权Gini、固定打破并列规则','只输入当前观测读出的七事实（UNKNOWN=-1）＋9维动作；真值仅作后继监督；旧96和新132各拟合一次，无C选深度']]))
 add('<p>类别权重为 held：[0.5625, 4.5]，in：[0.596386, 3.09375]，open：[1.375, 0.785714]，顺序为负／正。旧96每例呈现250次，新132每例181或182次；每运行总呈现均24,000次。已有一条旧AB当前读出错误仍保留，未替换为真值。C前后读出全部42/42正确。</p>')
 rows=[]
 for r in sel['runs']:
  n=r['run']+'_selected';rows.append([r['run'],r['selected']['update'],f'{r["selected"]["tune_root_macro_BCE"]:.6f}',ratio(ev['results'][n]['C_old']['all']['all7'],24),ratio(ev['results'][n]['C_new']['all']['all7'],18),ratio(cnew[n]['C_new']['both_target_correct'],6),ratio(cnew[n]['C_new']['both_all7_correct'],6)])
 add(table(['运行','预定规则选择的更新','C tune按根BCE','旧C全对','新C全对','两侧目标事实正确根','两侧七事实正确根'],rows));add('<p>代表候选是aug132_seed0、1500更新，由冻结的C tune标准选定；未根据最终C结果另挑模型。三种子不是三份独立数据。C已经用于开发，结果不等于盲测泛化或完整指南验收。</p>')
 add('<h2>曲线与预算</h2><figure><img src="figures/training_curves.png"><figcaption>每列一个种子。横轴为更新次数；上排是训练总损失的100更新滑动平均，纵轴为对数尺度，不能当成功率。下排是预定五个检查点的C tune按根加权BCE，星号标记选择结果。橙线旧96、绿线新132。旧组训练损失低却未解决新条件，说明仅延长旧分布训练不够。</figcaption></figure>')
 add(f'<p>六个小型缓存潜状态训练循环合计约{sel["total_wall_seconds"]:.2f}秒，不含此前仿真、图像采集和DINO特征缓存，不能当成完整项目训练耗时。30个检查点、9000条更新日志和144,000个采样索引均保留；GPU训练串行执行。</p>')
 add('<h2>浅层树对照的含义与限制</h2><p>补数据的固定树在新C为12/18、六组关键对照两侧全对为0/6；原复制规则为10/18。新神经模型在本轮的固定对照中更好。但这棵树在AB训练数据也只有125/132次全对，35个叶子中9个仍混合正负监督，已经存在拟合不足。差距不能证明符号输入不够，也不能排除更合适的符号模型同样拟合这些小规模条件。本轮遵守预先固定深度，没有查看C后加深、剪枝搜索或更换简单模型。</p><p>树的新C六次错误全部位于关键拒绝条件：持物关抽屉、已持另一物再提起、抽屉关闭时放入。两个提起条件出现多个held同时为真，按原互斥规则转为4项UNKNOWN。可查看学习出的完整树及每叶AB支持数，所有条件均由AB拟合，没有写入C专用规则。</p>')
 add('<h2>三类实测配对：修正了哪些事实</h2><p>继续使用上一轮已确定的每类第一个C根；不是根据本轮成绩挑案例。同一根的b0与b1声明动作相同、目标事实初值相同，实际后值不同；准备也改变机器人姿态，因而不是只改变一个物理变量的严格因果实验。</p>')
 for case in cases:add('<figure><img src="figures/case_'+case['family']+'.png"><figcaption>'+case['root_id']+'。左侧可执行，右侧控制器拒绝。表格完整保留七项：持红／蓝／粉、红／蓝／粉入屉、抽屉开。蓝灰是真实标签；绿色正确、橙色错误、灰色UNKNOWN。神经模型同为1500更新、种子0。图片来自同一批真实双相机记录，仅放大显示锚点帧，未生成未来图像。</figcaption></figure>')
 add('<h2>成功与拒绝分开看，旧数据不被新结果掩盖</h2>')
 rows=[]
 for name,label,_ in methods:
  rr=ev['results'][name]['C_new'];rows.append([label]+[ratio(rr[sub]['all7'],rr[sub]['pairs'])for sub in ['execution:success','execution:refused','role:positive_reference','role:core_state_dependence','role:nochange_semantics']])
 add(table(['方法','实际成功','控制器拒绝','可执行参考b0','关键条件b1','不变语义b2'],rows));add('<p>拒绝宏仍包含96步真实保持。学习到的是该固定控制器的接口语义，不是“物理上不可能运动”的普遍规律。增量训练模型C的UNKNOWN为0、全部改变／不变／其他实体事实正确；没有通过丢弃未知项或只评分目标物得到成绩。</p>')
 add('<figure><img src="figures/C_roots.png"><figcaption>每行一个独立C根，上方旧成功数据每根4次，下方新数据每根3次。颜色由浅橙到深绿表示根内全七事实正确比例从0到1，格内保留准确分子／分母。神经列均1500更新、种子0，全部种子的完整结果见CSV。</figcaption></figure>')
 add('<h2>符号正确与潜状态预测仍有明显分离</h2><p>所有42个C当前观测与真实后继观测读出均正确；增量模型的42个预测后继也能被同一共享头读成正确七事实。因此本轮没有剩余C符号错误可作首错分析。旧96对照仍有新C14次错误，且均在双端真实读出正确的情况下；真实后继编码替换预测后继、保持同一头后可恢复正确，仅用于诊断，不进入预测输入。</p>')
 add('<figure><img src="figures/latent_error.png"><figcaption>横轴分开旧／新训练数据组和种子，均1500更新；橙旧96、绿新132。左、中纵轴是相对冻结编码器真实后继的平均绝对误差，单位是潜坐标而非米；右轴是预测坐标超出GRU真实输出[-1,1]范围的比例。符号成绩改善时，场景潜状态误差反而增大。</figcaption></figure>')
 rows=[]
 for r in sel['runs']:
  n=r['run']+'_selected';m=ev['latent_metrics'][n]['C_new'];rows.append([n,f'{m["object_MAE"]:.4f}',f'{m["scene_MAE"]:.4f}',f'{100*m["outside_GRU_range"]:.2f}%'])
 add(table(['所选检查点：新C','物体MAE','场景MAE','超范围坐标比例'],rows));add('<p>代表候选场景MAE为1.5273，约49.49%的整体潜坐标超范围。这说明符号头在偏离真实后继编码的位置仍能输出正确事实。损失中潜状态项权重仅0.05，与这一现象相容，但本轮没有调权重验证唯一因果解释。没有裁剪潜状态、美化指标或为了这一问题继续训练；这些数值是继续多步滚动的实质风险。</p>')
 add('<h2>按预先标准保存候选，下一步仅建议独立单步验证</h2>')
 dec=ev['decision'];rows=[['代表模型旧C24/24、新C18/18，六根两侧均全对',dec['representative_correctness_pass']],['三个种子旧C不退化、关键条件稳定',dec['three_seed_stability_pass']],['同1500更新下新C根配对收益、旧C不退化',dec['same1500_update_data_benefit_pass']],['合并C calibration原谓词召回/FPR/UNKNOWN保护条件',dec['representative_Ccal_original_predicate_guard_pass']]];add(table(['训练前固定的本轮开发条件','结果'],[[a,'通过'if b else'未通过']for a,b in rows]));add('<p>这些是本轮单步符号开发标准，潜状态误差没有被事后改成通过，也不是完整原指南的验收。已保存精确选中的候选、原读出依赖和树对照哈希。独立验证建议为一次新D的九个根（三动作族×三色），每根三次、最多27次诊断，全程锁定候选与树、原复制和保持基线，保留失败及未覆盖，不重采。该建议尚未执行，必须后续另行授权；本轮止于训练与开发评估。</p>')
 add('<h2>核验与交付</h2><p>独立核验通过：419个此前交付文件哈希与只读状态未变；174对合并缓存逐项匹配来源；9000更新和全部采样索引可重建、没有C进入梯度；30个检查点的优化器步数及选择规则一致；树的每个叶子支持只来自AB；12,000组事实统计、全部根配对和潜状态误差重算一致。候选文件与预先选择的检查点哈希相同。</p>')
 links=[('全部逐次预测CSV','results/all_pair_predictions.csv'),('全部指标CSV','results/all_metrics.csv'),('全部根CSV','results/all_root_metrics.csv'),('18组同根对照','results/root_contrasts.json'),('读出／预测误差分层','results/error_partition.csv'),('完整结果JSON','results/evaluation.json'),('学习得到的浅树文本','results/tree_aug132_readable.txt'),('冻结训练协议','configs/protocol.json'),('检查点选择记录','configs/model_selection_frozen.json'),('独立核验记录','results/final_validation.json'),('开发标准判定','results/decision.json'),('候选说明','candidate/candidate.json'),('候选权重','candidate/transition.pt'),('独立验证建议（未执行）','INDEPENDENT_VALIDATION_PROPOSAL.json'),('上一轮冻结诊断','../state_predicates_state_dependence_20261014/report.html')];add('<div class="links">'+''.join('<a href="'+p+'">'+html.escape(label)+'</a>'for label,p in links)+'</div>')
 css='body{font-family:"Noto Sans CJK SC",sans-serif;max-width:1280px;margin:0 auto;padding:36px 26px;line-height:1.8;color:#203449;background:#f5f7fa}h1{font-size:34px;line-height:1.4}h2{font-size:24px;margin-top:40px}.lead{font-size:19px}.note{background:#e7eff6;border-left:5px solid #3a6c91;padding:20px}.eyebrow,figcaption{color:#5b6c7c}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:10px;border:1px solid #dae2e9;text-align:left;vertical-align:top}th{background:#e8eef4}tr:nth-child(even){background:#f7f9fb}.scroll{overflow-x:auto}figure{margin:22px 0;padding:14px;background:white;border:1px solid #dae2e9}figure img{width:100%;height:auto}figcaption{padding:8px;font-size:14px}.links{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.links a{padding:12px;background:white;color:#235879;border:1px solid #dae2e9}@media(max-width:800px){body{padding:15px}.links{grid-template-columns:1fr}h1{font-size:28px}}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>有限条件转移训练与开发评估</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>')
 summary={'created_utc':now(),'scope_complete':True,'training_runs':6,'total_updates':9000,'all_three_augmented_seeds_Cold_all7':24,'all_three_augmented_seeds_Cnew_all7':18,'all_three_old96_controls_Cnew_all7':4,'augmented_Cnew_both_sides_all7':6,'old96_Cnew_both_sides_all7':0,'tree_aug132_Cnew_all7':12,'tree_aug132_Cnew_both_sides_all7':0,'copy_Cnew_all7':10,'candidate':aug,'candidate_saved':True,'development_pass':True,'latent_warning':{'representative_Cnew_scene_MAE':ev['latent_metrics'][aug]['C_new']['scene_MAE'],'representative_Cnew_outside_GRU_fraction':ev['latent_metrics'][aug]['C_new']['outside_GRU_range']},'no_D_no_loop_no_upload':True,'no_more_training':True,'figures':7,'report':str(E/'report.html')};save(E/'results/summary.json',summary)
 (E/'HANDOFF.md').write_text('''# 有限条件转移训练与开发评估交付

用户授权的新轮次完成。项目 `/home/robot/research/projects/wm_symbolic`，实验 `experiments/state_predicates_condition_training_20261015`。未采D、未仿真/采集、未闭环、未上传、未改控制器或冻结读出。

## 训练与因果对照范围

- 固定种子0/1/2；旧96对与旧新132对各1500更新，同批量16、初始化、优化器和共同AB类别权重。总9000更新，144000样本呈现；训练循环约15.20秒，仅缓存潜状态小MLP，不含之前采集/DINO。
- 唯一可训练神经模块是原76224参数full潜转移；所有视觉/状态编码器、共享读出及.5阈值冻结。七事实与所有实体均监督，保留一条旧AB当前读出错误。
- 旧AB96/24根，新AB36/12根；旧C24/6根，新C18/6根；根不可拆。C tune21对/6根按根BCE选检查点，Ccal21对不选模型。检查点300/600/900/1200/1500，均训练满1500。另报同1500结果，避免选择效应。
- 简单模型为预先固定的七棵独立CART，深度4、叶至少2、加权Gini；仅观测读出的七事实三值+9动作输入，AB监督，不用GT当前状态、不手写C条件、不用C调结构。旧96/新132各拟合一次。实现纯numpy，含已通过的精确小例子与叶支持独立检查。

## 主要结果

三个增量种子：旧C24/24、新C18/18、合并42/42七事实全对；UNKNOWN0；新C六组两侧目标和全七事实均6/6。旧96三个种子所选与固定1500：旧C24/24，新C4/18，两侧0/6。补数据收益不能仅由延长训练解释；同权重、同更新与同初始状态对照支持这一点。
固定浅树旧96新C8/18，增量树12/18、两侧0/6、UNKNOWN4；原复制10/18、两侧0/6、UNKNOWN16。增量树AB也仅125/132，35叶中9叶标签混合，属于此固定结构的拟合不足；不能据此排除更合适的纯符号模型或证明潜模型普遍必要。未因C结果扩树/补模型。
全部42个C前后真实读出正确，增量预测无C七事实错误；旧96仍14个新C错误。真实后继替换只用于诊断，不作为预测输入。

## 必须保留的限制

代表候选aug132_seed0@1500通过预先单步符号开发门槛；candidate/transition.pt与所选检查点完全相同。新C物体MAE0.2393、场景MAE1.5273，49.49%整体预测潜坐标超GRU[-1,1]。场景潜误差反而增大：符号正确不等于真实后继动力学可靠。原损失0.05潜项未改，未追加调参。开发标准并未把潜状态误差当通过；不可升级为多步/TS-LTL/闭环验收。
拒绝只能解释该固定控制器的接口语义。所有C已用于开发，不是盲测泛化证据。

## 下一步建议，未执行

INDEPENDENT_VALIDATION_PROPOSAL.json：一次未来独立单步验证，建议9个新D根（三族×三色）、每根3次、最多27诊断；预先锁定候选、增量树、原复制与保持基线，保持真实2.1秒窗口和96posthold，保留失败/未覆盖、不补采、不训练。需用户后续明确授权才执行。不是闭环建议。

## 核验和文件

report.html含七张中文图（训练曲线、等预算C、潜误差、三类同根双相机事实对照、全部C根）。results下完整逐次/逐根/指标CSV、树可读文本、误差分层、门槛判定。configs/protocol.json训练前冻结；model_selection_frozen.json最终评估前冻结。source保留完整复现代码。
final_validation.json通过：419个旧交付文件哈希/只读不变，174对合并源数组一致，全部采样重构且无C梯度，30个检查点与优化器步数一致，树叶AB支持独立核验，12000组事实统计及根对照/潜状态重算一致，候选哈希与选择相同。没有核验失败或训练重跑。所有训练/评估进程均结束，后续不要重跑训练来追分。
''')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
