from config import *
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
PATHS=[('actual','真实后继读出','#2471a3','o'),('one_step','真实前态单步','#d68910','s'),('recursive','仅初态递归','#b03a2e','^')]
NAMES={'original_head':'原冻结头'}|{f'query_{a}_seed{s}':f'{"旧覆盖"if a=="old_coverage"else"新增覆盖"}头{s}'for s in[0,1]for a in['old_coverage','new_coverage']}|{f'transition_{a}_seed{s}':f'{"旧转移"if a=="old_coverage"else"新增转移"}{s}'for s in[0,1]for a in['old_coverage','new_coverage']}
def seq(d,name,method,branch='normal'):return '、'.join(str(r[method]['full9_correct'])for r in d['trajectory_prediction'][name]['branches'][branch]['all_available'])
def main():
 c=read(E/'results/metrics_C_validation.json');d=read(E/'results/metrics_D3.json');covs=[read(E/'results/coverage_ABC.json'),read(E/'results/coverage_D3.json')];eps=read(E/'results/collection_ABC.json')+read(E/'results/collection_D3.json');heads=['original_head','query_old_coverage_seed0','query_new_coverage_seed0','query_old_coverage_seed1','query_new_coverage_seed1'];tmodels=[f'transition_{a}_seed{s}'for s in[0,1]for a in['old_coverage','new_coverage']];h=np.arange(1,5)
 # Readout figure: all values from independent D3 roots, original head retained as reference.
 fig,ax=plt.subplots(figsize=(12,5.5));strata=['pre_grasp','held_open','hover_above_drawer','before_release','post_place','held_closed'];sn=['空手开屉\n抓取前','持物开屉\n抓取后','抽屉上方\n悬停','松手前','放置后\n稳定等待','持物关屉']
 for j,(name,color,label)in enumerate([('original_head','#7d8597','原冻结头'),('query_new_coverage_seed0','#249c8b','新增覆盖头0'),('query_new_coverage_seed1','#9a71b4','新增覆盖头1')]):
  vals=[d['true_state_readout'][name]['strata'][s]['full9_correct']for s in strata];bars=ax.bar(np.arange(6)+(j-1)*.25,vals,width=.24,color=color,label=label);ax.bar_label(bars,fontsize=10)
 ax.set(xticks=np.arange(6),xticklabels=sn,ylim=(0,14),ylabel='真实状态九命题全对数 / 12',title='新D根：中间态读出改善，但空手开屉有退化');ax.legend(loc='upper center',ncol=3);ax.grid(axis='y',alpha=.2);fig.text(.055,.02,'每类每根1帧，共12根；同一根各帧相关。灰=原冻结头；绿/紫=新增覆盖头的两个种子。\n编码器、转移、命题定义与阈值相同。本图只评价真实状态读出，不代表动作预测或任务成功率。',fontsize=10);fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(R/'D3_readout.png',dpi=170);fig.savefig(R/'D3_readout.svg');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(13,9))
 for seed in[0,1]:
  ax=axs[0,seed];rows=d['trajectory_prediction'][f'query_new_coverage_seed{seed}']['branches']['normal']['all_available']
  for method,label,color,marker in PATHS:ax.plot(h,[r[method]['full9_correct']for r in rows],color=color,marker=marker,label=label,lw=2)
  ax.set(title=f'仅更新查询头 · 种子{seed}',xlabel='技能步：1开屉 / 2抓取 / 3放入 / 4关屉',ylabel='九命题全对数 / 同12根',xticks=h,ylim=(-.5,13.5));ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
  ax=axs[1,seed]
  for name,label,color,ls in [(f'query_new_coverage_seed{seed}','固定原转移','#7d8597','--'),(f'transition_old_coverage_seed{seed}','等预算旧转移训练','#2471a3','-'),(f'transition_new_coverage_seed{seed}','等预算新增转移训练','#249c8b',':')]:
   rr=d['trajectory_prediction'][name]['branches']['normal']['all_available'];ax.plot(h,[r['recursive']['full9_correct']for r in rr],marker='o',lw=2.5,ls=ls,color=color,label=label)
  ax.set(title=f'再固定查询头 · 种子{seed} · 只比较递归',xlabel='连续技能步（不中途纠正）',ylabel='九命题全对数 / 同12根',xticks=h,ylim=(-.5,13.5));ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
 fig.suptitle('正常动作主报告：新数据没有解决首步和递归预测错误',fontsize=16);fig.text(.04,.018,'上排：同一原转移模型，用新增覆盖查询头比较真实读出、单步、递归。下排：查询头固定后比较三种转移模型。\n所有点使用同12个新D根和同一实际动作序列；技能执行失败仍保留。下排曲线重合表示结果相同，不能把旧转移继续训练的收益归给新增数据。',fontsize=10);fig.tight_layout(rect=[0,.08,1,.95]);fig.savefig(R/'D3_normal_prediction.png',dpi=170);fig.savefig(R/'D3_normal_prediction.svg');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,5))
 for seed,ax in enumerate(axs):
  rows=d['trajectory_prediction'][f'transition_new_coverage_seed{seed}']['branches']['held_closed']['all_available']
  for method,label,color,marker in PATHS:ax.plot(h,[r[method]['full9_correct']for r in rows],color=color,marker=marker,label=label,lw=2)
  ax.set(title=f'新增转移 + 已冻结查询头{seed}',xlabel='调用步：1抓取 / 2放入 / 3开屉 / 4关屉',ylabel='九命题全对数 / 同12根',xticks=h,ylim=(-.5,13.5));ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
 fig.suptitle('拒绝分组：第一步实际抓取；后面三次调用均被控制器拒绝',fontsize=14);fig.text(.04,.02,'蓝=真实后继读出；橙=真实前态单步；红=仅初态递归。拒绝后仍记录96个真实稳定控制步。\n递归第2–4步均为0/12；拒绝记录与正常链分开，不能合并成24个独立初始配置。',fontsize=10);fig.tight_layout(rect=[0,.12,1,.93]);fig.savefig(R/'D3_refusal_prediction.png',dpi=170);plt.close(fig)
 # Detailed numerical supplement, with every condition, path, group and paired category.
 sup=['# 全量分组统计','','所有“全对”均要求九项真值有效且全部匹配；Holding/In/drawer计数是组内至少一项错误。真实读出是单独控制，不中途输入递归。','']
 csvrows=[]
 for phase,m in [('C验证',c),('D3测试',d)]:
  for branch,title in [('normal','正常链'),('held_closed','持物关屉/拒绝分支')]:
   sup+=['## '+phase+' · '+title,'','| 条件 | 步 | 观测/计划根 | 真实 / 一步 / 递归 全对 | 真值读出错组 H/I/屉 | 一步错组 H/I/屉 | 递归错组 H/I/屉 | 真读对/一步错 | 两者均错 | 真读对/递归错 | 两者均错 | 一步/递归 原MSE |','|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|']
   for name,x in m['trajectory_prediction'].items():
    for r in x['branches'][branch]['all_available']:
     nums=[r[k]['full9_correct']for k in ['actual','one_step','recursive']];grp=lambda k:'/'.join(str(r[k]['groups'][g]['wrong_states'])for g in ['Holding','In','drawer']);a=r['one_step_paired']['states'];b=r['recursive_paired']['states'];sup.append(f'| {NAMES[name]} | {r["horizon"]} | {r["observed_roots"]}/{r["planned_roots"]} | '+ '/'.join(map(str,nums))+' | '+' | '.join(grp(k)for k in ['actual','one_step','recursive'])+f' | {a["actual_right_prediction_wrong"]} | {a["both_wrong"]} | {b["actual_right_prediction_wrong"]} | {b["both_wrong"]} | {r["one_step_raw_MSE"]:.4f}/{r["recursive_raw_MSE"]:.4f} |');csvrows.append({'phase':phase,'branch':branch,'condition':name,'horizon':r['horizon'],'observed_roots':r['observed_roots'],'actual_all9':nums[0],'one_step_all9':nums[1],'recursive_all9':nums[2],'one_step_actual_right_prediction_wrong':a['actual_right_prediction_wrong'],'recursive_actual_right_prediction_wrong':b['actual_right_prediction_wrong'],'one_step_raw_MSE':r['one_step_raw_MSE'],'recursive_raw_MSE':r['recursive_raw_MSE']})
 (R/'SUPPLEMENT.md').write_text('\n'.join(sup)+'\n')
 with(R/'prediction_summary.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 lines=['# 补齐中间态数据后的预测评测','','完成时间：'+now(),'','## 本轮证明了什么','','完整姿态和接触记录补齐了此前无法核验的入屉标签；新增中间态覆盖使真实状态读出在两个种子上都改善。但改善有取舍，尤其是空手开屉状态。新增转移训练没有解决正常链首步及长链递归失配，其正常链结果多数与等预算旧转移训练相同。不能把读出提高当成预测或规划问题已经解决。','','本轮按既有批准的物理控制器采集，保持网络结构、9命题、技能接口、损失和阈值不变；只进行了预测评测，没有规划或任务成功率评测，也未更改原指南的总体验收要求。','','## 数据、隔离和执行预算','','| 用途 | 独立初始配置根 | 三色每色根数 | 真实调用/转移 | 保存的观测 |','|---|---:|---|---:|---:|']
 for split,label in [('train','A/B训练'),('dev_select','C选择'),('dev_validate','C验证'),('test_D3','新D测试')]:
  ep=[r for r in eps if r['split']==split];lines.append(f'| {label} | {len({r["root_id"]for r in ep})} | '+('4（A/B各2）'if split=='train'else'2'if split.startswith('dev')else'4')+f' | {sum(r["actual_calls"]for r in ep)} | {sum(r["saved_states"]for r in ep)} |')
 counts=collections.Counter(s['status']for ep in eps for s in ep['slots']);lines+=['',f'共36根、72次复位、288次真实调用（成功{counts["success"]}、拒绝{counts["refused"]}、失败{counts["failed"]}，未达0），1649份保存观测全部9项真值可判定。连续视频和逐控制物理轨迹均保留。原始采集约530秒、423MB；首50转移去重边界观测约3.55MB。后续特征与模型文件另计。','','每根两条不同动作分支来自同一初始配置，不能当作两个独立根。所有派生帧、标签和特征跟随根划分。D3仅指同一物理环境D的新初态批次，不是新环境。D3在全部模型/阈值封存后才采集并评测；现在已经看过，不得再称未见测试。','','正常分支固定为开屉、抓取目标、放入、关屉；持物关屉分支固定为抓取、放入、开屉、关屉，后3步通常拒绝。设置初态只发生于原生复位，不算动作转移；调用期间只通过原机器人控制器推进物理，没有瞬移或强制附着。每次最多900控制步、96步后等待，内部最多2次尝试、外部0次重试，失败未换根。','','| 实际达到的关键条件（每色独立根数） | A/B训练 红/蓝/粉 | C选择 红/蓝/粉 | C验证 红/蓝/粉 | 新D 红/蓝/粉 |','|---|---|---|---|---|']
 allcov=covs[0]['coverage']|covs[1]['coverage']
 for st,title in [('held_open','持物且开屉'),('hover_above_drawer','持物悬停在开屉上方'),('post_place','放置后确实满足严格入屉'),('held_closed','持物且关屉')]:
  vals=['/'.join(str(allcov[s][colour][st]['condition_confirmed_roots'])for colour in COLORS)for s in ['train','dev_select','dev_validate','test_D3']];lines.append('| '+title+' | '+' | '.join(vals)+' |')
 lines+=['','未达到严格入屉的放置后样本仍在数据中，标签为实际值，不标未知、不剔除；未知掩码本轮样本恰为0。C验证红色严格入屉只有1根，仍不足以声称每项条件均有多个成功根，未为补齐数量越过预算。','','## 先只更新查询头','','两臂各144个训练状态、1200次更新、38400次抽样；96个旧状态共享，新增臂另用48个新状态（每A/B根各取持物开屉、悬停、放置后、持物关屉1帧）。每批相同20个共享样本，另12个为旧池或四种新状态各3个。两个种子共享配对初始化；原冻结头另外作参考。新增覆盖与抽样构成整体处理，不能拆成单独的“更多根”或“更多标签”效应。','','用旧C选择根和新C选择根的等权损失选检查点；全部更新均完成。新C验证6根、旧C保留集以及D未参与选择。','','| 查询头 | 新C验证全对/273 | 新D全对/549 | 新D根平均 | 旧C保留集全对/110 |','|---|---:|---:|---:|---:|']
 for name in heads:
  cm=c['true_state_readout'][name]['all_states'];dm=d['true_state_readout'][name]['all_states'];old=c['old_C_retention'][name];lines.append(f'| {NAMES[name]} | {cm["full9_correct"]} | {dm["full9_correct"]} | {dm["full9_root_macro"]:.1%} | {old["full9_correct"]} |')
 lines+=['','273/549是相关的观测帧，独立根数为6/12；根平均先在各根内部算正确率，再平均。旧C保留集只列110个完整标签状态，另6个部分标签状态单独存储评分，不混作完整九项。','','![新D真实状态读出](D3_readout.png)','','| 新D关键帧（每项12根、各1帧） | 原冻结头 | 旧覆盖头0/1 | 新增覆盖头0/1 |','|---|---:|---:|---:|']
 for st,title in zip(strata,['空手开屉、抓取前','抓取后持物开屉','抽屉上方悬停','松手前','放置后稳定','持物关屉']):
  get=lambda n:d['true_state_readout'][n]['strata'][st]['full9_correct'];lines.append(f'| {title} | {get("original_head")}/12 | {get("query_old_coverage_seed0")}/{get("query_old_coverage_seed1")} | {get("query_new_coverage_seed0")}/{get("query_new_coverage_seed1")} |')
 lines+=['','新增覆盖相对同种子旧覆盖，在D的完整状态配对中：种子0有146帧错→对、18帧对→错；种子1有142帧错→对、27帧对→错。整体保留集改善不能掩盖特定退化：原头空手开屉为12/12，新头为9/12和2/12；在新C验证上则由原头6/6降为两个新头均0/6，主要把抽屉打开错判为关闭。','','## 再看预测，区分读出更新与转移更新','','下表先固定原转移，只更换查询头；每格按技能1至4列出全九项正确的根数，分母始终12。它们不是整条任务成功率。','','| 查询头 | 真实后继读出 | 真实前态单步预测 | 仅初态递归预测 |','|---|---|---|---|']
 for name in heads:lines.append('| '+NAMES[name]+' | '+' | '.join(seq(d,name,k)for k in ['actual','one_step','recursive'])+' |')
 lines+=['','![正常链逐步预测及独立转移对照](D3_normal_prediction.png)','','最早有影响的错误并不只在长链后段：原头的第1步读出/预测均12/12，新增头在相同固定转移的第1步预测变为0/12，两个种子都有12条抽屉命题失配。种子0此时9条真实后继仍能读对，因此有9条明确的“真实读对、预测错”；种子1对应2条。只替换查询头的对照证明这次读出更新带来了首步预测兼容性退化，不能全部归给递归累积。','','第3步，新增头两种子均为真实后继11/12、一步9/12、递归2/12。逐根配对均有2条真实读对/一步错、9条真实读对/递归错；递归主要仍把已松手对象判为持有。这些是配对结果，不是总体准确率相减。','','查询头按C选择后固定：两个种子均选新增覆盖头。随后各跑旧/新增转移对照，都是从相同原转移出发、1000次更新、64000次抽样，48个旧样本共享，16个辅助样本来自旧或新增A/B；辅助旧样本匹配同一技能分布。编码器、解码器、查询头与阈值全冻结，保持原损失和旧链多步项。新增臂只使用96个新增A/B技能边界转移，不把悬停截帧伪装成新技能。','','| 固定查询头后的转移条件 | 正常链一步 /12 | 正常链递归 /12 | 持物关屉分支一步 /12 | 持物关屉分支递归 /12 |','|---|---|---|---|---|']
 for name in tmodels:lines.append('| '+NAMES[name]+' | '+' | '.join([seq(d,name,'one_step'),seq(d,name,'recursive'),seq(d,name,'one_step','held_closed'),seq(d,name,'recursive','held_closed')])+' |')
 lines+=['','正常链第3步递归从固定原转移的2/12提高到7/12、4/12，但等预算旧转移训练也达到相同7/12、4/12；不能把这部分收益归功于新增转移。新增转移在持物关屉分支第1步从旧转移的1/12提高到5/12、6/12；但后续拒绝调用的递归仍全部0/12。','','![拒绝分组预测](D3_refusal_prediction.png)','','原空间潜状态MSE、所有Holding/In/抽屉分组错误、真读对/预测错及两者均错均列在[完整统计](SUPPLEMENT.md)。MSE为固定256维潜表示的平方误差，不是语义准确率；降低不能自动证明九命题预测改善。','','## 真实画面解释关键混淆','','![固定C验证根的真实时刻](real_intermediate_states.png)','','每行同一个C验证根，每格左固定相机、右腕相机。新覆盖头能区分手持悬停与多数放置后状态，但出现空手开屉判关的退化。红色案例放置后最高角点高出抽屉沿约10.4毫米，虽有抽屉承重接触，按原严格定义In仍为0；这说明完整姿态标签为何必要。原视频保留，以下仅裁取连续真实帧：','','- [红块悬停至放置后](videos/C_intermediate_v3_red_2_hover_to_after_place.mp4)','- [蓝块悬停至放置后](videos/C_intermediate_v3_blue_2_hover_to_after_place.mp4)','- [粉块悬停至放置后](videos/C_intermediate_v3_pink_2_hover_to_after_place.mp4)','','## 交付与界限','','- [采集协议](PROTOCOL.md)；详细冻结配置、根清单、来源哈希与模型选择在实验configs目录。','- [D3逐命题配对7776行](paired_propositions_D3.csv)；[C验证配对3888行](paired_propositions_C_validation.csv)；[逐步摘要CSV](prediction_summary.csv)。','- 全量源码、原始记录、标签、特征、模型与日志：`'+str(E)+'`。','- 独立复核结果在`results/verification.json`，逐段视频帧数在`physical_media_verification.json`；最终清单为`configs/delivery_manifest.json`。','','本次有限补数与测试已按既定规模完成。结论是数据缺口确实影响读出，但补齐数据尚未解决预测；首步抽屉读出兼容性和拒绝后的递归状态保持仍失败。本轮不追加调参、不复用这批测试作未见测试，不以局部收益替代总体项目验收。']
 (R/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT_READY',flush=True)
if __name__=='__main__':main()
