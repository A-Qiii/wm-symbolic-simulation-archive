from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import imageio.v2 as imageio
from PIL import Image
import shutil,html,collections
F=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BLUE='#267ab3';ORANGE='#e38b2c';GREEN='#3c9d78';RED='#ce6666';ARMS=['canonical_control','canonical_matched']
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)
def frac(x):return f"{x['all_valid_facts_correct_states']}/{x['states']}"
def title(ax,s):ax.set_title(s,fontproperties=F,fontsize=12)
def finish(fig,path):fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
def main():
 R.mkdir(exist_ok=True);c=read(E/'results/C_metrics.json');d=read(E/'results/D2_metrics.json');loops=read(E/'results/closed_loop.json');den=read(E/'results/D2_denominators.json');coll=read(E/'results/D2_collection.json');diag=read(E/'results/single_substitution.json');roots=read(E/'configs/D2_roots.json');budget=read(E/'configs/protocol.json')['execution_limits'];train={f'{a}_seed{s}':read(E/'results'/f'train_{a}_{s}.json')for a in ARMS for s in[0,1]}
 comparisons=[]
 for seed in[0,1]:
  x=d[f'{ARMS[0]}_seed{seed}']['condition_held_closed'];y=d[f'{ARMS[1]}_seed{seed}']['condition_held_closed'];shared=sorted(set(x['root_errors'])&set(y['root_errors']));delta=[y['root_errors'][r]-x['root_errors'][r]for r in shared];comparisons.append({'seed':seed,'roots':len(shared),'matched_better_roots':sum(v<0 for v in delta),'control_error':x['weighted_observation_MSE_root_mean'],'matched_error':y['weighted_observation_MSE_root_mean'],'relative_error_reduction':1-y['weighted_observation_MSE_root_mean']/x['weighted_observation_MSE_root_mean'],'paired_root_deltas':dict(zip(shared,delta))})
 save(E/'results/primary_comparison.json',comparisons)
 # C root weighting and D2 consequence error; all bars have a labelled denominator.
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,(group,label)in zip(axs,[('dataset_original','原C：35根 / 300次'),('dataset_matched','新增C：2根 / 6次'),('all','全部C：37根等权')]):
  for k,arm in enumerate(ARMS):ax.bar(np.arange(2)+(k-.5)*.32,[c[f'{arm}_seed{s}'][group]['weighted_observation_MSE_root_mean']for s in[0,1]],width=.3,color=[BLUE,ORANGE][k],label=['control','matched'][k])
  ax.set_xticks([0,1],['seed0','seed1']);ax.set_ylabel('Weighted observation MSE');title(ax,label);ax.set_ylim(bottom=0);ax.legend()
 finish(fig,R/'C_weighting.png')
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,(group,label)in zip(axs,[('condition_held_closed','D2持物关闭：真实place后果'),('outcome_success','D2实际执行成功：回归检查')]):
  for k,arm in enumerate(ARMS):
   vals=[d[f'{arm}_seed{s}'][group]['weighted_observation_MSE_root_mean']for s in[0,1]];bars=ax.bar(np.arange(2)+(k-.5)*.32,vals,.3,color=[BLUE,ORANGE][k],label=['control','matched'][k]);ax.bar_label(bars,fmt='%.3f',fontsize=9)
  ax.set_xticks([0,1],['seed0','seed1']);ax.set_ylabel('Weighted observation MSE');title(ax,label);ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.legend()
 finish(fig,R/'D2_consequences.png')
 fig,ax=plt.subplots(figsize=(8,7));arr=np.array([[int(next(r for r in loops if r['root_id']==root['root_id']and r['arm']==a)['physical_success'])for a in ARMS]for root in roots]);from matplotlib.colors import ListedColormap
 ax.imshow(arr,cmap=ListedColormap([RED,GREEN]),vmin=0,vmax=1,aspect='auto');ax.set_xticks([0,1],['control / seed0','matched / seed0']);ax.set_yticks(range(12),[r['root_id'].replace('D2_place_repair_','')+(' / open'if r['drawer_reset_m'] else' / closed')for r in roots]);title(ax,'同12根匹配闭环：绿=实际目标成功，红=失败')
 for i,root in enumerate(roots):
  for j,a in enumerate(ARMS):v=next(r for r in loops if r['root_id']==root['root_id']and r['arm']==a);ax.text(j,i,('OK'if v['physical_success']else'FAIL')+f" / {v['calls']} calls",ha='center',va='center',color='white')
 finish(fig,R/'closed_loop_grid.png')
 (R/'videos').mkdir(exist_ok=True);media=[]
 for arm in ARMS:
  for success in[True,False]:
   candidates=[r for r in loops if r['arm']==arm and r['physical_success']==success]
   if not candidates:continue
   row=candidates[0];src=Path(row['video']);name=arm+'_'+('success'if success else'failure');dst=R/'videos'/f'{name}.mp4';shutil.copy2(src,dst);reader=imageio.get_reader(dst);meta=reader.get_meta_data();frames=reader.count_frames();assert frames==row['frames'];img=reader.get_data(frames//2);reader.close();Image.fromarray(img).save(R/'videos'/f'{name}.png');media.append({'name':name,'root_id':row['root_id'],'arm':arm,'physical_success':success,'calls':row['calls'],'status':row['native_statuses'],'video':str(dst),'source_video':str(src),'frames':frames,'fps':meta['fps'],'duration':meta['duration'],'sha256':sha(dst),'continuous':True,'type':'autonomous_closed_loop'})
 if not any(r['physical_success']for r in media):
  candidates=[]
  for r in coll:
   if r['condition']!='reference'or not r['tasks']or not r['tasks'][-1].get('after'):continue
   l=r['tasks'][-1]['after']['labels'];j=3+COLORS.index(r['target'])
   if l['y'][j]and l['y'][6]and l['v'][j]and l['v'][6]:candidates.append(r)
  if candidates:
   r=candidates[0];src=Path(r['folder'])/'continuous.mp4';dst=R/'videos/reference_executor_success.mp4';shutil.copy2(src,dst);reader=imageio.get_reader(dst);meta=reader.get_meta_data();n=reader.count_frames();Image.fromarray(reader.get_data(128)).save(R/'videos/reference_executor_success.png');reader.close();media.append({'name':'reference_executor_success','root_id':r['root_id'],'arm':'fixed_reference','physical_success':True,'calls':sum(t.get('native')is not None for t in r['tasks']),'status':[t['status']for t in r['tasks']],'video':str(dst),'source_video':str(src),'frames':n,'fps':meta['fps'],'duration':meta['duration'],'sha256':sha(dst),'continuous':True,'type':'fixed_reference_NOT_autonomous_success'})
 save(E/'results/media_manifest.json',media)
 cr=[];dr=[];colors=[]
 for arm in ARMS:
  for seed in[0,1]:
   key=f'{arm}_seed{seed}';sel=read(E/'models'/key/'selected.json');v=c[key];cr.append([key,sel['update'],*[f"{v[g]['weighted_observation_MSE_root_mean']:.4f}"for g in['dataset_original','dataset_matched','all','original_success']]])
   for g in['condition_held_closed','outcome_success','outcome_refused','all']:
    if g not in d[key]:continue
    v=d[key][g];dr.append([key,g,v['pairs'],v['roots'],frac(v['current']),frac(v['actual_successor']),frac(v['predicted_successor']),f"{v['target_In_false_positive']}/{v['target_In_valid_pairs']}",f"{v['weighted_observation_MSE_root_mean']:.4f}"])
   for split,mm in[('C',c),('D2',d)]:
    for color in COLORS+['unspecified']:
     g='colour_'+color
     if g in mm[key]:v=mm[key][g];colors.append([split,key,color,v['pairs'],v['roots'],f"{v['weighted_observation_MSE_root_mean']:.4f}",frac(v['predicted_successor'])if'predicted_successor'in v else'无完整标签'])
 looprows=[]
 for a in ARMS:
  rr=[r for r in loops if r['arm']==a];looprows.append([a,f"{sum(r['physical_success']for r in rr)}/12",f"{sum(r['physical_success']for r in rr if r['initial_open'])}/6",f"{sum(r['physical_success']for r in rr if not r['initial_open'])}/6",sum(r['false_completion']for r in rr),sum(r['calls']for r in rr)])
 failrows=[]
 for r in diag:
  if r['physical_success']:continue
  f=r['first_supported_error'];first=next((x for x in r['replacements']if x['acceptance_removed']),None);failrows.append([r['arm'],r['root_id'],f['kind']if f else'该替换未定位',f.get('step','')if f else'',('是'if first['current_all_valid_correct']else'否')if first else'',f"{first['original_planned_score']:.3f} → {first['replace_first_with_actual_observation_latent_score']:.3f}"if first else'无阈值翻转'])
 taskcalls=den['executed_tasks']+sum(r['calls']for r in loops);prepcalls=den['preparation_calls'];frames=sum(r['frames']for r in coll)+sum(r['frames']for r in loops);controlsteps=frames-len(coll)-len(loops);assert taskcalls<=234 and prepcalls<=24 and controlsteps<=267636
 audit={'offline':den,'closed_loop_episodes':len(loops),'closed_loop_roots':12,'closed_loop_task_calls':sum(r['calls']for r in loops),'all_task_calls':taskcalls,'preparation_calls':prepcalls,'all_primitive_calls':taskcalls+prepcalls,'total_control_steps':controlsteps,'physics_ticks':controlsteps*8,'continuous_video_frames':frames,'all_video_control_frequency_Hz':30,'training_updates':4000,'training_wall_seconds_sum':sum(r['seconds']for r in train.values()),'GPU_peak_training_bytes':max(r['peak_GPU_bytes']for r in train.values()),'old_D_reexecutions':0,'D2_retraining_or_reselection':False,'all_frozen_parameter_hashes_unchanged':all(r['final_frozen_hash']==r['initial']['frozen_hash']and r['final_query_hash']==r['initial']['query_hash']for r in train.values())};save(E/'results/execution_audit.json',audit)
 summary='；'.join(f"seed{r['seed']}持物关后继观测误差下降{r['relative_error_reduction']*100:.1f}%（{r['matched_better_roots']}/{r['roots']}根改善）"for r in comparisons);ls='；'.join(f"{a}实际闭环{sum(r['physical_success']for r in loops if r['arm']==a)}/12"for a in ARMS)
 report=f'''# 最小转移修复：一次冻结D2评估

**本轮修复未改善实际闭环，并出现正常动作语义预测退化。** {summary}。{ls}。持物关闭的完整九命题预测仍是两臂、两种子均0/12；正常成功动作的预测由对照31/50、39/50降为匹配26/50、34/50。目标In假阳性虽从12/12降为6/12和8/12，仍未消除。连续后果局部改善成立，但主任务尚未修好；不以MSE下降替代符号正确或真实完成。

## 实验边界与冻结检查

两臂均使用无对象参数`place_in_drawer()`；lift保留颜色参数，目标仍是In(指定颜色)且Closed。低层为未修改的PhysicalOracle脚本，不是HULC。高层所有候选后继仍来自学习转移，beam32、深度4、阈值0.8、每调用后重观察，最多6调用；没有GT前提过滤或手工拒绝效果。

同一seed的两臂从完全相同权重开始。DINO、预处理、编码器、解码器与查询头冻结；仅67,328个转移参数训练。预先保存1000步基础48样本、16辅助样本与真实链索引；两臂的基础batch/链完全一致，差别只有辅助place数据来源。对照用旧AB888；匹配臂增加AB4根12条，共900。共2臂×2种子×1000更新，所有运行完成；冻结参数哈希与无梯度检查均通过。训练文件访问守卫拒绝私有标签和测试路径，损失不使用命题标签。

准备动作、预算、数据与训练源在`configs/training_frozen.json`冻结。选点及全部执行代码在`configs/before_D2_frozen.json`冻结后才打开D2。原首版和只读复核交付保持原位。本次没有再次调参、增加种子或改query。

## C选择偏置与原任务回归

{table(['模型','选中更新','原C35根MSE','新增C2根MSE','C37根等权MSE','原C成功249次MSE'],cr)}

![C权重对照](C_weighting.png)

横轴是种子；蓝=对照，橙=补充匹配数据；纵轴为四组等权、变化加权的标准化观测预测MSE，越低越好。三面板依次是旧35根、新2根、全部37根等权。选点实际给新增2根50%权重，这不是全体根等权；两个根没有红色，不能凭其下降断言泛化。所有C根等权及旧任务成功调用回归均在表中，代价不隐藏。

旧匹配C记录缺少新版几何In评分所需物体姿态，其3个In标为无效，只评持物和抽屉6命题；原C完整9命题子集另列于机器结果。混合有效事实得分不能与全9命题得分直接比较。C颜色表中unspecified表示动作没有显式颜色，未用GT猜补。

## 一次性D2实际分母

12个新根、3颜色×4布局、初始开/关各6。离线60个分支，计划90任务槽位；实际执行{den['executed_tasks']}次，状态计数{den['status_counts']}。准备调用{prepcalls}/24；全部任务调用{taskcalls}/234；总控制步{controlsteps}/267636。准备失败不换根、不重试；未到达槽位仍在`D2_slots.json`及90分母内。连续视频记录每一个控制观测，共{frames}帧，30fps对应仿真时间，模型等待不计入视频时间。

{table(['模型','条件/结果组','实际调用','根','真实前态全对','真实后态全对','预测后态全对','目标In假阳性/有效','后果MSE'],dr)}

![D2后果预测](D2_consequences.png)

左图是主条件“已持物但抽屉关闭”的place，右图是所有实际成功调用，检查正常动作是否退化。蓝对照、橙匹配，横轴种子，纵轴同上无量纲观测MSE，越低越好。这不是RGB像素MSE，也不是物理完成率。每个根下的多个分支相关，不能把90槽位当90独立根。两种子也共享测试根。

语义表中每个分子表示该状态全部有效命题正确。真实前态、真实后态来自真实图像；预测后态来自学习转移，再交给同一冻结查询头。D2存有完整物理快照，使用新版9命题评分。目标In假阳性表示预测认为目标在抽屉内而实际不是；它尤其能揭示拒绝调用被误判为成功效果。即使连续预测改善，冻结query的当前识别错误仍可能存在，不会通过本轮暗改头来修复。

## 自主高层闭环

{table(['模型seed0','实际成功','初始打开','初始关闭','错误完成声明','实际调用'],looprows)}

![匹配根闭环](closed_loop_grid.png)

横轴两臂，纵轴12个相同根和初始开/关；绿=独立物理目标成功，红=失败，格内同时列调用数。只有seed0闭环，不能声称跨种子闭环复现。操作计划没有使用参考链；参考链仅验证底层执行与离线转移，是不同实验。搜索可重复拒绝动作；没有增加额外规则使结果变好。

## 第一个有影响的错误

{table(['模型','失败根','定位','0起始步骤','该步当前读出全对','原计划分数→单项替换'],failrows)}

替换仅使用该次真实执行后已保存RGB/本体感知重新编码的隐状态，替换计划首后继；后续动作、转移参数、查询头和阈值固定。未注入GT后继、未重新物理执行。阈值翻转定位有影响的预测错误，不能唯一排除表示/查询误差；无翻转不等于没有转移错误。错误完成声明则由独立物理真值核验停止。这里只诊断并封存，不在D2上修第二轮。

## 各色与完整交付

{table(['划分','模型','颜色','调用','根','后果MSE','预测后态全有效命题正确'],colors)}

[连续视频图集](index.html)包含代表案例及全部24段闭环链接。真实视频没有补帧；固定相机保持原始200×200，腕部图放大到200×200，H264压缩。绿/红结果来自独立评价，不来自视频主观判断。无自主成功时，图集中的固定参考链成功会明确标注，不能冒充自主规划成功。

训练主体测得合计{audit['training_wall_seconds_sum']:.2f}秒（特征已缓存，仅训练小转移网络；不含进程启动/缓存/采集），峰值已分配GPU内存{audit['GPU_peak_training_bytes']/1024**2:.1f}MiB。完整预算估算与实际调用在`protocol.json`、`execution_audit.json`。所有权重、采样索引、配置、结果、原生调用原因、连续视频、命题有效掩码和逐根差异均保存在本机新实验目录。

本轮只回答：在相同无对象place接口及冻结表示/查询条件下，增加这批交互覆盖是否改变转移与闭环。不能据此隔离接口修正相对首版的效应；不能将不同测试根的成功率直接当成首版3/6的严格提升。原HULC与正式6000转移指南仍未完成，本轮不更改其标准。
'''
 (R/'REPORT.md').write_text(report)
 cards=[]
 for r in media:
  kind='固定参考链成功（不是自主规划成功）'if r['type'].startswith('fixed')else('自主高层闭环成功'if r['physical_success']else'自主高层闭环失败');v=Path(r['video']);cards.append(f'<section><h2>{kind} · {html.escape(r["arm"])} · {r["root_id"]}</h2><p>真实{r["calls"]}次任务调用，{r["frames"]}帧，{r["duration"]:.2f}秒仿真时间。结果序列：{html.escape(str(r["status"]))}</p><video controls preload="metadata" poster="videos/{v.stem}.png" src="videos/{v.name}"></video></section>')
 links=''.join(f'<li><a href="../../experiments/interaction_latent_repair_20261016/data/closed_loop/{r["root_id"]}/{r["arm"]}/continuous.mp4">{r["arm"]} · {r["root_id"]} · {"成功"if r["physical_success"]else"失败"} · {r["calls"]}调用</a></li>'for r in loops)
 (R/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>转移修复连续视频与结果</title><style>body{font-family:system-ui,sans-serif;background:#f2f5f8;color:#172c40;max-width:1050px;margin:32px auto;padding:0 24px;line-height:1.65}section{background:white;border-radius:12px;padding:24px;margin:24px 0}video{width:min(100%,800px)}img{max-width:100%}a{color:#1b67a1}</style><h1>转移修复：真实连续视频与一次D2结果</h1><p>视频按每控制步记录，30fps表示仿真时间，不含模型等待。无补帧或事后重新演示。左图固定相机，右图腕部相机；物块红/蓝/粉代表对象身份。</p><p><a href="REPORT.md">完整分析与分母</a> · <a href="../../reports/interaction_latent_repair_review_20261016/index.html">首版端点动画（保留）</a></p><p>'''+summary+'。'+ls+'。</p>'+''.join(cards)+'<section><h2>全部24段自主闭环</h2><ul>'+links+'</ul></section><section><h2>C选择权重</h2><img src="C_weighting.png"><p>蓝=旧数据对照；橙=补充匹配条件。纵轴标准化观测MSE越低越好；左右分别查看旧35根与新增2根，第三图37根等权。</p></section><section><h2>D2实际后果与正常调用回归</h2><img src="D2_consequences.png"><p>左图持物关闭place，右图实际成功调用；横轴种子，蓝对照橙匹配，纵轴观测MSE越低越好。</p></section><section><h2>12个相同根的真实闭环结果</h2><img src="closed_loop_grid.png"><p>绿成功、红失败，格内为结果与真实调用数；只运行seed0。</p></section></html>')
 print('REPORT_COMPLETE',summary,ls,flush=True)
if __name__=='__main__':main()
