from pathlib import Path
import json,html,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patches
from scipy.spatial import ConvexHull
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_semantics_audit_20261014'
def read(p):return json.loads(Path(p).read_text())
def table(h,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table>'
def img(n,c):return '<figure><img src="data:image/png;base64,'+base64.b64encode((E/'figures'/n).read_bytes()).decode()+'" alt="'+html.escape(c)+'"><figcaption>'+html.escape(c)+'</figcaption></figure>'
def main():
 j=read(E/'results/audit.json');raw=read(E/'results/raw_place_coverage.json');b,p=j['two_acceptance_cases'];cov=j['coverage'];font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,3,figsize=(14,4.9),gridspec_kw={'width_ratios':[1,1,1.25]});g=b['geometry']['blue'];corn=np.array(g['world_corners_m'])*1000
 for k,(dims,title)in enumerate([((0,1),'蓝块：俯视投影'),((0,2),'蓝块：正视高度')]):
  ax=axs[k];points=corn[:,dims];hull=ConvexHull(points);ax.fill(points[hull.vertices,0],points[hull.vertices,1],color='#357db4',alpha=.7,label='蓝块真实包围面',zorder=3)
  if k==0:ax.add_patch(patches.Rectangle((0,-380),360,200,fc='#dcece5',ec='#34836a',label='抽屉内开口'));ax.set(xlim=(-10,370),ylim=(-390,-170),xlabel='世界x（mm）',ylabel='世界y（mm）');ax.plot(points[hull.vertices,0],points[hull.vertices,1],color='#357db4')
  else:
   ax.add_patch(patches.Rectangle((0,340),360,60,fc='#dcece5',ec='#34836a',label='抽屉视觉内腔'));ax.axhline(400,color='#34836a',lw=1.3,label='视觉边沿400mm');ax.axhline(401.2,color='#b47e35',ls='--',lw=1.3,label='碰撞网格最高点401.2mm');ax.annotate('最低点407.61mm\n仍高于边沿7.61mm',xy=(180,407.61),xytext=(12,447),fontsize=9,arrowprops={'arrowstyle':'->','color':'#23374b'});ax.set(xlim=(-10,370),ylim=(332,465),xlabel='世界x（mm）',ylabel='世界z（mm）')
  ax.set_title(title);ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.12)
 with np.load(p['image_paths'][-1])as z:rgb=z['rgb_gripper']
 ax=axs[2];ax.imshow(rgb,interpolation='nearest');pr=p['projection_blue_pink']
 for c,col in [('blue','#4ab7ff'),('pink','#f98ce3')]:
  x,y,x2,y2=pr[c]['box'];ax.add_patch(patches.Rectangle((x-.5,y-.5),x2-x,y2-y,fill=False,ec=col,lw=1.8,label=c+'可见像素框'))
 ax.set(title='粉块静止：腕部当前帧',xlabel='图像横坐标（像素）',ylabel='图像纵坐标（像素）');ax.legend(loc='lower center',fontsize=8);fig.tight_layout();fig.savefig(E/'figures/physical_semantics.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4.8));names=['A/B旧边界','A/B补充','A/B长窗','C旧边界','C补充','C长窗'];xx=np.arange(6);total=[q['pink_held_blue_supported']for q in cov];close=[q['pink_blue_close']for q in cov];supervised=[q['pink_blue_close_blue_supervised']for q in cov]
 axs[0].bar(xx-.24,total,.24,label='粉held且蓝受支撑',color='#9eb9cc');axs[0].bar(xx,close,.24,label='其中腕部两框≤5px',color='#ddad54');axs[0].bar(xx+.24,supervised,.24,label='接近且蓝held纳入监督',color='#358b77');axs[0].set(xticks=xx,xticklabels=names,ylabel='实际输入窗口数',title='物理关系见过，但长静止近投影组合未覆盖');axs[0].tick_params(axis='x',rotation=25);axs[0].legend(fontsize=8)
 masked=[sum(q['all_loss_mask_counts'])for q in cov];full=[sum(q['physical_valid_counts'])for q in cov];ratios=np.array(masked)/full*100;axs[1].bar(xx,np.ones(6)*100,color='#e5eaee',label='全部有效held/in/open事实');axs[1].bar(xx,ratios,color='#477fa4',label='实际损失/原评分掩码');axs[1].set(xticks=xx,xticklabels=names,ylabel='计入掩码的有效事实比例（%）',ylim=(0,128),title='补充与长窗只计目标held/in及open');axs[1].tick_params(axis='x',rotation=25);axs[1].legend(fontsize=8,loc='upper right')
 for i,(m,f)in enumerate(zip(masked,full)):axs[1].text(i,ratios[i]-5,f'{m}/{f}',ha='center',va='top',fontsize=8,color='white')
 for ax in axs:ax.grid(axis='y',alpha=.15)
 fig.tight_layout();fig.savefig(E/'figures/coverage_masks.png',dpi=170,bbox_inches='tight');plt.close(fig)
 rows=[]
 for name,q in zip(names,cov):rows.append([name,f"{q['windows']}窗 / {q['roots']}根",q['span_seconds'],q['held_over_open_drawer_windows'],f"{q['pink_held_blue_supported']} / {q['pink_held_blue_supported_roots']}根",f"{q['pink_blue_close']} / {q['pink_blue_close_roots']}根",q['pink_blue_close_blue_supervised'],'0'if q['source']=='long'else'不是2.1秒输入'])
 trainraw,devraw=raw['summary'];seed1=[q for q in j['C_saved_prediction_scores']if q['seed']==1];extra=raw['existing_C_held_down6'];extraover=sum(q['geometry']['center_in_opening_XY']for q in extra)
 nextrows=[
 ['先修可追溯输入合同','新增版本化清单，直接记录真正[t−63,t−42,t−21,t]路径、缓存、标签时刻和掩码。当前54条长窗元数据仍指短窗；本次已另存正确清单。','仅清单/断言修复；实际旧训练张量逐项完全匹配，不应据此重训或改旧结果。'],
 ['明确关系与监督','保留primary_location==drawer作为现接口in的含义；它不是一般几何包含，也不是完整动作成功。未来使用每物体共享的8类位置/归属头；同一头用于当前后验与未来先验。全部实体有效标签都计分，UNKNOWN与无效GT分开。','本轮不改标签、头、损失或解码；独立sigmoid读出与主WM原有类别头不等价，不能直接搬种子1权重冒充共享读出。'],
 ['先补足可用数据，再一次固定实验','现有720条A/B、240条C宏转移有短边界观测；长窗仅36/18个抓取状态，不能当成完整长历史宏转移。未来从既有布局重放/补采缺失公开历史时记录为复用根，不伪装成新独立数据。优先覆盖持物移至开口、低位仍抓持、真实静止且旁物投影接近；按原指南配额推进。','D失败仅界定缺失条件，不回灌D图像/姿态/布局；训练限A/B，选型与校准限C。现有日志私有位姿或有损视频不能补造缺失的掩码/特征。'],
 ['输入与训练目标','冻结DINO、双相机可见ROI及本体感觉，统一因果历史时间跨度；主WM继续对象Transformer+RSSM。只输入当前公开观测与声明动作，未来真值只作监督。优化当前与一步共享符号读出、原方案重建/KL/结果损失，预算以指南为准。','若统一2.1秒历史，必须为宏转移前后都获得真实公开窗口；不能让973个短窗和36个长窗被含糊称为统一长历史。'],
 ['一份新独立D接口测试（建议）','实现与C选型全部冻结后，仅再建一份全新D根集：12根（三色×初始开/关×两布局），每根预定3分支×4宏动作=144转移，并按预定动作取静止、下移等状态窗。先对所有已用布局/根排重，冻结种子与序列，一次运行，保留失败与缺失。','这是下一步离线误差分离测试建议，不是已经采集，也不替代原指南约6000总转移、1/3/5步、4–8技能闭环或独立评估的验收。当前D验收已被诊断，不得再次称盲测。'],
 ['分清读出和转移','同一条新转移分别评：①g(实际当前后验)对当前GT；②从实际当前后验出发的一步先验读出对下一GT；③g(实际下一后验)对下一GT；④先验与实际下一后验的符号差异。后续再比较自由多步与每步真实观测重启。','①或③错误先归当前观测/读出；①③正确而②错指向转移/潜状态分布问题，不能武断归纯动力学。只把GT符号替换给下游的诊断不修复潜状态。TS后继必须来自学习转移，不写人工效果。'],
 ['有限修正与停止条件','先报告首个影响决策的错误，只允许一次有单项替换证据支持的修正；重新评估必须按预先冻结流程，不在同一D根上循环调参。','这里提出可审查方案，不启动训练、世界模型、规划或采集。']]
 parts=['<h1>状态谓词：语义核对、覆盖审计与下一步方案</h1><p class="muted">2026-10-14 · 本机CPU只读分析 · 既有独立验收完整保留 · 无新推理、训练、阈值调整或采集</p>',
 '<p class="lead">两处问题均首先发生在<strong>当前状态读出</strong>，还不能归因于世界模型转移。蓝块一例在几何上仍完全高于抽屉内腔；粉块一例是桌面蓝块被误报held。实际输入覆盖存在缺口，但<strong>没有证据证明单一原因</strong>：种子1在已保存的355个C开发窗口、全部2485个有效事实上仍无原始误判或UNKNOWN。</p>',
 table(['核对项','证据','判断'],[
 ['in究竟表示什么','指南第275–279行要求每物体单一位置/归属。现目标由8类中的drawer类别导出；真实held覆盖为held类别。drawer通常由直接接触或向上支撑路径给出。','这是互斥的主要归属，不是通用几何in，也不是整条place动作成功。几何在容器内与仍抓持在一般情况下可以同时为真。'],
 ['蓝块t598','XY投影及整个XY包围框在开口内；中心z=428.04mm、最低点407.61mm；视觉边沿400mm，整个碰撞网格最高401.2mm。双侧夹持、外部支撑0。','该例不属于已进入内腔仍未释放；旧标签与完整3D几何一致为in假。p(in)=0.999928、p(held)=0.000246，远离0.5分界。可定位为提前判抽屉归属，不能证明内部只看XY投影。'],
 ['粉块t471','蓝块无机器人接触，桌面支撑3点；粉块被持有。两者中心距离170.75mm、高差130.92mm，腕部框仅差1px。p(held蓝)=0.999708，p(held粉)=0.999951。','原始错误是非目标蓝块假阳性，粉块原始判断正确；原解码器使两者UNKNOWN。不是物理双持、实体映射交换或解码器创造的误报。'],
 ['直接drawer标签的实现边界','直接接触标签分支未统一使用向上受力过滤；category7补全分支才使用该过滤。本次1364个输入状态中的drawer正标签全部有向上支撑路径。','这是接口说明和后续标签审计需保留的边界，当前两个失败不能据此判为标签错误。']]),
 img('physical_semantics.png','左图与中图是保存姿态和官方URDF/网格计算出的蓝块包围面：蓝色代表物体，绿色为抽屉开口/内腔；轴为世界毫米。中图绿色实线是视觉边沿，棕色虚线是整个碰撞网格最高点，物体最低点仍在其上。右图为粉块失败时84×84腕部原图，青框为蓝块、粉框为粉块可见像素范围；二维接近不等于三维接触。两个几何图采用独立轴比例，读数以坐标为准。'),
 '<h2>模型真正见过什么</h2><p>下表统计实际训练/开发输入，不把完整动作日志里的状态算成已经训练过。旧边界和补充窗口为连续4帧、0.1秒；长窗为间隔21控制步的4帧、2.1秒。近投影是本次统计前固定的描述分箱：腕部蓝/粉各≥4像素、包围框距离≤5像素；不是新增决策阈值。各来源可能复用根，表中根数不能相加当独立样本。</p>',
 table(['来源','实际输入','跨度秒','持物在开抽屉上方','粉held且蓝受支撑','其中近投影','近投影中蓝有监督','2.1秒真实静止'],rows),
 img('coverage_masks.png','左图纵轴为实际输入窗口数，灰蓝柱是粉held且蓝受支撑，橙柱是其中腕部投影接近，绿柱是接近窗口中蓝held也纳入原监督/评分的数量。右图纵轴为有效事实被掩码计入的比例，浅灰为全部有效held/in/open标签（100%），深蓝为实际损失或原开发评分比例；柱内标注计入事实数/全部有效事实数。两个图的横轴均对应A/B训练与C开发的三类来源；窗口数不代表独立根数。'),
 f'<p><strong>关系状态并非完全没见过。</strong>A/B实际输入有100个粉held且蓝受支撑窗口；其中24个近投影窗口来自15个旧边界根，且蓝held假标签已监督。C有34个对应窗口，近投影11窗/6根。长窗54个中没有一窗满足全程2.1秒真实持物静止；其中粉held窗口的蓝held标签也未计入原掩码。因此缺的是时间与关系的组合覆盖，不能简单说“没有非目标负例”。</p>',
 f'<p><strong>日志存在不等于可直接训练。</strong>另读现有296段A/B/C放置日志、{trainraw["frames_scanned"]+devraw["frames_scanned"]}个记录控制时刻：A/B在{trainraw["held_over_drawer_roots"]}根、{trainraw["held_over_drawer_trajectories"]}段中有{trainraw["held_over_drawer_frames"]}个持物投影位于开口内的时刻；C有{devraw["held_over_drawer_frames"]}个。两域都没有对应完整[-63,-42,-21,0]四个原始公共NPZ窗口。此前单独恢复的6个C持物下移窗口中{extraover}个位于开口投影内，但它们仅为开发诊断，不属于种子1训练；它们也不是本次严格低位连续下降验收的替代品。详见raw_place_coverage.json。此处没有穷举所有A/B动作的长静止日志。</p>',
 '<p>既有A/B日志还出现了仍抓持而物体中心低于抽屉视觉边沿的状态（蓝块中心最低约395.53mm，边沿约400mm）。因此“held ⇒ 几何in为假”不能作为一般逻辑规则；中心进入内腔也不等于完整物体被包含。本次D蓝块的结论来自其自身完整姿态与网格核对。</p>',
 '<h2>错误是否在开发中已经出现</h2>',
 table(['对象','已保存预测重算结果','解释'],[
 ['冻结种子1','355窗：原评分2265个事实；加回被掩掉的220个有效非目标事实后共2485个，原始错误0、UNKNOWN 0。','没有在已有C预测中找出同一蓝块误报；不能把掩码缺口宣称为已证明因果。训练集没有保存预测，本轮不新推理，所以训练误判率未知。'],
 ['历史种子0/2，仅旁证','C_core1200_14_b0边界1：真实持粉时，非目标红held被误报；p=.594549和.999783，均触发双held UNKNOWN，且原掩码本已计入。','同类非目标误报曾在C的其他种子出现，但不是冻结种子1、不是同一蓝物体、也不是2.1秒静止条件。'],
 ['实体、张量与泄漏检查','1364窗/4092次实体注册与私有ID比对无错；由原缓存重建objects/context/robot与实际数组完全相同；ROI框最大数值误差2.56e−8。72个A/B根与26个C根不重叠。','未发现错实体/错拼接，不能归咎于已检查的这些实现层。公开输入仅图像、可见掩码、深度、本体感觉及相机标定；私有数据仅用于本次评分核对。'],
 ['长窗元数据','54/54条原metadata仍指短窗与短窗缓存；训练代码实际读temporal.npz中的long_history数组，正确路径在cache_audit.json。','确有可追溯性缺陷，但实际训练没有用错短窗。本次新window_inventory.json写出真实路径，未覆盖旧元数据。']]),
 '<h2>有限的下一步方案（未执行）</h2>',table(['工作','输入/目标与方法','边界'],nextrows),
 '<p class="muted">本轮交付：report.html；figures/两张说明图；results/audit.json完整数值、C错误列表与两例几何；results/window_inventory.json逐窗输入/掩码清单；results/raw_place_coverage.json既有日志可复用性；source/审计脚本；configs/冻结范围；results/validation.json与delivery_manifest.json。原验收report.html和所有数据/模型均未改写。本结论不代表原指南全流程完成。</p>']
 css='body{font:16px/1.7 "Noto Sans CJK SC",sans-serif;background:#f1f5f7;color:#203546;margin:0}main{max-width:1250px;margin:auto;background:white;padding:34px}h1{font-size:29px}h2{font-size:23px;margin-top:32px}.muted{font-size:13px;color:#617789}.lead{padding:18px;background:#eaf3f8;border-left:4px solid #477fa4}table{width:100%;border-collapse:collapse;font-size:13px;margin:22px 0}th,td{border:1px solid #d8e2e9;padding:9px;vertical-align:top;text-align:left}th{background:#edf3f7}tr:nth-child(even){background:#fafcfd}img{width:100%;height:auto}figure{margin:26px 0}figcaption{font-size:13px;color:#617789}'
 (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>状态谓词语义与覆盖诊断</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
 (E/'NEXT_STEP_PROPOSAL.json').write_text(json.dumps({'status':'proposal_only_not_executed','steps':[dict(zip(['work','proposal','boundary'],r))for r in nextrows]},ensure_ascii=False,indent=2)+'\n');print('REPORT_COMPLETE',E/'report.html',flush=True)
if __name__=='__main__':main()
