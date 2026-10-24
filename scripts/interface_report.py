from pathlib import Path
import json,html,base64,datetime
from collections import defaultdict,Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wm_symbolic.symbol_decoder import Calibration
from wm_symbolic.proposition_interface import decode
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/proposition_interface_20261014';F=R/'figures'
def read(p):return json.loads(p.read_text())
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def pc(n,d):return f'{n}/{d}（{100*n/d:.1f}%）' if d else '—（分母为0）'
def main():
 out=read(R/'evaluation/RESULT.json');off=read(R/'evaluation/ABC_paired_metrics.json');cal=read(R/'evaluation/threshold_freeze.json');assert out['complete']
 font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f5f8fc','savefig.facecolor':'#f5f8fc'})
 # Whole-root resampling describes uncertainty; never reselects the threshold.
 records=torch.load(R/'evaluation/ABC_probabilities.pt',map_location='cpu',weights_only=False);cc=Calibration(**cal['calibration']);counts=defaultdict(lambda:np.zeros(2,dtype=int))
 for row in records:
  if row['cohort']!='C_calibration':continue
  p={k:v.mean(0) for k,v in row['probs'].items()};names=[c+'_'+n for c in ['red','blue','pink'] for n in ['drawer','held']]+['drawer_open','drawer_closed'];facts=decode(p,row['events'].mean(0),names,cc)[0]
  for c in ['red','blue','pink']:
   for loc in ['drawer','held']:
    for drawer in ['open','closed']:
     ns=[c+'_'+loc,'drawer_'+drawer]
     if all(facts[n] is True and row['truths'][n] is not None for n in ns):counts[row['root']]+=[int(not all(row['truths'][n] for n in ns)),1]
 arr=np.stack(list(counts.values()));rng=np.random.default_rng(824);boots=[]
 for _ in range(2000):
  t=arr[rng.integers(0,len(arr),len(arr))].sum(0)
  if t[1]:boots.append(t[0]/t[1])
 ci=np.quantile(boots,[.025,.975]).tolist();(R/'evaluation/calibration_root_uncertainty.json').write_text(json.dumps({'per_root_false_claims':{k:v.tolist() for k,v in counts.items()},'root_bootstrap95':ci,'resamples':2000,'threshold_selection_unchanged':True,'scope':'Terminal fact pairs only; current and prior records clustered together by eight physical roots. Not full LTL certification.'},indent=2))
 fig,ax=plt.subplots(1,2,figsize=(13,4.6));rows=[next(x for x in off['rows'] if x['cohort']=='C_development' and x['kind']==kind and x['mode']==mode) for kind in ['current','prior'] for mode in ['legacy','proposition']];x=np.arange(4);bottom=np.zeros(4)
 for key,label,color in [('correct','确定且正确','#36816c'),('wrong','确定但错误','#bf5654'),('unknown','UNKNOWN','#a7b2c2')]:
  vals=np.array([100*r.get(key,0)/r['valid'] for r in rows]);ax[0].bar(x,vals,bottom=bottom,color=color,label=label)
  for i,v in enumerate(vals):
   if v>3:ax[0].text(i,bottom[i]+v/2,f'{v:.1f}',ha='center',va='center',color='white',fontsize=10)
  bottom+=vals
 ax[0].set_xticks(x,['当前\n旧门控','当前\n新命题','一步未来\n旧门控','一步未来\n新命题']);ax[0].set_ylim(0,110);ax[0].set_ylabel('16类命题判断占有效记录的百分比');ax[0].set_title('12个C开发根：读出覆盖增加');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,1.00),ncol=3,fontsize=9)
 curve=[x for x in cal['threshold_curve'] if x['event_tau']==.9];xx=[x['tau'] for x in curve];false=[100*x['terminal_claims']['false']/x['terminal_claims']['claims'] for x in curve];coverage=[100*x['terminal_claims']['true_claims']/x['terminal_claims']['true_combinations'] for x in curve]
 ax[1].plot(xx,false,'o-',label='终态组合误报率',color='#bf5654');ax[1].plot(xx,coverage,'s-',label='真实终态组合确认覆盖率',color='#2876ad');ax[1].axhline(5,ls='--',color='#777',label='经验误报目标5%');ax[1].axvline(.95,ls=':',color='#36816c',label='C选定阈值0.95');ax[1].set_xticks(xx);ax[1].set_ylim(0,80);ax[1].set_xlabel('符号阈值（两接口相同）');ax[1].set_ylabel('百分比（各曲线分母不同，见说明）');ax[1].set_title('8个独立C校准根：不能只放宽拒绝');ax[1].legend(fontsize=9);fig.tight_layout();fig.savefig(F/'readout_and_calibration.png',dpi=155);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.6));mode_names={'legacy':'旧门控','proposition':'新命题读取','reference':'同执行器参考'}
 for ax,co,title in zip(axes,['ABC_development','fresh_D'],['A/B/C开发：6个任务','新D初态：6根、8个任务']):
  rr=[next(x for x in out['counts'] if x['cohort']==co and x['method']==m) for m in mode_names];bottom=np.zeros(3)
  for key,label,color in [('success','真实目标与过程完成','#36816c'),('no_plan_within_budget','候选预测仍无法支持计划','#d49a36'),('unknown_current_state','当前必要状态未知','#a7b2c2'),('other','物理过程失败','#bf5654')]:
   vals=np.array([r['success'] if key=='success' else r['tasks']-r['success']-r['stops'].get('unknown_current_state',0)-r['stops'].get('no_plan_within_budget',0) if key=='other' else r['stops'].get(key,0) for r in rr]);ax.bar(range(3),vals,bottom=bottom,color=color,label=label)
   for i,v in enumerate(vals):
    if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white',fontsize=13)
   bottom+=vals
  ax.set_xticks(range(3),list(mode_names.values()));ax.set_ylim(0,9);ax.set_yticks(range(0,10,2));ax.set_ylabel('任务数');ax.set_title(title)
 axes[1].legend(loc='upper center',bbox_to_anchor=(-.10,-.15),ncol=2,fontsize=10);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(F/'closed_loop.png',dpi=155);plt.close(fig)
 summary=[]
 for x in out['counts']:
  summary.append(['开发' if x['cohort']=='ABC_development' else '新D',mode_names[x['method']],pc(x['success'],x['tasks']),x['requests'],f'{x["requests"]/x["tasks"]:.2f}',x['queries'],pc(x['false_claims'],x['claims']) if x['method']!='reference' else '不使用模型确认'])
 fields=[]
 for name,title in [('red_table','红块在桌面'),('red_drawer','红块在抽屉'),('red_held','红块被拿着'),('blue_table','蓝块在桌面'),('blue_drawer','蓝块在抽屉'),('blue_held','蓝块被拿着'),('pink_table','粉块在桌面'),('pink_drawer','粉块在抽屉'),('pink_held','粉块被拿着'),('hand_empty','手空'),('drawer_closed','抽屉关闭'),('drawer_open','抽屉打开'),('slider_right','滑门在右')]:
  rr=[next(v for v in off['per_proposition'] if v['cohort']=='C_development' and v['kind']=='current' and v['mode']==mode and v['name']==name) for mode in ['legacy','proposition']];fields.append([title,*[f'未知 {v.get("unknown",0)}/{v["valid"]}；错误 {v.get("wrong",0)}/{v["valid"]-v.get("unknown",0)}个确定判断' for v in rr]])
 paired=[]
 for co,title in [('ABC_development','A/B/C根观测'),('fresh_D','新D根观测')]:
  rr=[x for x in out['paired_initial_states'] if x['cohort']==co];n=sum(len(x['scope']) for x in rr);old=sum(v is None for x in rr for v in x['legacy'].values());new=sum(v is None for x in rr for v in x['proposition'].values());paired.append([title,pc(old,n),pc(new,n),sum(x['rescued_correct'] for x in rr),sum(x['rescued_wrong'] for x in rr)])
 taskrows=[]
 for task in read(R/'tasks.json'):
  if task['cohort']!='fresh_D':continue
  rr=[next(x for x in out['results'] if x['task_id']==task['task_id'] and x['method']==m) for m in mode_names];text=task['text']
  for en,zh in [('red','红'),('blue','蓝'),('pink','粉')]:text=text.replace(en,zh)
  taskrows.append([task['root_id'],html.escape(text),*[('完成' if x['physical_task_success'] else '状态未知' if x['status']=='unknown_current_state' else '未找到计划' if x['status']=='no_plan_within_budget' else '过程失败')+f'；{x["physical_skills"]}次技能' for x in rr]])
 videos=[]
 for f in sorted((R/'media').glob('*.mp4')):
  info=read(f.with_suffix('.json'));videos.append([f'<a href="media/{f.name}">{f.stem}</a>',f'{info["duration_seconds"]:.3f}秒；{info["frames"]}帧',f'{f.stat().st_size/1e6:.2f} MB'])
 causal=read(R/'evaluation/causal_audit.json');norows=[]
 titles={'D_interface_00_red_closed':'根00：红块入抽屉并关','D_interface_00_blue_open':'根00：蓝块入抽屉保持开','D_interface_01_pink_held_closed':'根01：持粉块且抽屉关','D_interface_03_red_close_then_blue':'根03：红块收纳后持蓝块','D_interface_04_pink_open':'根04：粉块入抽屉保持开'}
 for case in causal['no_plan_cases']:
  aa=case['attempts'];last=aa[-1];norows.append([titles[case['task']],'/'.join(str(a['queries']) for a in aa),'/'.join(str(a['max_explored_depth']) for a in aa),sum(a['rejections'].get('unknown',0) for a in aa),sum(a['rejections'].get('candidate_input_unknown',0) for a in aa),sum(a['rejections'].get('process_constraint_violation',0) for a in aa),'第8层仍有候选，无终点接受' if 'horizon_limit' in last['termination_inferred_from_logged_edges_and_frozen_search'] else '第2层无可保留候选'])
 no_plan_section='<h2>五个no-plan具体停在哪里</h2>'+table(['新D任务','两次查询数','两次最大深度','未来命题未知拒绝','前置观察未知拒绝','过程违规拒绝','实际搜索终止'],norows)+'<p>每任务两次搜索之间只做一次4控制步重新观察，表中拒绝数合计两次、按候选边计数，不能当作独立物理失败次数。三个任务在第2层因没有可保留候选而结束；两个任务搜索到8步时域上限，仍有未达到接受条件的候选。因此，后两项明确带有搜索时域限制，不能只归因于模型错误。五项均未触及5000次查询上限；多持物互斥冲突拒绝均为0；因“前提明确为假”而被规则拒绝的次数为0。前置观察未知表示无法确定相关输入事实，不表示已经证明动作前提不满足。所有动作的后继仍实际查询RSSM。</p><p>搜索未找到计划不证明物理任务不可行：其中四项同根参考任务完整成功；另一项参考虽达到终点，却违反禁动约束。预测不确定、经验阈值和有限时域共同限定了当前搜索能接受什么；本轮没有继续调整其中任一项。</p>'
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>命题读取接口实验</title><style>body{{font:16px/1.7 system-ui,sans-serif;color:#1d334b;background:#f5f8fc;max-width:1220px;margin:35px auto;padding:0 22px}}h1{{font-size:29px}}h2{{margin-top:32px}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #d9e2eb;vertical-align:top;text-align:left}}th{{background:#e6edf5}}img{{width:100%;margin:12px 0}}.note{{background:#e5eef6;padding:15px}}small{{color:#697a8c}}a{{color:#1d6390}}</style><h1>命题读取减少了不必要的 UNKNOWN，但尚未解决新初态规划</h1><p>冻结同一个第1000步模型、DINO特征、1200条原数据和六技能v1控制端，只比较两种符号读取接口。A/B/C开发由0/6变为1/6；六个新D根上的八个任务仍为0/8。参考序列在新D完成7/8。已停止新增实验，没有依据新D结果继续降低阈值或修改模型。</p>
 <h2>这次究竟改了什么</h2>{table(['元素','本次处理','原因与边界'],[['八类位置输出','保留神经网络原输出；桌面概率=p左+p中+p右，抽屉=p抽屉，持物=pheld','任务若明确要求左/中/右，仍按该精确类别读取。'],['UNKNOWN','新接口按所需命题的概率判断真/假/未知；旧接口额外要求全类别最大值过阈值','两者共用同一任务范围与阈值，当前后验和未来后继使用同一解码函数。'],['必要事实','目标、LTL约束、全部持物身份和空手；每个候选动作再检查其相关前置观察变量','不只筛最终目标；多个held冲突、未见过对象及必要不确定性均保留。'],['动作前提','这里只检查相关事实是否可判断；前提为假仍由RSSM预测执行后果','没有人工Pick/Place效果，也没有用前提真值直接屏蔽动作或替代学习转移。'],['公平比较','双方共用更完整的事实范围和新的0.95/0.9阈值','旧门控组指复现原最大类别门控的配对基线；它比首轮仅目标/DFA的范围更严格，不能直接拿本次0/8与旧实验1/12作效果比较。']])}
 <p class="note">原指南第279行明确写了最大类别概率不足时相关事实返回UNKNOWN，旧门控有文档依据。本次是用户授权的设计假设修订。原指南第420行要求独立校准、优先控制完成误报不高于5%并提高覆盖；“至少10根、30次声明”是后加工程规则，之前将其称为原文要求不准确。</p>
 <h2>A/B/C离线配对：读出改善与误报同时检查</h2><img src="figures/readout_and_calibration.png"><p>左图来自12个C开发根、每根3个四步分支。当前共有180个边界×16命题=2880条有效判断；一步未来为144个端点×16=2304条。绿色表示确定且正确、红色表示确定但错误、灰色表示未知。相同八条潜变量样本同时送入两解码器，不是两次不同采样。新接口把当前UNKNOWN从1631降至304，增加1321个正确判断及6个错误判断；未来UNKNOWN从1262降至261，增加997个正确判断及4个错误判断。不能把所有新增确定值都称为修复。</p>
 <p>右图仅用8个独立C校准根。红线分母是发出的终态组合声明数，蓝线分母是174个真实满足的终态组合记录；当前/未来与同根分支有关联。阈值0.8时误报18/138，0.95时4/95（4.2%），故预定网格选0.95，事件阈值保持0.9，温度参数保持不变。所选阈值的整根重采样95%区间约为{100*ci[0]:.1f}%–{100*ci[1]:.1f}%；这不是全LTL流程的安全认证，继续标记为未认证。没有使用D选择阈值。</p>{table(['当前命题（C开发）','旧门控','新命题读取'],fields)}
 <h2>真实闭环：通过根状态并不等于找到可信计划</h2><img src="figures/closed_loop.png"><p>横轴是方法，纵轴是冻结任务数。绿色满足真实终点与全过程约束，黄色未找到满足八样本门槛的计划，灰色当前必要状态未知，红色物理过程违规。A/B/C与新D分别报告，零次确认的误报率未定义。</p>{table(['范围','接口/参照','完整物理成功','总技能请求','每任务平均请求','后继查询','模型误报/确认'],summary)}
 <p>两接口的14个配对任务初始h/z、随机状态、读出概率完全相同；同根物理复位误差均小于1e-5。根时刻的匹配事实比较如下，不把较长成功轨迹额外增加的边界与短停止轨迹混为同分母：</p>{table(['同一根时刻','旧UNKNOWN','新UNKNOWN','从未知变正确','从未知变错误'],paired)}
 {table(['新D根','任务（含过程要求）','旧门控','新命题读取','同执行器参考'],taskrows)}
 <h2>第一个阻塞点与物理失败证据</h2><p><b>接口确有作用：</b>C_core1200_01的收纳红块任务，旧接口在根时刻停止，新接口实际执行抓红块→放入→关闭三步，1085个真实控制步，任务及过程均满足；控制器和初始潜状态相同。</p>
 <p><b>新D仍被必要不确定性阻塞：</b>八项中五项从根UNKNOWN转为进入搜索，但没有发出技能；另三项仍无法确定当前持物身份。以D_interface_00收纳红块为例，“开抽屉”候选可通过，但后续“抓红块”的第6条预测样本只给出P(red_held)=0.9277，空手/滑门相关事实仍未知，低于冻结0.95阈值，整条候选被拒绝。其余七条样本或最大类别正确不能覆盖这条不确定样本。这证明未来预测及其确认尺度仍不足以支持现有保守搜索，不能把拒绝直接说成已证实的动力学错误；没有在测试上放宽门槛。</p>
 <p><b>控制端问题单独保留：</b>新D参考序列的失败发生于D_interface_03：第一次开抽屉把禁止移动的粉块推动34.19毫米，超过已有20毫米事件标准，因此即使终点满足也判过程失败。该任务后来还有一次控制器报告成功、独立放置端态未即时满足的差异，记录未改写。另一条参考请求引起滑门14.82毫米连续位移，单独报告，不能写作无副作用。其余参考任务的真实目标及LTL评分保留原定义；轻微接触本身不自动判失败。</p>
 <h2>新测试与旧D的关系</h2><p>六个新根采用预先固定的新随机种子617000–617005，任务与初态均在A/B/C验证前保存。与旧D实际根以及彼此按既定位置/朝向规则检查，无近重复；旧v1控制验收仅覆盖A/B/C，因此同环境可比较历史D根为旧核心实验20根。两任务共享初态的情况不增加物理独立性。场景仍是已经见过诊断结果的CALVIN D、相同布局生成范围和执行器，因此这里检验修订假设在新初态上的表现，不能称为全新环境泛化或未被既往D经验影响的原始盲测。</p>
 <h2>完整真实视频</h2>{table(['视频','原时长与帧数','文件大小'],videos)}<p>静态相机在左、夹爪相机在右，中文标题说明方法与结果。成功视频来自C；过程失败视频明确标为D参考控制序列，不冒充模型已执行的计划。D模型停止视频只有一次4控制步的重新观察，共5帧、0.167秒，真实没有技能动作；没有复制末帧延长视频。所有视频使用H.264，保留原始30Hz时间与全部帧。</p>
 <h2>结论与文件</h2><p>命题级读取是有证据支持的接口改进，但本次新D闭环仍未通过。下一步需要另行决定是否研究未来分布及确认尺度；本轮停在这个结论，不重训、不换控制器、不追加测试调参。旧源码、配置、结果保持不变，新文件集中在 runs/proposition_interface_20261014。</p><p><a href="protocol.json">预注册协议</a> · <a href="evaluation/threshold_freeze.json">C阈值曲线</a> · <a href="evaluation/new_test_freeze.json">新D前冻结清单</a> · <a href="evaluation/ABC_paired_metrics.json">逐命题离线统计</a> · <a href="evaluation/RESULT.json">配对结果与逐步失败证据</a></p><small>生成：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。总计18组A/B/C和24组新D运行；全部由固定v1执行器在本机仿真真实执行或停止，无新增训练。</small></html>'''
 (R/'report.html').write_text(body)
 contribution='<h2>可分离的贡献与未分离的因素</h2><p><b>配对两组内可以识别：</b>双方都暴露相同的桌面求和概率、相同完整事实清单、相同当前/未来持物检查和0.95/0.9阈值，唯一不同是旧组额外施加最大细分类别门控。因此本轮直接识别的是，在统一接口下移除该门控的贡献；不能独立声称“新增概率求和”本身带来多少闭环提升。</p><p><b>相对旧core1200不能分离：</b>本轮两组共同增加了所有持物/空手与候选前置观察检查，增加滑门保持约束，并把符号阈值从0.8改为独立C选择的0.95。这些共同变化没有另做消融，因此不能用旧实验1/12与本轮0/8之差归因于任一因素。</p><p>代表性C成功根中，红块最大细分类别概率只有0.578，但桌面三类之和为0.975、抽屉概率0.00765、持物概率0.00492。旧门控把相关事实都封为未知，新接口可判断“在桌面、尚未入抽屉、未持有”。这是有数值和同初态实际三步成功支持的局部修正；新D仍无闭环成功。</p>'
 body=body.replace('<h2>第一个阻塞点与物理失败证据</h2>',contribution+no_plan_section+'<h2>第一个阻塞点与物理失败证据</h2>')
 body=body.replace('未来预测及其确认尺度仍不足以支持现有保守搜索','未来预测的置信度、确认尺度与有限时域仍不足以支持现有保守搜索')
 body=body.replace('<a href="evaluation/RESULT.json">配对结果与逐步失败证据</a>','<a href="evaluation/RESULT.json">配对结果与逐步失败证据</a> · <a href="evaluation/causal_audit.json">五项no-plan与贡献可识别性</a>')
 (R/'report.html').write_text(body)
 standalone=body
 for f in F.glob('*.png'):standalone=standalone.replace('src="figures/'+f.name+'"','src="data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()+'"')
 import re
 standalone=re.sub(r'<a href="([^"]+)">(.*?)</a>',lambda m:m[2]+' <small>（本机目录：'+m[1]+'；视频另附Library）</small>',standalone)
 (R/'report_standalone.html').write_text(standalone)
 print(json.dumps({'report':str(R/'report.html'),'calibration_root_bootstrap95':ci,'figures':2},ensure_ascii=False))
if __name__=='__main__':main()
