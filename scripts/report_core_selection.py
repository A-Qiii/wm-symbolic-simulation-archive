from pathlib import Path
import json,datetime,re,hashlib
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'runs/core_executor_selection_20261014';R=P/'reports'
def load(p):return json.loads(p.read_text())
def link(p,label):return '<a href="../'+str(p.relative_to(P))+'">'+label+'</a>'
def table(h,rs):return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rs)+'</table>'
s=load(O/'selection.json');p=load(O/'scoring_protocol.json');a=load(O/'preservation_audit.json')
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>六技能执行器与评分协议固定</title><style>body{max-width:1080px;margin:35px auto;padding:0 24px;font:16px/1.7 system-ui,sans-serif;color:#263748}h1{font-size:30px}h2{margin-top:32px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;border:1px solid #ccd6df;text-align:left}th{background:#edf2f6}.box{background:#eaf5f0;border-left:5px solid #298775;padding:16px}code{overflow-wrap:anywhere}a{color:#17649e}.note{color:#53657a}</style><body><h1>六技能执行器已选定，研究重心转回世界模型</h1>'''
body+='<div class="box">选定冻结的 <b>physical_oracle_v1_candidate3</b>。采用用户明确认可的任务级处理保留这次轻微滑门副作用；停止继续优化控制器，也不扩大技能范围。正式采集与训练仍暂停。</div>'
body+='<p>记录日期：'+p['recorded_utc']+'。用户明确指示：“误碰的程度很小可以忽略不计，认为是成功”。这是看到验收结果后的评分修订，不能称为预先定义标准下的18/18。</p>'
body+=table(['同一批v1原18初态','原严格评分（保留）','用户认可后的事后任务级评分'],[['四步收纳链','17/18','18/18'],['可执行技能请求','88/90','90/90'],['缺少前提时正确拒绝','90/90','90/90']])
body+='<p class="note">没有重新挑选成功样本，也没有重跑以替换失败。原来的报告、评分、位移、动作、接触和视频保持原样；新增评分记录通过原始文件哈希关联。</p>'
body+='<h2>两次严格失败是否来自同一副作用</h2>'+table(['回合','滑门最大位移','目标结果','非目标状态核查'],[[e['case'],f'{e["slider_max_change_mm"]:.3f} mm','只持有指定粉块，官方目标与稳定检查通过','滑门始终为right；红/蓝物体归属与区域不变；灯状态不变'] for e in p['exceptions']])
body+='<p>两次都发生在同一 C_acceptance_02 初态的粉块抓取：一次在收纳链中，一次是前提缺失测试的抓取准备。滑门仍在原有逻辑类别内，没有越过已冻结的5.5 cm右侧状态边界；非目标方块最大漂移分别只有0.017 mm和0.054 mm。因此，此次评分变化不掩盖抓错物体或状态迁移错误。</p>'
body+='<h2>评分许可的边界</h2>'+table(['规则','固定处理'],[['当前命名技能','必须实际达到官方目标及原有稳定终态。'],['轻微滑门位移','接受这两条已审查记录中的同类副作用；没有据此新增可忽略任意碰撞的通用数值门槛。'],['逻辑状态和对象归属','非请求逻辑状态、对象归属变化仍失败；这两例逐帧离线核查通过。'],['任务禁止条件','将来的明确任务约束仍然有效；本轮没有验证任意未来LTL禁止式，不能泛化豁免。'],['原始标签','实际连续位移、图像和接触保持真实；不把9.47 mm改写成0。'],['输入隔离','真值只用于控制内部、标签和评价；成功标记、拒绝原因及GT状态不作为WM或高层输入。']])
body+='<h2>为什么选v1，如何处理后续试验</h2>'+table(['版本','实际证据','当前决定'],[['v1冻结版','90个有效请求均完成目标；仅两例被原副作用阈值扣分，现由用户认可','选定为六技能核心验证固定控制端。'],['v2避障版','旧18根回归通过；新27链通过，但135个有效请求中1次候选搜索失败；3项后续负例检查未到达','全部结果保留，不强制采用较保守的控制器。'],['v3开发探针','增加前方空域绕行后，同一关抽屉红块抓取单例实际成功；证明v2失败是路径搜索不足，未证明物理不可达','已停止；未做45根回归或27根新验收，不作为完整验证版本。']])
body+='<p>当前支持开/关抽屉、红/蓝/粉桌面抓取、放入抽屉，共6技能。适用证据来自原生A/B/C、分散桌面物体及四步收纳链；D、拥挤堆叠、柜内抓取和剩余19技能未由此验证。所选代码未改动，SHA-256为 <code>'+s['controller_sha256']+'</code>。</p>'
body+='<h2>可转入世界模型的结论</h2><p>可以把该六技能控制端固定下来，进入缩小范围的世界模型数据方案与实验准备；不再把消除这次微小滑门位移作为前置门槛。这不等于原指南25技能、6000条方案已完成，也不表示已经启动新数据采集或训练。</p>'
body+='<p>1200条提案仍为100根×3分支×4技能（训练/开发/测试60/20/20根）。建议每个对照分支最多安排一次有意义的前提错误，避免“关抽屉时先持物”造成连续无动作拒绝；同根分支全部同split，另预先固定组合测试。若保留原A/B→C→D含义，需明确环境映射；若做独立置信度校准，开发根还需预分调参与校准。以上只是方案检查，没有执行授权。</p>'
body+='<h2>固定文件与原始证据</h2><p>'+' · '.join(link(f,t) for f,t in [(P/'configs/controllers/selected_core_executor.json','所选执行器配置'),(O/'selection.json','版本选择依据'),(O/'scoring_protocol.json','评分协议与两例证据'),(O/'amended_scores.json','逐条旧/新评分'),(O/'preservation_audit.json','原文件保留核查'),(R/'physical_oracle_v1.html','原v1严格报告'),(R/'physical_oracle_v2.html','保留的v2报告'),(P/'runs/physical_oracle_v2/wm_1200_proposal_review.json','数据方案检查')])+'</p>'
body+='<p>两段v1代表视频已保存并交付，无重复上传：成功链 <code>libfile_012a0cbbf8c081918501dc014fc578d9</code>；副作用链 <code>libfile_b66d5fd6c2ec8191a592e534b84625da</code>。二者保持连续双相机实际帧及原标注。</p>'
body+=table(['保留/暂停状态','核验结果'],[['旧正式数据',str(a['main_transitions'])+'条，清单SHA保持不变'],['原严格评分/报告/控制器','哈希保持不变'],['正式采集暂停标记',a['collection_hold']],['正式训练暂停标记',a['training_hold']]])+'</body></html>'
f=R/'core_executor_selection_20261014.html';f.write_text(body)
links=re.findall(r'href="([^"]+)"',body);assert all((R/x).resolve().exists() for x in links)
h=load(P/'runs/continuation_handoff.json');h.update(latest_report='reports/core_executor_selection_20261014.html',priority='User accepts the reviewed small incidental displacement; corev1 selected, stop controller optimization, return focus to WM proposal without starting collection/training.',controller_status={'status':'selected_for_six_skill_core_validation','version':s['selected_version'],'post_hoc_scoring':True,'original_strict_chains':[17,18],'user_amended_chains':[18,18]},executor_reliability_status='Observed scope only: v1 goal90/90, correct refusals90/90; task-level acceptance changed posthoc for2 documented slider side-effect calls. No universal guarantee.',next='Discuss/finalize WM core dataset design. Proposed1200 and original6000 collection/training remain unstarted/paused.',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'runs/continuation_handoff.json').write_text(json.dumps(h,indent=2))
print(json.dumps({'report':str(f),'verified_links':len(links),'selected':s['selected_version'],'scoring_post_hoc':True}))
