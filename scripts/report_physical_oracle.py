"""Human-readable, traceable core physical-controller acceptance report."""
from pathlib import Path
import json, hashlib, html, datetime, math
from collections import Counter,defaultdict
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

P=Path('/home/robot/research/projects/wm_symbolic');B=P/'runs/physical_oracle_v1';R=P/'reports'
FP=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=FP.get_name();plt.rcParams['axes.unicode_minus']=False
LABEL={'open_drawer':'打开抽屉','close_drawer':'关闭抽屉','lift_red_block_table':'抓取红块','lift_blue_block_table':'抓取蓝块','lift_pink_block_table':'抓取粉块','place_in_drawer':'放入抽屉'}
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def link(p,label):return f'<a href="{html.escape(str(p.relative_to(P)) if False else "../"+str(p.relative_to(P)))}">{html.escape(label)}</a>'
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
def wilson(k,n):
 if not n:return [0,1]
 z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-w,c+w]

def main():
    assert (B/'independent_acceptance/COMPLETE.json').exists()
    roots=load(B/'roots_manifest.json');freeze=load(B/'controller_freeze.json')
    datasets={split:[dict(load(f),result_path=str(f)) for f in sorted((B/folder).glob('*/*/*/result.json'))] for split,folder in [('开发','frozen_development'),('独立验收','independent_acceptance')]}
    episodes={split:load(B/folder/'episode_summaries.json') for split,folder in [('开发','frozen_development'),('独立验收','independent_acceptance')]}
    checks={'main_manifest_unchanged':sha(P/'data/data_manifest.json')==freeze['main_manifest_sha256'],
            'main_transitions':load(P/'data/data_manifest.json')['real_transitions'],
            'controller_unchanged':sha(P/'src/wm_symbolic/physical_oracle.py')==freeze['controller_sha256'],
            'evaluator_unchanged':sha(P/'scripts/evaluate_physical_oracle.py')==freeze['evaluation_script_sha256'],
            'roots_unchanged':sha(B/'roots_manifest.json')==freeze['roots_manifest_sha256'],
            'main_collection_hold_exists':(P/'runs/PAUSED_BY_USER.json').exists(),
            'formal_training_hold_exists':(P/'runs/FORMAL_TRAINING_HOLD.json').exists()}
    totals={}
    for split,rows in datasets.items():
        positive=[r for r in rows if r['expected']=='success'];negative=[r for r in rows if r['expected']!='success'];chains=[r for r in episodes[split] if r['case']=='chain'];n=len(chains);k=sum(r['complete'] for r in chains)
        totals[split]={'chain_complete':k,'chains':n,'chain_wilson95':wilson(k,n),'feasible_success':sum(r['conservative_pass'] for r in positive),'feasible_requests':len(positive),'correct_refusals':sum(r['correct_refusal'] for r in negative),'negative_requests':len(negative),'retries':sum(r['retries'] for r in rows),'retry_skills':sum(r['retries']>0 for r in rows),'total_controls':sum(r['controls'] for r in rows),'max_controls':max(r['controls'] for r in rows),'max_unrequested_drift_mm':max(r['max_unrequested_object_displacement_m'] for r in rows)*1000,'max_robot_penetration_mm':max(r['max_robot_contact_penetration_m'] for r in rows)*1000,'max_robot_force_N':max(r['max_robot_contact_force_N'] for r in rows),'unrequested_contact_samples':sum(r['unrequested_robot_contact_samples'] for r in rows),'device_change_skills':sum(r['unrequested_device_state_change'] for r in rows)}
    summary={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'totals':totals,'checks':checks,'core_skills':list(LABEL),'family_count':3,'catalogue_total':25,'coverage_complete':False,'test_D_used':False,'formal_collection_resumed':False,'formal_training_started':False,'note':'Fixed physical GT-assisted executor core acceptance only; no WM or planner run in these episodes.'}
    (B/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    # Two views: root-level real outcomes and skill-level physical control cost.
    rows=datasets['独立验收'];chainrows=[r for r in rows if r['case']=='chain'];rootids=[r['root_id'] for r in roots if r['split']=='acceptance']
    fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1,1.15]})
    mat=np.full((18,4),np.nan);labels=['开抽屉','抓指定色','放入','关抽屉']
    for r in chainrows:
        col=0 if r['skill']=='open_drawer' else 1 if r['skill'].startswith('lift_') else 2 if r['skill']=='place_in_drawer' else 3
        mat[rootids.index(r['root']),col]=1 if r['conservative_pass'] else 0
    from matplotlib.colors import ListedColormap
    axes[0].imshow(mat,cmap=ListedColormap(['#cd5146','#238b74']),vmin=0,vmax=1,aspect='auto');axes[0].set_xticks(range(4),labels);axes[0].set_yticks(range(18),[x.replace('_acceptance_',' · ') for x in rootids]);axes[0].set_title('独立初态 × 四步收纳链');axes[0].set_ylabel('原生场景与预先固定的初态编号')
    for i in range(18):
      for j in range(4):axes[0].text(j,i,'通过' if mat[i,j]==1 else '失败' if mat[i,j]==0 else '未到达',ha='center',va='center',fontsize=8,color='white' if np.isfinite(mat[i,j]) else '#333')
    keys=list(LABEL)
    for i,k in enumerate(keys):
        v=[r['controls'] for r in rows if r['skill']==k and r['expected']=='success'];axes[1].scatter(np.full(len(v),i),v,color='#2867a8',alpha=.65,s=26)
        if v:axes[1].plot([i-.2,i+.2],[np.median(v)]*2,color='#222',lw=2)
    axes[1].axhline(900,color='#bb4848',ls='--',label='冻结上限 900 步');axes[1].set_xticks(range(6),[LABEL[k] for k in keys],rotation=25,ha='right');axes[1].set_ylabel('实际控制步数（30 步约为 1 秒仿真时间）');axes[1].set_title('可执行请求的实际代价，含重试');axes[1].set_ylim(0,950);axes[1].grid(axis='y',alpha=.2);axes[1].legend(loc='upper right')
    fig.tight_layout();(R/'figures').mkdir(exist_ok=True);fig.savefig(R/'figures/physical_oracle_v1_acceptance.png',dpi=160);plt.close(fig)
    # Exact same-root before/after causal evidence, not a HULC comparison.
    fig,ax=plt.subplots(1,2,figsize=(13,4.2))
    for sub,color,label in [('candidate1_C0','#d27d24','原向下抓持关闭'),('candidate2_C0','#287f73','改为前方水平推压')]:
        t=load(B/'development'/sub/'03_close_drawer/trace_private.json');ax[0].plot(np.arange(1,len(t)+1),[r['scene_obs'][1]*100 for r in t],color=color,label=label)
    ax[0].axhline(2,color='#555',ls='--',label='严格关闭上界 2 cm');ax[0].set(xlabel='当前关闭技能内控制步',ylabel='抽屉打开距离（cm）',title='相同起点：原轨迹撤离后又钩开抽屉');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
    for sub,color,label in [('candidate2_B_development_02','#d27d24','首次朝向：关节限位'),('candidate3_B2','#287f73','加入一次对称朝向重试')]:
        d=B/'development'/sub/'01_lift_pink_block_table';t=load(d/'trace_private.json');c=load(d/'controls.json');err=[np.linalg.norm(np.array(r['robot_obs'][:3])-a['target_xyz'])*100 for r,a in zip(t,c)];ax[1].plot(np.arange(1,len(t)+1),err,color=color,label=label)
    ax[1].set(xlabel='当前抓取技能内控制步（包括重试）',ylabel='TCP 与当步目标距离（cm）',title='相同起点：对称夹取方向解除限位');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2);fig.tight_layout();fig.savefig(R/'figures/physical_oracle_v1_causes.png',dpi=160);plt.close(fig)
    failed=load(B/'independent_acceptance/C_acceptance_02/chain/01_lift_pink_block_table/trace_private.json')
    fig,ax=plt.subplots(figsize=(12,4));xx=np.arange(1,len(failed)+1);ax2=ax.twinx()
    l1=ax.plot(xx,[(r['scene_obs'][0]-failed[0]['scene_obs'][0])*1000 for r in failed],color='#c47821',label='未请求的滑门位移')[0]
    l2=ax2.plot(xx,[(r['objects']['block_pink'][2]-failed[0]['objects']['block_pink'][2])*100 for r in failed],color='#9646af',label='目标粉块升高')[0]
    ax.axvline(208,color='#b74b43',ls='--');ax.axhline(3,color='#666',ls=':');ax.text(210,1.0,'第208步：手掌首次碰到滑门',fontsize=9,color='#b74b43')
    ax.set(xlabel='粉块抓取技能内控制步',ylabel='未请求滑门位移（mm）',title='独立验收失败：目标抓起，但引入了滑门副作用');ax2.set_ylabel('目标粉块相对初始高度（cm）',color='#9646af');ax.legend(handles=[l1,l2],loc='upper left');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(R/'figures/physical_oracle_v1_acceptance_failure.png',dpi=160);plt.close(fig)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>物理技能控制器 · 核心验收</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1160px;margin:35px auto;padding:0 24px;color:#233346;background:#fbfcfd}h1{font-size:30px}h2{margin-top:35px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd5df;padding:8px;text-align:left}th{background:#eaf0f5}img{width:100%;border:1px solid #ddd}a{color:#17649e}code{overflow-wrap:anywhere}video{width:48%;max-width:540px}small,.caption{color:#53647a}details{padding:10px;border:1px solid #ccd5df;margin:12px 0}.box{padding:16px;background:#eaf5f0;border-left:5px solid #238b74}</style><body><h1>真值辅助物理技能控制器：核心验收</h1>']
    t=totals['独立验收'];parts.append(f'<div class="box">独立验收完成 {t["chain_complete"]}/{t["chains"]} 条四步收纳链；可执行请求 {t["feasible_success"]}/{t["feasible_requests"]} 通过；前提缺失请求 {t["correct_refusals"]}/{t["negative_requests"]} 正确拒绝。该结论只覆盖这批预先固定的 A/B/C 桌面初态和六项核心技能，尚未覆盖全部 25 项技能。</div>')
    parts.append('<p>当前目标是先获得可检验的固定控制端。控制器读取模拟器中的物体位姿、接触及机器人状态，使用原 CALVIN 逆运动学和电机动作；夹取、拉动、推压与释放全部发生在原物理仿真中。世界模型与高层规划器没有参与本次控制测试，也没有获得这些真值。</p>')
    parts.append('<h2>具体元素、作用与边界</h2>'+table(['元素','实际实现','为什么需要 / 不代表什么'],[
      ['命名技能','开/关抽屉；抓桌面红、蓝、粉块；放入抽屉','一次只执行当前请求，不自动补做未请求的前提。'],['双相机与本体状态','RGB 静态/手眼、双深度和 robot_obs；公开入口 execute 只返回这些观测','GT 诊断保存在 private 文件中，尚未接入正式 WM 采集流程。'],['物理动作','原 IK、关节电机、夹爪摩擦与接触；原 URDF/动力学参数','运行时拦截物体重置、直接驱动桌面机构、改碰撞、外力、人工附件等调用。'],['成功','官方任务判定 + 更严格终态 + 额外 12 控制步稳定检查','例如关闭不仅要求相对位移，还要求最终打开距离 < 2 cm。'],['有界恢复','最多 2 次尝试、900 控制步；接触前 IK 失败可改用 180° 对称夹取姿态','不是无限修补；掉物、前提不再成立或预算耗尽均明确失败。'],['冻结划分','9 个开发根；18 个独立验收根；A/B/C 各环境覆盖三色、随机位置与朝向','验收根先于控制器开发保存；代码与阈值在验收前冻结；D 未使用。'],['保留状态','HULC、4215 条旧数据与原指南保留；正式采集和训练仍暂停','新执行器协议未混入旧数据，尚不能宣称原实验方案完成。']]))
    parts.append('<h2>实际结果</h2>'+table(['集合','完整四步链','可执行请求','正确拒绝','发生重试的技能','最大步数'],[[s,f'{v["chain_complete"]}/{v["chains"]}',f'{v["feasible_success"]}/{v["feasible_requests"]}',f'{v["correct_refusals"]}/{v["negative_requests"]}',v['retry_skills'],v['max_controls']] for s,v in totals.items()]))
    parts.append('<p class="caption">可执行请求包含收纳链中的四步，以及每个负例回合中用于建立“已持物、抽屉仍关”的一次正常抓取；拒绝次数单独计数，不并入成功率。18 条链的 Wilson 95% 区间为 '+f'{t["chain_wilson95"][0]*100:.1f}%–{t["chain_wilson95"][1]*100:.1f}%'+ '，不能据此保证任意初态成功。该控制预算与原 HULC 不同，未用这些数字做不公平的直接优劣比较。</p>')
    parts.append('<img src="figures/physical_oracle_v1_acceptance.png"><p class="caption">左图每行是一条预先固定的独立初态，每列是链中一步；绿色为通过，红色为失败，空白为未到达。右图蓝点是每次可执行请求的实际控制步数，黑短线为中位数，红虚线为 900 步预算。更多步数表示更慢，不能解释为更准确；重试已经计入。</p>')
    skillrows=[]
    for k in LABEL:
        subset=[r for r in rows if r['skill']==k and r['expected']=='success'];skillrows.append([LABEL[k],len(subset),sum(r['conservative_pass'] for r in subset),sum(r['retries'] for r in subset),f'{min(r["controls"] for r in subset)}–{max(r["controls"] for r in subset)}' if subset else '未执行'])
    parts.append(table(['技能','可执行请求数','通过数','重试总数','实际步数范围'],skillrows))
    parts.append('<h2>接触和意外变化没有省略</h2>'+table(['审计量','独立验收实际值','解释'],[
        ['非目标方块最大位移',f'{t["max_unrequested_drift_mm"]:.3f} mm','关闭时抽屉带动其中目标块属于预期；其余块全部检查，阈值 10 mm。'],['机器人接触最大穿透深度',f'{t["max_robot_penetration_mm"]:.3f} mm','PyBullet 接触求解器的数值穿透，不是关闭碰撞；冻结阈值 10 mm。'],['机器人接触峰值法向力',f'{t["max_robot_force_N"]:.1f} N','保留原 CALVIN 电机/夹爪力；不是对现实机器人的安全论证。'],['非指定接触采样条数',t['unrequested_contact_samples'],'包含手掌/指垫轻触桌面等，每帧可能有多个接触点；不声称无碰撞。'],['非请求的机构/灯状态变化技能数',t['device_change_skills'],'检查滑门、按钮、开关及灯的终态变化；原始轨迹保留。']]))
    parts.append('<p class="caption">接触审计采样在每个 30 Hz 控制边界，不覆盖控制步内部全部物理子步。初态较分散，未检验堆叠、拥挤、极端朝向、遮挡柜内或任意任务组合。</p>')
    parts.append('<h2>独立验收仍有一个明确失败，不能宣布可靠端已完成</h2><p>C_acceptance_02 抓粉块时，第 208 控制步手掌链接 8 接触滑门链接 2，法向力约 125 N；第 209 步滑门变化超过冻结的 3 mm 限制，最终滑动 9.466 mm。粉块被正确夹起，其余方块基本未动，随后放入、关抽屉也完成，但这条链仍判失败。同根的“抽屉关闭时抓取”准备动作也出现同样副作用，因此可执行请求的严格通过数是 88/90。</p><p>这次首先发生的错误是下降轨迹的手掌与滑门相碰，不是抓错颜色、没有抓稳或高层漏做前提。当前仅有直线末端路径与 IK，缺少整个夹爪/手掌对邻近机构的避障检查。验收后未用此根继续调参数，未放宽标准，候选版本不自动启用到正式实验。</p><img src="figures/physical_oracle_v1_acceptance_failure.png"><p class="caption">横轴为该次抓取内控制步。橙线对应左轴的未请求滑门位移（mm）；紫线对应右轴的粉块升高（cm）。红虚线标记第 208 步的首次有影响接触，灰点线是 3 mm 验收限制。目标升高并不能消除同时发生的副作用。</p>')
    faildir=B/'independent_acceptance/C_acceptance_02/chain/01_lift_pink_block_table'
    parts.append('<p>该次完整视频（左：静态相机；右：手眼相机）：</p>'+''.join(f'<video controls preload="metadata" src="../{(faildir/(cam+".mp4")).relative_to(P)}"></video>' for cam in ['rgb_static','rgb_gripper']))
    parts.append('<h2>开发中失败的首个原因及单项替换</h2><img src="figures/physical_oracle_v1_causes.png"><p>关闭失败：向下抓把手时手掌撞到桌沿，未能完全推回；张开后的后侧指头在撤离时又钩开抽屉。保留相同初态，只把关闭轨迹改为前方水平推压，桌面基座接触采样从 54 条降为 0，最终开度从 6.27 cm 降至近零。</p><p>B 场景抓粉块失败：接触前两个关节到 +2.8973 rad 上限，末端距离物体上方目标仍有 11.39 cm。保留相同初态，增加一次 180° 对称抓姿重试后完成。两个替换对的初始公开观测均逐元素完全相同。</p><p class="caption">两图橙线是失败实现，绿线是对应的单项替换；横轴是技能内控制步。左纵轴是抽屉开度，虚线为 2 cm 严格关闭边界；右纵轴是 TCP 与该步目标位置误差，绿色后段包含唯一一次重试。它们是物理轨迹与关节约束证据，不是对 HULC 感知错误的推断。</p>')
    parts.append('<h2>逐条独立验收与原始视频</h2><p>每项技能均保存静态相机和手眼相机完整视频、动作、私有接触轨迹和起止公开观测。下面按原先的根编号顺序列出全部记录，包括任何失败。</p>')
    for rid in rootids:
        subset=[r for r in rows if r['root']==rid];ep=[e for e in episodes['独立验收'] if e['root']==rid and e['case']=='chain'][0]
        parts.append(f'<details><summary>{rid} · 四步链 {"通过" if ep["complete"] else "失败"}</summary>')
        parts.append(table(['回合/技能','状态/原因','步数/重试','双相机','记录'],[[r['case']+' / '+LABEL[r['skill']],r['status']+' / '+str(r['reason']),f'{r["controls"]} / {r["retries"]}',link(Path(r['result_path']).parent/'rgb_static.mp4','静态')+' · '+link(Path(r['result_path']).parent/'rgb_gripper.mp4','手眼'),link(Path(r['result_path']),'结果')+' · '+link(Path(r['result_path']).parent/'controls.json','动作')+' · '+link(Path(r['result_path']).parent/'trace_private.json','私有轨迹')] for r in subset]));parts.append('</details>')
    parts.append('<h2>复查入口与尚未完成的范围</h2><p>'+ ' · '.join(link(x,y) for x,y in [(B/'protocol.json','原始控制器协议'),(B/'roots_manifest.json','初态清单'),(B/'controller_freeze.json','代码与验收冻结记录'),(B/'causal_development_evidence.json','同初态替换证据'),(B/'summary.json','汇总数据'),(P/'src/wm_symbolic/physical_oracle.py','控制器源码'),(P/'scripts/evaluate_physical_oracle.py','验收脚本')])+'</p>')
    parts.append('<p>当前覆盖 6/25 个技能、3 个技能族（抽屉、抓取、放置）。滑门、灯/按钮、平推、柜内抓取与其他放置仍需独立开发和验收。此结果不授权恢复正式采集或训练；后续必须使用独立的新执行器身份与数据协议。</p>'+table(['完整性检查','结果'],[[k,v] for k,v in checks.items()])+'</body></html>')
    (R/'physical_oracle_v1.html').write_text(''.join(parts))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
