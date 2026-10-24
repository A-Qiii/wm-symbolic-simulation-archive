from pathlib import Path
import json,hashlib,html,math,datetime
from collections import Counter
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

P=Path('/home/robot/research/projects/wm_symbolic');B=P/'runs/physical_oracle_v2';V1=P/'runs/physical_oracle_v1';R=P/'reports'
plt.rcParams['font.family']=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
LABEL={'open_drawer':'打开抽屉','close_drawer':'关闭抽屉','lift_red_block_table':'抓取红块','lift_blue_block_table':'抓取蓝块','lift_pink_block_table':'抓取粉块','place_in_drawer':'放入抽屉'}
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def link(p,label):return '<a href="../'+html.escape(str(p.relative_to(P)))+'">'+html.escape(label)+'</a>'
def table(head,rows):return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
def main():
    assert (B/'independent_acceptance/COMPLETE.json').exists()
    folders={'旧18根回归':'development_old_regression_candidate0','新27根独立验收':'independent_acceptance'}
    data={name:[dict(load(p),result_path=str(p)) for p in sorted((B/folder).glob('*/*/*/result.json'))] for name,folder in folders.items()}
    episodes={name:load(B/folder/'episode_summaries.json') for name,folder in folders.items()};stats={}
    for name,rows in data.items():
        pos=[x for x in rows if x['expected']=='success'];neg=[x for x in rows if x['expected']!='success'];chains=[x for x in episodes[name] if x['case']=='chain'];k=sum(x['complete'] for x in chains);n=len(chains);z=1.96;den=1+z*z/n;mean=(k/n+z*z/(2*n))/den;half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/den
        choices=[e['candidate'] for r in rows for e in r.get('events',[]) if e['phase']=='selected_clearance_checked_grasp']
        stats[name]={'strict_chains':[k,n],'chain_Wilson95':[mean-half,mean+half],'official_feasible_success':[sum(x['official_success'] for x in pos),len(pos)],'strict_feasible_success':[sum(x['conservative_pass'] for x in pos),len(pos)],'correct_refusals':[sum(x['correct_refusal'] for x in neg),len(neg)],'retries':sum(x['retries'] for x in rows),'max_controls':max(x['controls'] for x in rows),'total_controls':sum(x['controls'] for x in rows),'max_unrequested_object_drift_mm':max(x['max_unrequested_object_displacement_m'] for x in rows)*1000,'max_robot_penetration_mm':max(x['max_robot_contact_penetration_m'] for x in rows)*1000,'max_robot_force_N':max(x['max_robot_contact_force_N'] for x in rows),'unrequested_robot_contact_samples':sum(x['unrequested_robot_contact_samples'] for x in rows),'device_side_effect_calls':sum(x['unrequested_device_state_change'] for x in rows),'runtime_physics_guards_all_passed':all(x['physics_guard_passed'] for x in rows),'selected_grasp_tilts':dict(Counter(str(x['tilt_world_x_degrees']) for x in choices)),'planning_candidates_checked':sum(len(x.get('planning_private',[])) for x in rows),'planning_rejection_reasons':dict(Counter(q['reason'] for r in rows for q in r.get('planning_private',[]) if not q['valid']))}
    freeze=load(B/'final_freeze.json');checks={'controller_unchanged':sha(P/'src/wm_symbolic/physical_oracle_v2.py')==freeze['controller_sha256'],'base_v1_unchanged':sha(P/'src/wm_symbolic/physical_oracle.py')==freeze['base_v1_controller_sha256'],'evaluation_script_unchanged':sha(P/'scripts/evaluate_physical_oracle_v2.py')==freeze['evaluation_script_sha256'],'fresh_roots_unchanged':sha(B/'roots_manifest.json')==freeze['fresh_roots_sha256'],'main_manifest_unchanged':sha(P/'data/data_manifest.json')==freeze['main_manifest_sha256'],'main_transitions':load(P/'data/data_manifest.json')['real_transitions'],'collection_hold':(P/'runs/PAUSED_BY_USER.json').exists(),'training_hold':(P/'runs/FORMAL_TRAINING_HOLD.json').exists(),'v1_report_preserved':(R/'physical_oracle_v1.html').exists()}
    allpass=all(x['strict_chains'][0]==x['strict_chains'][1] and x['strict_feasible_success'][0]==x['strict_feasible_success'][1] and x['correct_refusals'][0]==x['correct_refusals'][1] for x in stats.values())
    summary={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'core_acceptance_passed':allpass,'coverage_skills':6,'catalogue_skills':25,'statistics':stats,'integrity':checks,'formal_collection_resumed':False,'formal_training_started':False,'test_D_used':False,'library_video_ids':['libfile_012a0cbbf8c081918501dc014fc578d9','libfile_b66d5fd6c2ec8191a592e534b84625da']}
    (B/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for dir_,color,label in [(V1/'independent_acceptance','#c67b2c','v1：旧验收失败'),(B/folders['旧18根回归'],'#258472','v2：同初态回归')]:
        trace=load(dir_/'C_acceptance_02/chain/01_lift_pink_block_table/trace_private.json');axes[0].plot(np.arange(1,len(trace)+1),[(x['scene_obs'][0]-trace[0]['scene_obs'][0])*1000 for x in trace],color=color,label=label)
    axes[0].axhline(3,color='#666',ls='--',label='原验收上限3 mm，保持不变');axes[0].set(xlabel='粉块抓取技能内控制步',ylabel='未请求滑门位移（mm）',title='同初态修复：目标抓取之外，也消除副作用');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    keys=list(LABEL)
    for name,color,shift in [('旧18根回归','#258472',-.1),('新27根独立验收','#3274b5',.1)]:
      for i,key in enumerate(keys):
        vals=[r['controls'] for r in data[name] if r['skill']==key and r['expected']=='success'];axes[1].scatter([i+shift]*len(vals),vals,color=color,s=17,alpha=.6,label=name if i==0 else None)
    axes[1].axhline(900,color='#bb4848',ls='--',label='900步上限');axes[1].set_xticks(range(6),[LABEL[k] for k in keys],rotation=25,ha='right');axes[1].set(ylabel='真实物理控制步数（30 Hz，包含重试）',title='回归与新验收分开记录控制代价',ylim=(0,950));axes[1].legend(fontsize=9);axes[1].grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(R/'figures/physical_oracle_v2_comparison.png',dpi=160);plt.close(fig)
    roots=load(B/'roots_manifest.json');rows=data['新27根独立验收'];fig,axes=plt.subplots(1,3,figsize=(13,5.3));from matplotlib.colors import ListedColormap
    for ax,env in zip(axes,'ABC'):
        rr=[r for r in roots if r['environment']==env];mat=np.full((9,4),np.nan)
        for r in rows:
            if r['environment']!=env or r['case']!='chain':continue
            col=0 if r['skill']=='open_drawer' else 1 if r['skill'].startswith('lift_') else 2 if r['skill']=='place_in_drawer' else 3
            mat[[x['root_id'] for x in rr].index(r['root']),col]=r['conservative_pass']
        ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#ca5248','#287daf']));ax.set_xticks(range(4),['开','抓','放','关']);ax.set_yticks(range(9),[x['root_id'].split('_')[-1]+' · '+{'red':'红','blue':'蓝','pink':'粉'}[x['chain_color']] for x in rr]);ax.set_title('原生场景 '+env)
        for i in range(9):
            for j in range(4):ax.text(j,i,'通过' if mat[i,j]==1 else '失败' if mat[i,j]==0 else '未到达',ha='center',va='center',color='white',fontsize=9)
    axes[0].set_ylabel('新独立初态编号与本链目标颜色');fig.tight_layout();fig.savefig(R/'figures/physical_oracle_v2_fresh_acceptance.png',dpi=160);plt.close(fig)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>物理技能控制器v2 · 回归与独立验收</title><style>body{max-width:1160px;margin:35px auto;padding:0 24px;font:16px/1.65 system-ui,sans-serif;color:#243447;background:#fbfcfd}h1{font-size:30px}h2{margin-top:34px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ced7df;padding:8px;text-align:left}th{background:#edf2f6}img{width:100%;border:1px solid #ddd}.caption{color:#53647a}details{border:1px solid #ced7df;padding:10px;margin:12px 0}a{color:#17649e}.box{padding:16px;border-left:5px solid #248373;background:#eaf5f0}code{overflow-wrap:anywhere}</style><body><h1>物理技能控制器 v2：回归与新独立验收</h1>']
    fresh=stats['新27根独立验收'];parts.append('<div class="box">'+('六项核心技能通过本轮预先固定的验收。' if allpass else '本轮仍有未通过项，未宣布可靠性完成。')+f'旧回归链 {stats["旧18根回归"]["strict_chains"][0]}/18；新独立链 {fresh["strict_chains"][0]}/27。正式世界模型采集和训练继续暂停。</div>')
    parts.append('<p>v1 的 18 条验收已经暴露过失败，因此这里全部标为回归集。新增 27 个 A/B/C 初态在 v2 开发前固定，每场景每种目标颜色各 3 个；未使用 D。v2 在旧回归与新验收之间没有改变源码、阈值或运动预算。</p>')
    parts.append(table(['条件','严格四步链','有效请求严格通过','缺前提正确拒绝','物理重试','最大步数'],[[name,f'{s["strict_chains"][0]}/{s["strict_chains"][1]}',f'{s["strict_feasible_success"][0]}/{s["strict_feasible_success"][1]}',f'{s["correct_refusals"][0]}/{s["correct_refusals"][1]}',s['retries'],s['max_controls']] for name,s in stats.items()]))
    parts.append('<p class="caption">一条链为打开抽屉→抓指定颜色→放入→关闭。有效请求还包括负例回合中的一次正常抓取；正确拒绝单独统计。新27条链的 Wilson 95% 区间为 '+f'{fresh["chain_Wilson95"][0]*100:.1f}%–{fresh["chain_Wilson95"][1]*100:.1f}%'+ '，不构成任意位置、任意场景或现实机器人的成功保证。</p>')
    parts.append('<h2>改动了什么，为什么能处理旧失败</h2>'+table(['元素','实现','原因与边界'],[
      ['完整碰撞几何','在独立 DIRECT 查询客户端加载相同机器人、桌子和方块的碰撞模型；检查整条机械臂、手掌、手指','原失败是手掌碰滑门，不能只检查末端点或目标方块。'],['固定候选集合','0/15/25/35° 前倾，3/10 mm 抓取高度偏移，两个180°对称朝向；最多16个候选','候选先在隔离几何世界查询，不通过真实试撞筛选。'],['路径检查','检查抬高、移至目标上方、下降、抬起；滑门/按钮/开关保留12 mm余量，其余障碍2 mm','余量属于控制规划参数；原验收阈值没有调宽。每6 mm平移或0.07 rad转动取样，不能称为连续空间完备证明。'],['真实执行','选定路径仍交给原 IK 与电机，方块靠手指接触与摩擦移动','实际执行世界禁止物体重置、直接驱动桌子、外力、人工粘连和关闭碰撞；仅查询客户端可重置几何。'],['副作用监视','任何非请求机构变化>3 mm或非目标方块漂移>10 mm时，返回执行失败','不再仅以抓起目标作为完成；不会自动把误碰机构拨回去掩盖错误。'],['公开输入输出','execute 只接收当前命名技能并返回双 RGB、深度、robot_obs','真值、碰撞候选、成功或拒绝原因只留在 private 审计，不给 WM 或高层规划器。'],['预算与验收','最多900真实控制步、2次物理尝试；保持官方任务+严格终态+12步稳定检查','查询候选数量不计作物理重试；两者分别记录。']]))
    parts.append('<p>旧失败根的初始公开观测逐元素完全相同。v1 的直下姿态引起9.466 mm滑门位移；v2 预查排除了该姿态，选用前倾15°、接触点抬高至方块中心上方10 mm的抓姿，真实滑门位移为0，四步任务完成。</p><img src="figures/physical_oracle_v2_comparison.png"><p class="caption">左图橙线是v1旧失败，绿线是v2同根回归；横轴为抓取内控制步，纵轴为未请求滑门位移，灰线为保持不变的3 mm限制。右图绿点是旧回归、蓝点是新验收的真实控制步数，红虚线是900步上限；不是训练损失或准确率。</p>')
    parts.append('<h2>新初态逐步结果</h2><img src="figures/physical_oracle_v2_fresh_acceptance.png"><p class="caption">三个面板分别是原生A/B/C场景，每行是开发前固定的新初态及目标颜色，每列依次是开、抓、放、关。蓝色为严格通过，红色为失败，空白为尚未到达。颜色表示验收结果，与方块实际颜色无关。</p>')
    parts.append('<h2>副作用、接触与候选筛选</h2>'+table(['集合','最大非目标漂移','最大接触穿透','非请求机构变化次数','非指定接触采样','选择的前倾角度计数'],[[name,f'{s["max_unrequested_object_drift_mm"]:.3f} mm',f'{s["max_robot_penetration_mm"]:.3f} mm',s['device_side_effect_calls'],s['unrequested_robot_contact_samples'],str(s['selected_grasp_tilts'])] for name,s in stats.items()]))
    parts.append('<p class="caption">非指定接触保留原始记录，不声称整个运动完全无接触。数值穿透来自原PyBullet接触求解器，审计仅采样30 Hz控制边界；关闭时抽屉内目标块随抽屉移动属于预期效果。碰撞查询中的假想几何位置与真实控制轨迹分开保存。</p>')
    parts.append('<h2>完整视频与动作记录</h2>')
    for name,folder in folders.items():
      parts.append('<details><summary>'+name+'：全部逐条记录</summary>')
      for root in sorted({x['root'] for x in data[name]}):
        rr=[x for x in data[name] if x['root']==root];parts.append('<details><summary>'+root+'</summary>'+table(['回合 / 技能','严格结果','步数 / 重试','视频','动作与私有审计'],[[x['case']+' / '+LABEL[x['skill']],str(x['conservative_pass'])+' / '+str(x['reason']),f'{x["controls"]} / {x["retries"]}',link(Path(x['result_path']).parent/'rgb_static.mp4','静态相机')+' · '+link(Path(x['result_path']).parent/'rgb_gripper.mp4','手眼相机'),link(Path(x['result_path']),'结果与候选')+' · '+link(Path(x['result_path']).parent/'controls.json','动作')+' · '+link(Path(x['result_path']).parent/'trace_private.json','真值轨迹')] for x in rr])+'</details>')
      parts.append('</details>')
    parts.append('<h2>已保存的两个原生视频附件</h2>'+table(['视频','确认的 Library ID'],[['收纳任务_双相机_成功链.mp4','libfile_012a0cbbf8c081918501dc014fc578d9'],['收纳任务_双相机_误碰滑门副作用链.mp4','libfile_b66d5fd6c2ec8191a592e534b84625da']])+'<p>这两段附件是已交付的v1成功与副作用失败代表，明确保留历史结果；不是本轮v2新验收视频。两视角均来自逐步实际渲染，没有用冻结末帧代替连续观测。</p>')
    parts.append('<h2>复查与尚未执行的工作</h2><p>'+' · '.join(link(x,y) for x,y in [(B/'summary.json','机器可读汇总'),(B/'final_freeze.json','最终代码冻结'),(B/'roots_manifest.json','新初态清单'),(B/'causal_clearance_development_evidence.json','同初态因果证据'),(B/'wm_1200_proposal_review.json','1200条提案检查'),(R/'physical_oracle_v1.html','保留的v1失败报告')])+'</p>'+table(['完整性检查','实际结果'],[[k,v] for k,v in checks.items()]))
    parts.append('<p>此次只验证6/25个技能及该分布下的四步收纳链。其余技能路径另行评估；未恢复正式采集、未训练WM、未运行高层符号规划。1200条提案仍是缩小到六技能的核心验证建议，不能替代原指南6000条完整方案。</p></body></html>')
    (R/'physical_oracle_v2.html').write_text(''.join(parts));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
