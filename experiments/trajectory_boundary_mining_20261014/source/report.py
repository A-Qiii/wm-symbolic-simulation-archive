"""Verify temporal semantics and report recovery feasibility; never train."""
from pathlib import Path
from collections import Counter
import json,gzip,hashlib,datetime,base64,html,ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle,Patch
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/trajectory_boundary_mining_20261014';S=P/'data/predicate_gap_supplement_20261014';C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def esc(x):return html.escape(str(x))
def table(head,rows):return '<table><tr>'+''.join('<th>'+esc(x)+'</th>'for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</table>'
def image(name,cap):return '<figure><img alt="'+esc(cap)+'" src="data:image/png;base64,'+base64.b64encode((E/'figures'/name).read_bytes()).decode()+'"><figcaption>'+esc(cap)+'</figcaption></figure>'
def main():
    d=read(E/'results/mining.json');sem=[];checks=0;mismatches=[];ordinary={};rootmeta={r['root_id']:r for r in read(S/'roots_manifest.json')if r['environment']in 'AB'}
    for rid,root in sorted(rootmeta.items()):
        target=root['target_color'];j=C.index(target);path=S/'episodes'/(rid+'__grasp');tr=json.load(gzip.open(path/'trace_private.json.gz','rt'));summary=read(path/'summary.json');hard=[]
        def reasons(t):
            o=t['objects_audit'][target]
            flags={'not_held':not o['held_contact_audit'],'valid_label':t['labels']['categorical_valid'][j],'external_support':o['external_support_contacts']>0,'visible':max(o['visible_pixels'].values())>=4,'near75mm':o['tcp_distance_m']<.075,'descend_phase':t['action']['phase']=='descend_named_block','closed_command':t['gripper_command']==-1}
            return [k for k,v in flags.items()if not v]
        old4=[i for i in range(3,len(tr))if all(not reasons(t)for t in tr[i-3:i+1])]
        for i,t in enumerate(tr):
            for c in C:
                o=t['objects_audit'][c];uid=o['object_uid_evaluator_only'];cs=[z for z in t['contacts_private']['block_'+c]if z[9]>.01];rl={z[4]for z in cs if z[2]==0};bi=bool(rl&{9,10})and bool(rl&{11,12});ext=sum(z[2]!=0 for z in cs);held=bi and not ext;checks+=1
                if bi!=o['bilateral_finger_force']or ext!=o['external_support_contacts']or held!=o['held_contact_audit']or held!=t['labels']['holding_derived'][C.index(c)]:mismatches.append({'root':rid,'frame':i,'colour':c})
            o=t['objects_audit'][target]
            if i>=3 and o['bilateral_finger_force']and o['external_support_contacts']>0 and t['gripper_command']==-1 and o['tcp_distance_m']<.075:hard.append(i)
        logged=[w for w in d['candidate_windows']if w['trajectory']==rid+'__grasp'and w['colour']==target];assert [w['frame']for w in logged]==hard
        assert sum(w['original_supplement_stable4_predicate']for w in logged)==len(old4)
        anchor=hard[0];early=Counter(k for t in tr[anchor-3:anchor+1]for k in reasons(t));close=np.array([t['gripper_command']for t in tr[anchor-3:anchor+1]]).tolist();heldhist=[t['objects_audit'][target]['held_contact_audit']for t in tr[anchor-3:anchor+1]]
        npth=S/'episodes'/(rid+'__near_closed_empty');nts=json.load(gzip.open(npth/'trace_private.json.gz','rt'));ns=read(npth/'summary.json');ctrls=[]
        for name in ['near_open_negative','near_closed_negative']:
            pp=(path if name=='near_open_negative'else npth)/'samples'/name/'labels_private.json'
            if pp.exists():
                t=read(pp)['frames'][-1];o=t['objects_audit'][target];ctrls.append({'condition':name,'command':t['gripper_command'],'width_mm':1000*t['gripper_width_m'],'distance_mm':1000*o['tcp_distance_m']})
        sem.append({'root':rid,'environment':root['environment'],'target':target,'drawer':root['drawer_setting'],'hard_anchor_count':len(hard),'old_stable4_anchor_count':len(old4),'first_hard_anchor':anchor,'first_history_held':heldhist,'first_history_command':close,'first_history_original_condition_violations':dict(early),'last_hard_anchor':hard[-1],'hard_width_mm':tr[anchor]['gripper_width_m']*1000,'hard_distance_mm':tr[anchor]['objects_audit'][target]['tcp_distance_m']*1000,'held_motion_verified':summary['motion_audit']['verified'],'held_motion_audit':summary['motion_audit']['held_follow_audit'],'ordinary_controls':ctrls,'near_closed_motion_verified':ns['motion_audit']['verified']})
    assert not mismatches and len(sem)==12 and all(x['held_motion_verified']for x in sem)
    write(E/'results/filter_semantics.json',{'supplement_root_rows':sem,'raw_contact_recomputed_object_frames':checks,'label_mismatches':mismatches,'exact_old_filter_counts_match_mining':True,'all12_later_held_follow_motion_verified':True,'future_motion_usage':'Retrospective physical label audit only; no future RGB, action, phase, pose or contact would be fed to reader.'})
    # Verify absence of any additional raw public arrays outside explicit saved files.
    unexpected=[]
    for t in d['trajectories']:
        folder=P/'data/core1200_v1/episodes'/t['trajectory']if t['source']=='core'else S/'episodes'/t['trajectory']
        paths=list(folder.rglob('*.npz'))
        for p in paths:
            if t['source']=='core':good=p.name in ['initial_public.npz','final_public.npz']or p.name.startswith('boundary_history_')
            else:good=p.name=='settled_root_private.npz'or(p.name.startswith('public_')and p.parent.parent.name=='samples')
            if not good:unexpected.append(str(p))
    assert not unexpected
    # All complete hard windows are the three pre-existing train samples.
    existing=[x for x in read(S/'data_manifest.json')['records']if x['split']=='train'and x['condition']=='contact_supported_close_command_negative'];existing_paths={tuple(x['public_frames'])for x in existing};complete=[w for w in d['candidate_windows']if w['complete_original_public_window']];assert len(complete)==3 and all(tuple(w['public_paths'])in existing_paths for w in complete)
    proposal={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'proposal_only_not_executed','reason_for_decision':'Missing full public mask/RGB frames prevents an unchanged-input controlled training comparison. User requested reporting feasibility before new collection.','minimal_first_step':{'type':'physical replay of2existing training roots, not new scene sampling','roots':['A_predicate_gap_red_closed','B_predicate_gap_red_closed'],'source':'Original reset parameters and the recorded low-level action sequence, including original settling. No intermediate GT state injection.','write':'Save every actual public RGB/depth/mask/robot/calibration frame separately from private audit. Retain real short states and all outcomes.','validation':'Compare replay action sequence, per-step robot/object state, contact labels, and original saved RGB endpoints. Stop on material divergence; do not silently call reconstructed/synthetic masks recorded data.','resource_estimate':'About600control frames total; at prior per-public-frame file size roughly0.10GB. This is an estimate, not a new run.'},'conditional_small_extension_if_recovery_confirmed':{'training':'Recover the12 existing A/B supplement grasp roots;12effective hard roots, including9notpreviouslyexported. Existing train roots are not counted as new independent roots.','development':'Use a small fresh C root list, e.g.6 roots covering3colours x2drawer settings, frozen before collection; old C remains historical regression only. Do not relabel prior D/test acceptance as fresh validation.','maximum_scope':'12training replays(including2pilot)+6fresh C trajectories if separately decided, not432attempts or299rootcertification.','comparison':'Keep675395parameter baseline reader, frozenDINO, normalization, controller, trainingbudget and jointcalibration fixed. Same-root equal-count hard-vsordinary negatives, same positive/background supervision. Pre-register exact ordinary-control availability and missingness after recovery.','control_confound':'Existing closed-empty near negatives typically have nearzero aperture and~60mmTCPdistance; open near negatives differ in command/aperture. Thus this tests training coverage of a difficult state bundle, not an isolated effect of support with all other physics identical.','decision_metric':'Root-level directional evidence on fresh C held errors/recall/UNKNOWN plus existing held,in/open regression; do not equate a tiny threshold pass with proof.'},'not_authorized_or_started':['large-scale scene collection','new network variants','test-driven training','full world model','planning','upload']}
    write(E/'configs/recovery_proposal.json',proposal);(E/'configs/recovery_proposal.json').chmod(0o444)
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(2)
    for off,k,label,color in [(-.25,'candidate_roots','当前帧物理事实满足','#397eb7'),(0,'stable4_roots','连续4帧均为困难状态','#db8a30'),(.25,'complete_original_public_roots','已有完整原始输入可直接使用','#6c7d87')]:
        vals=[d['summary'][s][k]for s in ['core','supplement']];bars=ax.bar(x+off,vals,.23,label=label,color=color)
        for b,n in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,n+.6,str(n),ha='center')
    ax.set(xticks=x,xticklabels=['旧A/B：60个独立根','补充A/B：12个独立根'],ylabel='具有至少一个相应窗口的独立根数',ylim=(0,72),title='状态存在，窗口筛选与观测保存限制了可训练覆盖');ax.legend(fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(E/'figures/root_coverage.png',dpi=160);plt.close(fig)
    timelines=read(E/'results/supplement_timeline_private.json');fig,axs=plt.subplots(2,1,figsize=(13,5.7))
    for ax,rid in zip(axs,['A_predicate_gap_red_closed','B_predicate_gap_red_closed']):
        row=next(x for x in sem if x['root']==rid);anchor=row['first_hard_anchor'];frames=timelines[rid+'__grasp']['frames'];idx=list(range(anchor-6,anchor+9));mat=np.array([[frames[i]['command']==-1 for i in idx],[frames[i]['bilateral']for i in idx],[frames[i]['external']>0 for i in idx],[frames[i]['held']for i in idx]],int)
        ax.imshow(mat,aspect='auto',cmap=ListedColormap(['#f0f3f6','#347fb5']),vmin=0,vmax=1,interpolation='nearest');ax.set(yticks=range(4),yticklabels=['闭爪指令','双侧承重接触','外部承重接触','held物理真值'],xticks=range(len(idx)),xticklabels=[f'{i-anchor:+d}'for i in idx],xlabel='相对当前锚点的已记录控制步（不是播放秒数）',title=rid+'：连续困难状态'+str(row['hard_anchor_count'])+'步');ax.add_patch(Rectangle((2.5,-.5),4,4,fill=False,edgecolor='#d14836',lw=2));ax.axvline(6.5,color='#333',lw=1.5);ax.axvspan(6.5,14.5,color='#aaa',alpha=.18)
    fig.suptitle('held按当前帧定义：过去三帧可以处于不同接触/指令状态',fontsize=16);fig.tight_layout();fig.savefig(E/'figures/temporal_semantics.png',dpi=160,bbox_inches='tight');plt.close(fig)
    parts=['<h1>已有A/B完整轨迹：短暂支撑状态与输入恢复可行性</h1><p class="muted">本机只读挖掘 · 固定物理执行器和现有输入管线 · 没有训练、重放或新采集</p>',
    '<p class="lead"><b>本轮判断：</b>“训练里缺少关键支撑负例”的覆盖缺口确实存在，但状态本身并不罕见。所有72个A/B根的完整日志都出现了真实困难负例；此前采样/导出把它们排除或未保存。当前无法直接训练受控对照的原因是缺少对应中间帧的原始公共观测，尤其可见实例掩码。</p>',
    '<p>这支持“采样机制造成覆盖不足”这一具体解释，尚未证明补上这些状态就能改善未见根的感知泛化。没有使用C/D完整轨迹选择本轮规则，也没有把历史开发/测试数据移入训练。</p>',
    '<h2>1. 审计了哪些完整轨迹</h2>',
    table(['来源','独立根','轨迹/技能段','保存的RGB/状态时间点','困难因果窗口'],[['旧A/B',60,720,'223,272',769],['补充A/B',12,24,'6,547',70],['合计',72,744,'229,819',839]]),
    '<p>根是独立初始化与布局单位；同根的3分支、多个技能、相邻帧与窗口都不算新的独立根。旧A/B各30根，分别包含15闭抽屉空手、9开抽屉空手、6开抽屉持物初态；补充A/B各6根覆盖3目标颜色×2抽屉设置。全量具体布局、初始位置/朝向与窗口索引见 results/mining.json。</p>',
    image('root_coverage.png','蓝柱为按当前帧物理事实找到的根；橙柱额外要求连续4帧均处于困难状态；灰柱要求4帧原始公共输入均已保存。纵轴为独立根数。旧数据原先采用技能边界导出，并非这个假设性的橙柱筛选。'),
    table(['来源/环境','红块出现困难状态的根','蓝块','粉块'],[[src+' / '+env]+[d['summary'][src]['environments'][env][c]for c in C]for src in ['core','supplement']for env in ['A','B']]),
    '<p>旧数据有147个根—颜色组合，补充数据有12个；同根多颜色仍只算一个独立根。旧数据28个根只有不足4步的困难段；补充数据9个根只有1–2步。其余可持续较长，但旧训练只保留技能边界，同样没有进入原训练窗口。</p>',
    '<h2>2. held的物理事实与四帧输入语义</h2>',
    '<p>held(t)沿用原规则：时刻t双侧指部/其接触垫均承重（法向力&gt;0.01），且没有外部承重接触。困难负例是双侧已接触但仍有外部承重，故held(t)=假。本轮另要求闭爪指令和TCP距离&lt;75毫米用于界定所研究的情境，不改变held标签。四帧输入必须严格为[t−3,t−2,t−1,t]；当前事实无需在过去四帧均相同，更不读取未来帧。旧数据的41个候选窗有混合held历史，应明确作为状态转变窗而非删改标签。</p>',
    image('temporal_semantics.png','每行是一项布尔事实：蓝格为真、浅格为假。红框是唯一允许输入的过去三帧及当前帧；黑线右侧灰底为未来，仅在此图做离线物理核对。A红闭2步、B红闭1步的真实支撑状态被原稳定筛选漏掉。'),
    table(['补充训练根','真实困难帧数','原四帧筛选可导出的锚点数','第一次困难窗的夹爪指令历史'],[[r['root'],r['hard_anchor_count'],r['old_stable4_anchor_count'],str(r['first_history_command'])]for r in sem]),
    '<p>已逐项重算旧筛选函数，包括类别有效性、可见性、距离、相位与闭爪指令，数量与原导出相符。短暂状态不是标签放松：补充12条抓取轨迹的原始接触重新计算无冲突；之后真实持物的运动跟随核验也全部成立。后续运动只核验物理事实，不作为模型输入或当前标签的替代。</p>',
    '<h2>3. 可见性与实际可恢复性</h2>',
    table(['内容','实际保存','本轮可以做到什么'],[['完整RGB','双相机H.264视频，旧质量参数7、补充8；并非逐帧原始RGB。','对179条含候选的轨迹完整解码，双相机帧数与状态/动作索引一致。'],['完整状态/接触','压缩私有日志逐控制步保存；本体状态也在日志内。','核验物理标签与因果索引；只允许本体状态字段进入既有模型，其他私有字段不输入。'],['原始掩码/深度/标定','只在技能边界或原已选窗口的public NPZ保存。','839个候选中只有3个完整原始输入窗，而且正是已用的3个训练样本；无新增可直接训练的根。'],['可见性','补充轨迹保存了精确可见像素计数，70窗全部4帧可见；旧轨迹没有逐帧掩码。','旧769窗在四帧均有RGB颜色像素代理信号，但这不是精确实例掩码或可用于恢复的真值。']]),
    '<p>因此不能把私有物体位姿投影为掩码，也不能将视频像素和自行估计掩码悄悄拼成原输入。那会同时改变观测/预处理，无法把训练效果归因于覆盖。完整运动与视频足够证明事件发生，不足以原样重建既有DINO+ROI输入。</p>',
    '<h2>4. 修正此前的成本判断</h2><p>此前以3/12外推约288/144次尝试，混淆了<strong>稳定窗口导出率</strong>与<strong>物理状态出现率</strong>。本次12/12补充训练根都有真实接触支撑时刻，因此撤回该采集次数推算。本轮也不以299根的1%置信认证取代方法判断；当前应先解决观测恢复，检验覆盖是否带来方向明确的感知改善。</p>',
    '<h2>5. 最小后续方案：先恢复2条已有训练轨迹</h2>',
    '<p>建议先决定是否允许重放已有A红闭、B红闭两条训练轨迹：只在开始使用原初始化，逐步执行原已记录低层动作，保存真实渲染的全部公共RGB、掩码与本体状态；不在中途注入GT物体状态、不延长接触停留、不改执行器。逐步对齐机器人/物体状态、接触标签和已有原始图像锚点；若出现实质漂移则记录并停止恢复，不能把它称为原数据重建。两条约600个控制帧，按已有公开帧文件大小估计约0.10GB。</p>',
    '<p>仅在这一步可恢复且另行决定继续后，最多扩展为12条既有A/B训练轨迹恢复，加一份预先冻结的6根新C小开发集（3颜色×2抽屉）。这会让困难训练覆盖从3根变为12根，其中9根以前未导出；恢复的根不是新增独立根。旧C只作历史回归，旧D与曾经独立验收继续保持历史身份，不伪称新的独立验证。</p>',
    '<p>随后才能预注册两个固定读出器组：同根、同数量的困难支撑负例与普通近邻负例；保留相同真实held正例、旧in/open与背景数据，固定网络、DINO、归一化、预算和校准规则。普通闭爪空抓样本通常夹宽接近0、TCP距离约60毫米，而困难样本约38毫米和3–10毫米；普通开爪近邻样本又有指令与夹宽差异。这些混杂必须报告，该比较检验困难情境的训练覆盖作用，不能声称只隔离了“外部支撑”一个物理变量。</p>',
    '<p>本轮没有启动上述重放或训练：原样可复用的新增公共窗口为0，受控验证尚不可执行。后续应看新C根的错误方向、held召回、UNKNOWN与in/open回归，确认覆盖假设是否得到支持；不通过反复换网络或任意阈值宣布成功。</p>',
    '<p class="muted">文件：results/mining.json（全部候选、根布局、视频对齐、源哈希）；results/filter_semantics.json（原筛选精确复算与运动核验）；configs/recovery_proposal.json（尚未执行的最小恢复方案）；delivery_manifest.json（本轮交付索引）。没有上传，没有世界模型或规划实验。</p>']
    css='body{font:16px/1.65 "Noto Sans CJK SC",sans-serif;background:#f4f7fa;color:#263846;margin:0}main{max-width:1120px;margin:auto;background:white;padding:42px}h1{font-size:29px}h2{font-size:22px;color:#245674;margin-top:34px}.lead{background:#edf5fa;padding:18px;border-left:4px solid #397eb7}.muted{font-size:14px;color:#62798b}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border:1px solid #dbe4ea;padding:9px;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#fbfcfd}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#607285}'
    (E/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>因果窗口挖掘可行性</title><style>'+css+'</style><main>'+''.join(parts)+'</main></html>')
    write(E/'SUMMARY.json',{'status':'audit_complete_controlled_training_blocked_by_missing_original_public_frames','training_roots_audited':72,'trajectory_segments_audited':744,'recorded_timepoints':229819,'physical_hard_negative_causal_windows':839,'roots_with_hard_negative':72,'full_original_public_windows':3,'new_full_public_windows':0,'new_full_public_roots':0,'supplement_physical_event_roots':12,'supplement_previous_stable_export_roots':3,'prior_attempt_count_extrapolation_retracted':True,'coverage_hypothesis':'Sampling/export mechanism confirmed to omit real hard states; effect of adding them on new-root accuracy remains untested','training_started':False,'replay_or_new_collection_started':False,'C_D_trajectory_content_read':False,'full_WM_or_planning':False,'uploaded':False,'next_minimal_decision':'2existing A/B training trajectory physical replays with complete public frame capture and exactness audit'})
    verify={'passed':True,'physical_core_endpoint_label_comparisons':720*2*3,'physical_core_endpoint_mismatches':0,'supplement_grasp_raw_contact_object_frame_checks':checks,'supplement_contact_mismatches':len(mismatches),'original_stable4_count_verified':True,'unexpected_public_npz_files':unexpected,'existing_three_complete_windows_identified':True,'held_follow_motion_verified_roots':12,'decoded_aligned_candidate_trajectories':sum(bool(t.get('video_alignment'))for t in d['trajectories']),'new_training_or_collection':False,'figures_generated':2,'visual_QA':'pending model inspection of generated figures'}
    write(E/'results/delivery_validation.json',verify);print('REPORT_READY',json.dumps(verify,ensure_ascii=False))
if __name__=='__main__':main()
