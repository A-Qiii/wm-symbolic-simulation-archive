"""Summarize already saved runs; no model forward, training, or collection."""
from pathlib import Path
import json,hashlib,datetime,base64,ast,io,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from PIL import Image
from metrics import known
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'experiments/state_predicates_20261014';A=P/'experiments/state_predicates_augmented_20261014';E=P/'experiments/state_predicates_joint_20261014';B=P/'data/predicate_joint_acceptance_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(h,rows):return '<table><tr>'+''.join('<th>'+str(x)+'</th>'for x in h)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in r)+'</tr>'for r in rows)+'</table>'
def img(path):return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def main():
    ev=read(E/'results/evaluation.json');models=ev['models'];qa=read(E/'results/acceptance_qa.json');protocol=read(E/'configs/protocol.json');freeze=read(E/'results/candidate_freeze.json');existing=read(E/'results/existing_joint.json');a=read(E/'results/in_supervision.json');b=read(E/'results/balanced_loss.json');meta=read(E/'data/metadata_acceptance.json');om=[r for r in read(O/'data/metadata.json')if r['split']=='diagnostic_D']
    with np.load(E/'data/supplement_acceptance.npz')as z:y=z['y'].copy();v=z['eval_valid'].copy()
    failure=[];conflicts=[]
    for seed in range(3):
        pp=np.load(E/'results'/f'balanced_loss_seed{seed}_acceptance.npy');tau=models[f'balanced_loss_seed{seed}']['thresholds'];kk=known(pp,tau)
        for i,j in np.argwhere((((pp>=.5)!=y)|~kk)&v):
            r=meta[i];f=read(r['labels_private'])['frames'][-1];q=f['objects_audit'][r['target_color']]
            failure.append({'seed':seed,'qualified_before_acceptance':seed in freeze['qualified_seeds'],'sample_id':r['sample_id'],'root_id':r['root_id'],'condition':r['condition'],'index':int(i),'column':int(j),'truth':bool(y[i,j]),'probability':float(pp[i,j]),'known':bool(kk[i,j]),'raw_error':bool((pp[i,j]>=.5)!=y[i,j]),'width_mm':1000*f['gripper_width_m'],'object_tcp_mm':1000*q['tcp_distance_m'],'external_support_contacts':q['external_support_contacts'],'bilateral_finger_contacts':q['bilateral_finger_force'],'visible_pixels':q['visible_pixels'],'public_frames':r['public_frames'],'private_label_sha256':sha(r['labels_private'])})
        for i in range(len(pp)):
            if(pp[i,:3]>=tau[0]).sum()>1:
                r=meta[i];lab=read(r['labels_private'])['frames'][-1];target=r['target_index'];wrong=[j for j in range(3)if pp[i,j]>=tau[0]and lab['labels']['categorical'][j]!=6]
                q=pp[i:i+1].copy();before=known(q,tau)[0,target]
                for j in wrong:q[0,j]=0.
                after=known(q,tau)[0,target]
                conflicts.append({'seed':seed,'sample_id':r['sample_id'],'target_column':target,'original_held_probabilities':pp[i,:3].tolist(),'GT_confirmed_false_held_columns':wrong,'target_known_before':bool(before),'target_known_after_single_output_correction':bool(after),'target_probability_unchanged':float(q[0,target]),'diagnostic_only_no_primary_result_changed':True,'no_GT_in_model_or_deployment':True})
    save(E/'results/failure_analysis.json',{'acceptance_errors_or_unknowns':failure,'one_factor_output_conflict_diagnostic':conflicts,'facts':['Both frozen candidates falsely infer in(red,drawer) for the same visibly supported tabletop red object in a closed-drawer root.','Seed1 also falsely infers held on two bilateral-contact states still externally supported.','Seed2 raw target held is correct, but a false held(blue) creates a multiple-held conflict and suppresses true held(red).','Old D retains three identical-boundary held(pink) false negatives in D_core1200_09_b2 for both qualified candidates.'], 'causal_limit':'Output-conflict replacement isolates the immediate rejection mechanism only. It does not establish the cause of the blue false positive or a valid model repair. Shared representation and limited cross-root coverage remain hypotheses.','no_new_training_or_model_inference_in_analysis':True})
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained');x=np.arange(3)
    for ax,(pred,title)in zip(axes,[('held','held误报 / 71负例'),('in_drawer','in误报 / 77负例'),('open_drawer','open全部错误 / 83例')]):
        for pref,offset,color,label in [('original',-.19,'#347CB6','旧基线'),('balanced_loss',.19,'#DC7A38','损失平衡后')]:
            values=[models[f'{pref}_seed{s}']['acceptance'][pred]['FP']+(models[f'{pref}_seed{s}']['acceptance'][pred]['FN']if pred=='open_drawer'else 0)for s in range(3)];ax.bar(x+offset,values,.36,color=color,label=label)
            for xx,val in zip(x+offset,values):ax.text(xx,val+.15,str(val),ha='center')
        ax.set_xticks(x,['种子0\n仅诊断','种子1\n冻结候选','种子2\n冻结候选']);ax.set_title(title);ax.set_ylabel('错误记录数（越少越好）');ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(axis='y',alpha=.18);ax.set_ylim(0,max(3,max(t.get_height()for t in ax.patches)+3))
    axes[0].legend();fig.suptitle('独立验收：12根、83窗口；原始0.5判定，UNKNOWN另列',fontsize=14);fig.savefig(E/'figures/acceptance_counts.png',dpi=150);plt.close(fig)
    # Deterministic diagnostic cases: all unique qualified-candidate acceptance failures, then old regression.
    cases=[];seen=set()
    for r in failure:
        key=(r['index'],r['column'])
        if r['qualified_before_acceptance']and key not in seen:cases.append(r);seen.add(key)
    cases.append({'old':True,'index':1347,'column':2,'public_frames':next(r for r in om if r['index']==1347)['public_inputs'],'sample_id':'D_core1200_09_b2 边界2','truth':True})
    fig,axes=plt.subplots(len(cases),3,figsize=(14,3.1*len(cases)),gridspec_kw={'width_ratios':[1,1,1.6]},layout='constrained')
    atoms=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)']
    for row,r in enumerate(cases):
        with np.load(r['public_frames'][-1])as z:
            for j,cam in enumerate(['static','gripper']):axes[row,j].imshow(z['rgb_'+cam]);axes[row,j].axis('off');axes[row,j].set_title('静态相机'if j==0 else'夹爪相机')
        cohort='old_D'if r.get('old')else'acceptance';i=next(i for i,q in enumerate(om)if q['index']==1347)if r.get('old')else r['index'];j=r['column']
        oldp=[float(np.load(E/'results'/f'original_seed{s}_{cohort}.npy')[i,j])for s in [1,2]];newp=[float(np.load(E/'results'/f'balanced_loss_seed{s}_{cohort}.npy')[i,j])for s in [1,2]]
        ax=axes[row,2];ax.axis('off');ax.text(0,.98,atoms[j]+' 真值：'+str(r['truth']),va='top',weight='bold',color='#B44232',fontsize=12)
        sid=r['sample_id'].replace('__','\n');text=sid+'\n种子1 / 种子2\n旧P(True)：'+' / '.join(f'{p:.6f}'for p in oldp)+'\n新P(True)：'+' / '.join(f'{p:.6f}'for p in newp)
        if not r.get('old'):text+=f"\n外部支撑接触：{r['external_support_contacts']}\n夹爪开度 / 距TCP：{r['width_mm']:.2f} / {r['object_tcp_mm']:.2f} mm"
        if not r.get('old')and not r['known']:text+='\n种子2：非目标blue误判held → 冲突拒判red'
        ax.text(0,.84,text,va='top',fontsize=10.2,linespacing=1.45)
    fig.suptitle('保留的失败状态：同帧旧基线与冻结候选对照；非随机样本',fontsize=14);fig.savefig(E/'figures/failures.png',dpi=135);plt.close(fig)
    seq=lambda pref,co,g,k:'、'.join(str(models[f'{pref}_seed{s}'][co][g][k])for s in range(3))
    stage_rows=[]
    for label,results in [('仅增加目标in监督',a),('再改变损失聚合',b)]:
        for r in results['seeds']:stage_rows.append([label,r['seed'],r['selected_update'],r['tune']['gate']['failed_domain_predicates'],r['calibration']['raw_gate']['failed_domain_predicates'],'通过'if r['qualified']else'未通过',str(r['calibration']['thresholds'])])
    ar=[]
    for pred,label in [('held','held（目标）'),('in_drawer','in（目标,抽屉）'),('open_drawer','open（抽屉）')]:
        q=models['original_seed0']['acceptance'][pred];ar.append([label,f"{q['positive']} / {q['negative']}",seq('original','acceptance',pred,'FP')+' → '+seq('balanced_loss','acceptance',pred,'FP'),seq('original','acceptance',pred,'FN')+' → '+seq('balanced_loss','acceptance',pred,'FN'),seq('balanced_loss','acceptance',pred,'unknown')])
    regression=[]
    for co,title in [('old_D','旧D：20根，300边界'),('seen_test','已看过补采测试：6根，38窗口')]:
        for pred,label in [('held','held'),('in_drawer','in'),('open_drawer','open')]:
            q=models['original_seed0'][co][pred];regression.append([title,label,f"{q['positive']} / {q['negative']}",seq('original',co,pred,'FP')+' / '+seq('original',co,pred,'FN'),seq('prior',co,pred,'FP')+' / '+seq('prior',co,pred,'FN'),seq('in_supervision',co,pred,'FP')+' / '+seq('in_supervision',co,pred,'FN'),seq('balanced_loss',co,pred,'FP')+' / '+seq('balanced_loss',co,pred,'FN')])
    cond=[]
    for name,q in models['balanced_loss_seed1']['acceptance_conditions'].items():
        g='in_drawer'if name=='in_positive'else'held';metricname='FN'if name in ['in_positive','held_positive']else'FP'
        cond.append([name,q['roots'],q['windows'],'、'.join(str(models[f'balanced_loss_seed{s}']['acceptance_conditions'][name]['metrics'][g][metricname])for s in [1,2]),'in漏判'if name=='in_positive'else'held漏判'if name=='held_positive'else'held误报'])
    blocks=['<h1>联合开发与独立验收：命题感知改进简报</h1><p class="sub">2026-10-14 · 本机执行 · 冻结DINOv2与控制器 · 只研究感知→命题</p>',
      '<div class="callout"><b>已有进步，但两个开发合格候选都未通过独立验收。</b><p>目标in负监督与损失平衡使种子1、2兼顾新旧开发集；独立12根上保住12个held与6个in真阳性的原始判定，并清零抽屉开关错误。但两个候选仍在同一桌面红块上高置信误报in；种子1还有2次held误报，种子2有1次真实held因多held冲突被拒判。旧D还保留3条同状态held漏判。候选、验收数据和失败均保留，没有验收后调参。</p></div>',
      '<h2>1. 本轮做了什么，预算是否一致</h2>',
      table(['步骤','唯一改变或用途','固定预算/限制'],[['现有权重联合选模','同一种子比较原模型、上轮selected与final1500，模型参数不变','9个已有checkpoint；不重建未保存的中间权重'],['对照A：目标in监督','审计已有73训练窗口后，增加目标in=false监督；不改标签定义','三种子，各1500更新×60窗口'],['对照B：损失平衡','保留A的监督；改为held/in/open等权，组内正负类按训练集频率加权','三种子，各1500更新×60窗口；无权重网格搜索'],['一次独立验收','12新D根；每根原有lift与空夹爪测量轨迹，6个开抽屉根增加原有place技能','30轨迹、83窗口、60段视频；失败与缺项不补抽']]),
      '<p>两项对照总计9000次更新、540,000次窗口呈现，均从头以种子0、1、2初始化；每项与旧训练拥有相同更新/窗口预算。训练池保持原900＋新增73，自然900:73配比，未加入验收数据。新增窗口由2个监督事实增为3个，故A相对上轮的变化包括增加in标签及随之改变的有效事实分母；不能称为完全相同的监督事实预算。B只改变损失聚合。公式、精确类别权重、每一步采样索引和全部评估checkpoint已保存。</p>',
      '<h2>2. 联合开发标准及选模混杂</h2>',
      '<p>选模使用原C tune的180窗口／12根和新增C dev的37窗口／6根。校准使用原C calibration的120窗口／8根及同一新增dev；新增dev被重复使用，不能把校准约束当作统计保证。旧D、已看过的补采test均不参与挑选或调参。</p>',
      '<p>每个旧/新域、每个谓词都要求原始0.5假阳性率≤1%，有正例时召回≥95%；再要求校准后的覆盖率≥95%、已知错误≤2%、有效真阳性召回≥95%。没有正例的新dev in只能验证误报。checkpoint先按违反标准的域/谓词数量，再按超限量及根/类别平均BCE排序。阈值不能把原始错误藏掉。</p>',
      '<p>现有池中，联合选模对种子0、1仍选回上轮selected；种子2选到与其selected相同权重的final1500。三个都未满足完整标准。所有现有checkpoint在旧C的同一个桌面蓝块状态上产生in误报，因重复边界计4/308；目标在双相机可见，标签审计未见矛盾。<b>因此，仅把选模改成新旧联合，不能解释或修复全部问题。</b>中间历史权重未保存，不能断言联合规则在整个旧训练轨迹上也一定找不到更好模型。</p>',
      table(['对照','种子','选中更新步','tune失败域/谓词数','cal原始失败数','联合标准','held/in/open阈值'],stage_rows),
      '<p>A消除了新dev的in误报，但不能同时解决旧C in与新dev held问题。B的种子1、2通过，种子0仍失败。候选与阈值于独立采集前冻结；种子0仅报告作诊断，即使其验收某项较好也不据此晋升。</p><p>为检查早期checkpoint的混杂，冻结后另外对已保存的第1500步权重做了开发集只读诊断：A三个种子仍各有超限项，B种子1、2均无原始超限。因此两项方法在开发集上的差异不只来自选中步数不同。此项是事后固定预算诊断，没有使用验收输入，也未改变候选或阈值。</p>',
      '<h2>3. 一次独立验收的完整结果</h2>',
      table(['谓词','正例 / 负例','FP：旧 → B','FN：旧 → B','B的UNKNOWN'],ar),
      '<p>每组三个数依次为种子0、1、2，不能相加当作更多独立样本。蓝色旧基线与橙色B模型读取同一四帧输入，错误按原始0.5判定。三个种子的原始held召回均12/12、in召回均6/6；但种子2有1个真held被冲突规则变为UNKNOWN，有效已知召回只有11/12（91.7%），低于95%。其held覆盖率82/83很高，也不能掩盖这一点。</p>',
      img(E/'figures/acceptance_counts.png'),
      '<p class="caption">横轴为种子，种子0仅诊断，1和2为事先冻结候选。纵轴是错误记录数；标题注明分母。第三图包含open的FP与FN。12个根才是独立分组；同根窗口、相同样本上的三个种子均有关联。</p>',
      table(['实际条件','根数','窗口数','种子1、2错误数','统计错误'],cond),
      '<p>两个冻结候选的in误报均1/77＝1.30%，超过预定1%上限；种子1 held误报2/71＝2.82%，也超限。候选均未通过，未放宽标准。close/open、接触支撑、远离等条件窗口全部保留；7根没有稳定四帧的接触但仍支撑片段，被记为缺项，未重采或伪造。全部12个held和6个in正例实际采到并核验。</p>',
      '<h2>4. 旧分布和已观察测试的诊断回归</h2>',
      table(['数据','谓词','正 / 负','原模型FP / FN','上轮补采FP / FN','A：in监督FP / FN','B：平衡损失FP / FN'],regression),
      '<p>这些成绩在候选冻结后查看，仅用于回归说明。B种子1、2已修复原先两条近距离held误报，但都在D_core1200_09_b2的粉块上保留3条held漏判；它们是同一物理状态的重复边界，不是三个独立失败。种子1还出现旧D的1/117个in漏判。对照A/B没有在这些结果出现后再训练。</p>',
      '<h2>5. 具体失败与有证据的解释</h2>',
      img(E/'figures/failures.png'),
      '<p class="caption">展示所有冻结候选独立验收中的不同失败/拒判状态，再加旧D重复漏判的代表帧。每行左/中为静态/夹爪相机，右为同帧旧基线和B模型的种子1、2概率；红字仅标示被分析的失败事实，图内红、蓝、粉是物体颜色。案例按保存索引选取，非随机样本，不能代替全量计数。接触和距离为评分侧记录，从未输入感知网络。</p>',
      '<ul><li><b>桌面红块被读成in：</b>D_joint_accept_red_closed_0的空夹爪移动后窗口，红块仍有4个外部支撑接触，双相机可见，夹爪距它63.56 mm。候选1、2的in概率分别约0.999992、0.999986。其真实状态并非入抽屉，阈值置信不能保证正确。</li><li><b>接触被误当held：</b>种子1在蓝块和红块两个场景误报；虽有双侧手指受力，但仍有2个外部支撑接触，未满足held定义。开度40.54/40.41 mm、距TCP6.75/4.32 mm，外观与真实夹持接近，单凭靠近和闭合不足以区分。</li><li><b>真held被拒判的直接链条：</b>种子2对red的held概率约0.999994，同时对未held的blue给出0.728789，触发多held冲突。只在离线输出诊断中将错误blue事实置false、保持red概率不变，red便从UNKNOWN恢复为True。这定位了拒判的直接原因；没有改动主结果，也没有把GT写入模型或部署规则。</li><li><b>旧粉块漏判：</b>物体抬起、双相机可见、抽屉关闭，B种子1、2仍给held约0.000476、0.000033。它与新测试正例同属held，但跨根视觉状态覆盖不足或共享表示影响仍可能存在。</li></ul>',
      '<p><b>事实与假设分开：</b>对照显示，增加in监督与改变损失权重改善了部分开发行为；不能证明原损失不平衡是独立验收错误的唯一原因。独立根仍出现容器归属和支撑状态错误，说明开发通过不足以证明泛化。共享表示、有限训练覆盖、相机视角和接触外观的作用尚未被单独消融。不能据目前结果断言唯一原因是遮挡、容量不足或某个颜色。</p><p>另做了不运行模型的实体对应审计：83个验收窗口的249个颜色注册对象均与独立评分对象一致，未发现red/blue/pink身份串位。私有对象UID仅用于这次核对，未输入感知网络。这排除了当前记录中的身份映射错误，不能解释或修复学习表示本身的误判。</p>',
      '<h2>6. 停止位置与保留建议</h2>',
      '<p>本轮已完成预定两项轻量对照及一次独立验收，预算内的联合选模混杂得到说明，合格开发候选也被独立验证出能力边界。<b>旧模型继续保留为可复现基线；A/B模型保留为候选，均不升级为可靠LTL事实接口。</b>没有开启世界模型或完整规划训练。</p>',
      '<p>若继续，需要新一轮方法/数据预算决定：优先补足跨根的“仍受支撑／真实抬起”和“桌面／抽屉内”成对覆盖，并在独立开发集比较谓词专用关系表示；当前证据还不能证明这比增加覆盖更有效。不能拿本轮验收的几个错误反复调阈值或挑checkpoint。本轮没有进行这些后续尝试，也未把新验收数据回灌训练。</p>',
      '<h2>7. 本机交付与复现</h2>',
      table(['位置','内容'],[['configs/protocol.json','开始前冻结的预算、标准和两项方法'],['results/existing_joint.json','9个既有权重的联合开发核查'],['results/in_supervision.json / balanced_loss.json','每个种子的选模与校准记录'],['models/','两项方法每75步的checkpoint及第1步，保留完整候选轨迹'],['results/candidate_freeze.json','验收前冻结的候选、阈值及源代码哈希'],['results/evaluation.json / predictions.csv','所有开发后诊断与独立验收预测、错误和UNKNOWN'],['results/failure_analysis.json','实际失败状态及单项输出替换诊断'],['data/predicate_joint_acceptance_20261014（项目级）','12根、30轨迹、双相机视频、公共观测与分离私有评分记录'],['report.html','本简报，两幅图全部内嵌；无需网络打开']]),
      '<p class="sub">全部文件仅保存在本机。旧项目、旧checkpoint、控制器、DINO缓存与宿主环境保持不变。无上传、无执行阻塞；研究结果为未通过独立可靠性验收。</p>']
    css='body{margin:0;background:#f1f5f8;color:#1a2a3b;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1320px;margin:auto;background:white;padding:36px}h1{font-size:30px}h2{margin-top:36px;border-bottom:2px solid #dae4eb;padding-bottom:8px;font-size:23px}table{width:100%;border-collapse:collapse;font-size:13px;margin:16px 0}th,td{border:1px solid #d8e2ea;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf1f6}tr:nth-child(even){background:#fafcfd}.callout{background:#fff1df;padding:20px;border-left:5px solid #d08038}img{max-width:100%;height:auto;display:block;margin:20px auto}.caption,.sub{font-size:14px;color:#526576}li{margin:12px 0}'
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>命题感知联合开发与独立验收</title><style>'+css+'</style><main>'+''.join(blocks)+'</main></html>'
    (E/'report.html').write_text(html)
    summary={'status':'completed_bounded_study_independent_acceptance_failed','qualified_dev_seeds':[1,2],'independent_passed_seeds':ev['primary_passed_seeds'],'updates':9000,'new_training_variants':2,'training_windows':'900old+73supplement natural ratio','acceptance_roots':12,'acceptance_trajectories':30,'acceptance_windows':83,'new_data_used_for_training':False,'baseline_recommendation':'Keep old models as reproducible baseline; new models candidates only, no reliable LTL-interface claim.','report':str(E/'report.html'),'remaining_execution_blockers':[],'upload_status':'not_uploaded'};save(E/'SUMMARY.json',summary)
    # Delivery integrity, not a scientific pass.
    for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
    for p,h in freeze['source_sha256'].items():assert sha(p)==h,p
    for m in freeze['models']:assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
    for s in range(3):
        x=np.load(E/'logs'/f'in_supervision_seed{s}_batch_indices.npy');z=np.load(E/'logs'/f'balanced_loss_seed{s}_batch_indices.npy');prior=np.load(A/'logs'/f'seed{s}_batch_indices.npy');assert x.shape==(1500,60)and np.array_equal(x,z)and np.array_equal(x,prior)
    for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
    for f in ['acceptance_counts.png','failures.png']:
        with Image.open(E/'figures'/f)as im:im.verify()
    assert html.count('data:image/png;base64,')==2
    save(E/'results/delivery_validation.json',{'passed':True,'scope':'Integrity, bookkeeping and saved-image decoding; not scientific reliability','original_frozen_assets_unchanged':True,'candidate_weights_and_frozen_sources_unchanged':True,'identical_sample_index_streams_across_prior_and_two_new_variants':True,'two_variants_three_seeds_1500x60_each':True,'python_sources_parse':True,'embedded_pngs':2,'acceptance_qa_passed':True,'full_60_videos_decoded':True,'all_12held_and6in_positive_roots_observed':True,'training_inputs_exclude_acceptance':True,'scientific_acceptance_passed':False,'entity_registry_audit_249_matches':read(E/'results/registry_audit.json')['entity_matches']==249,'visual_review':'Both figures individually viewed: labels, denominators, actual camera pixels and failure explanations readable; no visible clipping.','no_upload':True})
    save(E/'results/progress.json',{'stage':'complete','scientific_result':'independent_acceptance_failed','training_finished':True,'collection_finished':True,'report_finished':True,'no_more_training_or_collection_scheduled':True})
    entries=[]
    for root,label in [(E,'experiment'),(B,'acceptance_data')]:
        for p in sorted(root.rglob('*')):
            if p.is_file()and p.name!='delivery_manifest.json'and'__pycache__'not in p.parts:entries.append({'root':label,'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
    save(E/'delivery_manifest.json',{'experiment_directory':str(E),'acceptance_data_directory':str(B),'status':summary['status'],'files':entries,'file_count':len(entries),'total_bytes':sum(r['bytes']for r in entries),'no_upload':True,'remaining_execution_blockers':[]})
    print('REPORT_AND_MANIFEST_READY',len(entries),sum(r['bytes']for r in entries),flush=True)
if __name__=='__main__':main()
