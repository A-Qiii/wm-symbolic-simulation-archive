from coverage_common import *
import gzip, collections, shutil, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False})

def table(head, rows):
    return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def history(row):
    with gzip.open(row['candidate']['physics_history_path'],'rt') as f:
        return [json.loads(line) for line in f]

def main():
    assert not (E/'configs/delivery_manifest.json').exists()
    refs={}
    def reg(p): refs[str(p)]=sha(p);return p
    source=read(reg(EXP/'results/results.json'))
    source_by={(r['root_id'],r['frame_id']):r for r in source}
    private=read(reg(E/'data/private/labels.json'))
    meta={r['state_id']:r for r in private}
    val=np.load(reg(E/'data/evaluation/official_validation.npz'))
    score={}
    for seed in (0,1):
        for arm in ARMS:
            score[seed,arm]=np.load(reg(E/f'results/actual_probabilities/official_validation_{arm}_seed{seed}.npz'))['probability']
    # Complete error set, not a search for the most illustrative failure.
    indices=sorted(set(int(i) for seed in (0,1) for i in np.where((val['y'][:,0]==1)&(val['v'][:,0]==1)&(score[seed,'augmented'][:,0]<.5))[0]))
    assert len(indices)==2
    errs=[]
    for i in indices:
        m=meta[str(val['state_ids'][i])];src=source_by[m['root_id'],m['frame_id']]
        hist=history(src);window=[x for x in hist if 32<=x['physics_step']<=96];last=hist[-1]
        reg(src['candidate']['physics_history_path']);reg(src['source_path']);reg(src['candidate']['observation_path'])
        distances=[float(np.linalg.norm(np.asarray(x['state']['robot_obs'][:3])-x['physical']['objects']['block_red']['position']))*1000 for x in window]
        forces=np.array([x['confidence']['red']['family_total_normal_force_N'] for x in window])
        ss=[]
        for seed in (0,1):
            c=float(score[seed,'control'][i,0]);a=float(score[seed,'augmented'][i,0]);ss.append({'seed':seed,'control_score':c,'augmented_score':a,'new_FN':c>=.5,'positive_weighted_BCE_control':-8.545454545454545*np.log(c),'positive_weighted_BCE_augmented':-8.545454545454545*np.log(a),'logit_change':float(np.log(a/(1-a))-np.log(c/(1-c)))})
        errs.append({'state_id':m['state_id'],'root_id':m['root_id'],'frame_id':m['frame_id'],'scene':m['scene'],'hints':m['candidate_hints'],'scores':ss,
                     'stable_Holding_ticks':sum(x['labels']['y'][0] and x['labels']['v'][0] and x['confidence']['red']['Holding_confident'] for x in window),'window_ticks':len(window),
                     'finger_total_force_range_N':np.stack([forces.min(0),forces.max(0)]).tolist(),'finger_total_force_CV':src['candidate']['diagnostics'][0]['force_stability']['family_total_force_CV'],
                     'external_force_max_N':max(x['confidence']['red']['max_external_normal_force_N'] for x in window),
                     'tcp_to_red_center_mm_range':[min(distances),max(distances)],'tcp_to_red_center_mm_final':distances[-1],
                     'gripper_width_mm':last['state']['robot_obs'][6]*1000,'gripper_action':last['state']['robot_obs'][-1],
                     'red_min_world_corner_z_mm':float(np.min(np.array(last['labels']['objects']['red']['world_corners'])[:,2]))*1000,
                     'red_drift_mm':last['object_drift_from_initial']['red']['translation_m']*1000,'RGB_MAE_original_to_new':last['new_RGB_vs_original_MAE_0_255'],
                     'source_RGB_path':src['source_path'],'new_RGB_path':src['candidate']['observation_path']})
    assert sum(e['scores'][0]['new_FN'] for e in errs)==1
    pairs=read(reg(E/'results/official_matched_pairs.json'));pairchanges=[]
    for seed in (0,1):
        c=next(x for x in pairs if(x['head'],x['seed'],x['arm'],x['color'])==('Holding',seed,'control','red'))
        a=next(x for x in pairs if(x['head'],x['seed'],x['arm'],x['color'])==('Holding',seed,'augmented','red'))
        for cc,aa in zip(c['records'],a['records']):
            assert cc['annotation_index']==aa['annotation_index']
            if cc['paired_correct']!=aa['paired_correct']:
                rows=[]
                for i,sid in enumerate(val['state_ids']):
                    if any(h['annotation_index']==cc['annotation_index'] for h in meta[str(sid)]['candidate_hints']):
                        rows.append({'state_id':str(sid),'y':bool(val['y'][i,0]),'v':bool(val['v'][i,0]),'control_score':float(score[seed,'control'][i,0]),'augmented_score':float(score[seed,'augmented'][i,0])})
                pairchanges.append({'seed':seed,'control_correct':cc['paired_correct'],'augmented_correct':aa['paired_correct'],**{k:v for k,v in cc.items()if k!='paired_correct'},'states':rows})

    oldmeta=read(reg(REP/'data/public/states.json'));newmeta=read(reg(PRIOR/'data/ABC/public/states.json'))
    combo={str(x['state_index']):x for x in read(reg(EX/'simultaneous_relation_probe_20261017/data/train/states.json'))}
    def old_public(sid):
        if sid.startswith('oldpool:'):sid=sid[len('oldpool:'):]
        kind,idx=sid.split(':')
        if kind=='combo':return {'path':combo[idx]['public_path'],'root_id':combo[idx]['root_id']}
        return (oldmeta if kind=='old' else newmeta)[int(idx)]
    datasets={}
    for name in ('old_Holding','old_In'):
        d=np.load(reg(E/f'data/train/{name}.npz'));rows=[]
        for sid,root,y,v in zip(d['state_ids'],d['root_ids'],d['y'],d['v']):
            pub=old_public(str(sid));assert pub['root_id']==root
            path=reg(pub['path']);obs=np.load(path)['robot_obs'];rows.append({'state_id':str(sid),'root_id':str(root),'width_mm':float(obs[6]*1000),'close_command':bool(obs[-1]<0),'y':y[:3].astype(bool).tolist(),'v':v[:3].astype(bool).tolist(),'path':str(path)})
        datasets[name]=rows
    official=[]
    for m in private:
        if m['split']!='train_candidate':continue
        src=source_by[m['root_id'],m['frame_id']];hist=history(src);last=hist[-1];obs=last['state']['robot_obs'];reg(src['candidate']['physics_history_path'])
        official.append({'state_id':m['state_id'],'root_id':m['root_id'],'scene':m['scene'],'width_mm':float(obs[6]*1000),'close_command':bool(obs[-1]<0),'y':m['y'][:3],'v':m['v'][:3],
                         'distances_mm':[float(np.linalg.norm(np.asarray(obs[:3])-last['physical']['objects']['block_'+co]['position']))*1000 for co in COLORS],
                         'annotations':[h['annotation_index']for h in m['candidate_hints']]})
    assert len(official)==120
    train=np.load(reg(E/'data/train/official_train.npz'))
    assert [x['state_id']for x in official]==train['state_ids'].tolist()
    assert np.array_equal(np.array([x['y']for x in official]),train['y'][:,:3])
    assert np.array_equal(np.array([x['v']for x in official]),train['v'][:,:3])
    datasets['official_train']=official
    coverage=[];colorcoverage=[]
    for name,rows in datasets.items():
        closing=[x for x in rows if x['close_command']]
        empty=[x for x in closing if all(x['v']) and not any(x['y'])]
        physically=[x for x in empty if x['width_mm']<=5]
        coverage.append({'pool':name,'states':len(rows),'roots':len({x['root_id']for x in rows}), 'close_command_states':len(closing),'close_command_empty':len(empty),'close_command_empty_roots':len({x['root_id']for x in empty}),'width_le5mm_empty':len(physically),'width_le5mm_empty_roots':len({x['root_id']for x in physically}),
                         'empty_width_mm_range':[min(x['width_mm'] for x in empty),max(x['width_mm'] for x in empty)] if empty else None})
        for j,color in enumerate(COLORS):
            pos=[x for x in closing if x['v'][j]and x['y'][j]];neg=[x for x in closing if x['v'][j]and not x['y'][j]]
            bothroots={x['root_id']for x in pos}&{x['root_id']for x in neg}
            # Fixed descriptive 5mm width strata, not a training criterion or outcome-selected gate.
            posbins={(x['root_id'],int(np.floor(x['width_mm']/5)))for x in pos};negbins={(x['root_id'],int(np.floor(x['width_mm']/5)))for x in neg}
            colorcoverage.append({'pool':name,'color':color,'closing_positive':len(pos),'closing_negative':len(neg),'roots_with_both':len(bothroots),'same_root_width5mm_bins_with_both':len(posbins&negbins)})
    draws=[]
    for head in HEADS:
        rows=datasets['old_'+head]
        oldempty=np.array([x['close_command']and all(x['v'])and not any(x['y'])for x in rows]);oldclosed=np.array([b and x['width_mm']<=5 for b,x in zip(oldempty,rows)])
        newempty=np.array([x['close_command']and all(x['v'])and not any(x['y'])for x in official]);newclosed=np.array([b and x['width_mm']<=5 for b,x in zip(newempty,official)])
        for seed in (0,1):
            sc=np.load(reg(E/f'configs/schedule_{head}_seed{seed}.npz'))
            for arm in ARMS:
                inds=sc['control']if arm=='control'else sc['common_old']
                draws.append({'head':head,'seed':seed,'arm':arm,'closing_empty_draws':int(oldempty[inds].sum()+(newempty[sc['official']].sum()if arm=='augmented'else 0)),
                              'width_le5mm_empty_draws':int(oldclosed[inds].sum()+(newclosed[sc['official']].sum()if arm=='augmented'else 0))})
    neighborhood=[]
    for e in errs:
        for restrict in (False,True):
            near=[x for x in official if x['close_command'] and (not restrict or x['scene']==e['scene']) and abs(x['width_mm']-e['gripper_width_mm'])<=5 and abs(x['distances_mm'][0]-e['tcp_to_red_center_mm_final'])<=10]
            neighborhood.append({'validation_example':e['state_id'],'same_scene':restrict,'description':'official TRAIN only; closing, width +/-5mm and TCP-red center distance +/-10mm around reviewed state; posthoc descriptive neighborhood',
                                 'red_positive':sum(x['v'][0]and x['y'][0]for x in near),'red_negative':sum(x['v'][0]and not x['y'][0]for x in near),'roots':len({x['root_id']for x in near}),'states':[x['state_id']for x in near]})
    annotationpairs=[]
    for j,color in enumerate(COLORS):
        anns=sorted({a for x in official for a in x['annotations']});knownpairs=[];closedpairs=[]
        for ann in anns:
            rows=[x for x in official if ann in x['annotations']and x['v'][j]]
            if any(x['y'][j]for x in rows)and any(not x['y'][j]for x in rows):knownpairs.append(ann)
            rows=[x for x in rows if x['close_command']]
            if any(x['y'][j]for x in rows)and any(not x['y'][j]for x in rows):closedpairs.append(ann)
        annotationpairs.append({'color':color,'same_segment_both':len(knownpairs),'same_segment_both_closing':len(closedpairs),'closing_annotation_ids':closedpairs})

    flips=read(reg(E/'results/decision_flips.json'));representatives=[]
    # One lexicographically first seed0 regression for each degraded cohort/head.
    for cohort,head in [('C_validation','In'),('D12_seen_actual135','Holding')]:
        selected=sorted([x for x in flips if x['cohort']==cohort and x['head']==head and x['seed']==0 and x['effect']=='regression'],key=lambda x:(x['root_id'],x['state_id'],x['color']))[0]
        if cohort=='C_validation':pub=newmeta[int(selected['state_id'].split(':')[1])]
        else:
            dp=read(reg(D/'data/combo/public/pairs.json'));ds=read(reg(D/'data/combo/public/states.json'));pair=next(p for p in dp if 'D12pair:'+p['id']==selected['state_id']);pub=ds[pair['after']]
        obs=np.load(reg(pub['path']));base=Path(pub['path']).parent.parent;raw=read(reg(base/'_private/states.json'));physical=next(x for x in raw if x['public_path']==pub['path'])['physical'];color=selected['color']
        representatives.append(selected|{'public':pub,'gripper_width_mm':float(obs['robot_obs'][6]*1000),'gripper_action':float(obs['robot_obs'][-1]),'tcp_object_center_mm':float(np.linalg.norm(obs['robot_obs'][:3]-physical['objects']['block_'+color]['position']))*1000,'native_held_names':physical['held']})
    save(E/'results/error_review.json',{'utc':now(),'method':'Read-only posthoc review; no fitting or threshold change; legacy probability fields are uncalibrated scores','official_red_FN_states':errs,'red_pair_changes':pairchanges,'coverage_unique':coverage,'coverage_by_color':colorcoverage,'coverage_draws':draws,'official_train_annotation_matching':annotationpairs,'official_train_geometry_neighborhood':neighborhood,'representatives':representatives,'source_sha256':refs})
    for name,rows in [('error_review_training_coverage',coverage),('error_review_color_coverage',colorcoverage),('error_review_draws',draws)]:
        with (R/f'{name}.csv').open('w')as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    fig,axes=plt.subplots(4,4,figsize=(13,11))
    for row,e in enumerate(errs):
        old=np.load(e['source_RGB_path']);new=np.load(e['new_RGB_path'])
        for col,(im,title)in enumerate([(old['rgb_static'],'官方原静态'),(old['rgb_gripper'],'官方原腕部'),(new['rgb_static'],'第96步新静态'),(new['rgb_gripper'],'第96步新腕部')]):
            axes[row,col].imshow(im,interpolation='nearest');axes[row,col].set_title(f"{e['scene']}帧{e['frame_id']}：{title}",fontsize=10);axes[row,col].axis('off')
    for row,r in enumerate(representatives,start=2):
        obs=np.load(r['public']['path'])
        for col,key in enumerate(('rgb_static','rgb_gripper')):
            axes[row,col].imshow(obs[key],interpolation='nearest');axes[row,col].axis('off');axes[row,col].set_title(('旧C'if row==2 else'已见D')+' / '+key,fontsize=10)
        ax=axes[row,2];ax.bar([0,1],[r['control_probability'],r['augmented_probability']],color=['#357ba3','#d57a32']);ax.axhline(.5,color='#666',ls='--');ax.set(ylim=(0,1),xticks=[0,1],xticklabels=['旧覆盖','加覆盖'],ylabel='加权BCE分数',title=r['head']+'('+r['color']+')')
        axes[row,3].axis('off');axes[row,3].text(0,.8,f"{r['state_id']}\nGT={r['truth']}，新增{r['error_after']}\nseed0：{r['control_probability']:.3f}→{r['augmented_probability']:.3f}\n夹爪宽度 {r['gripper_width_mm']:.1f} mm\nTCP—物体中心 {r['tcp_object_center_mm']:.1f} mm",va='top',fontsize=10)
    fig.suptitle('只读错误审阅：两帧红Holding漏判，以及旧C / 已见D新增错误代表',fontsize=14)
    fig.text(.5,.02,'前两行原图不赋新物理标签，仅第96步新图对应稳定性标签；后两行同一实际状态，两臂只改训练数据。蓝/橙柱表示训练臂；灰虚线是原0.5阈值。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(R/'figures/error_review_contact_sheet.png',dpi=145);plt.close(fig)

    erows=[]
    for e in errs:
        for s in e['scores']:erows.append([e['root_id'],e['frame_id'],e['hints'][0]['annotation_index'],s['seed'],'新增'if s['new_FN']else'两臂均错',f"{s['control_score']:.6f}→{s['augmented_score']:.6f}",f"{s['positive_weighted_BCE_control']:.2f}→{s['positive_weighted_BCE_augmented']:.2f}"])
    phys=[[e['frame_id'],f"{e['stable_Holding_ticks']}/{e['window_ticks']}",' / '.join(f"{e['finger_total_force_range_N'][0][j]:.3f}–{e['finger_total_force_range_N'][1][j]:.3f}"for j in(0,1)),' / '.join(f'{x:.2e}'for x in e['finger_total_force_CV']),f"{e['external_force_max_N']:.4f}",f"{e['gripper_width_mm']:.2f}",f"{e['tcp_to_red_center_mm_final']:.2f}",f"{e['red_drift_mm']:.2f}"]for e in errs]
    paired=[[x['seed'],x['root_id'],x['annotation_index'],f"{x['control_correct']}→{x['augmented_correct']}",x['positive_states'],x['negative_states']]for x in pairchanges]
    cov=[[x['pool'],x['states'],x['close_command_states'],x['close_command_empty'],x['close_command_empty_roots'],x['width_le5mm_empty'],x['width_le5mm_empty_roots']]for x in coverage]
    colrows=[[x['pool'],x['color'],x['closing_positive'],x['closing_negative'],x['roots_with_both'],x['same_root_width5mm_bins_with_both']]for x in colorcoverage]
    nrows=[[x['validation_example'].split(':')[-2],'同场景'if x['same_scene']else'A/B合计',x['red_positive'],x['red_negative'],x['roots']]for x in neighborhood]
    (R/'ERROR_REVIEW.md').write_text(f'''# 已保存错误与训练覆盖的只读审阅

红Holding扩充后有两帧漏判，但只有A0134帧1547682是新增；B0053帧330560是两臂共有错误。两帧来自不同源episode、不同任务段，两seed重复出现，不能称为两个独立新增错误。以下不重新训练、不改标签、权重或阈值。

{table(['源episode','帧','标注段','seed','错误归属','旧→加分数','正例加权BCE项 旧→加'],erows)}

sigmoid输出是**未校准分数**。训练使用正例权重8.5454545，因此不能把0.87解释成87%的真实拿持概率。表内单事实正例损失为−8.5454545·ln(分数)，尚未除以状态中有效物体数或batch大小，也不是训练集损失。这里的变化是对未参与训练验证帧的事后计算，说明同一真值的分数下移，不能反推优化器忽略了正例权重。

{table(['帧','稳定Holding tick','两指族总法向力范围N','两指族CV','最大外部力N','夹爪宽mm','TCP到红中心mm','红物体漂移mm'],phys)}

两帧均在32–96步全部65tick保留原Holding阳性与置信条件。距离使用公共TCP位置到物体中心的欧氏距离，不是指尖接触距离；双指族力证明实际接触。新标签属于重新仿真的第96步图像，不能回贴到官方原图。原图与新图像素MAE、完整来源与物理记录见[审阅JSON](../../experiments/official_readout_coverage_control_20261018/results/error_review.json)。稳定性不等于独立实机真值；强制保持动作后的仿真接触状态仍是本次标签定义。

原图→新图的静态/腕部像素MAE（0–255单位）分别为A帧1.017/4.903、B帧0.884/2.695；红物体最低角点z分别440.539/440.182mm。两帧均为贴近台面、双指夹持的状态。数据支持稳定夹持而非高抬起；本轮不据此修改Holding定义或把视觉接近台面当成标签错误。

![原图新图和旧C/D代表错误](figures/error_review_contact_sheet.png)

第一、二行分别对应新增和原有红漏判，列为原静态、原腕部、新静态、新腕部。第三行取旧C的In新增错误，第四行取已见D的Holding新增错误；按seed0的root/state/color字典序选首例，未挑最漂亮或最严重的案例。两臂读取同一实际输入。柱为分数，横轴两训练臂，灰线原0.5阈值。图中物体颜色是实际RGB，柱色只区分训练臂。

红色同段配对逐项变化如下；完整状态与分数在审阅JSON中。seed0是2段改善、1段退化，3→4；seed1是1段改善、2段退化，4→3。段16860的退化来自新增红漏判；seed1段10285另有负例由正确变为误报，不能归为同一次漏判。段是标注身份，多个段/帧并非独立试验。

{table(['seed','源episode','段ID','整段全对 旧→加','正状态','负状态'],paired)}

## 训练里有没有“闭合但空手”

区分两个可核查条件：robot_obs末项<0表示发出闭合命令；实际开口宽度≤5mm表示近闭合。5mm只用于本次事后描述，未进入训练、标签或门槛。“空手”要求三个Holding标签全有效且全负，不把未知当负；夹着抽屉把手但未拿方块也属于此定义。

{table(['训练池','状态数','闭合命令','闭合命令且空手','对应根数','宽≤5mm且空手','对应根数'],cov)}

各头两臂真正抽取到的闭合空手次数见[抽取次数表](error_review_draws.csv)。旧池和新增池均实际含此类负例，不能由结果不好推断“没有空手负例”。发出闭合命令不保证手指已合拢，故两列不可混为一谈。

Holding两seed的闭合命令空手抽取次数分别982→1133、993→1158；开口≤5mm的空手抽取分别699→764、694→769，扩充臂没有减少这类抽取。近零开口的数值残差（少量略负值）按原robot_obs保留，没有截断或修标签。

{table(['训练池','颜色','闭合命令正例','闭合命令负例','同根正负都存在','同根同5mm开口档正负都存在'],colrows)}

同根和开口档只匹配粗条件，不代表完全相同视觉/姿态反事实。旧池156/210所有样本均核对公共robot_obs；没有为了这次审阅重新统一历史物理位置记录，因此下面精细几何邻域只统计120个官方新增训练状态，不把未统计的旧几何当缺失。

官方训练同一标注段、同色兼有可靠正负的段数为{[x['same_segment_both']for x in annotationpairs]}（红/蓝/粉）；进一步要求正负两侧均发出闭合命令后为{[x['same_segment_both_closing']for x in annotationpairs]}。它证明部分同段匹配存在，不能证明所有失败外观已被覆盖。

围绕两帧失败，事后只读检查“闭合命令、开口宽度相差≤5mm、TCP到红中心距离相差≤10mm”的官方训练邻域：

{table(['验证帧（只用于邻域描述）','场景条件','红Holding正','红Holding负','训练根数'],nrows)}

邻域边界是描述性选择，不是预注册检验、模型选点或自动追加数据依据；若局部计数为0，只证明当前120个新增训练状态在这些明确条件下没有该类样本。不能泛化为整套训练缺负例，也不能据此认定退化原因。所有计数和源ID保留，未作因果单例修补。

可以确认的局部缺口是：B帧330560的同场景几何邻域在两个新增训练根中有6个红正例、0个红负例。A新增错误对应同场景邻域已有4正/1负，不能用“没有负例”解释；跨A/B两个邻域也都有负例。旧池的几何邻域尚未统一复核，因此不能宣布全训练缺少这种负例。旧C代表例是蓝块In由0.814降至0.287；D代表例实际拿着粉块，却把相距314.4mm的蓝块Holding分数从0.311升至0.691。它们说明仍有关系判读错误，没有证明T或某个像素因素是原因。
''')
    shutil.copy2(__file__,E/'source/error_review.py')
    print('READONLY_ERROR_REVIEW_COMPLETE',json.dumps({'red_errors':[(x['frame_id'],x['scores'][0]['new_FN'])for x in errs],'coverage':coverage,'geometry_neighborhood':[{k:v for k,v in x.items()if k!='states'}for x in neighborhood],'same_segment':annotationpairs},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
