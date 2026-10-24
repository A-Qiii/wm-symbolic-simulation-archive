from diagnostic import *
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def validation():
    protocol=read(E/'configs/diagnostic_protocol.json');frozen=read(E/'configs/rules_frozen.json');rules=frozen['rules'];oldmanifest=read(OLD/'delivery_manifest.json')
    assert sha(OLD/'delivery_manifest.json')==protocol['original_manifest_sha256']
    for r in oldmanifest['entries']:assert sha(OLD/r['path'])==r['sha256'],r['path']
    assert sha(OLD/'results/decision.json')==protocol['original_decision_sha256']
    assert read(OLD/'results/decision.json')['worth_one_new_independent_test']is False
    assert sha(P/'src/wm_symbolic/physical_oracle.py')==protocol['controller_sha256']
    assert sha(E/'source/diagnostic.py')==protocol['source_sha256']
    assert sha(E/'data/AB_rule_fit_only.npz')==protocol['AB_data_sha256']
    tr=read(E/'data/AB_rule_fit_pairs.json');z=dict(np.load(E/'data/AB_rule_fit_only.npz'))
    fresh,_=fit_rules(tr,z['before'],z['after'],z['before_valid'],z['after_valid']);assert fresh==rules
    rev,_=fit_rules(tr[::-1],z['before'][::-1],z['after'][::-1],z['before_valid'][::-1],z['after_valid'][::-1]);assert rev==rules
    rejected=False
    try:fit_rules([dict(r,split='dev')for r in tr],z['before'],z['after'],z['before_valid'],z['after_valid'])
    except AssertionError:rejected=True
    assert rejected
    def rec(skill,args={}):return {'skill':skill,'declared_arguments':args,'pair_id':'in_memory_contract_check_NOT_empirical_data'}
    tests=[]
    for name,cur,row,expected_unknown in [('empty_hand_place',[0,0,0,0,0,0,1],rec('place_in_drawer'),6),('conflicting_held_place',[1,1,0,0,0,0,1],rec('place_in_drawer'),6),('unknown_held_place',[1,np.nan,0,0,0,0,1],rec('place_in_drawer'),6),('unsupported_action',[0,0,0,0,0,0,1],rec('unimplemented_action'),7),('copy_unrelated_UNKNOWN',[0,0,0,0,np.nan,0,1],rec('lift_red_block_table',{'color':'red'}),1)]:
        p,k,_=predict(np.array([cur],float),[row],rules);assert int((~k).sum())==expected_unknown;tests.append({'test':name,'UNKNOWN':expected_unknown,'passed':True,'not_an_empirical_transition':True})
    data=dict(np.load(OLD/'data/_private/training_targets_and_audit.npz'));pairs=read(OLD/'data/public/pairs.json');base=dict(np.load(E/'results/copy_effect_predictions.npz'));p,k,_=predict(data['observed_current_probabilities'],pairs,rules);assert np.array_equal(p,base['probabilities'],equal_nan=True)and np.array_equal(k,base['known'])
    altered=[dict(r,root_id='ignored',environment='ignored',macro_index=-999,dev_role='ignored',intended_root_colour_evaluator_only='ignored',future_truth='ignored',primitive_status='ignored')for r in pairs]
    ap,ak,_=predict(data['observed_current_probabilities'],altered,rules);assert np.array_equal(ap,p,equal_nan=True)and np.array_equal(ak,k)
    try:predict(data['observed_current_probabilities'],pairs,rules,future_truth=data['after_y'])
    except TypeError:pass
    else:raise AssertionError('Forbidden future input accepted')
    ev=read(E/'results/metrics.json');old=read(OLD/'results/evaluation.json');n=0
    mapping={'AB':'AB_all','C':'C_all','C_tune':'C_tune','C_calibration':'C_calibration'}
    for name,sets in ev['results'].items():
        if name=='copy_effect':continue
        for sn,sub in sets.items():
            a=sub['all'];b=old['results'][name][mapping[sn]]['all'];assert a['all7_known_correct']==b['all7_known_correct_pairs']and a['all4_correct_roots']==b['all_pairs_correct_roots']
            for mask,q in a['masks'].items():
                ref=b['mask_metrics'][mask];assert(q['strict_correct'],q['facts'],q['UNKNOWN'])==(ref['strict_known_correct'],ref['facts'],ref['UNKNOWN']);n+=1
    yy=data['after_y'].astype(bool);vv=data['after_valid'];full=((p>=.5)==yy)&k&vv
    c=np.array([r['split']=='dev'for r in pairs]);ab=~c
    assert int(full[c].all(1).sum())==24 and int(full[ab].all(1).sum())==95
    assert int((~k[ab]).sum())==2 and int((~k[c]).sum())==0
    independent_counts={}
    before=data['before_y'].astype(bool)
    for label,sel in [('AB',ab),('C',c)]:
        independent_counts[label]={}
        for maskname,mask in [('changed',before!=yy),('unchanged',before==yy)]:
            mask=mask&vv&sel[:,None];count=int(mask.sum());good=int((mask&full).sum());u=int((mask&~k).sum());ref=ev['results']['copy_effect'][label]['all']['masks'][maskname]
            assert(count,good,u)==(ref['facts'],ref['strict_correct'],ref['UNKNOWN']);independent_counts[label][maskname]={'facts':count,'correct':good,'UNKNOWN':u}
    legacy=read(E/'results/read_only_legacy_sources.json')
    for r in legacy:
        path=Path(r['path']);root=path.relative_to(P/'data/core1200_v1/episodes').parts[0];assert root.startswith(('A_','B_','C_'));assert sha(path)==r['sha256']
    assert frozen['created_utc']<ev['created_utc']
    val={'created_utc':now(),'passed':True,'original_delivery_101_manifest_entries_plus_manifest_unchanged':True,'original_preregistered_decision_unchanged':True,'original_controller_unchanged':True,'AB_rule_rederivation_identical':True,'AB_row_order_invariance':True,'non_AB_training_rows_rejected':True,'inference_signature':list(inspect.signature(predict).parameters),'private_metadata_perturbation_invariance':True,'future_kwarg_rejected':True,'UNKNOWN_contract_tests':tests,'existing_model_metric_cells_reproduced':n,'new_baseline_C_pairs':24,'new_baseline_C_all7_correct':24,'new_baseline_AB_all7_correct':95,'new_baseline_AB_UNKNOWN':2,'legacy_AB_C_readonly_source_files_verified':len(legacy),'D_paths_opened':0,'neural_training':False,'new_simulator_collection':False,'same_original120_pairs':True}
    val['new_baseline_changed_unchanged_independent_recount']=independent_counts
    save(E/'results/validation.json',val);return val

def main():
    val=validation();ev=read(E/'results/metrics.json');rules=read(E/'configs/rules_frozen.json')['rules'];legacy=read(E/'results/legacy_ABC_summary.json');audit=read(E/'results/current120_audit_summary.json')
    # Minimum condition types, not an invented empirical quota or new execution.
    controller=str(P/'src/wm_symbolic/physical_oracle.py')
    needs=[
      {'priority':'核心1','same_action':'close_drawer','contrast':'抽屉打开：空手 vs 持有一块','implemented_outcome':'空手可执行；持物拒绝 requires_empty_gripper','source':controller+':139-140','legacy_ABC_refusals':80,'discriminative':'是：固定close效果把open置0，但拒绝时应保留打开','current_protocol_coverage':'0拒绝；需要原2.1秒历史+96后停留的配对','construction':'可由打开抽屉后空手/执行已实现lift得到；不调用持物关门来构造起始状态'},
      {'priority':'核心2','same_action':'lift_<指定颜色>_block_table','contrast':'指定目标在桌上：空手 vs 持有另一颜色','implemented_outcome':'空手可执行；已持物拒绝 requires_empty_gripper','source':controller+':144-147','legacy_ABC_refusals':16,'discriminative':'是：固定lift效果会新增目标held；拒绝应保留原持有物','current_protocol_coverage':'0拒绝；需各目标颜色的独立根对照','construction':'另一物体通过已实现lift获得；指定目标仍放桌上'},
      {'priority':'核心3','same_action':'place_in_drawer','contrast':'持有同色块：抽屉充分打开 vs 关闭','implemented_outcome':'打开时可执行；关闭时拒绝 requires_open_drawer','source':controller+':135-138','legacy_ABC_refusals':0,'discriminative':'是：目标绑定相同，但固定放置效果不应在拒绝状态发生','current_protocol_coverage':'没有关闭且持物的放置调用','construction':'在初始开/闭抽屉各自执行已实现lift；lift无需抽屉打开。这里只是源码支持，尚未采集验证'},
      {'priority':'语义自检','same_action':'open_drawer','contrast':'空手：关闭 vs 已充分打开','implemented_outcome':'已打开拒绝 drawer_already_open','source':controller+':139-142','legacy_ABC_refusals':24,'discriminative':'单独不足：无条件open=1也是幂等，7谓词可同样正确','current_protocol_coverage':'0整状态不变调用','construction':'直接沿用已实现打开后的状态；不是承诺可重复执行成功'},
      {'priority':'语义自检','same_action':'place_in_drawer','contrast':'同为打开抽屉：持物 vs 空手','implemented_outcome':'空手拒绝 requires_exactly_one_held_block','source':controller+':135-137','legacy_ABC_refusals':41,'discriminative':'验证不变与UNKNOWN：本次对照无唯一目标时输出6项UNKNOWN，不假设成功/自环','current_protocol_coverage':'0空手放置；旧日志40个关闭、1个打开，需控制起始开闭','construction':'空手打开抽屉是已有执行能力'},
      {'priority':'语义自检','same_action':'lift_<指定颜色>_block_table','contrast':'同一目标：在桌上空手 vs 已被持有','implemented_outcome':'已持有目标也拒绝 requires_empty_gripper，不会再次抓取','source':controller+':144-147','legacy_ABC_refusals':0,'discriminative':'单独不足：固定held(target)=1也是幂等；状态预测与拒绝状态码不同','current_protocol_coverage':'0同目标重复抓取','construction':'先执行该已实现lift，再请求相同skill；不得描述为新抓取能力'}]
    csvsave(E/'results/minimum_condition_types.csv',needs);save(E/'results/minimum_condition_types.json',{'meaning':'six condition types; three core contrasts plus three semantics controls; not a sample-size or acceptance quota','conditions':needs,'future_protocol':'Keep model and controller frozen; determine independent-root quotas before any newly authorized collection. Preserve failures, UNKNOWN, and all96physical posthold controls. Do not promote symbolic guards into the learned WM planner.'})
    oldrows=list(csv.DictReader((E/'results/legacy_ABC960_audit.csv').open(encoding='utf-8-sig')))
    empties=[r for r in oldrows if r['skill']=='place_in_drawer'and r['status']=='refused'and r['reason']=='requires_exactly_one_held_block'];assert all(json.loads(r['before_held'])==[]for r in empties)
    extra={'empty_place_refusals_by_start_open_fact':dict(collections.Counter(r['before_7'][-1]for r in empties)),'failed_place_label_only_effect_table_agreements':sum(r['skill']=='place_in_drawer'and r['status']=='failed'and r['GT_before_fixed_table_explains_diagnostic_only']=='True'for r in oldrows)};save(E/'results/additional_scope_checks.json',extra)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11})
    models=[('persistence','保持当前','#89979f'),('action_only_seed1','仅动作','#d78038'),('copy_effect','复制＋AB效果（事后）','#3376b7'),('full_seed2','完整模型','#087f8c')];xx=np.arange(3);fig,ax=plt.subplots(figsize=(12,4.8))
    for j,(name,label,col)in enumerate(models):
        r=ev['results'][name]['C']['all'];yy=[r['all7_rate']*100,r['masks']['changed']['root_macro_strict_accuracy']*100,r['masks']['unchanged']['root_macro_strict_accuracy']*100];pos=xx+(j-1.5)*.19;ax.bar(pos,yy,.18,color=col,label=label)
        for x,y in zip(pos,yy):ax.text(x,y+2,f'{y:.1f}',ha='center',fontsize=9)
    ax.set_xticks(xx,['全7项同时正确（24对）','变化事实（30项）','不变事实（138项）']);ax.set_ylim(0,112);ax.set_ylabel('C严格正确率（%）');ax.set_title('同一C开发集：简单复制＋效果覆盖与完整模型并列');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.25),ncol=4,frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(E/'figures/C_copy_effect_comparison.png',dpi=160);plt.close(fig)
    def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</table></div>'
    report=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>复制与动作效果：事后诊断</title><style>body{max-width:1100px;margin:32px auto;padding:0 24px;background:#f7fafb;color:#153446;font:16px/1.7 "Noto Sans CJK SC",sans-serif}h1{font-size:29px}h2{font-size:21px;margin-top:32px}.note{padding:16px 20px;background:#e5f0f5;border-left:4px solid #3376b7}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{padding:9px;border-bottom:1px solid #d7e1e6;text-align:left}th{background:#e4eef2}.scroll{overflow:auto}img{max-width:100%}figcaption,.small{font-size:13px;color:#496573}a{color:#146e94}code{font-size:13px}</style><body><h1>复制当前事实＋A/B动作效果：事后诊断</h1><p class="small">单独补充分析 · 原预注册结论与全部旧文件保持不变 · 无训练、无采集、无D、无控制器修改</p><div class="note"><b>结论：</b>新简单对照在C同样24/24，现有成功链不能区分完整模型与固定动作效果表。<b>下一步选“先补当前时序协议下的状态依赖/不变样本”</b>，模型保持冻结。依据是强基线并列和覆盖缺口，原“严格超过100%”门槛不作为后续研究的永久停止理由。</div><h2>规则仅从A/B的96对归纳</h2><p>先复制当前冻结读出的全部事实，仅覆盖A/B中发生变化、且后继取值在该动作族全部24对上一致的谓词角色。未导入控制器的前提或效果表。抓取目标来自动作声明；放置没有声明颜色，只能用当前读出的唯一held绑定，不能用根任务颜色或后继真值。</p>']
    rule_rows=[]
    for fam,g in rules.items():
        for r in g['effects']:rule_rows.append([fam,r['slot'],r['set_to'],str(r['after_support'])+'/'+str(r['total_AB_pairs']),r['observed_transitions']])
    report.append(table(['动作族','覆盖事实','覆盖值','AB后继一致支持','AB观察到的变化'],rule_rows));report.append('<p>复制事实中的UNKNOWN保留。放置时只有三个held均已知且恰好一个为真才绑定目标；空手、冲突或未知时，所有可能受该规则影响的6个held/in事实为UNKNOWN，抽屉状态照抄。未知动作全7项UNKNOWN；重新应用多个held为真的原冲突规则。AB全成功不足以识别必要执行前提，因此这里除目标绑定外不做适用性门控：显式抓取及开关抽屉仍应用已学效果，即使真实适用性尚不确定。这一限制由后面的状态依赖挑战检验；不能把它称为已学会拒绝语义。</p><h2>相同C配对的比较</h2>')
    rows=[]
    for name,label in [('full_seed2','完整模型（3种子相同）'),('copy_effect','复制＋AB效果'),('action_only_seed1','仅动作（3种子相同）'),('persistence','保持当前')]:
        r=ev['results'][name]['C']['all'];rows.append([label,f"{r['all7_known_correct']}/24",f"{r['masks']['changed']['strict_correct']}/30",f"{r['masks']['unchanged']['strict_correct']}/138",r['masks']['all7']['UNKNOWN'],f"{r['all4_correct_roots']}/6"])
    report.append(table(['方法','全7项正确','变化正确','不变正确','UNKNOWN事实数','根内4对均正确'],rows));report.append('<figure><img src="figures/C_copy_effect_comparison.png"><figcaption>横轴为整状态、变化与不变三种评分条件；纵轴为严格正确率，UNKNOWN计错。蓝色为本次事后对照，青色为完整模型，橙色仅动作、灰色保持当前。完整模型三个种子相同；柱图显示代表种子。C为开发集，不是新的独立验收。</figcaption></figure>')
    rr=list(csv.DictReader((E/'results/all_root_metrics.csv').open(encoding='utf-8-sig')));rr=[r for r in rr if r['set']=='C'and r['model']=='copy_effect'];report.append(table(['C根','新对照全7项','变化','不变','UNKNOWN'],[[r['root_id'],r['all7_known_correct']+'/4',r['changed_correct']+'/'+r['changed_facts'],r['unchanged_correct']+'/'+r['unchanged_facts'],r['UNKNOWN']]for r in rr]));report.append('<p>C tune与calibration各12/12。AB为95/96：原有“误报握持粉”的当前读出被复制，再加上抓取蓝的效果，产生2个held UNKNOWN；该例保留。C没有当前读出错误样本，不能据此证明修复能力。以上预测只用当前读出；下述标签结构审计与之分开。</p><h2>当前数据与旧日志分别证明什么</h2><p>当前120对全为成功，无整状态不变调用。按同动作及绑定目标分为8组，每组只有一种相对变化模式；开关抽屉虽有不同初始内容，其差别可完全通过复制保留。标签层面120/120转移均能由同一固定效果表解释，尚无“同动作/目标因前提不同而不应发生该效果”的区分样本。</p>')
    report.append(table(['旧ABC日志中的拒绝条件','条数','与固定效果表的区分力'],[['持物时关抽屉',80,'强：抽屉应保持打开，固定close会错误关闭'],['持有另一物体时抓取',16,'强：应保留原held，不能新增目标held'],['空手放置',41,'检验不变与未知目标；本次对照输出UNKNOWN'],['已打开再打开',24,'单独不足：把open置1同样得到幂等状态']]))
    report.append('<p>只读审计旧ABC共960条/80根：782成功、161拒绝、17失败；161条拒绝均有同复位根、同声明动作的成功记录可对照，但先前执行历史不同，不能称为纯单因素干预。16条失败放置在这7项标签上仍符合固定效果，说明“执行失败”标签不等于7谓词错误。旧收集器的历史可能补帧、拒绝耗时为零且无96步后停留；这些日志只用于设计依据，未参与新规则归纳或模型评分，不能混入当前评估。</p><h2>最少需要补齐的条件类型</h2>')
    report.append(table(['用途','相同动作下的条件对照','现有控制器语义','旧ABC拒绝证据'],[[r['priority'],r['contrast'],r['implemented_outcome'],r['legacy_ABC_refusals']]for r in needs]));report.append('<p>前三项是能使固定成功效果表出错的核心对照；后三项用于不变/拒绝语义自检，不能单独证明状态推理。持有同一目标再抓取会被拒绝，不存在“再次抓取成功”的新能力。持物且抽屉关闭可由关闭初始状态下执行既有lift构造，无需新增控制器动作；源码支持不等于已采集成功。这里列的是结构覆盖下限，不是统计验收样本量或缩小原计划的配额。</p><p><b>唯一下一步：</b>先冻结上述状态依赖/不变挑战的根级配额与评分，再补齐符合原2.1秒历史及96步真实后停留的样本；模型继续冻结。拒绝后的漂移或掉落必须实测。这个步骤可检验是否超越固定成功效果及是否保留状态；常规成功链独立验证目前只能证明同类分布复现。即使新挑战通过，也仍不等于复杂世界推理或多步动力学已成立。</p><p>潜空间风险保持：完整代表seed2在C的对象MAE=0.1422，场景MAE=0.0529，33.0%预测坐标超出原GRU[-1,1]；seed1对象MAE约1.50。符号正确不能替代潜空间正确。当前没有1/3/5步滚动或闭环验证。</p><h2>本地详细证据</h2><ul>')
    for path,label in [('configs/diagnostic_protocol.json','事后诊断范围与未知规则'),('configs/rules_frozen.json','C评分前冻结的AB归纳规则'),('results/all_metrics.csv','所有方法、集合、事实掩码及当前读出错误子集'),('results/all_root_metrics.csv','各根完整结果'),('results/all_copy_effect_pair_facts.csv','120对×7事实逐项输出'),('results/current120_state_audit.csv','当前120对状态依赖审计'),('results/current120_action_target_groups.csv','动作/目标分组覆盖'),('results/legacy_ABC960_audit.csv','旧ABC960条审计，含原文件路径'),('results/legacy_same_root_action_contrasts.csv','161条同根同动作成功/拒绝对照'),('results/minimum_condition_types.csv','最少条件类型与控制器源码位置'),('results/validation.json','独立验证与UNKNOWN接口检查'),('delivery_manifest.json','交付SHA256清单')]:report.append(f'<li><a href="{path}">{label}</a></li>')
    report.append('</ul><p class="small">所有旧文件与原预注册决定已逐项核对未变；此对照只用于事后诊断，不接入学习世界模型或TS/LTL搜索。</p></body></html>');(E/'report.html').write_text('\n'.join(report))
    save(E/'results/next_step_decision.json',{'created_utc':now(),'analysis_type':'post_hoc','original_preregistered_decision_preserved':True,'next_step':'first_add_state_dependent_and_nochange_samples_under_existing_time_contract_with_models_frozen','reason':'new_copy_effect_baseline_ties_full_model_on_C_and_current120_contains_zero_effect_modulation_contrasts','not_based_on_permanent_stop_from_saturated_changed_metric':True,'ordinary_independent_success_validation_now':'would test replication on similar successful trajectories; would not distinguish simple effects from state-dependent outcome prediction','new_collection_executed':False,'new_training_executed':False,'D_used':False,'latent_fidelity_and_multistep_unresolved':True})
    print(json.dumps({'report':str(E/'report.html'),'validation':val,'legacy_empty_place_breakdown':extra},ensure_ascii=False))
if __name__=='__main__':main()
