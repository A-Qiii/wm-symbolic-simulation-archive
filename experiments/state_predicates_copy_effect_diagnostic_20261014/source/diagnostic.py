"""Post-hoc AB-only effect induction; CPU evaluation/audit, no model training."""
from pathlib import Path
import argparse,json,hashlib,datetime,csv,collections,inspect
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
OLD=P/'experiments/state_predicates_one_step_20261014'
E=P/'experiments/state_predicates_copy_effect_diagnostic_20261014'
LONG=P/'data/predicate_long_pairs_20261014'
COLORS=['red','blue','pink']
NAMES=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)']
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def csvsave(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows)
def family(r):return 'lift'if r['skill'].startswith('lift_')else r['skill']
def decode(prob):
    prob=np.asarray(prob); known=np.isfinite(prob); hat=prob>=.5
    conflict=hat[...,:3].sum(-1)>1;known[...,:3]&=~(conflict[...,None]&hat[...,:3]);return hat,known
def target(r,before,known):
    if family(r)=='lift':
        c=r['declared_arguments'].get('color');return COLORS.index(c)if c in COLORS else None
    if family(r)=='place_in_drawer':
        jj=np.flatnonzero(before[:3]);return int(jj[0])if known[:3].all()and len(jj)==1 else None
    return None
def slot(col,t):
    if col==6:return 'open(drawer)'
    if t is not None and col%3==t:return 'held(target)'if col<3 else'in(target,drawer)'
    return 'absolute:'+NAMES[col]
def column(s,t):
    if s=='open(drawer)':return 6
    if s=='held(target)':return t
    if s=='in(target,drawer)':return t+3 if t is not None else None
    return NAMES.index(s.split(':',1)[1])
def fit_rules(train_rows,before,after,valid_before,valid_after):
    """All inputs are AB subset arrays. C outcomes are not accepted here."""
    assert len(train_rows)==96 and all(r['split']=='train'and r['environment']in ['A','B']for r in train_rows)
    groups={}; evidence=[]
    for fam in sorted(set(map(family,train_rows))):
        ii=[i for i,r in enumerate(train_rows)if family(r)==fam];ts={i:target(train_rows[i],before[i],valid_before[i])for i in ii}
        if fam in ['lift','place_in_drawer']:assert all(ts[i]is not None for i in ii)
        changes=set()
        for i in ii:
            for j in np.flatnonzero(valid_before[i]&valid_after[i]&(before[i]!=after[i])):changes.add(slot(int(j),ts[i]))
        effects=[]
        for sl in sorted(changes):
            cols={i:column(sl,ts[i])for i in ii};seen=[i for i in ii if valid_after[i,cols[i]]];values={bool(after[i,cols[i]])for i in seen};val=int(next(iter(values)))if len(values)==1 and len(seen)==len(ii)else None
            transitions=collections.Counter(f'{int(before[i,cols[i]])}->{int(after[i,cols[i]])}'for i in ii if valid_before[i,cols[i]]and valid_after[i,cols[i]])
            effects.append({'slot':sl,'set_to':val,'after_support':len(seen),'total_AB_pairs':len(ii),'observed_transitions':dict(transitions)})
        groups[fam]={'AB_pairs':len(ii),'AB_roots':len({train_rows[i]['root_id']for i in ii}),'binding':'declared_lift_colour'if fam=='lift'else'unique_current_held_from_readout'if fam=='place_in_drawer'else'drawer','effects':effects,'non_effect_facts':'copy current ternary readout; no overwrite'}
        evidence +=[{'family':fam,'pair_id':train_rows[i]['pair_id'],'resolved_training_target_from_AB_labels':COLORS[ts[i]]if ts[i]is not None else None,'before':before[i].astype(int).tolist(),'after':after[i].astype(int).tolist()}for i in ii]
    return groups,evidence
def predict(current_prob,declared_rows,rules):
    """Only current readout, declared actions and already frozen AB rules."""
    h,k=decode(current_prob);out=np.asarray(current_prob).copy();ok=k.copy();audit=[]
    for i,r in enumerate(declared_rows):
        fam=family(r);t=target(r,h[i],k[i]);reason=None;over=[]
        if fam not in rules:out[i]=np.nan;ok[i]=False;reason='unseen_action_family'
        else:
            group=rules[fam]
            for eff in group['effects']:
                sl=eff['slot'];col=column(sl,t)
                if col is None:
                    cols=list(range(3))if sl=='held(target)'else list(range(3,6));out[i,cols]=np.nan;ok[i,cols]=False;reason='target_binding_unknown_or_unseen_no_unique_held';over+=cols
                elif eff['set_to']is None:out[i,col]=np.nan;ok[i,col]=False;reason='AB_effect_not_unanimous';over.append(col)
                else:out[i,col]=eff['set_to'];ok[i,col]=True;over.append(col)
        # Preserve pre-existing UNKNOWN and apply the unchanged multiple-held rule.
        hh,newk=decode(out[i]);ok[i]&=newk
        audit.append({'pair_id':r['pair_id'],'family':fam,'resolved_target_from_current_public_input':COLORS[t]if t is not None else None,'overwritten_columns':sorted(set(over)),'abstention_reason':reason})
    return out,ok,audit
def signature(values):return ''.join('1'if v else'0'for v in values)
def main_fit():
    assert not(E/'configs/rules_frozen.json').exists()
    pairs=read(OLD/'data/public/pairs.json');ii=[i for i,r in enumerate(pairs)if r['split']=='train'];tr=[pairs[i]for i in ii]
    with np.load(OLD/'data/_private/training_targets_and_audit.npz')as z:
        b=z['before_y'][ii].astype(bool);a=z['after_y'][ii].astype(bool);bv=z['before_valid'][ii].astype(bool);av=z['after_valid'][ii].astype(bool)
    np.savez_compressed(E/'data/AB_rule_fit_only.npz',before=b,after=a,before_valid=bv,after_valid=av)
    save(E/'data/AB_rule_fit_pairs.json',tr);rules,evidence=fit_rules(tr,b,a,bv,av)
    config={'created_utc':now(),'status':'POST_HOC_DIAGNOSTIC_NOT_AMENDED_PREREGISTRATION','original_result_preserved':True,'no_neural_training':True,'no_collection':True,'no_D':True,'no_controller_changes':True,'fit':'AB96 paired ground-truth labels only; C labels never an argument to fit_rules','input':'current frozen readout probabilities plus declared skill/arguments only','effect_induction':'Copy all current facts; collect AB changed predicate roles per action family, overwrite only roles whose post-value is unanimous across all AB examples in that family. Other facts never reset. Conflicting/unsupported effects become UNKNOWN.','target_binding':'lift: explicit declaration; place: all three current held facts known and exactly one true; never root mission colour, actual successor or executor outcome','preconditions':'No execution applicability rule is imported or inferred from controller. Learned assignments extrapolate the observed successful AB action effects. Only target binding is a prerequisite to role instantiation. This is a simple effect table, not learned failure/refusal dynamics.','unknown':'Keep copied UNKNOWN unless an AB-supported assignment determines that fact. Unresolved place target (including known empty hand) makes all possible held/in target effect facts UNKNOWN; open is copied. Unseen action makes all seven UNKNOWN. Reapply multiple-positive-held UNKNOWN rule. No zero-time self-loop is assumed.','evaluation':'All original120 pairs retained; C all24/tune12/cal12 reported. UNKNOWN counts as wrong in strict metrics. This design is post hoc because earlier C results were already known.','AB_data_sha256':sha(E/'data/AB_rule_fit_only.npz'),'AB_pairs_sha256':sha(E/'data/AB_rule_fit_pairs.json'),'source_sha256':sha(E/'source/diagnostic.py'),'original_manifest_sha256':sha(OLD/'delivery_manifest.json'),'original_decision_sha256':sha(OLD/'results/decision.json'),'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py')}
    save(E/'configs/diagnostic_protocol.json',config);save(E/'configs/rules_frozen.json',{'created_utc':now(),'rules':rules,'AB_fit_pair_count':96,'AB_fit_only':True,'protocol_sha256':sha(E/'configs/diagnostic_protocol.json')});save(E/'results/AB_induction_evidence.json',evidence)
    for p in [E/'configs/diagnostic_protocol.json',E/'configs/rules_frozen.json']:p.chmod(0o444)
    print(json.dumps(rules,ensure_ascii=False))

def metric(prob,known,y,v,idx,roots,masks):
    out={};hat=prob>=.5;correct=known&(hat==y);full=(correct&v).all(1);rr=sorted({roots[i]for i in idx})
    out.update(pairs=len(idx),roots=len(rr),all7_known_correct=int(full[idx].sum()),all7_rate=float(full[idx].mean())if len(idx)else None,all4_correct_roots=sum(bool(full[[i for i in idx if roots[i]==r]].all())for r in rr),masks={})
    for key,mask in masks.items():
        vv=v[idx]&mask[idx];n=int(vv.sum());good=int((correct[idx]&vv).sum());u=int((~known[idx]&vv).sum());rootrates=[]
        for r in rr:
            ri=[i for i in idx if roots[i]==r];rv=v[ri]&mask[ri]
            if rv.sum():rootrates.append(float((correct[ri]&rv).sum()/rv.sum()))
        out['masks'][key]={'facts':n,'strict_correct':good,'UNKNOWN':u,'strict_accuracy':good/n if n else None,'root_macro_strict_accuracy':float(np.mean(rootrates))if rootrates else None}
    return out
def main_evaluate():
    frozen=read(E/'configs/rules_frozen.json');protocol=read(E/'configs/diagnostic_protocol.json');assert sha(E/'source/diagnostic.py')==protocol['source_sha256'];rules=frozen['rules'];pairs=read(OLD/'data/public/pairs.json')
    targetarrays=dict(np.load(OLD/'data/_private/training_targets_and_audit.npz'));savedpred=dict(np.load(OLD/'results/predictions.npz'));p0=targetarrays['observed_current_probabilities'];b=targetarrays['before_y'].astype(bool);y=targetarrays['after_y'].astype(bool);v=targetarrays['after_valid'];bv=targetarrays['before_valid']
    pred,known,audit=predict(p0,pairs,rules);np.savez_compressed(E/'results/copy_effect_predictions.npz',probabilities=pred,known=known);save(E/'results/prediction_input_audit.json',audit)
    metas=read(OLD/'data/_private/evaluation_metadata.json');targetmask=np.zeros_like(v)
    for i,r in enumerate(metas):
        if r['skill']in ['open_drawer','close_drawer']:targetmask[i,6]=True
        else:
            j=COLORS.index(r['declared_arguments']['color']if family(r)=='lift'else r['intended_root_colour_evaluator_only']);targetmask[i,[j,j+3]]=True
    masks={'all7':np.ones_like(v),'changed':(b!=y)&bv,'unchanged':(b==y)&bv,'target_entity':targetmask,'other_entity':~targetmask}
    sets={'AB':np.array([r['split']=='train'for r in pairs]),'C':np.array([r['split']=='dev'for r in pairs]),'C_tune':np.array([r['dev_role']=='tune'for r in pairs]),'C_calibration':np.array([r['dev_role']=='calibration'for r in pairs])}
    pp={'copy_effect':pred,**savedpred};kk={n:decode(p)[1]for n,p in pp.items()};kk['copy_effect']=known;results={};rows=[];rootrows=[];details=[];roots=[r['root_id']for r in pairs]
    h0,k0=decode(p0);currentgood=(k0&(h0==b)&bv).all(1)
    for name,p in pp.items():
        results[name]={}
        for sn,sel in sets.items():
            results[name][sn]={}
            for sub,submask in [('all',np.ones(120,bool)),('current_readout_wrong_or_UNKNOWN',~currentgood)]:
                q=metric(p,kk[name],y,v,np.flatnonzero(sel&submask),roots,masks);results[name][sn][sub]=q
                for mask,m in q['masks'].items():rows.append({'model':name,'set':sn,'subset':sub,**{k:q[k]for k in ['pairs','roots','all7_known_correct','all7_rate','all4_correct_roots']},'mask':mask,**m})
            for rid in sorted({roots[i]for i in np.flatnonzero(sel)}):
                ri=[i for i in np.flatnonzero(sel)if roots[i]==rid];q=metric(p,kk[name],y,v,ri,roots,masks);rootrows.append({'model':name,'set':sn,'root_id':rid,'pairs':len(ri),'all7_known_correct':q['all7_known_correct'],'changed_correct':q['masks']['changed']['strict_correct'],'changed_facts':q['masks']['changed']['facts'],'unchanged_correct':q['masks']['unchanged']['strict_correct'],'unchanged_facts':q['masks']['unchanged']['facts'],'UNKNOWN':q['masks']['all7']['UNKNOWN']})
    for i,r in enumerate(pairs):
        for j in range(7):details.append({'pair_id':r['pair_id'],'root_id':r['root_id'],'split':r['split'],'dev_role':r['dev_role'],'skill':r['skill'],'predicate':NAMES[j],'current_probability':float(p0[i,j]),'predicted_probability':float(pred[i,j])if np.isfinite(pred[i,j])else None,'predicted_known':bool(known[i,j]),'predicted_value':int(pred[i,j]>=.5)if known[i,j]else'UNKNOWN','after_truth':int(y[i,j]),'valid':bool(v[i,j]),'changed':bool(masks['changed'][i,j]),'strict_correct':bool(known[i,j]and(pred[i,j]>=.5)==y[i,j])})
    save(E/'results/metrics.json',{'created_utc':now(),'post_hoc':True,'rules_sha256':sha(E/'configs/rules_frozen.json'),'all120_pairs_retained':True,'results':results});csvsave(E/'results/all_metrics.csv',rows);csvsave(E/'results/all_root_metrics.csv',rootrows);csvsave(E/'results/all_copy_effect_pair_facts.csv',details)
    # Label-only structural audit: never substitute this for observation-conditioned scores.
    structural,sk,_=predict(b.astype(float),pairs,rules);structuralcorrect=(sk&((structural>=.5)==y)&v).all(1)
    auditrows=[];grouprows=[]
    for i,r in enumerate(pairs):
        t=target(r,b[i],bv[i]);key=r['skill']+'|'+(COLORS[t]if t is not None else'drawer');delta=[f'{slot(j,t)}:{int(b[i,j])}->{int(y[i,j])}'for j in np.flatnonzero(masks['changed'][i])]
        auditrows.append({'pair_id':r['pair_id'],'root_id':r['root_id'],'split':r['split'],'dev_role':r['dev_role'],'skill':r['skill'],'declared_target':r['declared_arguments'].get('color'),'audit_resolved_target_from_before_GT':COLORS[t]if t is not None else'drawer','action_resolved_target_key':key,'before_7':signature(b[i]),'after_7':signature(y[i]),'relative_delta':delta,'whole_state_nochange':bool(not masks['changed'][i].any()),'label_only_fixed_effect_table_explains':bool(structuralcorrect[i]),'primitive_status':None})
    lp=read(LONG/'data_manifest.json');priv={r['pair_id']:r for r in lp['private_pairs']}
    for r in auditrows:r['primitive_status']=priv[r['pair_id']]['primitive']['status'];r['macro_controls']=priv[r['pair_id']]['macro_controls']
    for key in sorted({r['action_resolved_target_key']for r in auditrows}):
        selected=[r for r in auditrows if r['action_resolved_target_key']==key]
        grouprows.append({'action_target':key,'pairs':len(selected),'AB_pairs':sum(r['split']=='train'for r in selected),'C_pairs':sum(r['split']=='dev'for r in selected),'distinct_before_7':len({r['before_7']for r in selected}),'distinct_after_7':len({r['after_7']for r in selected}),'distinct_relative_delta':len({tuple(r['relative_delta'])for r in selected}),'statuses':dict(collections.Counter(r['primitive_status']for r in selected)),'nochange_pairs':sum(r['whole_state_nochange']for r in selected),'fixed_effect_explained':sum(r['label_only_fixed_effect_table_explains']for r in selected)})
    csvsave(E/'results/current120_state_audit.csv',auditrows);csvsave(E/'results/current120_action_target_groups.csv',grouprows)
    save(E/'results/current120_audit_summary.json',{'groups':grouprows,'all120_fixed_effect_explained_from_GT_before':int(structuralcorrect.sum()),'not_a_predictive_GT_input_result':True,'whole_state_nochange_pairs':sum(r['whole_state_nochange']for r in auditrows),'statuses':dict(collections.Counter(r['primitive_status']for r in auditrows)),'groups_with_different_effect_pattern_under_same_action_resolved_target':sum(r['distinct_relative_delta']>1 for r in grouprows)})
    # Existing ABC development evidence only. Never open a D root or a mixed split manifest.
    oldrows=[];sources=[];controller_hash=sha(P/'src/wm_symbolic/physical_oracle.py')
    for d in sorted((P/'data/core1200_v1/episodes').iterdir()):
        if not(d.is_dir()and d.name.startswith(('A_','B_','C_'))):continue
        records=read(d/'committed_rows.json');sources.append(str(d/'committed_rows.json'))
        assert all(r['environment']in ['A','B','C']and r['split']in ['train','dev']for r in records)
        for r in records:
            folder=Path(r['obs_before']).parent;rp=folder/'result_private.json';yp=folder/'labels_ownership_v2_private.json';rr=read(rp);lab=read(yp);sources +=[str(rp),str(yp)];assert rr['controller_sha256']==controller_hash
            def convert(a):
                cat=np.array(a['categorical'][:4]);cv=np.array(a['categorical_valid'][:4],bool);return np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2],np.r_[cv[:3],cv[:3],cv[3]]
            yy0,vv0=convert(lab['before']);yy1,vv1=convert(lab['after']);tt=COLORS.index(r['skill'].split('_')[1])if r['skill'].startswith('lift_')else None
            held=[COLORS[j]for j in np.flatnonzero(yy0[:3])];decl={'color':COLORS[tt]}if tt is not None else{};public={'skill':r['skill'],'declared_arguments':decl,'pair_id':r['episode_id']+'__'+str(r['step'])}
            op,ok,_=predict(yy0[None].astype(float),[public],rules);expl=bool((ok[0]&((op[0]>=.5)==yy1)&vv1).all());color=COLORS[tt]if tt is not None else held[0]if len(held)==1 and r['skill']=='place_in_drawer'else'drawer'if r['skill']in ['open_drawer','close_drawer']else'unresolved'
            oldrows.append({'pair_id':public['pair_id'],'root_id':r['root_id'],'episode_id':r['episode_id'],'environment':r['environment'],'split':r['split'],'skill':r['skill'],'declared_target':decl.get('color'),'before_held':held,'before_drawer_m':lab['before']['continuous']['drawer_m'],'before_7':signature(yy0),'after_7':signature(yy1),'all7_valid_both':bool(vv0.all()and vv1.all()),'status':rr['status'],'reason':rr.get('reason'),'controls':rr['controls'],'before_after_identical_7':bool(np.array_equal(yy0,yy1)),'zero_action_state_identical':rr.get('zero_action_state_identical',False),'target_already_held':bool(tt is not None and yy0[tt]),'other_object_held':bool(tt is not None and held and not yy0[tt]),'target_on_table_from_label':bool(tt is not None and int(lab['before']['categorical'][tt])in [0,1,2]),'resolved_audit_target':color,'GT_before_fixed_table_explains_diagnostic_only':expl,'GT_before_table_UNKNOWN_count':int((~ok[0]).sum()),'result_source':str(rp),'labels_source':str(yp),'same_controller_hash':True,'not_same_2p1s_plus96_macro_protocol':True})
    csvsave(E/'results/legacy_ABC960_audit.csv',oldrows)
    reasonrows=[]
    for key in sorted({(r['skill'],r['status'],r['reason']or'')for r in oldrows}):
        ss=[r for r in oldrows if(r['skill'],r['status'],r['reason']or'')==key];reasonrows.append({'skill':key[0],'status':key[1],'reason':key[2],'A':sum(r['environment']=='A'for r in ss),'B':sum(r['environment']=='B'for r in ss),'C':sum(r['environment']=='C'for r in ss),'total':len(ss),'distinct_roots':len({r['root_id']for r in ss}),'all7_unchanged':sum(r['before_after_identical_7']for r in ss),'label_only_fixed_table_explained':sum(r['GT_before_fixed_table_explains_diagnostic_only']for r in ss),'first_source':ss[0]['result_source']})
    csvsave(E/'results/legacy_ABC_reason_counts.csv',reasonrows)
    # Match evidence within the same reset root and same declared action/target.
    contrasts=[]
    for key in sorted({(r['root_id'],r['skill'],r['declared_target']or'')for r in oldrows}):
        ss=[r for r in oldrows if(r['root_id'],r['skill'],r['declared_target']or'')==key]
        successes=[r for r in ss if r['status']=='success']
        for neg in [r for r in ss if r['status']=='refused']:
            if not successes:continue
            pos=successes[0];contrasts.append({'root_id':key[0],'skill':key[1],'declared_target':key[2],'success_pair':pos['pair_id'],'refusal_pair':neg['pair_id'],'success_before':pos['before_7'],'success_after':pos['after_7'],'refusal_before':neg['before_7'],'refusal_after':neg['after_7'],'refusal_reason':neg['reason'],'states_different':pos['before_7']!=neg['before_7'],'success_source':pos['result_source'],'refusal_source':neg['result_source'],'scope':'same reset root with different preceding physical history; not isolated single-factor intervention'})
    csvsave(E/'results/legacy_same_root_action_contrasts.csv',contrasts)
    save(E/'results/legacy_ABC_summary.json',{'rows':len(oldrows),'roots':len({r['root_id']for r in oldrows}),'by_environment':dict(collections.Counter(r['environment']for r in oldrows)),'statuses':dict(collections.Counter(r['status']for r in oldrows)),'refusals':dict(collections.Counter(r['reason']for r in oldrows if r['status']=='refused')),'same_root_action_success_refusal_contrasts':len(contrasts),'contrast_reasons':dict(collections.Counter(r['refusal_reason']for r in contrasts)),'held_same_target_lift_refusals':sum(r['status']=='refused'and r['target_already_held']for r in oldrows),'held_other_target_lift_refusals':sum(r['status']=='refused'and r['other_object_held']for r in oldrows),'opened_D_paths':0,'used_for_rule_fit':False,'used_for_model_scoring':False,'can_merge_into_current_protocol':False,'mismatch':'old collector uses at most four recent frames and may pad; rejected primitive has zero controls; no96posthold. It supplies evidence of executable/refused conditions, not current-protocol endpoint labels.'})
    save(E/'results/read_only_legacy_sources.json',[{'path':p,'sha256':sha(p)}for p in sorted(set(sources))])
    print(json.dumps({'C_copy_effect':results['copy_effect']['C']['all'],'AB_copy_effect':results['copy_effect']['AB']['all'],'current_audit':read(E/'results/current120_audit_summary.json'),'legacy_summary':read(E/'results/legacy_ABC_summary.json')},ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','evaluate'],required=True);args=ap.parse_args();main_fit()if args.phase=='fit'else main_evaluate()
