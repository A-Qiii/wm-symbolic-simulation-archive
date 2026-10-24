"""Verify the completed fixed-budget experiment and seal its local deliverables.
No training, physics, threshold tuning, or candidate promotion occurs here.
"""
from common import *
import os, re, shutil, subprocess, sys

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_manifest(root, manifest):
    entries=read(manifest)['files']
    for item in entries:
        p=Path(root)/item['path']
        assert p.is_file(),str(p)
        assert p.stat().st_size==item['bytes'],str(p)
        assert digest(p)==item['sha256'],str(p)
    return {'manifest':str(manifest),'manifest_sha256':digest(manifest),'verified_files':len(entries),'verified_bytes':sum(x['bytes']for x in entries)}

def entries(paths, root):
    return [{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)}for p in sorted(paths)]

# This hook blocks project label/private files during actual CLI execution, and
# records project reads. The parent verifier may read evaluation metadata.
WRAPPER=r'''
import json, pathlib, runpy, sys
project='/home/robot/research/projects/wm_symbolic/'
reads=set()
def audit(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    p=str(pathlib.Path(args[0]).absolute())
    mode=args[1];flags=args[2]
    read_op=not (isinstance(mode,str)and ('w'in mode or 'a'in mode))
    if p.startswith(project)and read_op:
        path=pathlib.Path(p)
        if 'private'in path.parts or '_private'in path.parts or path.name in ['targets.npz','windows.json'] or 'targets_and_GT'in path.name:
            raise RuntimeError('GT/label read blocked during query: '+p)
        reads.add(p)
sys.addaudithook(audit)
script=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,str(pathlib.Path(script).parent))
runpy.run_path(script,run_name='__main__')
print('READ_AUDIT_JSON '+json.dumps(sorted(reads)),file=sys.stderr)
'''

def main():
    assert not(R/'delivery_manifest.json').exists(),'Existing sealed delivery: use verify_delivery.py, do not overwrite.'
    frozen=read(E/'configs/training_frozen.json')
    for path,h in frozen['sha256'].items():assert digest(path)==h,path
    provenance=read(E/'data/cache_provenance.json');assert len(provenance)==942
    for row in provenance:assert row['public_RGB_only'] and digest(row['source'])==row['sha256'],row['source']
    meta=read(BASE/'data/metadata.json');index={r['window_id']:i for i,r in enumerate(meta)}
    roots={s:{r['root_id']for r in meta if r['split']==s}for s in {r['split']for r in meta}}
    for a in roots:
        for b in roots:
            if a!=b:assert not(roots[a]&roots[b])
    schedule=np.load(E/'data/train_schedule.npy');assert schedule.shape==(1200,32)
    assert all(meta[int(i)]['split']=='train'and meta[int(i)]['environment']in ['A','B']for i in schedule.ravel())
    train={a:read(E/'results'/a/'training.json')for a in ['pooled','spatial']}
    for a,t in train.items():
        assert t['updates']==1200 and t['seed']==0 and t['trainable_parameters']==247569
        assert all(t[k]for k in ['frozen_encoder_unchanged','alias_query_permutation_checks','initial_function_same','AB_only_training_indices_checked','no_D'])
        assert np.array_equal(schedule,np.load(E/'logs'/a/'train_indices.npy'))
        assert [c['update']for c in t['checkpoints']]==[300,600,900,1200]
        for c in t['checkpoints']:assert digest(c['path'])==c['sha256']
        selected=min(t['checkpoints'],key=lambda x:(x['C_selection_loss'],x['update']))
        assert t['selected']==selected==read(E/'configs'/a/'reader_frozen.json')
    assert train['pooled']['initial_state_sha256']==train['spatial']['initial_state_sha256']
    assert read(E/'configs/candidate_decision.json')['candidates']==[]
    print('FROZEN_TRAINING_AND_CACHE_CHECKS_PASSED',flush=True)
    numerical=read(E/'results/numerical_validation.json');assert all(r['decoded_fact_changes_count']==0 for r in numerical.values())
    outputs=[];maxerr=0.;max_cpuerr=0.
    for c in read(R/'tables/query_commands.json'):
        cmd=[sys.executable,'-c',WRAPPER,str(E/'source/query.py'),'--variant',c['variant'],'--window-id',c['window_id'],'--query',c['query'],'--goal-object',c['goal_object']]
        proc=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        result=json.loads(proc.stdout);audit=[x for x in proc.stderr.splitlines()if x.startswith('READ_AUDIT_JSON ')];assert len(audit)==1
        assert not result['GT_loaded_at_inference'] and result['planning_status'].startswith('not_executed')
        saved=np.load(E/'results'/c['variant']/'probabilities.npy')[index[c['window_id']]]
        current=np.array([result['all_probabilities'][f]for f in FACTS]);err=float(np.abs(saved-current).max());maxerr=max(maxerr,err)
        cpu_saved=np.load(E/'results'/c['variant']/'cpu_validation_probabilities.npy')[index[c['window_id']]];cpuerr=float(np.abs(cpu_saved-current).max());max_cpuerr=max(max_cpuerr,cpuerr)
        print('QUERY_NUMERICS',c['variant'],c['window_id'],c['query'],'CPU_batch_difference',cpuerr,'GPU_saved_difference',err,flush=True)
        assert cpuerr<1e-5
        from baseline_model import decode
        assert np.array_equal(decode(saved[None])[0],np.array([result['decoded_state'][f]for f in FACTS]))
        fact=c['query'].replace('obj_','');assert abs(result['probability']-result['all_probabilities'][fact])<1e-5
        expected=bool(result['decoded_state']['Closed(drawer)']==1 and result['decoded_state'][f"In({c['goal_object']},drawer)"]==1)
        assert expected==result['joint_goal_currently_satisfied']
        outputs.append({'command_arguments':c,'output':result,'max_probability_difference_from_saved_GPU':err,'max_probability_difference_from_CPU_batch':cpuerr,'project_reads':json.loads(audit[0][len('READ_AUDIT_JSON '):]),'GT_read_guard_active':True})
    assert len(outputs)==12
    normal=[r for r in outputs if r['command_arguments']['window_id'].endswith('empty_closed__t00127')]
    assert all(r['output']['joint_goal_currently_satisfied']==(r['command_arguments']['variant']=='spatial')for r in normal)
    save(E/'results/query_cli_verification.json',{'created_utc':now(),'queries_completed':len(outputs),'max_probability_difference_from_saved_GPU':maxerr,'max_probability_difference_from_CPU_batch':max_cpuerr,'CPU_GPU_full942_validation':numerical,'normal_closed_spatial_false_positive_retained':True,'details':outputs})
    print('12_ACTUAL_CPU_QUERIES_PASSED_WITH_GT_READ_BLOCKER',maxerr,flush=True)
    old=[]
    for name in ['state_semantics_v2_20261015','containment_diagnostic_20261015']:
        rp=P/'reports'/name
        old.append({'experiment':name,'delivery':verify_manifest(rp,rp/'delivery_manifest.json'),'source_data':verify_manifest(P,rp/'tables/source_data_manifest.json')})
        print('OLD_FILES_UNCHANGED',name,old[-1]['source_data']['verified_files'],flush=True)
    integrity={'created_utc':now(),'frozen_training_files_verified':len(frozen['sha256']),'cached_feature_sources_verified':len(provenance),'matching_AB_only_schedule':True,'disjoint_split_roots':{k:len(v)for k,v in roots.items()},'total_training_updates':2400,'trainable_parameters_each':247569,'selected_checkpoint_updates':{a:t['selected']['update']for a,t in train.items()},'query_cli_runs':12,'max_cpu_GPU_probability_difference_in_12_queries':maxerr,'max_single_CPU_batch_CPU_difference':max_cpuerr,'full942_CPU_GPU_validation':numerical,'prior_reports':old,'old_source_data_files_verified':sum(r['source_data']['verified_files']for r in old),'old_deliverable_files_verified':sum(r['delivery']['verified_files']for r in old),'new_candidate_promoted':False,'new_D':False,'new_planning':False}
    save(E/'results/delivery_checks.json',integrity)
    for name in ['query_cli_verification.json','delivery_checks.json','numerical_validation.json']:shutil.copy2(E/'results'/name,R/'tables'/name)
    report=R/'REPORT.md';report.write_text(report.read_text()+f"\n## 可复核交付记录\n\n实际运行12次CPU查询，5种关系均覆盖；推理时阻止读取项目标签／私有GT文件。与同CPU批量推理概率最大差为{max_cpuerr:.3g}，与保存GPU概率最大差为{maxerr:.3g}，离散判断一致，空间版错误共同目标也如实复现。首次封存以1e-5直接比较CPU与GPU概率而停止，失败日志和原脚本已保留。随后对两组全部942窗口复核，CPU与GPU最大概率差分别为{numerical['pooled']['max_absolute_difference']:.3g}和{numerical['spatial']['max_absolute_difference']:.3g}，九项离散事实均无变化；没有修改模型、0.5阈值或候选门槛。训练前冻结的{len(frozen['sha256'])}份文件和942个缓存来源哈希一致；两组1200批训练索引一致且只含AB。此前{integrity['old_source_data_files_verified']}份源码／数据文件、{integrity['old_deliverable_files_verified']}份已交付文件校验均未改变。详见[tables/delivery_checks.json](tables/delivery_checks.json)和[查询运行记录](tables/query_cli_verification.json)。\n")
    for folder in ['source','configs']:
        for src in sorted((E/folder).rglob('*')):
            if src.is_file()and '__pycache__'not in src.parts:
                target=R/'source'/src.relative_to(E);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    for page in [R/'README.md',R/'REPORT.md',R/'gallery.html']:
        content=page.read_text();links=re.findall(r'\]\(([^)]+)\)',content)+re.findall(r'(?:src|href)="([^"]+)"',content)
        for href in links:
            if href.startswith(('http:','https:','#')):continue
            assert (page.parent/href).exists(),(page,href)
    paths=[p for p in E.rglob('*')if p.is_file()and '__pycache__'not in p.parts]
    source=entries(paths,P);save(R/'tables/source_data_manifest.json',{'project':str(P),'files':source,'count':len(source),'bytes':sum(x['bytes']for x in source)})
    delivery=entries([p for p in R.rglob('*')if p.is_file()and p.name!='delivery_manifest.json'and '__pycache__'not in p.parts],R)
    save(R/'delivery_manifest.json',{'created_utc':now(),'files':delivery,'count':len(delivery),'bytes':sum(x['bytes']for x in delivery)})
    print(json.dumps({'sealed':True,'source_files':len(source),'delivery_files':len(delivery),'old_source_files_unchanged':integrity['old_source_data_files_verified'],'old_deliverables_unchanged':integrity['old_deliverable_files_verified'],'new_candidates':0}),flush=True)
if __name__=='__main__':main()
