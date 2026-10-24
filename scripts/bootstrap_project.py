from pathlib import Path
import json, subprocess, hashlib, shutil, os, time, urllib.request, zipfile

P = Path('/home/robot/research/projects/wm_symbolic')
OLD = Path('/home/robot/research/projects/robot_state')

def save(rel, obj):
    (P / rel).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def setup():
    protocol = {
        'guide_sha256': 'f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4',
        'environment': 'local_only', 'executor': 'HULC_ABC_D_frozen',
        'splits': {'train': {'environments': ['A','B'], 'roots':240, 'branches':3, 'length':6, 'transitions':4320},
                   'dev': {'environments':['C'], 'roots':30, 'branches':2, 'length':6, 'transitions':360},
                   'calibration': {'environments':['C'], 'roots':30, 'branches':2, 'length':6, 'transitions':360},
                   'test': {'environments':['D'], 'roots':80, 'branches':2, 'length':6, 'transitions':960}},
        'pilot': {'train_roots':60, 'branches':2, 'length':6, 'transitions':720, 'dev_roots':10,
                  'storage_probe_transitions':50, 'training_leakage_probe_transitions':128, 'updates':6000},
        'training': {'updates':30000, 'one_step_until':5000, 'three_step_until':15000, 'final_horizon':5,
                     'batch':8, 'accumulate':2, 'window':6, 'lr':0.0003, 'min_lr':0.00003,
                     'warmup':1000, 'weight_decay':0.0001, 'clip_grad':1.0, 'checkpoint_every':1000,
                     'seeds_functional':1, 'seeds_comparison':3},
        'model': {'vision':'dinov2_vits14_frozen', 'input':[224,224], 'patch':[16,16,384],
                  'roi':[2,2], 'transformer_layers':2, 'transformer_dim':256, 'heads':4,
                  'obs_dim':512, 'rssm_gru':512, 'rssm_categories':[32,32], 'unimix':0.01,
                  'shared_symbol_head':True, 'task_text_in_transition':False},
        'loss': {'obs':1.0, 'sym':1.0, 'kl':0.5, 'out':0.5, 'multi':0.5,
                 'kl_balance':[0.8,0.2], 'free_nats_total':1.0, 'class_weight_cap':5.0},
        'search': {'beam':32, 'horizon':8, 'successor_query_budget':5000, 'samples_per_sequence':8,
                   'replan_after_every_skill':True, 'merge_by_symbols_only':False,
                   'dynamic_preconditions_from_gt':False},
        'execution': {'budget_candidates':[120,240,360], 'use_gt_early_stop':False,
                      'reset_hulc_each_skill':True, 'reset_rssm_each_skill':False,
                      'single_gpu_serial':True, 'keep_failed_transitions':True},
        'evaluation': {'closed_loop_roots':40, 'tasks_per_root':2, 'families':4, 'main_tasks':80,
                       'main_length':[4,8], 'stress_tasks':20, 'stress_length':[8,10],
                       'false_confirmation_target':0.05, 'reference_success_fraction_target':0.60,
                       'max_causal_corrections':1},
        'stage0_gate': {'skill_families':4, 'reference_chain_length':[3,4], 'dual_camera':True,
                        'reset':True, 'planning_regression':True}
    }
    save('configs/protocol.json', protocol)
    repos={}
    for rel in ['calvin','hulc','calvin/calvin_env','calvin/calvin_env/tacto']:
        q=P/'vendor'/rel
        repos[rel]={'commit':subprocess.check_output(['git','-C',str(q),'rev-parse','HEAD'],text=True).strip(),
                    'remote':subprocess.check_output(['git','-C',str(q),'remote','get-url','origin'],text=True).strip()}
    save('source/repositories.json',repos)
    save('runs/status.json',{'phase':0,'complete':False,'real_transitions':0,'trained_models':0,
                            'stage0':{'guide_complete':True,'gpu_inventory':True,'frozen_executor':False,
                                      'dual_camera':False,'reset':False,'four_families':False,'reference_chain':False}})
    (P/'scripts/bootstrap_project.py').write_bytes(Path(__file__).read_bytes())
    (P/'src/wm_symbolic/__init__.py').write_text('"""CALVIN macro-skill world model and symbolic planning experiment."""\n')
    (P/'README.md').write_text('# wm_symbolic\n\n本机独立实验；权威方案保存在 source/。当前结果以 runs/status.json 和 logs/ 为准。\n\nconfigs：冻结协议；vendor：官方源码；.envs：独立环境；data：真实数据与特征；models：检查点；runs：运行证据；reports：解释与图表；scripts/src：代码。\n\n旧 robot_state 项目只读，旧模型与结果不能充当本轮实验结果。原指南的 AutoDL 环境描述由用户明确要求的本机执行覆盖。\n')

def downloads():
    items=[('http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip','calvin_debug_dataset.zip',1299150917,'data/raw'),
           ('http://hulc.cs.uni-freiburg.de/model_weights/HULC_ABC_D.zip','HULC_ABC_D.zip',479026990,'models')]
    records=[]
    for url,name,expected,target in items:
        archive=P/'cache'/name
        if not archive.exists() or archive.stat().st_size!=expected:
            partial=archive.with_suffix('.zip.part'); start=partial.stat().st_size if partial.exists() else 0
            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {})
            with urllib.request.urlopen(req,timeout=45) as r:
                append=start>0 and r.status==206
                count=start if append else 0; last=time.monotonic()
                with partial.open('ab' if append else 'wb') as out:
                    while chunk:=r.read(1024*1024):
                        out.write(chunk); count+=len(chunk)
                        if time.monotonic()-last>20:
                            print(name,count,expected,flush=True); last=time.monotonic()
            assert partial.stat().st_size==expected,(name,partial.stat().st_size,expected)
            partial.rename(archive)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip(); assert bad is None,bad
            dest=P/target
            for n in z.namelist():
                if not (dest/n).resolve().is_relative_to(dest.resolve()): raise ValueError('Unsafe archive path')
            z.extractall(dest)
        records.append({'url':url,'archive':str(archive),'bytes':expected,'sha256':digest,'extracted_to':str(P/target)})
        save('source/assets.json',records)
        print('complete',name,flush=True)

if __name__=='__main__':
    import sys
    {'setup':setup,'downloads':downloads}[sys.argv[1]]()
