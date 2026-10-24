from study import *
from model import ObservationStateWorldModel,OriginalWorldModel,shared_parameters,load_shared,add_new_optimizer_group

def main():
    guard('train');start=time.monotonic();assert not(U/'configs/training_frozen.json').exists()
    old,new,dev,norm=prepared();assert len(old.episodes)==120 and len(new.episodes)==24 and len(dev.episodes)==18
    lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
    original_protocol=read(C/'configs/protocol.json')
    protocol={'utc':now(),'single_change':'observe returns h_post = h_prior + W LayerNorm(E_actual), with non-affine LayerNorm and zero-initialized bias-free W512x512. q and z are computed exactly as before from h_prior,E. initial() is unchanged. Corrected h persists into next imagine GRU and the existing shared symbol/outcome/reconstruction heads.','rationale':'Same-prefix actual success/failure E differs and q responds, but deterministic h is equal. On both final1000 control seeds, all24 nonzero local current-vs-representation-KL E-gradient comparisons oppose. This motivates testing a direct observation update; it does not establish a unique cause or exclude all undertraining.','undertraining_limit':'Control500->1000 h2 train pair4->6 and8->8; dev objectives worsen4.4776->5.5897 and4.6983->5.7314. No claim that longer training could never help. No added updates.','added_parameters':262144,'initialization_matching':'All original tensors AND AdamW moments copied from the same original6000 checkpoint; new W starts exactly zero with zero/absent AdamW moments. Added capacity and new optimizer state cannot be perfectly matched. Exact initial forward identity checked under shared RNG.','training':original_protocol['training'],'selection':original_protocol['selection'],'evaluation':original_protocol['evaluation'],'diagnostic_gates':original_protocol['diagnostic_gates'],'control':'Previous paired b1+b2, two1000-update fits. Reuse after source/schedule/initialization and candidate dev-score checks.','effect_scope':'Observation route plus262144 new parameters. Not an isolated causal claim that h dominance alone is responsible. Does not change loss or remove valid uncertain-prior supervision. No independent readout head.','h0_scope':'initial() has h=0 exactly as before; no explicit new observation-to-h path at h0. Shared training may still affect h0, evaluated separately.','budget':'Two1000-update fits only; 1200second training cap; no follow-on training.','no_new_simulation_C_D_upload':True}
    protocol['diagnostic_gates']=copy.deepcopy(protocol['diagnostic_gates'])
    protocol['diagnostic_gates']['development_generalizing_support']=protocol['diagnostic_gates']['development_generalizing_support'].replace('prior recovery control','prior paired b1+b2 control')
    save(U/'configs/protocol.json',protocol)
    for p,h in read(C/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
    for p,h in read(C/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h,p
    schedules={}
    for seed in [0,1]:
        p=C/f'configs/schedule_seed{seed}.npz';q=U/p.relative_to(C);q.write_bytes(p.read_bytes());assert sha(p)==sha(q)
        with np.load(q)as z:schedules[seed]={k:z[k]for k in z.files}
    oldcache=[old.get(i)for i in range(len(old.episodes))];newcache={'state_update':[new.get(i)for i in range(len(new.episodes))]};vcache=[dev.get(i)for i in range(len(dev.episodes))];w=[torch.tensor(v,device='cuda')for v in norm['class_weights']]
    checks=[];initials=[]
    for seed in [0,1]:
        cp=torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False);a=OriginalWorldModel(skill_count=6,event_count=3).cuda();b=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();a.load_state_dict(cp['model']);load_shared(b,cp['model']);a.eval();b.eval();receipt=read(C/f'results/train_paired_seed{seed}.json');assert statehash(a)==receipt['initial_hash']
        assert sum(p.numel()for p in b.parameters())-sum(p.numel()for p in a.parameters())==262144
        # Same real sequence and RNG, including future transitions after each observe.
        batch=new.batch([0,1],'cuda');out=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
          for model in [a,b]:
            torch.manual_seed(93101);ee=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot']);aa=model.action(batch['skill'],batch['arguments'],batch['budget']);posts,qs,priors,ps=model.history(ee,aa);out.append([x.detach().float().cpu()for state in posts+priors for x in [state.h,state.z]]+[q.detach().float().cpu()for q in qs+ps]+[v.detach().float().cpu()for state in posts+priors for v in model.symbols(state).values()])
        assert all(torch.equal(x,y)for x,y in zip(*out))
        # Existing AdamW moments copied by matching original parameter order; only the new matrix has no history.
        opt=torch.optim.AdamW(shared_parameters(b),lr=.0001,betas=(.9,.999),weight_decay=.0001);opt.load_state_dict(copy.deepcopy(cp['optimizer']));old_state=len(opt.state);add_new_optimizer_group(opt,b);assert len(opt.state)==old_state and b.observation_to_h.weight not in opt.state
        initials.append({'seed':seed,'shared_initial_hash':statehash(a),'all_shared_tensors_exact':all(torch.equal(a.state_dict()[k],b.state_dict()[k])for k in a.state_dict()),'zero_residual_forward_exact':True,'new_parameters':262144,'shared_parameters':sum(p.numel()for p in a.parameters()),'old_optimizer_state_entries':old_state,'new_optimizer_moments_absent':True})
        for candidate in receipt['candidates']:
            assert sha(candidate['path'])==candidate['sha256'];a.load_state_dict(torch.load(candidate['path'],map_location='cuda',weights_only=False)['model']);score=legacy_train.score(a,vcache,w);assert score==candidate['development_objective'];checks.append({'seed':seed,'update':candidate['update'],'stored':candidate['development_objective'],'recomputed':score})
        del a,b,opt,cp;torch.cuda.empty_cache()
    save(U/'results/control_reuse_audit.json',{'utc':now(),'reuse_allowed':True,'same_data_loss_mix_schedule_budget_dev_scorer':True,'checks':checks,'initialization':initials})
    # Preserve the old loop, with only new-model construction/loading and optimizer extension changed.
    text=(C/'source/exact_original_training_loop.py').read_text();text=text.replace('m = WorldModel(skill_count=6, event_count=3).cuda()','m = ObservationStateWorldModel(skill_count=6, event_count=3).cuda()').replace("m.load_state_dict(cp['model'])","load_shared(m, cp['model'])").replace('torch.optim.AdamW(m.parameters(),','torch.optim.AdamW(shared_parameters(m),').replace("opt.load_state_dict(copy.deepcopy(cp['optimizer']))","opt.load_state_dict(copy.deepcopy(cp['optimizer']))\n        add_new_optimizer_group(opt, m)")
    (U/'source/training_loop.py').write_text(text);compile(text,str(U/'source/training_loop.py'),'exec')
    refs=dict(read(C/'configs/training_frozen.json')['sha256']);refs.update(read(C/'configs/models_frozen.json')['sha256']);refs.update({str(p):sha(p)for p in(C/'results').glob('train_paired_seed*.json')});refs.update({str(C/'configs/protocol.json'):sha(C/'configs/protocol.json')});refs.update({str(p):sha(p)for p in(U/'source').glob('*.py')});refs.update({str(p):sha(p)for p in(U/'configs').glob('*')if p.is_file()});refs[str(U/'results/prechange_diagnostic.json')]=sha(U/'results/prechange_diagnostic.json')
    save(U/'configs/training_frozen.json',{'utc':now(),'before_first_update':True,'sha256':refs})
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('1200second two-fit cap')));signal.alarm(1200)
    env=dict(globals(),S=U,ARMS=['state_update'],oldcache=oldcache,newcache=newcache,vcache=vcache,w=w,schedules=schedules,results=[],selections=[],score=legacy_train.score)
    exec(compile(text,str(U/'source/training_loop.py'),'exec'),env)
    for p,h in refs.items():assert sha(p)==h,p
    save(U/'configs/models_frozen.json',{'utc':now(),'selection':env['selections'],'fits':2,'updates':2000,'control_reused':True,'sha256':{c['path']:c['sha256']for r in env['results']for c in r['candidates']}})
    save(U/'results/training_complete.json',{'utc':now(),'fits':2,'updates':2000,'seconds':time.monotonic()-start,'C_D_samples':0,'new_collection':0,'control_refits':0});signal.alarm(0);print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
