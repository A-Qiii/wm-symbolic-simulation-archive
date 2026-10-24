from config import *
import shutil,ast,collections
assert not E.exists(),E
for f in ['source','configs','results','data']: (E/f).mkdir(parents=True)
R.mkdir(parents=True,exist_ok=False)
for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
old=0
for folder in [SPAN,DIAG]:
 for p,h in read(folder/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p;old+=1
ts=read(DIAG/'data/public/trajectories.json');samples=[]
for ti,t in enumerate(ts):
 for step,r in enumerate(t['steps'],1):samples.append({k:t[k]for k in ['group','root_id','color','branch']}|{'sample_id':len(samples),'trajectory_id':ti,'global_step':step,'skill':r['skill'],'before':r['before'],'after':r['after'],'action':r['action'],'split':'AB_train'if t['group'].endswith('AB')else'C_development'})
assert len(ts)==60 and len(samples)==240
save(E/'configs/samples.json',samples)
models=read(SPAN/'configs/choices_frozen.json')
protocol={'frozen_utc':now(),'scope':'Frozen decoder/objective/query mechanism diagnostic only; no network optimization, simulation, data collection, loss change or newD.',
 'sampling':'All240 boundaries of60 existing diagnostic trajectories:oldAB12normal,newAB12x2branches,Cselect6x2,Cvalidate6x2. No sampling on outcome, prediction or label.',
 'selected_models':models['selected'],'query_heads':models['query_heads'],
 'initializations':'Each queryseed has240 actual successor latents. Each of4selectedtransitionmodels has240trueprevious single-step and240initial-only full4 recursive latents. 2400 cases total; two objectives share each exact start.',
 'losses':{'mixed':'F+0.25Z','latent_only':'0.25Z','F':'mean of4groupmeans:object_visual256x3,object_geometry12x3,scene_visual1024,robot15; percoordinate weight1+4*clamp(abs(actual_after-actual_before),max1)','Z':'raw latent MSE over4x64, no query normalization; coefficient held at0.25 in both arms to isolate removing F'},
 'optimization':{'parameters':'only independent per-sample latent tensor; all model parameters requires_grad=False','optimizer':'plain gradient descent; sum independent per-sample objective (no batch averaging)','learning_rate':4.0,'steps':32,'snapshots':[0,1,4,8,16,32],'clipping':False,'line_search':False,'early_stopping':False,'nonfinite':'stop and report; do not retune','selection':'no best-step or C-based selection; step32 is primary endpoint, earlier steps diagnostic'},
 'gradients':'At every initial case compute F gradient and0.25Z gradient, individual F-group gradients, norms/cosine; compute query logit JVP along -gradF without labels. True labels only in post-hoc scoring.',
 'checkpoint_pairs':'All20existingSPANcheckpoints:4runs x0/250/500/750/1000. Evaluate same240boundaries in single and fullrecursive paths. Adjacent-update sample pairs primary; no reselection.',
 'scoring':'Full9/group correctness, signed logit margin, baseline-error cohorts, true-state strata, AB/C and normal/refusal. Positive logit margin denotes correct side of boundary, not calibrated confidence. Report objective decrease with correct-to-wrong flips and target distance.',
 'thresholds':{'objective_decrease':1e-6,'gradient_nonzero_rms':1e-8,'margin_direction':1e-6},
 'limits':'Latent optimization points need not be physically realizable states; finite diagnostic cannot establish causal training improvement or latent information loss. C is repeatedly seen development data.',
 'stop':'Report whether controlled latent-only vs mixed training is justified; do not start it.'}
save(E/'configs/protocol.json',protocol)
inputs=[E/'configs/protocol.json',E/'configs/samples.json',DIAG/'data/public/inputs.npz',DIAG/'data/public/trajectories.json',REP/'data/public/normalization.npz',REP/'models/query_seed0.pt',REP/'models/canonical_matched_seed0/selected.json',REP/'models/canonical_matched_seed0/update1000.pt']
inputs+=list((E/'source').glob('*.py'))
inputs += [Path(q['path'])for q in models['query_heads'].values()]
inputs += sorted((SPAN/'models').glob('*/update*.pt'))
inputs += [REP/'source'/f for f in ['model.py','runtime.py','representation.py']]
save(E/'configs/frozen.json',{'frozen_utc':now(),'prior_sealed_files_verified':old,'sha256':{str(p):sha(p)for p in inputs}})
print('FROZEN',len(samples),'boundaries',old,'prior sealed files',flush=True)
