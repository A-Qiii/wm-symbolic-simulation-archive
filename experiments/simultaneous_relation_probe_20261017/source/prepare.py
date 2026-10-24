"""Offline structural/coverage audit and fixed finite boundary dataset; no simulation."""
from config import *
import sys,ast,copy,shutil,collections,torch
sys.path.insert(0,str(REP/'source'))
from model import Query,probs
from registry import extract
sys.path.insert(0,str(EV/'source'))
from native_evaluator import NativeEvaluator

def dual(y,v):return any(v[j]and y[j]and v[k+3]and y[k+3]for j in range(3)for k in range(3)if j!=k)
def main():
 assert not E.exists()
 sources={};seals={}
 def reg(p):p=Path(p);sources[str(p)]=sha(p);return p
 for mf in [COMBO/'configs/delivery_manifest.json',EV/'configs/delivery_manifest.json',FIX/'configs/delivery_manifest.json']:
  for p,h in read(mf)['sha256'].items():assert sha(p)==h,p
  seals[str(mf)]=sha(mf);reg(mf)
 for sub in ['source','configs','data/train','data/validation','data/evaluator_D','results','models','logs','audit']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True)
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text());shutil.copy2(p,E/'source'/p.name)
 for base,names in [(REP,['model.py','features.py','representation.py','registry.py']),(PRIOR,['prepare_query.py','train_query.py']),(FIX,['setup.py','train.py'])]:
  for name in names:reg(base/'source'/name)
 # Verify the exact query training pools against their source labels, including native corrections in this audit only.
 auditrows=read(reg(EV/'audit/query_pool_labels.json'));coverage=[];poolcopies=[]
 for pool,path in [('precursor144',PRIOR/'data/query_train/new_coverage.npz'),('current156',FIX/'data/query_train/empty_open.npz')]:
  data=dict(np.load(reg(path)));rr=[r for r in auditrows if r['pool']==pool];assert len(rr)==len(data['z'])
  for i,r in enumerate(rr):assert r['state_key']==str(data['state_ids'][i])and r['root_id']==str(data['root_ids'][i])and data['y'][i].astype(bool).tolist()==r['stored_y']and data['v'][i].astype(bool).tolist()==r['stored_v']
  for version in ['stored','native']:
   pos=[r for r in rr if dual(r[version+'_y'],r[version+'_v'])];coverage.append({'pool':pool,'labels':version,'states':len(rr),'root_N':len({r['root_id']for r in rr}),'joint_states':len(pos),'joint_root_N':len({r['root_id']for r in pos}),'holding_states':sum(any(y and v for y,v in zip(r[version+'_y'][:3],r[version+'_v'][:3]))for r in rr),'inside_states':sum(any(y and v for y,v in zip(r[version+'_y'][3:6],r[version+'_v'][3:6]))for r in rr)})
  poolcopies+=rr
  np.savez_compressed(E/'audit'/f'{pool}_native_label_copy.npz',root_ids=data['root_ids'],state_ids=data['state_ids'],old_y=data['y'],old_v=data['v'],native_y=np.array([r['native_y']for r in rr],np.float32),native_v=np.array([r['native_v']for r in rr],np.float32))
 save(E/'results/historical_pool_coverage.json',coverage)
 # Binding and probability/loss tests run on CPU, independent from any probe optimizer.
 torch.set_num_threads(4);torch.manual_seed(17073000);q=Query().eval();z=torch.randn(7,4,64);perm=[2,0,1]
 with torch.no_grad():
  a=q(z);b=q(torch.cat([z[:,perm],z[:,3:]],1));assert torch.allclose(b[:,:3],a[:,perm],atol=1e-6,rtol=1e-6);assert torch.allclose(b[:,3:6],a[:,[3+j for j in perm]],atol=1e-6,rtol=1e-6);assert torch.allclose(b[:,6:],a[:,6:],atol=1e-6,rtol=1e-6)
 logits=torch.full((1,9),-10.);logits[0,0]=10.;logits[0,4]=10.;logits[0,7]=10.;pp=probs(logits);assert pp[0,0]>.99 and pp[0,4]>.99
 tree=ast.parse((PRIOR/'source/train_query.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='loss');ns={'F':torch.nn.functional,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_loss','exec'),ns);loss=ns['loss']
 def grad(held):
  ll=torch.zeros(1,9,requires_grad=True);y=torch.zeros(1,9);y[0,0]=held;y[0,4]=1;y[0,7]=1;v=torch.ones(1,9);v[0,3]=0;loss(ll,y,v,torch.ones(9)).sum().backward();return ll.grad
 g0=grad(0);g1=grad(1);assert g1[0,3]==0 and g1[0,4]!=0 and torch.equal(g0[:,3:6],g1[:,3:6])
 rgb=np.zeros((20,60,3),np.uint8);rgb[:,:20]=[255,0,0];rgb[:,20:40]=[0,0,255];rgb[:,40:]=[255,0,255];masks,_=extract(rgb)
 for j in range(3):assert masks[j,:,j*20:(j+1)*20].all()and masks[j].sum()==400
 # Reproduce frozen actual-query probabilities and check all source pool latent/label bindings.
 roots=read(reg(COMBO/'configs/roots.json'));states=read(reg(COMBO/'data/combo/public/states.json'));pairs=read(reg(COMBO/'data/combo/public/pairs.json'));labels=read(reg(COMBO/'data/combo/_private/labels.json'));oldlabels=read(reg(COMBO/'data/combo/_private/legacy_labels.json'));inputs=np.load(reg(COMBO/'data/combo/public/inputs.npz'));raw=inputs['z'];pred=read(reg(COMBO/'results/predictions.json'));stats=torch.load(reg(REP/'models/query_seed0.pt'),map_location='cpu',weights_only=True)['normalization'];zn=((torch.tensor(raw)-stats['mean'])/stats['std']);probs_all={};max_delta=0.
 protocol=read(reg(COMBO/'configs/protocol.json'))
 for seed in [0,1]:
  q=Query().eval();q.load_state_dict(torch.load(reg(protocol['query_heads'][str(seed)]['path']),map_location='cpu',weights_only=True)['state'])
  with torch.inference_mode():p=probs(q(zn)).numpy()
  probs_all[str(seed)]=p
  for row in pred:
   if row['condition']==f'mixed_seed{seed}':max_delta=max(max_delta,float(np.max(np.abs(p[row['after']]-row['actual_probability']))))
 assert max_delta<1e-5,max_delta
 priorz=np.load(reg(PRIOR/'data/ABC/public/inputs.npz'))['z'];priorzn=((torch.tensor(priorz)-stats['mean'])/stats['std']).numpy();oldpath=EX/'prediction_query_diagnostics_20261016/data/head_train/original_coverage.npz';old=dict(np.load(reg(oldpath)));oldbyid={int(s):i for i,s in enumerate(old['state_ids'])}
 for pool,path in [('precursor144',PRIOR/'data/query_train/new_coverage.npz'),('current156',FIX/'data/query_train/empty_open.npz')]:
  data=dict(np.load(path))
  for i,sid in enumerate(data['state_ids']):
   src,k=str(sid).split(':');expected=priorzn[int(k)]if src=='new'else old['z'][oldbyid[int(k)]];assert np.array_equal(data['z'][i],expected)
 # Fixed AB 00 training / 01 validation; D only evaluator. No historical states train the probes.
 splits={r['root_id']:('diagnostic_D_seen'if r['environment']=='D'else'train'if r['replicate']==0 else'validation')for r in roots}
 for split in ['train','validation','diagnostic_D_seen']:assert sum(s==split for s in splits.values())==6
 root_sets={split:sorted(r for r,s in splits.items()if s==split)for split in set(splits.values())};save(E/'configs/root_split.json',root_sets)
 boundary=sorted({p[k]for p in pairs for k in ['before','after']});groups={};state_rows=[];features=[];native=[];legacy=[];prob0=[];prob1=[]
 for i in boundary:
  s=states[i]
  with np.load(reg(s['path']))as obs:h=hashlib.sha256(b''.join(obs[k].tobytes()for k in sorted(obs.files))).hexdigest()
  key=(s['root_id'],h)
  if key in groups:
   row=state_rows[groups[key]];assert labels[str(i)]==labels[str(row['combo_state_index'])];row['source_indices'].append(i);row['source_branches'].append(s['branch']);continue
  j=len(state_rows);groups[key]=j;lab=labels[str(i)];held=[c for k,c in enumerate(COLORS)if lab['v'][k]and lab['y'][k]];assert all(lab['v']);features.append(raw[i]);native.append(lab['y']);legacy.append(oldlabels[str(i)]['y']);prob0.append(probs_all['0'][i]);prob1.append(probs_all['1'][i]);state_rows.append({'state_index':j,'combo_state_index':i,'root_id':s['root_id'],'environment':s['environment'],'split':splits[s['root_id']],'observation_sha256':h,'source_indices':[i],'source_branches':[s['branch']],'held_color':held[0]if len(held)==1 else'none','holding_count':len(held),'dual_relation':dual(lab['y'],lab['v']),'public_path':s['path'],'native_all_valid':True})
 # Native physical recomputation for every selected real boundary, not sequence-derived labels.
 point_map={}
 for ep in read(reg(COMBO/'results/collection.json')):
  f=reg(Path(ep['folder'])/'_private/states.json')
  for pt in read(f):point_map[pt['public_path']]=pt
 ev={env:NativeEvaluator(env)for env in 'ABD'}
 for row in state_rows:assert ev[row['environment']].evaluate(point_map[row['public_path']]['physical'])==labels[str(row['combo_state_index'])]
 arrays={'z':np.stack(features),'y':np.array(native,bool),'legacy_y':np.array(legacy,bool),'query_seed0':np.stack(prob0),'query_seed1':np.stack(prob1)};meta=[]
 for row in state_rows:
  y=arrays['y'][row['state_index']]
  for j,c in enumerate(COLORS):meta.append({k:row[k]for k in ['root_id','environment','split','held_color','holding_count','dual_relation']}|{'state_index':row['state_index'],'object_index':j,'object_color':c,'In':bool(y[j+3]),'object_is_held':bool(y[j])})
 cov=[]
 for split in ['train','validation','diagnostic_D_seen']:
  ii=[j for j,r in enumerate(state_rows)if r['split']==split];folder=E/'data'/('evaluator_D'if split=='diagnostic_D_seen'else split);np.savez_compressed(folder/'states.npz',**{k:v[ii]for k,v in arrays.items()},state_indices=np.array(ii));save(folder/'states.json',[state_rows[i]for i in ii]);save(folder/'objects.json',[r for r in meta if r['split']==split]);rows=[r for r in meta if r['split']==split];cov.append({'split':split,'roots':6,'unique_observations':len(ii),'object_labels':len(rows),'In_positive':sum(r['In']for r in rows),'In_negative':sum(not r['In']for r in rows),'dual_observations':sum(state_rows[i]['dual_relation']for i in ii),'dual_roots':len({state_rows[i]['root_id']for i in ii if state_rows[i]['dual_relation']})})
 save(E/'results/probe_data_coverage.json',cov);save(E/'data/all_state_provenance.json',state_rows);save(E/'results/structure_audit.json',{'utc':now(),'hard_mutual_exclusion_between_objects':False,'Holding_In_independent_sigmoid':True,'drawer_only_softmax':True,'mask_uses_per_fact_validity_not_held_status':True,'query_permutation_equivariance_test_passed':True,'positive_Holding_red_and_In_blue_allowed':True,'invalid_In_cell_zero_gradient_other_In_cells_retained':True,'RGB_colour_binding_order_test_passed':True,'pool_z_and_label_binding_300_rows_exact':True,'actual_query_reproduction_max_abs_probability_error':max_delta,'definite_new_code_bug_found':False,'known_historical_geometry_labels_preserved_and_copied_native':True,'native_selected_boundary_labels_recomputed':len(state_rows),'simulator_calls':0,'D_used_for_training_or_selection':False})
 cfg={'utc':now(),'authorization':'Bounded offline diagnostic approved; no collection/WM updates/query replacement; D is previously observed diagnostic only.','question':'Is simultaneous In information linearly/nonlinearly readable from current actual-observation z and corresponding object token?','inputs':{'object':'current corresponding object token64','context':'same object64 + scene64 + mean of3objecttokens64; total192, same ingredients as frozenquery forconstantIn type'},'models':['linear','mlp32'],'seed':17073000,'fits':4,'training_roots':root_sets['train'],'validation_roots':root_sets['validation'],'seen_D_diagnostic_roots':root_sets['diagnostic_D_seen'],'selection':'No early stop, hyperparameter search or checkpoint selection; fixed final500updates, validation descriptive only, D evaluator loaded only after final weights frozen.','training':'CPU fullbatch AdamW lr0.001 wd0.001,500updates; BCEWithLogits, train-root-balanced and class-balanced weights; perinputstandardization fromtrainonly,std floor0.01; shared3objectclassifier,no colour/held/sequence/history/physics inputs.','label_budget':'All3objects at actual skill boundaries, exactRGB+robot duplicate states withinroot deduplicated; no dense intermediateframes; no historicalsamples added to probe training.','normal_and_combination_states':'All3frozenbranches provide positive/negative In underempty andheldgripper; original/secondtarget are metadata only.','metrics':'In TP/FN/FP/TN,recall,FPR,balancedaccuracy; joint3In stateexact; bysplit,objectcolour,heldidentity,dualcondition androot; majorityclass andtraincolourpriorbaselines; uniformincorrect/correct ifmissingsupport reported; rootmacro notframeindependence.','limits':'Smallprobe success onlysupports decodability; failuredoesnotproveinformationabsence. Probe exposure differs from originalquery; no causalattributionsolely to addedcoverage. Frozenencoder was previously trained on AB scene distribution; rootisolation appliesnewprobe labels, not newpretraining. D observed diagnostic only.'}
 save(E/'configs/protocol.json',cfg);save(E/'configs/source_references.json',{'sha256':sources,'prior_manifests':seals})
 for p,h in sources.items():assert sha(p)==h,p
 summary='''# 同时关系离线诊断：结构与覆盖初查\n\n固定query的Holding/In为独立sigmoid；只有抽屉三分类做softmax。没有发现跨物体互斥归一化、按Held屏蔽In标签或对象绑定索引错误。概率共存、逐事实梯度掩码、对象置换和RGB颜色绑定测试通过。历史query数据300行的标签/潜变量绑定核对一致。\n\n前驱池144状态与当前池156状态各30根；原标签和原生几何修正版均为0个“夹持一物、另一物在屉内”状态、0个对应根。覆盖缺口明确，但不能单凭缺口证明它是唯一原因。已知旧几何标签只在新诊断副本并列保存。\n\nprobe仅使用新组合实验已经存在的技能边界，全部三个对象都提供In正负标签。A/B固定每色00根训练、01根验证，共6/6根；D的6根仅为已观察诊断集。输入只有当前对应对象token或对象+场景+池化共享潜特征；没有动作计划、历史真值、物理坐标、颜色ID或Held身份。各输入分别运行同数据的线性分类器和32隐藏单元MLP，固定500更新，不搜索或按D选点。\n'''
 (R/'STRUCTURE_AND_COVERAGE.md').write_text(summary+'\n'+json.dumps(cov,ensure_ascii=False,indent=2)+'\n');print('STRUCTURE_COVERAGE_COMPLETE',json.dumps({'historical':coverage,'probe':cov,'query_reproduction_error':max_delta}),flush=True)
if __name__=='__main__':main()
