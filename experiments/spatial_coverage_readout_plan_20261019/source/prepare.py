from pathlib import Path
import os,sys,json,hashlib,datetime,csv,html,re,base64,xml.etree.ElementTree as ET
import numpy as np
import yaml
P=Path('/home/robot/research/projects/wm_symbolic')
Q=P/'experiments/spatial_coverage_readout_plan_20261019'
W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
os.environ['MPLCONFIGDIR']=str(Q/'cache/matplotlib');os.environ['PYTHONDONTWRITEBYTECODE']='1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def guard(event,args):
 if event in ['socket.connect','subprocess.Popen','os.system']:raise RuntimeError('Preparation only: no external process/network')
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  write=(isinstance(m,str)and any(x in m for x in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if write and p.is_relative_to(P)and not p.is_relative_to(Q):raise RuntimeError('Old project writes forbidden')
def seals():
 out={}
 for rel,base in [('experiments/rssm_scaling_run_20261019/configs/delivery_manifest.json',None),('experiments/rssm_h2_supplement_20261019/manifest.json',None),('experiments/object_identity_readout_20261019/manifest.json','experiments/object_identity_readout_20261019'),('experiments/roi_geometry_audit_20261019/manifest.json','experiments/roi_geometry_audit_20261019')]:
  m=P/rel
  for name,h in read(m)['sha256'].items():
   f=Path(name)if Path(name).is_absolute()else(P/base if base else m.parent)/name;assert sha(f)==h;out[str(f)]=h
  out[str(m)]=sha(m)
 return out
def render(md):
 def inline(x):
  x=html.escape(x);x=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',x);return re.sub(r'`(.*?)`',r'<code>\1</code>',x)
 ls=md.splitlines();out=[];i=0
 while i<len(ls):
  x=ls[i]
  if not x:i+=1;continue
  if x.startswith('|'):
   rr=[]
   while i<len(ls)and ls[i].startswith('|'):rr.append([v.strip()for v in ls[i].strip('|').split('|')]);i+=1
   out.append('<table><thead><tr>'+''.join('<th>'+inline(v)+'</th>'for v in rr[0])+'</tr></thead><tbody>');out.extend('<tr>'+''.join('<td>'+inline(v)+'</td>'for v in r)+'</tr>'for r in rr[2:]);out.append('</tbody></table>');continue
  m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',x)
  if m:out.append('<img alt="'+html.escape(m[1])+'" src="'+m[2]+'">')
  elif x.startswith('#'):
   n=len(x)-len(x.lstrip('#'));out.append(f'<h{n}>'+inline(x[n:].strip())+f'</h{n}>')
  else:out.append('<p>'+inline(x)+'</p>')
  i+=1
 return '\n'.join(out)
def main():
 assert Q.is_dir()and not(Q/'manifest.json').exists()
 # Font discovery may invoke read-only fc-list; load plotting before the strict process guard.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 sys.addaudithook(guard);before=seals();save(Q/'protected_before.json',before)
 scenes={};refs=[]
 for e in 'AB':
  file=P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml';refs.append(file);sc=yaml.safe_load(file.read_text());dims={}
  for c,obj in sc['objects']['movable_objects'].items():
   f=P/'vendor/calvin/calvin_env/data'/obj['file'];refs.append(f);root=ET.fromstring(f.read_text());size=np.array([float(x)for x in root.find('.//collision/geometry/box').get('size').split()])*sc['global_scaling'];dims[c]={'scaled_dimensions_m':size.tolist(),'max_xy_half_diagonal_m':float(np.linalg.norm(size[:2])/2)}
  f=P/'vendor/calvin/calvin_env/data'/sc['objects']['fixed_objects']['table']['file'];refs.append(f);urdf=ET.fromstring(f.read_text());button=np.array([float(x)for x in urdf.find("joint[@name='base__button']/origin").get('xyz').split()])*sc['global_scaling']
  lo,hi=sc['surfaces']['table'];scenes[e]={'native_center_sampling_bounds':sc['surfaces']['table'],'original_slot_centers_m':[lo[0]+(hi[0]-lo[0])*r for r in [.16,.5,.84]],'jitter_m':.014,'block_geometry':dims,'button_joint_origin_scaled_m':button.tolist(),'full_physical_right_region_safety_certified':False,'candidate_only':True}
 save(Q/'scene_constraints.json',scenes)
 original=P/'experiments/object_identity_readout_20261019';compute=read(original/'results/compute_complete.json');storage=read(P/'runs/core1200_v1/storage_first50.json');progress=read(P/'runs/core1200_v1/progress.json')
 drows=[]
 for line in(P/'logs/core1200_post_sealed_D_collection.log').read_text().splitlines():
  try:r=json.loads(line)
  except:continue
  if isinstance(r,dict)and 'episode'in r and 'step'in r:drows.append(r)
 assert len(drows)==240 and all(r['episode'].startswith('D_')for r in drows)
 budget={'control_train_roots_reused':40,'new_train_roots':40,'new_AB_eval_roots':{'old_band':10,'gap':10},'new_C_roots':0,'old_C_roots_exploratory':20,'new_transitions':720,'new_boundaries':900,'new_readout_fits':2,'updates_each':1500,'new_head_updates':3000,'new_RSSM_updates':0,'training_control_head_reuse_requires_exact_protocol_check':True,'historical_collection_reference':{'transitions':240,'seconds':progress['elapsed_this_run_seconds']},'collection_linear_estimate_minutes':progress['elapsed_this_run_seconds']/240*720/60,'planned_collection_minutes':[30,60],'cache_reference':{'new_boundaries':300,'seconds':13.562206579999838},'cache_linear_estimate_seconds':13.562206579999838*3,'cache_and_routing_allowance_minutes':[2,5],'old_head_seconds':[r['seconds']for r in compute['fits']],'heads_and_features_allowance_minutes':[2,5],'report_work_minutes':[15,30],'new_transition_bytes_linear':storage['bytes_per_transition']*720,'new_feature_bytes_linear':436642*900,'storage_expected_GB':[2,3],'storage_reserved_GB':5,'estimated_not_measured_new_run':True}
 save(Q/'budget.json',budget)
 protocol={'status':'PROPOSAL_ONLY_PENDING_EXECUTION_DECISION','independent_audit':'Parent relayed confirmation of class-conditional support gap; no execution authorization','factor':'Class-conditional x coverage within native A/B physically validated placement/reach sets','fixed':['two frozen roots40 encoders','public identity routing + object pooling','309165-parameter shared head','GT physical classes','40 roots per training arm','scene/color-to-class assignment','initial-state and held color','y and yaw','three branches of four skills','V1 executor and budgets','unweighted four-field CE','uniform root sampling and paired schedule','head initialization and 1500 updates'],'training_label_priors':'Exact h0 per-root match; realized downstream priors and TV distances reported; no automatic reweight/drop/resample. Conclusion is x-policy total effect, not isolated mediation-free geometry mechanism.','evaluation':'Fresh AB old-band10 + gap10 with pre-frozen recipes, both arms common. Existing C20 exploratory only. Freeze heads before new evaluation collection.','safety':'Native center envelope is a prior, not footprint/reach proof; full support/collision and fixed-y/yaw feasible set still unverified. B right cannot satisfy >5mm valid-margin under native max .15.','no_generated_roots':True,'no_sampling_execution':True,'no_collection':True,'no_training':True,'no_model_forward':True,'no_upload':True}
 save(Q/'protocol_draft.json',protocol)
 refs += [P/'scripts/core1200_prepare.py',P/'scripts/core1200_collect.py',P/'src/wm_symbolic/calibrated_labels.py',P/'src/wm_symbolic/physical_oracle.py',P/'runs/core1200_v1/progress.json',P/'runs/core1200_v1/cache_progress.json',P/'runs/core1200_v1/storage_first50.json',P/'experiments/acquisition_scene_surface_pilot8_20261018/configs/protocol.json',P/'experiments/acquisition_scene_surface_pilot8_20261018/results/ENGINEERING_PILOT_COMPLETE.json',original/'configs/protocol.json',original/'results/compute_complete.json']
 save(Q/'source_evidence.json',{'sha256':{str(f):sha(f)for f in refs},'prior_lesson':'23/96 centers outside native placement range; six monitored roots already touch button; corrected engineering8 succeeds7, not a universal reach certificate','reference_table':'source code/configs read; no new physics execution'})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':170})
 fig,ax=plt.subplots(2,1,figsize=(11,5),layout='constrained')
 for i,e in enumerate('AB'):
  a=ax[i];lo,hi=scenes[e]['native_center_sampling_bounds'];a.axvspan(lo[0],hi[0],color='#dfe3e8',alpha=.7)
  spans=[(-.1416667,.1416667,'#D97919'),(.1516667,.35,'#008F91')]if e=='A'else[(-.35,-.1516667,'#8B5AA3'),(-.1416667,.1416667,'#D97919')]
  for l,h,c in spans:a.plot([l,h],[.5,.5],color=c,lw=12,solid_capstyle='butt')
  for x in scenes[e]['original_slot_centers_m']:a.plot([x-.014,x+.014],[.25,.25],color='black',lw=6);a.plot(x,.25,'o',c='black',ms=4)
  for cut in [-.1466667,.1466667]:a.axvline(cut,ls='--',c='#555',lw=1)
  a.set(xlim=(-.44,.44),ylim=(.1,.7),yticks=[.25,.5],yticklabels=['Original three slots','Candidate within-class x'],title=f'Scene {e}: native centre prior [{lo[0]:.2f}, {hi[0]:.2f}] m',xlabel='Table-local / world x (metres)');a.spines[['top','right','left']].set_visible(False)
 fig.suptitle('Candidate domains only: footprint, collision and reach checks still required',fontsize=12);fig.savefig(Q/'design.png');plt.close(fig)
 md=(W/'spatial_coverage_PROPOSAL.md').read_text();marker='## 两组如何配对'
 md=md.replace(marker,'![采样范围示意](design.png)\n\n图示：横轴是物理桌面局部x（本场景也等于世界x）；灰底为场景声明的中心采样范围，黑短线为原三槽加扰动，彩色粗线为同类内连续候选范围（紫左、橙中、青右），虚线为固定物理类别边界。彩线尚未经完整支撑、碰撞和可达性检查，不能当作安全放置证书。\n\n'+marker)
 (Q/'PROPOSAL.md').write_text(md);body=render(md).replace('design.png','data:image/png;base64,'+base64.b64encode((Q/'design.png').read_bytes()).decode());(Q/'PROPOSAL.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>类内空间覆盖对照方案</title><style>body{max-width:1150px;margin:36px auto;padding:0 24px;font:17px/1.7 sans-serif;color:#18212b}table{border-collapse:collapse;width:100%;font-size:14px;margin:22px 0}td,th{border:1px solid #d0d7de;padding:7px;text-align:left}th{background:#edf4f4}img{max-width:100%;height:auto}strong{color:#006d77}code{font-size:.9em}</style>'+body+'</html>')
 assert before==seals();assert not any(x in sys.modules for x in ['torch','pybullet','transformers'])
 save(Q/'final_verification.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PROPOSAL_PREPARED_ONLY','old_sealed_files_unchanged':len(before),'new_roots_generated':0,'collection':0,'model_forwards':0,'fits':0,'uploads':0,'independent_audit_constraints_included':True,'source_references_checked':len(refs),'geometry_safety_not_yet_validated':True,'execution_not_started':True})
 files=sorted(f for f in Q.rglob('*')if f.is_file()and 'cache'not in f.relative_to(Q).parts);save(Q/'manifest.json',{'files':len(files),'sha256':{str(f.relative_to(Q)):sha(f)for f in files}})
 print(json.dumps({'proposal':str(Q/'PROPOSAL.html'),'files':len(files),'manifest_sha256':sha(Q/'manifest.json'),'old_files_unchanged':len(before),'budget':budget},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
