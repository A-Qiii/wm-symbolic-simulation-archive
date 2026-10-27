"""Read-only same-new-test comparison; preserve versions/seeds/layout groups."""
from paths import CJK_FONT
from paths import ROOT as B, RUNTIME as P
import json,csv,collections,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def boot(x,seed):
 x=np.array(x,float);rng=np.random.default_rng(seed);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1);return {'mean_difference':float(x.mean()),'percentile95':np.quantile(means,[.025,.975]).tolist(),'independent_layouts':len(x),'training_seed':seed,'training_randomness_not_covered':True}
def main():
 root=B/'reports/v3';assert(root/'live/complete.json').exists();rows=read(root/'confirmation/all_predictions.json');summary=read(root/'confirmation/summary.json');out=root/'analysis';out.mkdir(exist_ok=True);lookup={(r['version'],r['seed'],r['episode']):r for r in rows};deltas=[];perception=[]
 for seed in [0,1,2]:
  for recipe in sorted({r['recipe']for r in rows}):
   cases=[r for r in rows if r['version']=='v1'and r['seed']==seed and r['recipe']==recipe];assert len(cases)==20
   for metric in ['free_endpoint','free_interval','feedback_endpoint']:
    ds=[];gains=[];losses=[]
    for old in cases:
     new=lookup[('v3',seed,old['episode'])];a=bool(old.get('correct',{}).get(metric,False));b=bool(new.get('correct',{}).get(metric,False));ds.append(int(b)-int(a))
     if b and not a:gains.append(old['episode'])
     if a and not b:losses.append(old['episode'])
    deltas.append({'seed':seed,'recipe':recipe,'metric':metric,'paired_difference':boot(ds,seed),'gain_cases':gains,'loss_cases':losses})
  oldcases=[r for r in rows if r['version']=='v1'and r['seed']==seed and r['physical_valid']];differences=[];same_symbols=0;total=0
  for old in oldcases:
   new=lookup[('v3',seed,old['episode'])]
   for a,b in zip(old['current'],new['current']):
    differences.append(float(np.max(np.abs(np.array(a['xyz'])-np.array(b['xyz'])))));same_symbols+=int(a['region']==b['region']and a['held']==b['held']);total+=1
  perception.append({'seed':seed,'same_current_symbol_boundaries':same_symbols,'total':total,'max_current_XYZ_difference_m':max(differences),'perception_weights_unchanged':True})
 for r in deltas:r.update(reference_version='v1',model_version='v3')
 for seed in [0,1,2]:
  for recipe in sorted({r['recipe']for r in rows}):
   cases=[r for r in rows if r['version']=='v2'and r['seed']==seed and r['recipe']==recipe]
   for metric in ['free_endpoint','free_interval','feedback_endpoint']:
    ds=[];gains=[];losses=[]
    for old in cases:
     new=lookup[('v3',seed,old['episode'])];a=bool(old.get('correct',{}).get(metric,False));b=bool(new.get('correct',{}).get(metric,False));ds.append(int(b)-int(a))
     if b and not a:gains.append(old['episode'])
     if a and not b:losses.append(old['episode'])
    deltas.append({'reference_version':'v2','model_version':'v3','seed':seed,'recipe':recipe,'metric':metric,'paired_difference':boot(ds,seed),'gain_cases':gains,'loss_cases':losses})
 save(out/'paired_prediction_changes.json',deltas);save(out/'perception_invariance.json',perception)
 with(out/'prediction_summary.csv').open('w')as f:
  flat=[]
  for s in summary:
   for k in ['current','one','free_endpoint','free_interval','feedback_endpoint']:
    n,d=s[k];flat.append({**{q:s[q]for q in ['version','seed','recipe']},'metric':k,'correct':n,'denominator':d})
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 actual=[];pair={};parity=[]
 for r in read(root/'live/all_results.json'):
  p=read(Path(r['score_file']).parent/'policy.json');spec=p['task_spec'];key=(r['version'],spec['task'],spec['stratum_private'],spec.get('goal_id',0));actual.append((key,r,p));pair[(r['version'],r['case'])]=r
  parity.extend(b['feature_max_abs_diff_vs_reference']for b in p['boundaries']if'feature_max_abs_diff_vs_reference'in b)
 groups=collections.defaultdict(list)
 for key,r,p in actual:groups[key].append((r,p))
 actualsummary=[]
 for key,xs in sorted(groups.items()):
  actualsummary.append({'version':key[0],'task':key[1],'stratum':key[2],'goal':key[3],'success':sum(r['success']for r,p in xs),'denominator':len(xs),'stops':dict(collections.Counter(r['stop']for r,p in xs)),'physics_invalid':sum(not r['physics_valid']for r,p in xs),'initialization_invalid':sum(not r['initialization_valid']for r,p in xs),'actions':sum(len(p['selected_actions'])for r,p in xs)})
 save(out/'actual_summary.json',actualsummary);gains=[];losses=[]
 for (version,case),old in pair.items():
  if version!='v1':continue
  new=pair[('v3',case)]
  if new['success']and not old['success']:gains.append(case)
  if old['success']and not new['success']:losses.append(case)
 layout_cases=collections.defaultdict(dict)
 for key,r,p in actual:layout_cases[(r['version'],p['task_spec']['group'])][r['case']]=int(r['success'])
 differences=[]
 for (version,group),old in layout_cases.items():
  if version!='v1':continue
  new=layout_cases[('v3',group)];assert set(old)==set(new)and len(old)==6;differences.append(np.mean([new[k]-old[k]for k in old]))
 save(out/'actual_paired_changes.json',{'same_cases':120,'gains':gains,'losses':losses,'feature_parity_boundaries':len(parity),'max_feature_difference':max(parity)if parity else None,'layout_group_paired_success_difference':boot(differences,0)})
 font=FontProperties(fname=str(CJK_FONT)if CJK_FONT else None);plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':12,'svg.fonttype':'none'});fig,axs=plt.subplots(2,2,figsize=(12,9),sharey=True);colors={'v1':'#54789b','v2':'#d98a3c','v3':'#249c83'}
 for i,(recipe,title)in enumerate([('B_loaded_reverse','未见载物B→A'),('A_beside_remove_then_move','旁置后先放A，再搬大块：小块应留A')]):
  for j,metric in enumerate(['free_endpoint','feedback_endpoint']):
   ax=axs[i,j]
   for v,offset in [('v1',-.24),('v2',0),('v3',.24)]:
    ns=[next(s for s in summary if s['version']==v and s['seed']==seed and s['recipe']==recipe)[metric][0]for seed in [0,1,2]];bars=ax.bar(np.arange(3)+offset,ns,width=.22,color=colors[v],label=v)
    for bar,n in zip(bars,ns):ax.text(bar.get_x()+bar.get_width()/2,n+.3,str(n),ha='center',fontsize=11)
   ax.set_title(title+'\n'+('从起点连续预测'if j==0 else'真实拿起后读新画面再预测'),fontsize=13);ax.set_xticks([0,1,2],['种子0','种子1','种子2']);ax.set_ylim(0,22);ax.set_yticks([0,5,10,15,20]);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
   if j==0:ax.set_ylabel('两物块区域＋Held联合正确数／20')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.95));fig.suptitle('两层干预：同一新测试布局上的三版本对照',fontsize=18,y=.995);fig.text(.5,.02,'蓝：原v1；橙：统一自身运动v2；绿：逐轴交互v3。每柱20独立布局；视觉头和训练预算相同。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.05,1,.91]);figs=root/'figures';figs.mkdir(exist_ok=True)
 for ext in ['png','svg','pdf']:fig.savefig(figs/('before_after_predictions.'+ext),dpi=180)
 plt.close(fig)
 categories=[('transition','loaded',0,'载物：大块和小块B→A'),('transition','empty',0,'旁置：大块B→A，小块留B'),('basic','loaded',1,'叠放：大块到B，小块始终A'),('basic','empty',1,'旁置：大块到B，小块始终A'),('basic','loaded',2,'叠放：两块都到B'),('basic','empty',2,'旁置：两块都到B')]
 fig,ax=plt.subplots(figsize=(12,6))
 for version,offset in [('v1',-.16),('v3',.16)]:
  ns=[next(r for r in actualsummary if r['version']==version and(r['task'],r['stratum'],r['goal'])==c[:3])['success']for c in categories];bars=ax.barh(np.arange(6)+offset,ns,height=.29,color=colors[version],label=version)
  for bar,n in zip(bars,ns):ax.text(n+.15,bar.get_y()+bar.get_height()/2,str(n)+'/20',va='center',fontsize=11)
 ax.set_yticks(range(6),[c[3]for c in categories]);ax.invert_yaxis();ax.set_xlim(0,23);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('实际成功任务数／20（停机保留在分母）');ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False);fig.legend(*ax.get_legend_handles_labels(),frameon=False,loc='upper center',bbox_to_anchor=(.6,.93),ncol=2);fig.suptitle('真实执行对照：同20组新布局，冻结种子0',fontsize=17);fig.text(.5,.015,'蓝：原v1；绿：逐轴交互v3。成功同时要求区域/Held、全过程约束及技能/物理有效性。',ha='center',fontsize=10);fig.tight_layout(rect=[0,.045,1,.86])
 for ext in ['png','svg','pdf']:fig.savefig(figs/('actual_tasks.'+ext),dpi=180)
 plt.close(fig);save(out/'provenance.json',{'confirmation_sha256':hashlib.sha256((root/'confirmation/all_predictions.json').read_bytes()).hexdigest(),'live_sha256':hashlib.sha256((root/'live/all_results.json').read_bytes()).hexdigest(),'no_new_inference_or_fit':True,'seeds_separate':True});print('ANALYSIS_V3',out,flush=True)
if __name__=='__main__':main()
