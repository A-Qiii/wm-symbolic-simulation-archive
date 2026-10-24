from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64,html,re,ast
ARM_COLORS={'linear':'#356ab8','mlp':'#008b83','constant':'#888888','original_coverage':'#555555','original_control':'#aaaaaa','geometry_learned':'#cb8e18','geometry_rule':'#8d5a00'}
def ci(v,seed=261019):
 v=np.asarray(v,float)
 if not len(v):return [None,None]
 rng=np.random.default_rng(seed);b=v[rng.integers(0,len(v),(2000,len(v)))].mean(1);return np.quantile(b,[.025,.975]).tolist()
def fmt(x,n=2):return '—'if x is None else f'{x:.{n}f}'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v)for v in r)+'|'for r in rows])
def prediction(seed,split,kind,t):
 if kind in ['linear','mlp']:
  q=np.load(S/f'data/visual_{kind}_seed{seed}_{split}.npz');assert np.array_equal(q['episode_id'],t['episode_id'])and np.array_equal(q['step'],t['step']);x=q['prediction'][...,0]*WIDTH+LO;return classes(x),x
 if kind=='constant':x=np.full_like(t['x'],read(S/'configs/constant_baseline.json')['q']*WIDTH+LO);return classes(x),x
 if kind=='geometry_rule':return classes(t['x']),None
 if kind=='geometry_learned':return np.load(S/f'data/privileged_geometry_geometry_seed{seed}_{split}.npz')['prediction'].argmax(-1),None
 arm=kind.replace('original_','');q=dict(np.load(R/f'data/predictions_{arm}_seed{seed}_{split}.npz'));assert np.array_equal(q['episode_id'],t['episode_id'])and np.array_equal(q['step'],t['step'])and np.array_equal(q['cat'][:,:3],t['cat']);return q['prob'][:,:24].reshape(-1,3,8).argmax(-1),None
def main():
 sys.addaudithook(guard);start=time.monotonic();metrics=[];roots=[];pairs=[];confusions=[];bins=[];examples=[];lookup={};cache={};GT={s:target(s)for s in SPLITS};selected={x['seed']:x['selected']for x in read(S/'configs/development_selection.json')['selection']}
 for split,t in GT.items():
  for seed in [0,1]:
   arms=['linear','mlp','constant','geometry_rule','geometry_learned']+(['original_control','original_coverage']if split in ['old_band','gap']else[])
   for arm in arms:
    pred,x=prediction(seed,split,arm,t);cache[split,seed,arm]=(pred,x);margin=np.min(abs(t['x'][...,None]-CUTS),-1);error=abs(x-t['x'])*1000 if x is not None else None
    for cohort in ['h0','all_boundaries']:
     mask=t['mask']&(t['h0'][:,None]if cohort=='h0'else True);accuracy,ids=root_values(pred==t['cat'],mask,t['root_id']);lo,hi=ci(accuracy,261019+seed);row={'split':split,'seed':seed,'cohort':cohort,'arm':arm,'roots':len(ids),'objects':int(mask.sum()),'correct':int(((pred==t['cat'])&mask).sum()),'root_accuracy':float(accuracy.mean()),'accuracy_ci_low':lo,'accuracy_ci_high':hi,'MAE_mm':None,'RMSE_mm':None,'margin_sufficient':None};vals={'accuracy':accuracy}
     if x is not None:
      mae,_=root_values(error,mask,t['root_id']);mse,_=root_values(error**2,mask,t['root_id']);safe,_=root_values(error<margin*1000,mask,t['root_id']);row.update(MAE_mm=float(mae.mean()),RMSE_mm=float(np.sqrt(mse.mean())),margin_sufficient=float(safe.mean()));maeci=ci(mae,261019+seed);row.update(MAE_ci_low=maeci[0],MAE_ci_high=maeci[1]);vals.update(MAE_mm=mae,MSE_mm2=mse,margin_sufficient=safe)
     metrics.append(row);lookup[split,seed,cohort,arm]=row
     for name,v in vals.items():
      for rid,value in zip(ids,v):roots.append({'split':split,'seed':seed,'cohort':cohort,'arm':arm,'root_id':rid,'metric':name,'value':float(value)})
     for truth in range(3):
      for p in range(8):confusions.append({'split':split,'seed':seed,'cohort':cohort,'arm':arm,'truth':truth,'prediction':p,'count':int((mask&(t['cat']==truth)&(pred==p)).sum())})
     for lower,upper in [(5,10),(10,20),(20,40),(40,float('inf'))]:
      mm=mask&(margin*1000>=lower)&(margin*1000<upper);v,rr=root_values(pred==t['cat'],mm,t['root_id']);bins.append({'split':split,'seed':seed,'cohort':cohort,'arm':arm,'margin_low_mm':lower,'margin_high_mm':upper if upper<float('inf')else'Infinity','objects':int(mm.sum()),'roots':len(rr),'correct':int((mm&(pred==t['cat'])).sum()),'root_accuracy':float(v.mean())if len(v)else None})
    if arm in ['linear','mlp']:
     for i,j in zip(*np.where(t['mask'])):examples.append({'split':split,'seed':seed,'arm':arm,'root_id':str(t['root_id'][i]),'episode_id':str(t['episode_id'][i]),'boundary':int(t['step'][i]),'h0':bool(t['h0'][i]),'color':COLORS[j],'array_index':int(i),'object_index':int(j),'truth':int(t['cat'][i,j]),'prediction':int(pred[i,j]),'true_x_m':float(t['x'][i,j]),'predicted_x_m':float(x[i,j]),'signed_error_mm':float((x[i,j]-t['x'][i,j])*1000),'absolute_error_mm':float(error[i,j]),'boundary_margin_mm':float(margin[i,j]*1000),'margin_sufficient':bool(error[i,j]<margin[i,j]*1000),'availability':float(t['availability'][i,j]),'public_frame':str(t['public_frame'][i])})
   for cohort in ['h0','all_boundaries']:
    comparisons=[('mlp','linear','MAE_mm'),('mlp','linear','accuracy')]+([(a,b,'accuracy')for a in ['linear','mlp']for b in ['original_coverage','original_control']]if split in ['old_band','gap']else[])
    for a,b,metric in comparisons:
     aa={x['root_id']:x['value']for x in roots if (x['split'],x['seed'],x['cohort'],x['arm'],x['metric'])==(split,seed,cohort,a,metric)};bb={x['root_id']:x['value']for x in roots if (x['split'],x['seed'],x['cohort'],x['arm'],x['metric'])==(split,seed,cohort,b,metric)};assert aa.keys()==bb.keys();v=np.array([aa[r]-bb[r]for r in sorted(aa)]);factor=100 if metric=='accuracy'else 1.;lo,hi=ci(v*factor,261019+seed);pairs.append({'split':split,'seed':seed,'cohort':cohort,'arm_a':a,'arm_b':b,'metric':metric,'units':'pp'if metric=='accuracy'else'mm','difference_a_minus_b':float(v.mean()*factor),'ci_low':lo,'ci_high':hi,'independent_roots':len(v),'bootstrap_replicates':2000})
 save(S/'results/metrics.json',metrics);csvwrite(S/'results/metrics.csv',metrics);csvwrite(S/'results/root_statistics.csv',roots);csvwrite(S/'results/paired_differences.csv',pairs);csvwrite(S/'results/confusions.csv',confusions);csvwrite(S/'results/margin_bins.csv',bins);csvwrite(S/'results/per_object.csv',examples)
 # Planned scientific charts. h0 is the inferentially independent primary view.
 figures=[]
 def finish(fig,name):
  if not name.startswith('case_'):fig.tight_layout(pad=2.0)
  fig.savefig(S/'figures'/name,dpi=160,bbox_inches='tight');plt.close(fig);figures.append(name)
 fig,axes=plt.subplots(2,2,figsize=(11,9))
 for row,split in enumerate(['old_band','gap']):
  t=GT[split];mask=t['mask']&t['h0'][:,None]
  for seed in [0,1]:
   ax=axes[row,seed]
   for arm,marker in [('linear','o'),('mlp','x')]:ax.scatter(t['x'][mask],cache[split,seed,arm][1][mask],label=arm,color=ARM_COLORS[arm],marker=marker,alpha=.8)
   ax.plot([-.44,.44],[-.44,.44],'k--',lw=1,label='ideal x prediction')
   for cut in CUTS:ax.axvline(cut,color='gray',ls=':',lw=.8);ax.axhline(cut,color='gray',ls=':',lw=.8)
   ax.set(xlabel='True table-local x (m)',ylabel='Predicted table-local x (m)',title=f'{split} | E{seed} | h0: 10 roots / 28 objects');ax.legend(fontsize=8)
 finish(fig,'position_scatter.png')
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for row,split in enumerate(['old_band','gap']):
  for seed in [0,1]:
   ax=axes[row,seed];q=[x for x in roots if x['split']==split and x['seed']==seed and x['cohort']=='h0'and x['metric']=='MAE_mm'and x['arm']in ['linear','mlp']];rids=sorted(set(x['root_id']for x in q))
   for rid in rids:
    vals=[next(x['value']for x in q if x['root_id']==rid and x['arm']==a)for a in ['linear','mlp']];ax.plot([0,1],vals,color='#999999',alpha=.65,lw=1);ax.scatter([0,1],vals,color=[ARM_COLORS['linear'],ARM_COLORS['mlp']],s=22)
   ax.set(xticks=[0,1],xticklabels=['Linear','MLP'],ylabel='Per-root mean absolute x error (mm)',title=f'{split} | E{seed} | h0');ax.grid(axis='y',alpha=.2)
 finish(fig,'paired_root_error.png')
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for row,split in enumerate(['old_band','gap']):
  for seed in [0,1]:
   ax=axes[row,seed]
   for arm in ['linear','mlp','original_coverage']:
    bb=[b for b in bins if b['split']==split and b['seed']==seed and b['cohort']=='h0'and b['arm']==arm];yy=[np.nan if b['root_accuracy']is None else 100*(1-b['root_accuracy'])for b in bb];ax.plot(range(4),yy,'o-',label=arm,color=ARM_COLORS[arm])
   bb=[b for b in bins if b['split']==split and b['seed']==seed and b['cohort']=='h0'and b['arm']=='linear'];ax.set(xticks=range(4),xticklabels=[f'{lo}-{hi}\nn={b["objects"]}'for lo,hi,b in zip([5,10,20,40],[10,20,40,'inf'],bb)],ylim=(-3,103),ylabel='Root-weighted region error (%)',xlabel='True nearest-boundary margin (mm)',title=f'{split} | E{seed} | h0');ax.legend(fontsize=8)
 finish(fig,'boundary_margin_error.png')
 fig,axes=plt.subplots(2,4,figsize=(14,7))
 for ri,split in enumerate(['old_band','gap']):
  t=GT[split];mask=t['mask']&t['h0'][:,None]
  for seed in [0,1]:
   for ai,arm in enumerate(['linear','mlp']):
    ax=axes[ri,seed*2+ai];pred=cache[split,seed,arm][0];mat=np.array([[np.sum(mask&(t['cat']==yy)&(pred==pp))for pp in range(3)]for yy in range(3)]);ax.imshow(mat,cmap='Blues',vmin=0,vmax=22)
    for y in range(3):
     for x in range(3):ax.text(x,y,str(mat[y,x]),ha='center',va='center',color='white'if mat[y,x]>12 else'black')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=['L','M','R'],yticklabels=['L','M','R'],xlabel='Predicted region',ylabel='True region',title=f'{split} E{seed} {arm}')
 finish(fig,'region_confusions.png')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for seed in [0,1]:
  for arm in ['linear','mlp','geometry']:
   rr=read(S/f'results/fit_{arm}_seed{seed}.json');axes[seed].plot([x['update']for x in rr['curve']],[x['loss']for x in rr['curve']],label=arm)
  axes[seed].set(title=f'E{seed}: geometry seed paired by index',xlabel='Update',ylabel='Training objective (different losses)');axes[seed].legend()
 finish(fig,'training_losses.png')
 # Independent illustrative roots, prioritizing h0. Rules never select models.
 candidates=[]
 for split in ['old_band','gap']:
  t=GT[split]
  for i,j in zip(*np.where(t['mask'])):
   y=int(t['cat'][i,j]);old=[int(cache[split,k,'original_coverage'][0][i,j])for k in [0,1]];lin=[int(cache[split,k,'linear'][0][i,j])for k in [0,1]];mlp=[int(cache[split,k,'mlp'][0][i,j])for k in [0,1]];sel=[int(cache[split,k,selected[k]][0][i,j])for k in [0,1]];kind=[]
   if all(v!=y for v in old)and all(v==y for v in sel):kind.append('selected_common_fix')
   if all(v==y for v in old)and all(v!=y for v in sel):kind.append('selected_common_regression')
   if all(v!=y for v in sel):kind.append('selected_common_error')
   if all(v!=y for v in lin)and all(v==y for v in mlp):kind.append('mlp_fixes_both_linear_heads')
   if all(v==y for v in lin)and all(v!=y for v in mlp):kind.append('mlp_breaks_both_linear_heads')
   if len(set(sel))>1:kind.append('selected_seed_disagreement')
   for label in kind:candidates.append({'split':split,'root_id':str(t['root_id'][i]),'episode_id':str(t['episode_id'][i]),'boundary':int(t['step'][i]),'h0':bool(t['h0'][i]),'index':int(i),'object_index':int(j),'color':COLORS[j],'truth':y,'case_type':label,'true_x_m':float(t['x'][i,j]),'boundary_margin_mm':float(np.min(abs(t['x'][i,j]-CUTS))*1000),'availability':float(t['availability'][i,j]),'original_coverage':old,'linear':lin,'mlp':mlp,'selected':sel,'public_frame':str(t['public_frame'][i])})
 cases=[];used=set()
 for kind in ['selected_common_fix','selected_common_regression','selected_common_error','mlp_fixes_both_linear_heads','mlp_breaks_both_linear_heads','selected_seed_disagreement']:
  possible=[c for c in candidates if c['case_type']==kind and c['root_id']not in used]
  if not possible:continue
  c=min(possible,key=lambda c:(not c['h0'],c['boundary'],c['root_id'],c['object_index']));used.add(c['root_id']);cases.append(c)
 for z,c in enumerate(cases):
  t=GT[c['split']];i=c['index'];j=c['object_index'];frame=np.load(c['public_frame']);fig=plt.figure(figsize=(12,5));ax=fig.add_axes([.02,.2,.30,.70]);ax.imshow(frame['rgb_static']);ax.axis('off');ax.set_title('Static RGB (current frame)');ax=fig.add_axes([.34,.29,.22,.56]);ax.imshow(frame['rgb_gripper']);ax.axis('off');ax.set_title('Gripper RGB');lines=[];c['details']=[]
  for seed in [0,1]:
   lines.append(f'E{seed} original coverage: '+['left','middle','right','drawer','sliderL','sliderR','held','other'][c['original_coverage'][seed]])
   for arm in ['linear','mlp']:
    x=float(cache[c['split'],seed,arm][1][i,j]);error=(x-c['true_x_m'])*1000;pred=int(cache[c['split'],seed,arm][0][i,j]);original=cache[c['split'],seed,arm][0][i].copy();substituted=original.copy();substituted[j]=t['cat'][i,j];mask=t['mask'][i];detail={'seed':seed,'arm':arm,'pred_x_m':x,'signed_error_mm':error,'absolute_error_mm':abs(error),'pred_region':pred,'selected_by_dev':arm==selected[seed],'offline_one_object_x_substitution':{'table_exact_before':bool(np.all(original[mask]==t['cat'][i,mask])),'table_exact_after':bool(np.all(substituted[mask]==t['cat'][i,mask])),'only_focal_x_replaced_by_GT_for_scoring':True,'model_forward':False}};c['details'].append(detail);lines.append(f'  {arm}: x={x:.4f}m, error={error:+.1f}mm, '+['left','middle','right'][pred]+(' [dev selected]'if arm==selected[seed]else''))
   lines.append('')
  fig.text(.58,.78,'\n'.join(lines),fontsize=10,va='top');fig.suptitle(c['episode_id']+f' | boundary {c["boundary"]} | {c["case_type"]}',fontsize=12);fig.text(.02,.08,f'Focal object: {c["color"]} | true region: '+['left','middle','right'][c['truth']]+f' | true x={c["true_x_m"]:.4f}m\nNearest region-boundary margin={c["boundary_margin_mm"]:.1f}mm | public ROI fraction={c["availability"]:.3f}',fontsize=11);name=f'case_{z:02d}.png';c['figure']=name;finish(fig,name)
 save(S/'results/selected_cases.json',cases)
 # Mask-only visibility summaries, including unavailable observations rather than dropping them.
 visibility=[]
 for split,t in GT.items():
  for seed in [0,1]:
   for arm in ['linear','mlp']:
    p,x=cache[split,seed,arm]
    for available in [False,True]:
     m=t['mask']&((t['availability']>0)==available);v,rr=root_values(abs(x-t['x'])*1000,m,t['root_id']);acc,_=root_values(p==t['cat'],m,t['root_id']);visibility.append({'split':split,'seed':seed,'arm':arm,'available':available,'objects':int(m.sum()),'roots':len(rr),'root_MAE_mm':float(v.mean())if len(v)else None,'root_accuracy':float(acc.mean())if len(acc)else None})
 csvwrite(S/'results/visibility.csv',visibility)
 # Report uses predeclared metrics; narrative explicitly distinguishes privileged controls.
 pf=read(S/'results/preflight.json');co=read(S/'results/compute_complete.json');dev=read(S/'configs/development_selection.json');md=['# 固定观测特征的位置读出实验','本实验已实际完成：6 个小头、9000 次更新，全部使用已有 A/B 缓存。没有新增采集、没有编码器前向、没有 RSSM 更新、没有读取 C/D 样本，也没有上传文件。所有模型固定训练到第1500步；开发集仅在两个预定读出结构间选择，未根据历史评估结果重训。','这是 GT 确认物体在桌面上时的位置诊断，不是完整感知、持物关系或世界模型闭环验收。训练40根，开发10根，old_band与gap各10根，都是历史数据的探索性复用。', '## 现在证明了什么']
 for seed in [0,1]:
  d=dev['selection'][seed];md.append(f'E{seed} 开发集 h0 根平均 MAE：线性 {d["validation_h0_root_MAE_mm"]["linear"]:.2f} 毫米、MLP {d["validation_h0_root_MAE_mm"]["mlp"]:.2f} 毫米，因此预定规则选择 {d["selected"]}。选择发生在两个评估队列预测之前，两个读出头仍完整保留并分别报告。')
 md+=['E0 的 MLP 在 old_band/gap 的 h0 分别比线性降低18.81/23.39毫米误差，两项同根95%区间均低于0；E1对应差值为增加3.26/9.96毫米，区间跨0。复杂读出对这一冻结表示有局部帮助，但两套E未给出一致收益。E和读出初始化同时按种子变化，不能把二者差异全归因于编码器。','按开发选择，E0-MLP的 old_band/gap 区域准确率相对原覆盖头为−5.00/+5.00个百分点；E1-线性为−28.33/−3.33个百分点。只有E1在old_band的下降区间明确低于0。因此不能把位置探针当成已经优于原分类头的修复方案。','真实几何规则和可学习几何头在两个评估队列上均100%正确，支持这些观测点在准确几何给定时能够被简单下游区分；但可学习几何头在更宽的训练分布上仍有错误（见独立上限表），不能宣称所有边界都已学准。固定视觉表示可能保留可用的区域信息，同时仍不容易在当前训练目标和预算下精确回归坐标。']
 md+=['## 初始状态的主要结果','每个队列10个独立根、28个有效桌面物体。MAE先在每根内平均，再对根平均；准确率同样按根平均，因此不必等于正确物体数/28。95%区间使用2000次根重采样，不能把多个物体、分支、重复起点当作独立根。']
 rr=[]
 for split in ['old_band','gap']:
  for seed in [0,1]:
   for arm in ['constant','linear','mlp','original_coverage','original_control']:
    v=lookup[split,seed,'h0',arm];rr.append([split,f'E{seed}',arm,fmt(v['MAE_mm']),fmt(v['RMSE_mm']),fmt(v['root_accuracy']*100)+'%',f'{v["correct"]}/{v["objects"]}',fmt(v['margin_sufficient']*100 if v['margin_sufficient']is not None else None)+'%'if v['margin_sufficient']is not None else'—'])
 md.append(table(['队列','冻结E','读出','MAE mm','RMSE mm','根平均区域准确率','正确物体','误差小于边界余量'],rr))
 rr=[]
 for p in pairs:
  if p['split']in ['old_band','gap']and p['cohort']=='h0'and (p['metric']=='MAE_mm'or p['arm_b']=='original_coverage'):
   rr.append([p['split'],f'E{p["seed"]}',p['arm_a']+' − '+p['arm_b'],p['metric'],fmt(p['difference_a_minus_b'])+' '+p['units'],'['+fmt(p['ci_low'])+', '+fmt(p['ci_high'])+']'])
 md+=['## 同根配对差异','位置误差差值为负表示 MLP 更好；准确率差值为正表示位置探针优于原覆盖分类头。每项均使用同一批根和物体，原分类头沿用八类全 argmax，预测成 held/drawer 等也计错；没有只在三类内重新归一化提高原头成绩。',table(['队列','E','对比','指标','差值','根配对95%区间'],rr)]
 rr=[]
 for split in ['train','validation','old_band','gap']:
  for seed in [0,1]:
   for arm in ['geometry_rule','geometry_learned']:
    for cohort in ['h0','all_boundaries']:
     v=lookup[split,seed,cohort,arm];rr.append([split,cohort,f'seed {seed}',arm,fmt(v['root_accuracy']*100)+'%',f'{v["correct"]}/{v["objects"]}'])
 md+=['## 特权几何上限（单独列示）','下面直接输入 GT 的两个有符号边界距离。规则上限验证坐标与标签接线；可学习几何头检验已知准确位置时，小头能否学到边界。它们不是视觉模型成绩，也不进入世界模型或规划器。',table(['队列','视图','初始化','条件','根平均区域准确率','正确物体'],rr),'可学习几何头在训练集全部边界上分别为1322/1364、1312/1364个物体正确，未达到规则上限。评估h0队列的5–10毫米分箱没有样本，old_band的10–20毫米分箱也没有样本，因此评估100%不能外推为整个连续桌面的边界读出已解决。','## 训练拟合与外部误差','下表均为全部边界的根平均MAE（毫米），不是只报训练损失。MLP即使在训练集也未精确拟合全部坐标，因此评估失败不能直接证明编码器丢失了位置信息；固定容量、目标和优化预算仍是未分离因素。',table(['E','头','训练','开发','old_band','gap'],[[f'E{seed}',arm]+[fmt(lookup[split,seed,'all_boundaries',arm]['MAE_mm'])for split in ['train','validation','old_band','gap']]for seed in [0,1]for arm in ['linear','mlp']])]
 md+=['## 误差与边界的关系','坐标目标是物理桌面局部 x，范围[-0.44,0.44]米，分区边界固定±0.146667米。误差必须与到最近边界的真实余量一起看：几毫米误差也可能翻过很近的边界，而较大误差可能仍停留在同一区。误差小于余量是保持区域的充分条件，不是必要条件。','![位置散点](figures/position_scatter.png)','横轴为真实桌面x、纵轴为预测x，单位均为米；蓝圆为线性、青色叉为MLP，黑虚线为理想预测，灰点线为两条区域边界。按评估队列和冻结E分面，比较是否偏离理想线及是否跨区。','![同根误差](figures/paired_root_error.png)','每条灰线连接同一个根的线性与MLP误差，纵轴为毫米；蓝点和青点分别为两种读出。下降表示该根受益，上升表示退化，避免只看总体平均掩盖根间差异。','![边界余量](figures/boundary_margin_error.png)','横轴是真值到最近分区边界的距离分箱（毫米），n为该箱物体数；纵轴为根平均区域错误率。蓝线为线性、青线为MLP、灰线为原覆盖分类头。空箱不连出成绩，不把空分母视为零错误。','![混淆矩阵](figures/region_confusions.png)','行是真实区域、列是预测区域，L/M/R分别是左/中/右；数字是物体计数，蓝色越深计数越多，各面板使用相同色阶。对角线正确，非对角线显示具体跨界方向。','## 核验、范围与限制',f'3150条物体身份与标签记录对齐，错误身份0；70个根无重复初始物理状态，训练旧新配对根族未流入开发集。物理网格边界、世界位置减桌面原点、已保存局部坐标三者一致，最大差异 {pf["table_coordinate_max_difference_m"]} 米。原生A/B桌面原点和旋转均为零，但仍显式执行并核验变换。验证标签的有效性从原记录恢复，桌面近界无效数为 {pf["table_invalid_objects"]}。',table(['队列','根','边界记录','有效桌面物体','非桌面物体','桌面不可见'],[[v['split'],v['roots'],v['boundaries'],v['table_valid_objects'],v['non_table_objects'],v['eligible_unavailable_objects']]for v in pf['counts']]),'GT仅用于普通探针目标、有效性掩码与评分。普通forward输入仅有物体512维、全局512维、固定颜色one-hot和公开availability；不可见物体保留零物体向量。该冻结表示已经融合DINO、公开二维框、机器人状态，因此成功不能归功于纯DINO。','线性与MLP同时失败并不能证明位置信息已被不可逆删除；有限样本、固定预算、特征到几何的可读性和优化仍无法完全分离。相较旧八类分类头，本轮还改变了监督目标、结构和身份查询，因此不能把所有变化归因于单个因素。','六个模型均只拟合一次，无按评估结果修正、无追加种子。开发选择仅影响报告的预定首选模型，不丢弃另一个结果。完整原指南的学习转移与TS/LTLf闭环验收未由本实验替代。','## 具体独立根案例']
 for c in cases:
  md+=[f'案例：{c["episode_id"]}，边界{c["boundary"]}，{c["color"]}，{c["case_type"]}。'+('这是h0初始状态。'if c['h0']else'这是动作后的描述性案例，不增加独立样本数。')+f'真实x={c["true_x_m"]:.5f}米，最近区域边界余量={c["boundary_margin_mm"]:.1f}毫米。','!['+c['case_type']+'](figures/'+c['figure']+')','左侧为同一边界的静态与夹爪RGB，右侧逐项列出原分类头区域、两种探针的位置/有符号误差/区域和开发集选择。颜色名指实际物体；正误差表示向桌面+x方向偏移。这里只替换该物体的位置读出做评分诊断，没有执行后续动作或补跑轨迹。']
  why={'selected_common_fix':'两套E的开发首选头都修复原分类头，但E0-MLP仍有约79毫米误差；因为该物体离边界约121毫米，区域正确不等于定位精确。','selected_common_regression':'第一个有影响的错误是位置回归向左越过边界：真实余量26.2毫米，四个探针误差都超过这一余量，原分类头却正确。它反驳了“换成坐标监督一定更好”，不能被解释为物体不可见。','selected_common_error':'蓝块在静态图清楚可见，公开ROI可用。原分类头在同一冻结特征上判断正确，四个位置探针却向中间偏移约202–302毫米。这个反例更支持当前连续读出/训练方案不足，不能声称区域信息完全从表示中消失。','mlp_fixes_both_linear_heads':'两套E上MLP都把同一物体拉回正确区域，支持该具体观测存在可被有限非线性读出的信息；开发集仍为E1选择线性，不能凭此单例改选或追加训练。','mlp_breaks_both_linear_heads':'线性头正确且误差分别约8.7和0.7毫米，两个MLP却向左越过28.2毫米余量。增加非线性既能修复也能引入错误，不能只展示改善案例。','selected_seed_disagreement':'同一物体的两套E表现不同，但E与读出初始化同时变化；这只是模型组合差异，不能独立证明编码器或某个颜色机制是原因。'}
  md.append(why[c['case_type']]);md.append(table(['E','头','只替换该物体真实x前：该帧桌面物体全对','替换后全对'],[[f'E{v["seed"]}',v['arm'],v['offline_one_object_x_substitution']['table_exact_before'],v['offline_one_object_x_substitution']['table_exact_after']]for v in c['details']]));md.append('该单项替换仅在已保存预测上离线重评分，不向普通模型输入GT，不证明闭环控制成功；如果替换后仍错，说明同一帧还有其他物体错误。')
 md+=['## 训练与资源','![训练曲线](figures/training_losses.png)','横轴为更新数，纵轴为各自训练目标。线性/MLP使用SmoothL1，几何头使用交叉熵；损失曲线仅检查训练过程，不能跨不同目标比较数值大小判断能力。',f'实际计算墙钟 {co["seconds"]:.2f} 秒，PyTorch峰值分配 {co["peak_cuda_GiB"]:.3f} GiB；串行完成6个小头、总9000更新。此时间不含脚本编写、报告制作和输入核验。没有因实际耗时低于估计而扩大实验预算。','全量逐物体误差、根统计、配对差异、边界分箱、混淆计数、不可见子集统计分别保存在 results/per_object.csv、root_statistics.csv、paired_differences.csv、margin_bins.csv、confusions.csv、visibility.csv。模型、配置、目标与来源索引分目录保存。']
 text='\n\n'.join(md)+'\n';(S/'REPORT.md').write_text(text)
 source_text=(O/'source/report.py').read_text();node=next(n for n in ast.parse(source_text).body if isinstance(n,ast.FunctionDef)and n.name=='render_markdown');ns={'re':re,'html':html};exec(compile(ast.Module(body=[node],type_ignores=[]),'<render>','exec'),ns);body=ns['render_markdown'](text)
 for name in figures:body=body.replace('figures/'+name,'data:image/png;base64,'+base64.b64encode((S/'figures'/name).read_bytes()).decode())
 (S/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>固定观测特征的位置读出实验</title><style>body{max-width:1180px;margin:40px auto;font:16px/1.75 system-ui;color:#203047;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd6df;padding:7px;text-align:left}th{background:#edf4f6}img{max-width:100%}h1,h2{color:#126b6c}figure{margin:24px 0}p{overflow-wrap:anywhere}</style>'+body+'</html>')
 assert 'torch'not in sys.modules and 'pybullet'not in sys.modules;save(S/'results/report_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'figures':len(figures),'distinct_case_roots':len(cases),'metrics_rows':len(metrics),'paired_rows':len(pairs),'new_fits':0,'encoder_forwards':0,'C_D_data_access':0});print('REPORT_COMPLETE',len(metrics),len(cases),flush=True)
if __name__=='__main__':main()
