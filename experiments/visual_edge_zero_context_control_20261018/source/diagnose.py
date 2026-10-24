from common import *
import collections,csv,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})

def main():
 rows=read(BASE/'data/public/states.json');raw=np.load(BASE/'data/public/raw_features.npz');groups=[];drifts=[];examples=[]
 for path in sorted((BASE/'data/evaluation').glob('*.npz')):
  d=np.load(path);name=path.stem;vis=d['visibility'].astype(bool)
  for head in HEADS:
   n=np.load(BASE/f'configs/norm_{head}_edge.npz');x=(d[head]-n['mean'])/n['std']
   for part,slice_ in [('object',slice(0,74)),('endpoint',slice(74,202)),('relative',slice(202,207))]:
    zz=x[:,:,slice_];drifts.append({'cohort':name,'head':head,'part':part,'standardized_mean_shift_RMS':float(np.sqrt(np.square(zz.mean((0,1))).mean())),'per_state_RMS_median':float(np.median(np.sqrt(np.square(zz).mean((1,2))))),'per_state_RMS_p90':float(np.quantile(np.sqrt(np.square(zz).mean((1,2))),.9))})
  if name not in ['C_all_available','C_validation','official_validation','AB_heldout16','D12_seen_actual135']:continue
  for seed in(0,1):
   for arm in['node','edge']:
    pred=np.load(BASE/f'results/scores/{name}_{arm}_seed{seed}.npz')['scores']>=.5;counters={}
    for i,index in enumerate(d['indices']):
     rr=rows[int(index)];family='intermediate_v3'if'intermediate_v3'in rr['root_id']else'legacy_semantics'if'semantics'in rr['root_id']else'official_state'if rr['root_id'].startswith('official')else'identity_D12'if'identity'in rr['root_id']else'other'
     for j,color in enumerate(COLORS):
      visibility='both'if vis[i,:,j].all()else'static_only'if vis[i,0,j]else'wrist_only'if vis[i,1,j]else'neither'
      g=raw['geometry'][int(index),0,j];lo=np.maximum(g[:2],BOX[:2]);hi=np.minimum(g[2:4],BOX[2:]);area=np.prod(np.maximum(g[2:4]-g[:2],0));frac=float(np.prod(np.maximum(hi-lo,0))/area)if area>0 else None
      overlap='not_detected_static'if frac is None else'outside_ROI'if frac<=1e-8 else'fully_in_ROI'if frac>=1-1e-6 else'partial_ROI'
      for head,off in [('Holding',0),('In',3)]:
       if not d['v'][i,off+j]:continue
       truth=bool(d['y'][i,off+j]);p=bool(pred[i,off+j]);conf='TP'if truth and p else'FN'if truth else'FP'if p else'TN'
       for dimension,level in[('color',color),('origin',family),('visibility',visibility),('target_bbox_drawer_ROI',overlap),('color_x_origin',color+'/'+family),('origin_x_visibility',family+'/'+visibility)]:
        key=(head,dimension,level)
        if key not in counters:counters[key]=collections.Counter({'TP':0,'FN':0,'FP':0,'TN':0})
        counters[key][conf]+=1
    for (head,dimension,level),c in counters.items():groups.append({'cohort':name,'seed':seed,'arm':arm,'head':head,'dimension':dimension,'level':level,**dict(c)})
 save(E/'results/error_decomposition.json',groups);save(E/'results/feature_domain_drift.json',drifts)
 for name,rr in [('error_decomposition',groups),('feature_domain_drift',drifts)]:
  with(R/f'{name}.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
 flips=read(BASE/'results/decision_flips.json');cfirst=sorted([r for r in flips if r['cohort']=='C_validation'and r['head']=='In'and r['seed']==0 and r['effect']=='regression'and not r['truth']],key=lambda r:(r['root_id'],r['state_id'],r['color']))[0]
 for scene in ['A','B','C','D']:
  if scene=='C':rr=rows[cfirst['index']]
  elif scene in['A','B']:rr=next(r for r in rows if r['scene']==scene and r['train']and r['state_id'].startswith('official:'))
  else:rr=next(r for r in rows if r['scene']=='D')
  examples.append(rr)
 fig,axes=plt.subplots(2,4,figsize=(12,6.8));cols=['#df3434','#315ae5','#c832b6']
 for j,r in enumerate(examples):
  z=np.load(r['path'])
  for cam,key in enumerate(['rgb_static','rgb_gripper']):
   im=z[key];h,w=im.shape[:2];ax=axes[cam,j];ax.imshow(im,interpolation='nearest');ax.set(xticks=[],yticks=[],title=r['scene']+(' 静态抽屉区域'if cam==0 else' 腕部完整视野'))
   for k in range(3):
    g=raw['geometry'][r['index'],cam,k]
    if g[4]:ax.add_patch(Rectangle((g[0]*w,g[1]*h),(g[2]-g[0])*w,(g[3]-g[1])*h,fill=False,ec=cols[k],lw=1.2))
   box=BOX if cam==0 else np.array([0.,0.,1.,1.]);ax.add_patch(Rectangle((box[0]*w,box[1]*h),(box[2]-box[0])*w-1,(box[3]-box[1])*h-1,fill=False,ec='#00c4c9',ls='--',lw=2))
 fig.suptitle('固定像素区域跨场景审阅：包含抽屉，也混入桌面、机械臂和背景',fontsize=14)
 fig.text(.5,.015,'青色虚框为同一归一化ROI；实框由当前RGB颜色检测得到。A/B取训练源，C取首个新增In误报，D取既有诊断。并非仿真投影或分割质量真值。',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(R/'figures/ABCD_fixed_ROI_review.png',dpi=150);plt.close(fig)
 save(E/'results/ROI_review_sources.json',{'utc':now(),'examples':examples,'selection':'A/B first official training frame; C first seed0 In new FP; D first existing diagnostic','no_GT_projection_or_masks':True,'quantitative_quality_proxy':'2D target bbox overlap with fixed ROI is not drawer segmentation IoU; cropped context may contain other objects/background','network_ROIs_unchanged':True})
 shutil.copy2(__file__,E/'source/diagnose.py');print('READONLY_DOMAIN_DECOMPOSITION_COMPLETE',len(groups),'stratified cells',flush=True)
if __name__=='__main__':main()
