from common import *
from wm_symbolic.public_geometry import colour_masks,visible_world_points
def audit(split):
 count={v:collections.Counter()for v in ['static','gripper']};memo={}
 for e in read(S/f'cache/{split}_metadata.json'):
  for f in e['features']:
   path=f['public_inputs'][-1]
   if path not in memo:
    out={v:collections.Counter()for v in count}
    with np.load(path)as z:frame={k:z[k]for k in z.files}
    for view in count:
     c=out[view];c['object_opportunities']=3
     if any(k not in frame for k in ['rgb_'+view,'seg_'+view,'depth_'+view,'calib_'+view+'_fov','calib_'+view+'_viewMatrix']):c['missing_camera_depth_or_calibration']=3;continue
     dep=frame['depth_'+view];seg=frame['seg_'+view];xyz=visible_world_points(frame,view);masks=colour_masks(frame['rgb_'+view]);foreground=seg>=0;valid=np.isfinite(dep)&(dep>.01)&(dep<5)&np.isfinite(xyz).all(-1);c['foreground_pixels']=int(foreground.sum());c['invalid_depth_foreground_pixels']=int((foreground&~valid).sum())
     for color in COLORS:
      regions=[masks[color]&(seg==i)for i in np.unique(seg[masks[color]&foreground])];rgb=[r for r in regions if r.sum()>=6]
      if not rgb:c['RGB_region_under_6_pixels']=c.get('RGB_region_under_6_pixels',0)+1;continue
      points=[xyz[r&valid]for r in rgb if (r&valid).sum()>=6]
      if not points:c['visible_RGB_but_under_6_valid_depth_points']+=1;continue
      if not any((np.quantile(p,.95,axis=0)-np.quantile(p,.05,axis=0)).max()<=.13 for p in points):c['visible_depth_span_rejected']+=1;continue
      c['usable_RGBD_geometry']+=1
    memo[path]=out
   for view in count:count[view].update(memo[path][view])
 save(S/f'results/depth_visibility_breakdown_{split}.json',{'utc':now(),'split':split,'classification':'observable RGB region absence versus missing/invalid depth versus visible-span rejection; no private truth','counts':count,'no_model_inputs_changed':True});print('MISSING_AUDIT',split,count,flush=True)
if __name__=='__main__':
 guard()
 for s in ['train','development','historical_evaluation','diagnosed_confirmation','fresh_independent']:
  if(S/f'cache/{s}_metadata.json').exists():audit(s)
