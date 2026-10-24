"""Visible RGB-D points only. No simulator body metadata, contacts or scene pose."""
import numpy as np

def public_frame(obs):
 if 'rgb_static' in obs:return obs
 result={**obs['rgb_obs'],**obs['depth_obs'],'robot_obs':obs['robot_obs']}
 for camera,mask in obs.get('segmentation',{}).items():result['seg_'+camera]=mask
 for camera,values in obs.get('calibration',{}).items():
  for name,value in values.items():result['calib_'+camera+'_'+name]=np.asarray(value)
 return result

def visible_world_points(frame,camera='static'):
 depth=frame['depth_'+camera];h,w=depth.shape;fov=float(frame['calib_'+camera+'_fov']);focal=h/(2*np.tan(np.deg2rad(fov)/2));v,u=np.indices((h,w));view=np.asarray(frame['calib_'+camera+'_viewMatrix']).reshape(4,4).T
 camera_points=np.stack([(u-w//2)*depth/focal,-(v-h//2)*depth/focal,-depth,np.ones_like(depth)],-1)
 return (camera_points@np.linalg.inv(view).T)[...,:3]

def colour_masks(rgb):
 x=rgb.astype(float)/255.;r,g,b=x.transpose(2,0,1)
 return {'red':(r>.22)&(r>g*2.)&(r>b*1.7),'blue':(b>.15)&(b>r*1.7)&(b>g*1.5),'pink':(r>.24)&(b>.24)&(g<np.minimum(r,b)*.5)&(r<2*b)&(b<2*r)}

def visible_parts(frame,camera='static'):
 xyz=visible_world_points(frame,camera);mask=frame['seg_'+camera];result={}
 for identifier in np.unique(mask):
  if identifier<0:continue
  pixels=mask==identifier;points=xyz[pixels]
  if len(points)<8:continue
  lo=np.quantile(points,.05,axis=0);hi=np.quantile(points,.95,axis=0)
  result[int(identifier)]={'pixels':len(points),'median':np.median(points,axis=0).tolist(),'span_90':(hi-lo).tolist(),'bounds_90':[lo.tolist(),hi.tolist()]}
 return result

def visible_colours(frame):
 output={}
 for camera in ['static','gripper']:
  if 'calib_'+camera+'_viewMatrix' not in frame:continue
  xyz=visible_world_points(frame,camera);masks=colour_masks(frame['rgb_'+camera]);seg=frame.get('seg_'+camera)
  for color,pixels in masks.items():
   candidates=[]
   for identifier in (np.unique(seg[pixels]) if seg is not None else [0]):
    if identifier<0:continue
    selected=pixels&(seg==identifier) if seg is not None else pixels;points=xyz[selected]
    if len(points)<6:continue
    span=np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0)
    if max(span)>.13:continue
    candidates.append({'camera':camera,'pixels':len(points),'point':np.median(points,axis=0).tolist(),'span_90':span.tolist()})
   if candidates:
    candidate=max(candidates,key=lambda x:x['pixels'])
    if color not in output or output[color]['pixels']<candidate['pixels']:output[color]=candidate
 return output

def drawer_front(frame):
 """Identify a visible low rectangular cabinet part by geometry, never body/link ID."""
 candidates=[]
 for identifier,part in visible_parts(frame).items():
  x,y,z=part['median'];sx,sy,sz=part['span_90']
  if .25<=sx<=.52 and .28<=z<=.42 and sz<=.14 and sy<=.46 and part['pixels']>=80:
   candidates.append({'anonymous_mask':identifier,'front_y_m':part['bounds_90'][0][1],**part})
 return max(candidates,key=lambda x:x['pixels']) if candidates else None
