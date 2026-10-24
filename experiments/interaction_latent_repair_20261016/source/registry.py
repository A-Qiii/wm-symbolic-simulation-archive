"""Version 1: colour regions from the CURRENT RGB image only.

No simulator mask, identity, pose, label, future frame, or previous detection is
accepted by this API. Three distinct task colours provide semantic association.
"""
import numpy as np
RULE={'version':'rgb_colour_regions_v1','centres_degrees':[0,240,300],'hue_half_width_degrees':10,'min_saturation':.65,'min_value':.35,'min_component_pixels':4,'connectivity':8,'selection':'largest_connected_component_per_colour_per_current_camera_frame','tracking':'none; immutable colour names, no missing-region imputation or future-frame association'}
def hsv(rgb):
 a=np.asarray(rgb,dtype=np.float32)/255.;mx=a.max(-1);mn=a.min(-1);d=mx-mn;h=np.zeros(mx.shape,np.float32);idx=a.argmax(-1);ok=d>1e-8
 for j,add in [(0,0),(1,2),(2,4)]:
  mask=ok&(idx==j);h[mask]=(((a[..., (j+1)%3][mask]-a[..., (j+2)%3][mask])/d[mask])+add)*60%360
 return h,d/np.maximum(mx,1e-8),mx
def largest_component(mask):
 h,w=mask.shape;seen=np.zeros_like(mask);components=[]
 for y,x in zip(*np.where(mask)):
  if seen[y,x]:continue
  stack=[(int(y),int(x))];seen[y,x]=True;pixels=[]
  while stack:
   yy,xx=stack.pop();pixels.append((yy,xx))
   for dy in [-1,0,1]:
    for dx in [-1,0,1]:
     ny,nx=yy+dy,xx+dx
     if 0<=ny<h and 0<=nx<w and mask[ny,nx]and not seen[ny,nx]:seen[ny,nx]=True;stack.append((ny,nx))
  components.append(pixels)
 out=np.zeros_like(mask);components.sort(key=len,reverse=True)
 if components and len(components[0])>=RULE['min_component_pixels']:
  yy,xx=np.array(components[0]).T;out[yy,xx]=True
 return out,[len(c)for c in components]
def extract(rgb):
 assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[-1]==3
 h,s,v=hsv(rgb);masks=[];details=[]
 for centre in RULE['centres_degrees']:
  dist=np.abs((h-centre+180)%360-180);raw=(dist<=RULE['hue_half_width_degrees'])&(s>=RULE['min_saturation'])&(v>=RULE['min_value']);mask,components=largest_component(raw);masks.append(mask);details.append({'raw_colour_pixels':int(raw.sum()),'component_areas':components,'accepted_pixels':int(mask.sum()),'visible':bool(mask.any())})
 return np.stack(masks),details
