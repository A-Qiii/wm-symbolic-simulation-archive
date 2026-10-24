from common import *
import torch,torch.nn.functional as F,fcntl,sys,time
from transformers import Dinov2Model

def main():
 assert not(E/'results/inference.json').exists()
 for p,h in read(E/'configs/inference_frozen.json')['sha256'].items():assert sha(p)==h,p
 rows=read(E/'data/states.json');templates=read(E/'configs/templates.json')['annotations'];original=[];metadata=[]
 for r in rows:
  z=np.load(r['path']);assert set(z.files)=={'rgb_static','rgb_gripper'}
  for cam,key in enumerate(['rgb_static','rgb_gripper']):
   rgb=z[key].copy();assert rgb.dtype==np.uint8;original.append(rgb);metadata.append({**r,'camera':cam,'width':rgb.shape[1],'height':rgb.shape[0]})
 images=original.copy();contexts=[]
 for t in templates:
  ii=next(i for i,r in enumerate(metadata)if r['index']==t['index']and r['camera']==t['camera']);rgb=original[ii];x0,y0,x1,y1=t['box'];gray=np.full_like(rgb,128);gray[y0:y1,x0:x1]=rgb[y0:y1,x0:x1];c1=rgb[y0:y1,x0:x1].copy();c2=gray[y0:y1,x0:x1].copy();assert np.array_equal(c1,c2)
  contexts.append({'template':t['name'],'original_image':ii,'gray_image':len(images),'crop_original':len(images)+1,'crop_gray':len(images)+2});images.extend([gray,c1,c2])
 assert len(images)==40
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('Inference cannot read any other experiment')
   if p.is_relative_to(E/'data')and p.suffix=='.npz':pass # Only previously validated two-RGB files exist here.
 sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.init();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();model=Dinov2Model.from_pretrained(DINO,local_files_only=True).eval().requires_grad_(False).cuda();before=sha(DINO/'model.safetensors');mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 def statehash():return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes()for k,v in sorted(model.state_dict().items()))).hexdigest()
 init=statehash();tokens=[]
 prepared=[F.interpolate(torch.from_numpy(im).permute(2,0,1)[None].float(),size=(224,224),mode='bilinear',align_corners=False)[0]/255 for im in images]
 for t,c in zip(templates,contexts):
  r=metadata[c['original_image']];xx=(np.arange(16)+.5)*r['width']/16;yy=(np.arange(16)+.5)*r['height']/16;XX,YY=np.meshgrid(xx,yy);x0,y0,x1,y1=t['box'];mask=(XX>=x0)&(XX<x1)&(YY>=y0)&(YY<y1);pixels=torch.tensor(np.repeat(np.repeat(mask,14,0),14,1));orig=prepared[c['original_image']];modified=torch.full_like(orig,128/255);modified[:,pixels]=orig[:,pixels];assert torch.equal(modified[:,pixels],orig[:,pixels]);prepared[c['gray_image']]=modified
 for i in range(0,len(images),8):
  batch=torch.stack(prepared[i:i+8]).cuda()
  with torch.inference_mode():out=model(pixel_values=(batch-mu)/sd).last_hidden_state[:,1:]
  assert out.shape[1:]==(256,384);tokens.append(out.cpu().numpy())
 assert init==statehash()and before==sha(DINO/'model.safetensors');tok=np.concatenate(tokens);np.savez_compressed(E/'results/patch_tokens.npz',tokens=tok)
 def selector(box,w,h):
  xx=(np.arange(16)+.5)*w/16;yy=(np.arange(16)+.5)*h/16;X,Y=np.meshgrid(xx,yy);x0,y0,x1,y1=box;mask=(X>=x0)&(X<x1)&(Y>=y0)&(Y<y1);assert mask.any();return mask.reshape(-1)
 descriptors=[]
 for t in templates:
  i=next(i for i,r in enumerate(metadata)if r['index']==t['index']and r['camera']==t['camera']);r=metadata[i];mask=selector(t['box'],r['width'],r['height']);desc=tok[i,mask].mean(0);desc/=np.linalg.norm(desc);descriptors.append(desc)
 D=np.stack(descriptors);unit=tok[:28]/np.linalg.norm(tok[:28],axis=-1,keepdims=True);sims=np.einsum('id,npd->nip',D,unit);save(E/'results/image_metadata.json',metadata);predictions=[];heat=[]
 for target in ['drawer_handle','gripper_finger']:
  ti=[i for i,t in enumerate(templates)if t['target']==target];scores=sims[:,ti].max(1);heat.append(scores)
  for i,r in enumerate(metadata):
   z=scores[i];best=int(z.argmax());yy,xx=divmod(best,16);grid=np.indices((16,16));far=(np.maximum(abs(grid[0]-yy),abs(grid[1]-xx))>=3).reshape(-1);runner=float(z[far].max());predictions.append({'target':target,'index':r['index'],'state_id':r['state_id'],'scene':r['scene'],'camera':r['camera'],'x':float((xx+.5)*r['width']/16),'y':float((yy+.5)*r['height']/16),'top_cosine':float(z[best]),'far_runnerup_cosine':runner,'top_minus_far':float(z[best]-runner),'winning_template':templates[ti[int(sims[i,ti,best].argmax())]]['name'],'same_camera_as_reference':r['camera']==templates[ti[0]]['camera'],'is_reference_image':any(t['index']==r['index']and t['camera']==r['camera']for t in templates if t['target']==target),'valid_detection':None,'interpretation':'unconditional best patch, no calibrated detection/absence confidence or instance boundary'})
 np.savez_compressed(E/'results/matching_heatmaps.npz',scores=np.stack(heat));save(E/'results/part_matches.json',predictions)
 context=[]
 for t,c in zip(templates,contexts):
  r=metadata[c['original_image']];m=selector(t['box'],r['width'],r['height']);a=tok[c['original_image'],m].mean(0);b=tok[c['gray_image'],m].mean(0);u=tok[c['crop_original']].mean(0);v=tok[c['crop_gray']].mean(0)
  context.append({'template':t['name'],'unchanged_crop_RGB':True,'full_image_ROI_cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'full_image_ROI_relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)),'crop_reencoding_max_abs_diff':float(np.max(abs(tok[c['crop_original']]-tok[c['crop_gray']]))),'crop_descriptor_cosine':float(u@v/(np.linalg.norm(u)*np.linalg.norm(v))),'part_patch_count':int(m.sum()),'same_selected_patch_input_pixels':True,'limit':'after resize preserve entire selected14x14patch inputs exactly, gray only other patches; local crop excludes exterior by construction but still contains any background inside its box'})
 save(E/'results/context_sensitivity.json',context);torch.cuda.synchronize();save(E/'results/inference.json',{'utc':now(),'images':40,'source_RGB_states':14,'queries':len(predictions),'device':torch.cuda.get_device_name(0),'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'frozen_model_state_hash':init,'checkpoint_sha256':before,'model_state_unchanged':True,'input_schema':['rgb_static','rgb_gripper'],'new_downloads':0,'optimizer_updates':0,'simulator_calls':0,'detector_trained':False});print('READONLY_ENDPOINT_PROBE_COMPLETE',len(predictions),'unconditional patch matches',flush=True)
if __name__=='__main__':main()
