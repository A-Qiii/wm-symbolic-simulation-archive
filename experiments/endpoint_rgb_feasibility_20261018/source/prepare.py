from common import *
import shutil,importlib.util

def main():
 assert not E.exists()
 for d in ['configs','source','data','results','logs']:(E/d).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 rows=read('/tmp/endpoint_sample_rows.json');copied=[];refs={}
 for r in rows:
  src=Path(r['path']);dst=E/f"data/{r['index']:05d}.npz";z=np.load(src);assert set(z.files)=={'rgb_static','rgb_gripper'};shutil.copy2(src,dst);refs[str(src)]=sha(src);copied.append({k:r[k]for k in ['index','state_id','root_id','scene','train']}|{'path':str(dst),'source':str(src),'sha256':sha(dst)})
 save(E/'data/states.json',copied)
 # Human-visible part rectangles, drawn by assistant from RGB grids before any inference.
 templates=[{'name':'drawer_handle_A','target':'drawer_handle','index':210,'camera':0,'box':[48,169,110,196]}, {'name':'drawer_handle_B','target':'drawer_handle','index':270,'camera':0,'box':[53,175,120,200]}, {'name':'gripper_finger_A','target':'gripper_finger','index':210,'camera':1,'box':[15,5,29,27]}, {'name':'gripper_finger_B','target':'gripper_finger','index':270,'camera':1,'box':[18,5,34,27]}]
 save(E/'configs/templates.json',{'annotations':templates,'origin':'assistant rough RGB-only visible part boxes drawn before inference; not independently validated instance ground truth','selection':'first A/B official training RGB only; no C/D template or tuning','limits':'drawer handle is only a part; single finger is not whole gripper; boxes may include background; no simulator or robot inputs'})
 inventory=[]
 roots=[P/'models',Path('/home/robot/research/models'),Path('/home/robot/.cache/huggingface/hub'),Path('/home/robot/.cache/torch/hub/checkpoints')]
 for root in roots:
  if root.exists():
   for f in root.rglob('*'):
    if f.is_file()and f.suffix in ['.safetensors','.ckpt','.pt','.pth','.bin']:inventory.append({'path':str(f),'bytes':f.stat().st_size})
 save(E/'results/local_inventory.json',{'utc':now(),'searched_roots':[str(p)for p in roots],'weights':inventory,'packages_available':{n:importlib.util.find_spec(n)is not None for n in ['torch','torchvision','transformers','cv2','segment_anything','ultralytics']},'verified_vision_encoder':'local Dinov2Model384dimension patch14; real execution pending','detectors':'no drawer/gripper detector or SAM/Grounding/YOLO/OWL checkpoint located in these model/cache roots; not a whole-host absence claim','existing_RGB_localizer':'three known colour largest HSV components only','HULC':'existing fixed skill executor checkpoint, not a verified instance detector','text_model':'MiniLM text encoder is not a visual localizer'})
 save(E/'configs/protocol.json',{'utc':now(),'mode':'bounded read-only feasibility after all previous evaluations seen','states':14,'images':28,'sample_selection':'A/B first official train RGB plus each scene first,middle,last available metadata row; 14 unique states frozen before inference','network':'same frozen cached DINOv2-small, no model downloads','preprocess':'full image resize bilinear to224 square, ImageNet normalization; uses16x16 patches; previous pipeline padded84px wrist into224 giving ~6x6 effective content, so this is not directly interchangeable with old head inputs','matching':'mean normalized descriptor of patches with centres inside assistant template part box; two A/B templates perpart; maximum cosine template similarity over every candidate patch; report unconditional top1 and spatially distant runner-up, no validated threshold','targets':['drawer_handle','gripper_finger'],'camera_transfer':'static handle reference and wrist finger reference each applied to both cameras; correspondence not ground truth','no_object_boxes_from_matching':'a matching point is not an instance boundary or identity guarantee','context_probe':'four reference part boxes: compare pooled full-image tokens original vs gray outside selected14x14patches after resizing, preserving selected input patches exactly, and separately re-encoded identical local crops; no label scoring','budget':'one frozen inference pass over28 originals+4 outside-gray variants+8 local crops=40RGB tensors, no sweeps, no retries unless execution error','training':False,'collection':False,'network':False,'simulator_masks_or_coordinates_or_robot_obs':False,'independent_evaluation':False,'models_adopted':False})
 for p in [BASE/'configs/delivery_manifest.json',PREV/'configs/delivery_manifest.json',DINO/'model.safetensors',DINO/'config.json']:
  refs[str(p)]=sha(p)
 save(E/'configs/source_references.json',{'sha256':refs})
 shutil.copy2('/tmp/endpoint_reference_grid.png',R/'figures/reference_RGB_grid.png')
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 files=[p for sub in ['configs','source','data']for p in(E/sub).rglob('*')if p.is_file()]
 save(E/'configs/inference_frozen.json',{'utc':now(),'before_inference':True,'sha256':{str(p):sha(p)for p in files}})
 print('READONLY_ENDPOINT_PROBE_FROZEN',len(rows),len(templates),flush=True)
if __name__=='__main__':main()
