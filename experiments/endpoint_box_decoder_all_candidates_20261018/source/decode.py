from common import *
import torch,torch.nn.functional as F,sys,re

def candidates(raw,grid):
 n=grid*grid
 with torch.inference_mode():
  t=torch.tensor(raw);s=t[:,:2].sigmoid();g=t[:,2:].reshape(n,2,4);g=torch.cat([.5*g[...,:2].tanh(),g[...,2:].sigmoid()],-1);patch=torch.arange(n)[:,None].expand(n,2);center=torch.stack([patch%grid+.5,patch//grid+.5],-1).to(g.dtype)/grid+g[...,:2]/grid;boxes=torch.cat([center-g[...,2:]/2,center+g[...,2:]/2],-1).clamp(0,1)
  im=s.T.reshape(1,2,grid,grid);peaks=(im==F.max_pool2d(im,3,1,1)).reshape(2,n).T;s=s.numpy();boxes=boxes.numpy();peaks=peaks.numpy()
 out=[]
 for c in range(2):
  for k in sorted(np.where(s[:,c]>=.5)[0],key=lambda k:(-s[k,c],int(k))):
   b=boxes[k,c];out.append({'category':CATS[c],'score':float(s[k,c]),'box':b.tolist(),'patch':int(k),'valid_box':bool(b[2]>b[0]and b[3]>b[1]),'old_local_peak':bool(peaks[k,c])})
 return out
def predict(cs,original):
 selected=[];trace=[]
 for c in CATS:
  kept=[];cap=1 if c=='drawer_front'else 2
  for p in [x for x in cs if x['category']==c]:
   reason=None;blocker=None
   if original and not p['old_local_peak']:reason='local_peak_removed'
   elif original and len(kept)>=cap:reason='cap_not_processed'
   elif not p['valid_box']:reason='invalid_box'
   else:
    blocker=next((q for q in kept if iou(p['box'],q['box'])>.3),None)
    if blocker is not None:reason='NMS_suppressed'
    else:reason='kept';kept.append(p)
   trace.append({'category':c,'patch':p['patch'],'reason':reason,'blocking_patch':blocker['patch']if blocker else None})
  selected.extend({k:p[k]for k in ['category','score','box','patch']}for p in kept)
 return selected,trace
def main():
 assert not(E/'results/decoded_predictions.json').exists();check_manifest(E/'configs/pre_decode_frozen.json');torch.set_num_threads(4)
 reference={(r['resolution'],r['seed'],r['image_id']):r['predictions']for r in read(E/'data/reference/decoded_predictions.json')}
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('No external experiment or RGB during decoding')
   if p.is_relative_to(E/'data/annotations')or p==E/'data/images.json':raise RuntimeError('Decoder cannot read labels/scene metadata')
 sys.addaudithook(guard);outputs=[];traces=[];raw_seen=0;images=0
 for path in sorted((E/'data/raw_outputs').glob('*.npz')):
  match=re.fullmatch(r'r(224|448)_(.+)_seed([01])',path.stem);assert match;res=int(match[1]);split=match[2];seed=int(match[3]);assert split in SPLITS
  with np.load(path)as z:
   for raw,i in zip(z['raw'],z['image_ids']):
    i=int(i);assert raw.shape==((res//14)**2,10);raw_seen+=raw.shape[0]*2;cs=candidates(raw,res//14);old,ot=predict(cs,True);new,nt=predict(cs,False);assert old==reference[res,seed,i],('old decoder mismatch',res,seed,i)
    outputs.append({'resolution':res,'seed':seed,'image_id':i,'split':split,'original':old,'corrected':new})
    ot={(x['category'],x['patch']):x for x in ot};nt={(x['category'],x['patch']):x for x in nt}
    for c in cs:traces.append({'resolution':res,'seed':seed,'image_id':i,'split':split,**c,'original_disposition':ot[c['category'],c['patch']],'corrected_disposition':nt[c['category'],c['patch']]})
    images+=1
 assert images==160
 save(E/'results/decoded_predictions.json',outputs);save(E/'results/candidate_trace.json',traces)
 save(E/'results/decoding.json',{'utc':now(),'image_model_configurations':images,'raw_patch_class_candidates':raw_seen,'score_passing_candidates':len(traces),'original_predictions_exactly_reproduced':True,'score_threshold':.5,'NMS_suppression_IoU':.3,'new_cardinality_caps':None,'new_local_peak_filter':False,'annotation_reads_during_decoding':0,'new_forward_passes':0,'feature_extractions':0,'optimizer_steps':0,'device':'CPU'})
 files=list((E/'results').glob('*.json'))
 save(E/'configs/outputs_frozen_before_scoring.json',{'utc':now(),'sha256':{str(p):sha(p)for p in files}})
 print('ALL_CANDIDATE_DECODING_COMPLETE',images,len(traces),flush=True)
if __name__=='__main__':main()
