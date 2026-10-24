"""Five prespecified CPU linear probes; C results never select settings."""
from common import *
import sys,collections,importlib.util
import torch
def metrics(score,y,idx,meta):
 p=score>=0;z=y.astype(bool);ix=np.array(idx,int);out={'facts':len(ix),'positive':int(z[ix].sum()),'negative':int((~z[ix]).sum()),'TP':int((p[ix]&z[ix]).sum()),'TN':int((~p[ix]&~z[ix]).sum()),'FP':int((p[ix]&~z[ix]).sum()),'FN':int((~p[ix]&z[ix]).sum())};out['accuracy']=float((p[ix]==z[ix]).mean());out['balanced_accuracy']=.5*(out['TP']/max(1,out['positive'])+out['TN']/max(1,out['negative']));out['roots']=len({meta[i]['root_id']for i in ix});return out
def main():
 torch.set_num_threads(4);assert not(E/'results/probes.json').exists();wins=read(BASE/'data/windows.json');source=read(BASE/'data/metadata.json');assert [w['window_id']for w in wins]==[r['window_id']for r in source[:630]]
 save(E/'configs/probes_frozen.json',{'created_utc':now(),'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(E/'configs/protocol.json'),'no_C_selection':True,'source_features':'Existing DINO caches; no backbone run','no_D':True})
 spec=importlib.util.spec_from_file_location('old_query_model',BASE/'source/model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);net=mod.QueryRelationReader().eval();cp=read(BASE/'configs/reader_frozen.json');assert sha(cp['path'])==cp['sha256'];net.load_state_dict(torch.load(cp['path'],weights_only=False,map_location='cpu')['model']);inp=dict(np.load(BASE/'data/inputs.npz'))
 with torch.inference_mode():nodes=net.nodes(**{k:torch.tensor(v[:630])for k,v in inp.items()}).numpy()
 features={k:[]for k in ['frozen_edge_latent384','existing_pooled_visual3852','cached_spatial_patch_visual15372','RGB_bbox_geometry12','GT_relative_geometry_upper_bound']};meta=[];ys=[];valid=[]
 for i,w in enumerate(wins):
  with np.load(source[i]['raw_feature_path'])as z:raw={k:z[k].astype(np.float32)for k in z.files}
  for j,c in enumerate(COLORS):
   n=nodes[i];features['frozen_edge_latent384'].append(np.r_[n[j],n[3],n[j]-n[3],n[j]*n[3]])
   features['existing_pooled_visual3852'].append(np.r_[inp['objects'][i,-1,j],inp['context'][i,-1]])
   g=raw['geometry'][-1,:,j];bbox=np.c_[g[:,0,:5],g[:,:,5].mean(1)].ravel();features['RGB_bbox_geometry12'].append(bbox)
   features['cached_spatial_patch_visual15372'].append(np.r_[raw['local_tokens'][-1,:,j].ravel(),raw['global_grid'][-1].ravel(),bbox])
   gt=w['targets_v2_1']['objects'][c];features['GT_relative_geometry_upper_bound'].append(np.r_[gt['side_rim_clearances_m'],gt['floor_clearance_m'],float(bool(gt['load_bearing_paths'])),float(gt['held'])])
   ys.append(w['targets_v2_1']['y'][3+j]);valid.append(w['targets_v2_1']['v'][3+j]);meta.append({k:w[k]for k in ['window_id','root_id','split','condition','anchor']}|{'object':c,'outside_supported':gt['outside']and bool(gt['load_bearing_paths']),'indirect_protruding':gt['outside']and any(len(x)>2 for x in gt['load_bearing_paths'])})
 features={k:np.array(v,np.float32)for k,v in features.items()};y=np.array(ys,float);v=np.array(valid,bool);tr=np.array([i for i,r in enumerate(meta)if r['split']=='train'and v[i]]);te=np.array([i for i,r in enumerate(meta)if r['split']=='dev'and v[i]]);assert set(meta[i]['root_id']for i in tr).isdisjoint(meta[i]['root_id']for i in te)
 public={k:x for k,x in features.items()if not k.startswith('GT_')};np.savez_compressed(E/'data/public_probe_features.npz',**public);(E/'data/_private').mkdir(exist_ok=True);np.savez_compressed(E/'data/_private/probe_targets_and_GT_features.npz',y=y,v=v,GT_relative_geometry=features['GT_relative_geometry_upper_bound']);save(E/'data/object_query_metadata.json',meta)
 roots=sorted({meta[i]['root_id']for i in tr});weights=np.zeros(len(tr),float)
 for root in roots:
  for label in [0,1]:
   ii=[k for k,i in enumerate(tr)if meta[i]['root_id']==root and y[i]==label];assert len(ii)>0;weights[ii]=1/(len(roots)*2*len(ii))
 weights/=weights.sum();tw=torch.tensor(weights,dtype=torch.float64);ty=torch.tensor(y[tr]*2-1,dtype=torch.float64);base=float((tw*ty).sum());yw=ty-base;out={};preds={}
 pairlist=[]
 for root in sorted({r['root_id']for r in meta}):
  for c in COLORS:
   pos=next((i for i,r in enumerate(meta)if r['root_id']==root and r['object']==c and r['condition']=='single_inside'and r['anchor']==127 and y[i]and v[i]),None)
   if pos is None:continue
   for cond in ['stacked','outside_front_supported']:
    neg=next((i for i,r in enumerate(meta)if r['root_id']==root and r['object']==c and r['condition']==cond and r['anchor']==127 and not y[i]and v[i]),None)
    if neg is not None:pairlist.append({'positive':pos,'negative':neg,'root_id':root,'object':c,'split':meta[pos]['split'],'negative_condition':cond})
 for name,arr in features.items():
  tx=torch.tensor(arr,dtype=torch.float64);mean=(tx[tr]*tw[:,None]).sum(0);std=(((tx[tr]-mean)**2*tw[:,None]).sum(0)).sqrt();keep=std>=1e-6;tx=(tx[:,keep]-mean[keep])/std[keep];d=tx.shape[1];xx=tx[tr];sw=(tw*len(tr)).sqrt();wx=xx*sw[:,None];kernel=(wx@wx.T)/d;modelpath=E/'models'/(name+'.npz')
  if modelpath.exists():
   stored=dict(np.load(modelpath));assert np.array_equal(stored['mean'],mean.numpy())and np.array_equal(stored['std'],std.numpy())and np.array_equal(stored['keep'],keep.numpy());coef=torch.tensor(stored['coef'],dtype=torch.float64);assert float(stored['intercept'])==base
  else:
   alpha=torch.linalg.solve(kernel+.1*torch.eye(len(tr),dtype=torch.float64),yw*sw);coef=wx.T@alpha/d;np.savez_compressed(modelpath,mean=mean.numpy(),std=std.numpy(),keep=keep.numpy(),coef=coef.numpy(),intercept=base)
  score=(tx@coef+base).numpy();preds[name]=score
  res={'original_dimension':arr.shape[1],'nonconstant_AB_dimensions':int(keep.sum()),'AB':metrics(score,y,tr,meta),'C':metrics(score,y,te,meta),'C_outside_supported':metrics(score,y,[i for i in te if meta[i]['outside_supported']],meta),'C_indirect_protruding':metrics(score,y,[i for i in te if meta[i]['indirect_protruding']],meta),'paired_same_object':{}}
  for split in ['train','dev']:
   pp=[r for r in pairlist if r['split']==split];res['paired_same_object'][split]={'pairs':len(pp),'both_correct':sum(score[r['positive']]>=0 and score[r['negative']]<0 for r in pp),'positive_score_exceeds_negative':sum(score[r['positive']]>score[r['negative']]for r in pp),'score_differences':[float(score[r['positive']]-score[r['negative']])for r in pp]}
  out[name]=res;print(json.dumps({'probe':name,'C':res['C'],'C_outside':res['C_outside_supported'],'pairs_C':res['paired_same_object']['dev']},ensure_ascii=False,default=lambda x:x.item()),flush=True)
 save(E/'results/probes.json',{'protocol':read(E/'configs/protocol.json')['probes'],'train_valid_object_facts':len(tr),'C_valid_object_facts':len(te),'train_roots':len(roots),'C_roots':len({meta[i]['root_id']for i in te}),'results':out,'no_model_promoted':True,'no_neural_updates':True,'closed_form_fits':len(out)});np.savez_compressed(E/'results/probe_scores.npz',**preds);save(E/'results/probe_pairs.json',pairlist)
if __name__=='__main__':main()
