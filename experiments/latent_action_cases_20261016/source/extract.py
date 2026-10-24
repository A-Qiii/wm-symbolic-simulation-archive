from common import *
import torch,sys,fcntl
sys.path.insert(0,str(REPAIR/'source'))
from model import Model

def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 sel=read(REPAIR/'models/canonical_matched_seed0/selected.json');oldsel=read(OLD/'models/interaction_seed0/selected.json');assert sha(sel['path'])==sel['sha256']and sha(oldsel['path'])==oldsel['sha256']
 cp=torch.load(sel['path'],map_location='cpu',weights_only=True)['state'];ocp=torch.load(oldsel['path'],map_location='cpu',weights_only=True)['state'];assert all(torch.equal(v,ocp[k])for k,v in cp.items()if not k.startswith('transition.'))
 norm=dict(np.load(REPAIR/'data/public/normalization.npz'));m=Model(norm,'learned').cuda().eval().requires_grad_(False);m.load_state_dict(cp)
 cases=read(E/'data/cases.json');d=dict(np.load(REPAIR/'data/D2/public/inputs.npz'));indices=[r[role]for r in cases for role in['before','after']]
 with torch.no_grad():z=m.encode(torch.tensor(d['objects'][indices],device='cuda'),torch.tensor(d['scene'][indices],device='cuda')).cpu().numpy().reshape(4,2,256)
 bg=np.load(PREV/'data/main_latents.npz')['interaction'];rows=read(PREV/'data/main_states.json');lookup={(r['batch'],r['state_index']):i for i,r in enumerate(rows)};alias=[]
 for i,r in enumerate(cases):
  for j,role in enumerate(['before','after']):
   k=lookup.get(('D2',r[role]));
   if k is not None:
    delta=float(np.max(np.abs(z[i,j]-bg[k])));assert np.allclose(z[i,j],bg[k],atol=1e-5,rtol=1e-5);z[i,j]=bg[k];alias.append({'case_id':r['case_id'],'role':role,'background_index':k,'encoding_max_abs_diff':delta})
 pred=np.load(REPAIR/'results/D2_canonical_matched_seed0.npz');ix={int(v):i for i,v in enumerate(pred['pair_indices'])};predz=np.stack([pred['predicted_latent'][ix[int(r['id'][1:])]].reshape(256)for r in cases]);savedrecords=read(REPAIR/'results/D2_records.json');records={r['pair_id']:r for r in savedrecords if r['arm']=='canonical_matched'and r['seed']==0}
 co=np.load(PREV/'results/coordinates.npz');std=co['interaction_std'];results=[]
 for i,r in enumerate(cases):
  rec=records[r['id']];assert rec['before']==r['before']and rec['after']==r['after']and rec['skill']==r['skill'];pr=discrete(rec['predicted_successor']);truth=np.array(rec['true_after']['y']);valid=np.array(rec['true_after']['v']);wrong=np.flatnonzero((pr!=truth)&valid);mse=float(np.mean((predz[i]-z[i,1])**2));assert np.isclose(mse,rec['latent_error'],rtol=1e-4,atol=1e-6)
  results.append(r|{'current_truth':rec['true_before']['y'],'actual_truth':truth.tolist(),'truth_valid':valid.tolist(),'predicted_probabilities':rec['predicted_successor'],'predicted_labels':pr.tolist(),'current_semantics':semantics(rec['true_before']['y']),'actual_semantics':semantics(truth),'predicted_semantics':semantics(pr),'wrong_propositions':[{'name':FACTS[j],'predicted':bool(pr[j]),'actual':bool(truth[j])}for j in wrong],'correct_propositions':int(np.sum((pr==truth)&valid)),'valid_propositions':int(valid.sum()),'latent_mse_raw':mse,'latent_mse_AB_standardized':float(np.mean(((predz[i]-z[i,1])/std)**2))})
 np.savez_compressed(E/'data/latents.npz',background=bg,current=z[:,0],actual=z[:,1],predicted=predz,mean=co['interaction_mean'],std=std)
 save(E/'results/cases.json',results);save(E/'results/extraction.json',{'model':sel,'original_encoder_model':oldsel,'encoder_exactly_equal':True,'background_aliases':alias,'state_encoding_calls':1,'states_encoded':8,'predictions_reused_from_sealed_npz':4,'model_updates':0,'new_physical_calls':0,'query_thresholds':'Existing first6>=0.5; drawer3 argmax. Unchanged.','device':torch.cuda.get_device_name(0)})
 print(json.dumps([{k:r[k]for k in['case_id','id','latent_mse_AB_standardized','correct_propositions','predicted_semantics','actual_semantics']}for r in results],ensure_ascii=False))
if __name__=='__main__':main()
