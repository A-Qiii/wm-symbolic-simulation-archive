"""AB-only diagnostic: do frozen latent query inputs distinguish new containment labels?"""
from common import *
from model import QueryRelationReader
import torch,collections

def main():
 torch.set_num_threads(4);meta=read(E/'data/metadata.json');inp=dict(np.load(E/'data/inputs.npz'));targets=dict(np.load(E/'data/targets.npz'));net=QueryRelationReader().eval();cp=read(E/'configs/reader_frozen.json');net.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=False)['model']);ids=[i for i,r in enumerate(meta)if r['pool']=='new_semantics'and r['split']=='train'];x={k:torch.tensor(v[ids])for k,v in inp.items()}
 with torch.inference_mode():nodes=net.nodes(**x).numpy();prob=net.probabilities(**x).numpy()
 index={i:j for j,i in enumerate(ids)};pairs=[]
 for root in sorted({meta[i]['root_id']for i in ids}):
  target=next(r['color']for r in read(B/'roots_manifest.json')if r['root_id']==root);j=COLORS.index(target);ii=next((i for i in ids if meta[i]['root_id']==root and meta[i]['condition']=='single_inside'and meta[i]['anchor']==127),None)
  if ii is None:continue
  for case in ['stacked','outside_front_supported']:
   jj=next((i for i in ids if meta[i]['root_id']==root and meta[i]['condition']==case and meta[i]['anchor']==127),None)
   if jj is None:continue
   n1,n2=nodes[index[ii]],nodes[index[jj]];v1=np.r_[n1[j],n1[3],n1[j]-n1[3],n1[j]*n1[3]];v2=np.r_[n2[j],n2[3],n2[j]-n2[3],n2[j]*n2[3]];pairs.append({'root_id':root,'object':target,'conditions':['single_inside',case],'labels_In':[bool(targets['y'][ii,3+j]),bool(targets['y'][jj,3+j])],'valid_In':[bool(targets['v'][ii,3+j]),bool(targets['v'][jj,3+j])],'frozen_edge_input_max_abs_diff':float(np.abs(v1-v2).max()),'frozen_edge_input_L2':float(np.linalg.norm(v1-v2)),'object_latent_max_abs_diff':float(np.abs(n1[j]-n2[j]).max()),'scene_latent_max_abs_diff':float(np.abs(n1[3]-n2[3]).max()),'raw_object_input_L2':float(np.linalg.norm(inp['objects'][ii,:,j]-inp['objects'][jj,:,j])),'raw_context_input_L2':float(np.linalg.norm(inp['context'][ii]-inp['context'][jj])),'predicted_In':[float(prob[index[ii],3+j]),float(prob[index[jj],3+j])],'public_windows':[meta[ii]['window_id'],meta[jj]['window_id']]})
 opposites=[r for r in pairs if r['labels_In'][0]!=r['labels_In'][1]and all(r['valid_In'])];exact=sum(r['frozen_edge_input_max_abs_diff']==0 for r in opposites);near=sum(r['frozen_edge_input_max_abs_diff']<1e-5 for r in opposites);save(E/'results/AB_frozen_representation_audit.json',{'AB_only':True,'new_training_updates':0,'C_not_used_for_diagnostic_matching':True,'matched_pairs':pairs,'valid_opposite_label_pairs':len(opposites),'exact_same_query_inputs_opposite_labels':exact,'max_abs_diff_under1e_5_pairs':near,'interpretation':'Exact identical frozen query inputs with opposite containment labels would prove that this query head cannot separate those observations. Near saturation is evidence of weakly preserved geometric distinctions, not proof of mathematical impossibility. No additional training or threshold selection performed.'});print(json.dumps({'pairs':len(opposites),'exact':exact,'near1e5':near,'range_maxdiff':[min(r['frozen_edge_input_max_abs_diff']for r in opposites),max(r['frozen_edge_input_max_abs_diff']for r in opposites)]}),flush=True)
if __name__=='__main__':main()
