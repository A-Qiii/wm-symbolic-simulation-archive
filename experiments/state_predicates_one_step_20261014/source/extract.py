from common import *
import fcntl,time
def main():
 assert not(E/'data/public/transition_inputs.npz').exists();protocol=read(E/'configs/protocol.json')
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');m=reader();before=digest_model(m);start=time.perf_counter();pairs=read(DATA/'data/public/pairs_v1.json');windows=read(DATA/'data/public/windows_v1.json')
 with np.load(DATA/'data/public/paired_inputs_v1.npz')as z,torch.inference_mode():
  inputs={k:torch.as_tensor(z[k],device='cuda')for k in ['objects','context','robot']};h,s=encode(m,**inputs);recomposed=logits(m,h,s).sigmoid();direct=m.probabilities(**inputs);error=float((recomposed-direct).abs().max());assert error==0.;h=h.cpu().numpy();s=s.cpu().numpy();prob=recomposed.cpu().numpy();bi=z['pair_before_index'];ai=z['pair_after_index'];actions=z['declared_action_features'].copy()
 with np.load(DATA/'results/reader_probabilities.npz')as prev:assert np.array_equal(prob,prev['rgb_v1'])
 np.savez_compressed(E/'data/public/transition_inputs.npz',current_objects=h[bi],current_scene=s[bi],declared_actions=actions)
 with np.load(DATA/'data/_private/supervision.npz')as gt:target={k:gt[k].copy()for k in gt.files}
 target.update(next_objects=h[ai],next_scene=s[ai],observed_current_probabilities=prob[bi],observed_next_probabilities=prob[ai]);np.savez_compressed(E/'data/_private/training_targets_and_audit.npz',**target);np.savez_compressed(E/'data/_private/window_latents.npz',objects=h,scene=s,probabilities=prob);save(E/'data/public/pairs.json',pairs)
 roots={r['root_id']:r for r in read(B/'roots_manifest.json')};evalmeta=[]
 for r in pairs:evalmeta.append(r|{'intended_root_colour_evaluator_only':roots[r['root_id']]['target_color'],'intended_target_source':'explicit lifted colour'if r['skill'].startswith('lift_')else'root plan scoring only, NOT model input'if r['skill']=='place_in_drawer'else'drawer'})
 save(E/'data/_private/evaluation_metadata.json',evalmeta);assert digest_model(m)==before;result={'passed':True,'windows_encoded':150,'pairs':120,'latent_shapes':{'objects':[3,96],'scene':[96]},'recomposed_shared_heads_vs_original_reader_max_error':error,'prior_RGB_reader_probabilities_byte_equal':True,'candidate_digest_unchanged':before,'inputs_sha256':sha(E/'data/public/transition_inputs.npz'),'targets_sha256':sha(E/'data/_private/training_targets_and_audit.npz'),'exact_public_input_keys':['current_objects','current_scene','declared_actions'],'future_latent_only_in_separate_private_target_file':True,'seconds':time.perf_counter()-start};save(E/'results/extraction.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
