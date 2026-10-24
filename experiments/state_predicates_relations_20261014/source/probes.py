from common import *
import fcntl
J=P/'experiments/state_predicates_joint_20261014';E=P/'experiments/state_predicates_relations_20261014'
def main():
    assert not(E/'results/development_input_probes.json').exists()
    audit=read(E/'results/input_and_label_audit.json');assert not audit['entity_or_repacking_mismatches']and not audit['in_positive_without_upward_support']and not audit['in_negative_with_upward_support']
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');ds=dataset();out=[]
    for seed,r in enumerate(read(J/'results/balanced_loss.json')['seeds']):
        m=loadmodel(r['checkpoint_path'])
        for domain in ['old_tune','old_cal','new_dev']:
            d=ds[domain];base=infer(m,d)
            for perturb in ['zero_explicit_object_bbox_only','swap_left_right_context_cells','replace_normalized_proprioception_by_training_mean']:
                z={**d}
                if perturb=='zero_explicit_object_bbox_only':
                    z['objects']=d['objects'].copy();q=z['objects'].reshape(len(d['y']),4,3,2,390);q[:,:,:,:,384:388]=0
                elif perturb=='swap_left_right_context_cells':
                    q=d['context'].reshape(len(d['y']),4,2,2,2,384);z['context']=q[:,:,:,:,[1,0],:].reshape(len(d['y']),4,-1).copy()
                else:z['robot']=np.zeros_like(d['robot'])
                pp=infer(m,z);groups={}
                for g,cols in GROUPS.items():
                    v=d['v'][:,cols];yy=d['y'][:,cols];a0=base[:,cols];a1=pp[:,cols]
                    groups[g]={'valid':int(v.sum()),'mean_abs_probability_change':float(np.abs(a1-a0)[v].mean()),'raw_prediction_flips':int(((a0>=.5)!=(a1>=.5))[v].sum()),'original_errors':int(((a0>=.5)!=yy)[v].sum()),'perturbed_errors':int(((a1>=.5)!=yy)[v].sum()),'correct_to_wrong':int((((a0>=.5)==yy)&((a1>=.5)!=yy)&v).sum()),'wrong_to_correct':int((((a0>=.5)!=yy)&((a1>=.5)==yy)&v).sum())}
                out.append({'seed':seed,'domain':domain,'perturbation':perturb,'groups':groups})
        del m
    save(E/'results/development_input_probes.json',{'rows':out,'scope':'Input-feature perturbations on old/new C development only. No RGB editing or new DINO extraction.','interpretation_limit':'OOD substitutions can destroy correlations; sensitivity shows the trained function depends on these inputs, not unique causal reasoning or diagnosis of a test failure.','no_training':True,'no_test_read':True})
    for perturb in sorted({r['perturbation']for r in out}):
        print(perturb,{g:{'mean_abs_delta':float(np.mean([r['groups'][g]['mean_abs_probability_change']for r in out if r['perturbation']==perturb])),'raw_flips':sum(r['groups'][g]['raw_prediction_flips']for r in out if r['perturbation']==perturb)}for g in GROUPS},flush=True)
if __name__=='__main__':main()
