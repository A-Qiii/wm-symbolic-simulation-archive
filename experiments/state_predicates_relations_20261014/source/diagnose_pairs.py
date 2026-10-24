from common import *
from pair_model import PairReader
import fcntl
E=P/'experiments/state_predicates_relations_20261014'
def output(m,d):
    m.eval()
    with torch.no_grad():
        o,z=m(**{k:torch.as_tensor(d[k],device='cuda')for k in ['objects','context','robot','neighbors']});l=torch.cat([o[:,:,0],o[:,:,1],z],-1)
        return l.cpu().numpy(),l.sigmoid().cpu().numpy()
def main():
    ds=dataset()
    with np.load(E/'data/neighbors_train_dev.npz')as z:
        for n in ds:ds[n]['neighbors']=z[n].copy()
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');result=read(E/'results/pair_training.json');fixed={};abl=[]
    for arm in ['appearance_only','relative_geometry']:
        for seed in range(3):
            m=PairReader(arm=='relative_geometry').cuda();m.load_state_dict(torch.load(E/'models'/arm/f'seed{seed}/update1500.pt',map_location='cuda',weights_only=False)['model']);met={n:evaluate_predictions(output(m,ds[n])[1],ds[n])for n in ['old_tune','old_cal','new_dev']};fixed[f'{arm}_seed{seed}']={'metrics':met,'raw_gate':raw_gate(met)}
            if arm=='relative_geometry':
                m.load_state_dict(torch.load(result[arm]['runs'][seed]['checkpoint_path'],map_location='cuda',weights_only=False)['model']);weight=m.fusion.weight.detach().clone();norm=float(weight.norm())
                for domain in ['old_tune','old_cal','new_dev']:
                    l0,p0=output(m,ds[domain]);mask=ds[domain]['v']
                    for change in ['zero_relative_geometry','zero_entire_pair_residual']:
                        if change=='zero_relative_geometry':m.use_geometry=False
                        else:
                            with torch.no_grad():m.fusion.weight.zero_()
                        l1,p1=output(m,ds[domain]);abl.append({'seed':seed,'domain':domain,'change':change,'learned_fusion_weight_norm':norm,'mean_abs_logit_change':float(np.abs(l1-l0)[mask].mean()),'mean_abs_probability_change':float(np.abs(p1-p0)[mask].mean()),'prediction_flips':int(((p0>=.5)!=(p1>=.5))[mask].sum()),'original_errors':int(((p0>=.5)!=ds[domain]['y'])[mask].sum()),'changed_errors':int(((p1>=.5)!=ds[domain]['y'])[mask].sum())});m.use_geometry=True
                        with torch.no_grad():m.fusion.weight.copy_(weight)
            del m
    save(E/'results/pair_diagnostics.json',{'fixed1500_dev':fixed,'selected_relative_path_ablations':abl,'only_C_development':True,'no_training_or_saved_weight_change':True,'interpretation':'Ablations are OOD sensitivity probes, not causal identification. Final1500 checks diagnose selected-update confounding; neither diagnostic selects a replacement model.'})
    print('PAIR_DIAGNOSTICS_COMPLETE',{k:r['raw_gate']['failed_domain_predicates']for k,r in fixed.items()},flush=True)
if __name__=='__main__':main()
