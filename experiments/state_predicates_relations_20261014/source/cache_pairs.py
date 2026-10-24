from common import *
E=P/'experiments/state_predicates_relations_20261014'
def main():
    assert not(E/'data/neighbors_train_dev.npz').exists()
    audit=read(E/'results/input_and_label_audit.json');assert not audit['entity_or_repacking_mismatches']and not audit['in_positive_without_upward_support'];ds=dataset();stats=read(O/'data/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);arrays={};sources={}
    for name,d in ds.items():
        nn=[]
        for r in d['meta']:
            path=r['feature_path']
            with np.load(path)as z:
                # Use already cached visual ROI arrays; explicitly never load numeric visible_registry_ids.
                tok=((z['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384)[:,:,:48].reshape(4,2,12,4,384).mean(3);g=z['geometry'].reshape(4,2,64,8)[:,:,:48].reshape(4,2,12,4,8);visible=g[:,:,:,0,4:5];n=np.concatenate([tok*visible,g[:,:,:,0,:5],g[:,:,:,:,5].mean(3)[...,None]],-1);nn.append(n)
            sources[path]=sha(path)
        arrays[name]=np.stack(nn);assert arrays[name].shape==(len(d['y']),4,2,12,390)
    np.savez_compressed(E/'data/neighbors_train_dev.npz',**arrays);save(E/'data/neighbor_sources.json',sources)
    save(E/'configs/representation_decision.json',{'created_utc':now(),'evidence_before_training':'Exact packing and label audit passed. Old reader drops explicit non-target instance representations, while drawer ROI is retained in all1200 audited caches;12-slot truncation does not drop that drawer in this audit. Existing bbox/context/proprio preserve spatial information, so test an inductive bias rather than claim no relationship ability.','two_arms':['appearance_only','relative_geometry'],'same_parameters_and_inputs_except_8D_relative_geometry':'Two target-neighbor center deltas, two log size ratios, IoU, intersection/target area, intersection/neighbor area, occupancy difference; appearance control sets these8values tozero.','no_GT_or_numeric_ids':'Only cached normalized pooled DINO appearance, visible box/mask attributes. No semantic drawer identity or simulator coordinates.','extra_collection_criterion':'At least2/3 relational seeds pass full joint dev. Additionally fewer total tune+cal raw gate failures than appearance control, or if equal at least20% lower mean joint tune BCE. No test scores enter this rule.','reused_old_B_normalization_sampling_loss':True})
    (E/'configs/representation_decision.json').chmod(0o444);print('PAIR_CACHE_READY',{n:list(a.shape)for n,a in arrays.items()},flush=True)
if __name__=='__main__':main()
