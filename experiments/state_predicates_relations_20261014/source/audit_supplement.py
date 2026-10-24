from common import *
E=P/'experiments/state_predicates_relations_20261014'
def main():
    meta=read(A/'data/metadata_train_dev.json');stats=read(O/'data/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rows=[]
    with np.load(A/'data/supplement_train_dev.npz')as z:obj=z['objects'].copy()
    for i,r in enumerate(meta):
        f=read(r['labels_private'])['frames'][-1]
        with np.load(r['feature_path'])as z:tok=((z['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=z['geometry'].reshape(4,2,64,8);ids=z['visible_registry_ids'].tolist()
        for j,c in enumerate(['red','blue','pink']):
            uid=f['objects_audit'][c]['object_uid_evaluator_only'];assert uid in ids;slot=ids.index(uid);v=geo[:,:,slot*4,4:5];arr=np.concatenate([tok[:,:,slot*4:slot*4+4].mean(2)*v,geo[:,:,slot*4,:5],geo[:,:,slot*4:slot*4+4,5].mean(2)[...,None]],-1);err=float(np.abs(arr-obj[i,:,j].reshape(4,2,390)).max());assert err==0;rows.append({'sample_id':r['sample_id'],'colour':c,'match':True})
        j=r['target_index'];assert f['labels']['categorical'][j]!=3 and f['labels']['categorical_valid'][j]
    save(E/'results/supplement_audit.json',{'windows':110,'entity_matches':330,'max_repack_error':0,'target_in_negative_labels':110,'private_UID_used_only_for_posthoc_audit':True,'no_feature_or_label_change':True,'rows':rows});print('SUPPLEMENT_AUDIT_PASSED',len(rows))
if __name__=='__main__':main()
