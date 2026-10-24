from pathlib import Path
import json,hashlib
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'experiments/state_predicates_20261014';A=P/'experiments/state_predicates_augmented_20261014';E=P/'experiments/state_predicates_relations_20261014';C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ownership(info,uid):
    scene=info['scene_info'];objects=scene['movable_objects'];table=scene['fixed_objects']['table'];tid=table['uid'];drawer=table['links']['drawer_link'];byid={o['uid']:(name,o)for name,o in objects.items()}
    def support(u,seen):
        if u in seen:return []
        name,o=byid[u];paths=[]
        for c in o['contacts']:
            if c[9]<=.01:continue
            if c[1]==u:other,link,nz=c[2],c[4],c[7][2]
            elif c[2]==u:other,link,nz=c[1],c[3],-c[7][2]
            else:continue
            if nz<=.5:continue
            if other==tid and link==drawer:paths.append([name,'drawer'])
            elif other in byid:
                for rest in support(other,seen|{u}):paths.append([name]+rest)
        return paths
    return support(uid,set())
def main():
    meta=[r for r in read(O/'data/metadata.json')if r['split']!='diagnostic_D'];rows=read(P/'data/core1200_v1/data_manifest.json')['records'];by={}
    for r in rows:by.setdefault(r['episode_id'],{})[r['step']]=r
    stats=read(O/'data/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);labelrows=[];mismatch=[];maxerr=0.;example=None;slotcounts=[];overflows=[];drawer_coverage=[]
    with np.load(O/'data/inputs_targets.npz')as z:allobjects=z['objects'].copy()
    for r in meta:
        idx=r['index'];source=by[r['episode_id']][max(0,r['step']-1)];side='before'if r['step']==0 else'after';lp=Path(source['labels_path']);lab=read(lp)[side];info=read(lp.parent/'endpoint_info_private.json')[side]
        assert lab['categorical'][:4]==r['category']
        with np.load(r['feature_path'])as f:
            tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);ids=f['visible_registry_ids'].tolist();valid=f['valid'].reshape(4,2,64);slotcounts.append(len(ids))
            overflows.append(int(f['overflow_count']))
        tab=info['scene_info']['fixed_objects']['table'];drawer_id=int(tab['uid'])+((int(tab['links']['drawer_link'])+1)<<24)
        with np.load(r['public_inputs'][-1])as zz:drawer_pixels=sum(int((zz['seg_'+cam]==drawer_id).sum())for cam in ['static','gripper'])
        drawer_coverage.append({'index':idx,'drawer_pixels_last_views_total':drawer_pixels,'drawer_cached_slot':drawer_id in ids,'overflow_count':overflows[-1]})
        for j,c in enumerate(C):
            uid=info['scene_info']['movable_objects']['block_'+c]['uid'];arr=np.zeros((4,2,390),np.float32)
            if uid in ids:
                slot=ids.index(uid);sl=slice(slot*4,slot*4+4);vv=valid[:,:,slot*4];arr[:,:,:384]=tok[:,:,sl].mean(2)*vv[:,:,None];arr[:,:,384:389]=geo[:,:,slot*4,:5];arr[:,:,389]=geo[:,:,sl,5].mean(2)
            err=float(np.max(np.abs(arr-allobjects[idx,:,j].reshape(4,2,390))));maxerr=max(maxerr,err)
            if err>1e-6:mismatch.append({'index':idx,'object':c,'max_error':err})
            if lab['categorical_valid'][j]:
                paths=ownership(info,uid);labelrows.append({'index':idx,'root_id':r['root_id'],'split':r['split'],'object':c,'category':lab['categorical'][j],'in_label':lab['categorical'][j]==3,'upward_drawer_support_paths':paths,'held':lab['categorical'][j]==6,'source_label':str(lp),'source_side':side})
        if example is None:
            roi=tok[:,:,:4];permuted=roi[:,:,[3,2,1,0]];meanerr=float(np.abs(roi.mean(2)-permuted.mean(2)).max());grid=tok[:,:,48:].reshape(4,2,4,4,384);pooled=grid.reshape(4,2,2,2,2,2,384).mean((3,5));swapped=pooled[:,:, :, [1,0]]
            example={'index':idx,'ROI_cell_order_reversal_mean_difference':meanerr,'coarse_grid_left_right_swap_input_difference':float(np.abs(pooled-swapped).max()),'source':r['feature_path']}
    disagreements=[r for r in labelrows if r['in_label']and not r['upward_drawer_support_paths']]
    unexpected_neg=[r for r in labelrows if r['upward_drawer_support_paths']and not r['in_label']and not r['held']]
    out={'old_train_dev_windows':len(meta),'entity_checks':len(meta)*3,'entity_or_repacking_mismatches':mismatch,'max_repack_absolute_error':maxerr,'visible_slots_min_max':[min(slotcounts),max(slotcounts)],'cache_overflow_windows':sum(x>0 for x in overflows),'cache_overflow_max':max(overflows),'drawer_visible_but_not_in_cached_slots':[r for r in drawer_coverage if r['drawer_pixels_last_views_total']>=2 and not r['drawer_cached_slot']],'drawer_coverage_rows':drawer_coverage,'explicit_order_probe':example,'valid_object_records':len(labelrows),'in_positive_records':sum(r['in_label']for r in labelrows),'in_positive_without_upward_support':disagreements,'in_negative_with_upward_support':unexpected_neg,'active_label_pipeline':'core1200_collect.label -> calibrated_labels_at -> PhysicalOracle.held override -> drawer_ownership','label_definition_note':'in is valid category3: drawer contact/support ownership, not image bbox overlap or pure 3D point containment. Initial labels_at prioritizes drawer contact; upward support audit below checks the actual saved data. Held uses bilateral force-bearing descendant fingers and absence of external support.','input_findings':{'DINO':'Native RGB padded224;16x16 patch map384D, not just a global CLS vector.','object_input':'per-frame per-camera four ROIAlign cells averaged to384D, plus bbox4/visible1/occupancy1;2views concatenated780D','scene_input':'original4x4coarse per view averaged to2x2 and flattened;8ordered384D cells=3072D','history':'Each target gets shared MLP, concatenated scene/proprio encoding, shared GRU over4frames; no self-attention or learned entity query.','entities':'Pixel-colour registration selects an ROI; symbolic handle indexes output. No learned colour embedding; variableK supported.','shared':'held andin share object encoder, objectGRU, 64hidden outputMLP; scene128 context is shared across targets. Draweropen has separate sceneGRU/head.','proprio':'15values/4frames:TCPxyz,RPY,opening_width,7joint positions,command, AB-normalized.','discarded':'Nine non-target instance ROIs are not separately passed to old reader; no explicit target-to-robot/drawer pair representation.','not_used':'Depth and camera calibration are saved but unused by old reader. No objectGTxyz, contacts, futureframes or success inputs.','caution':'Object bbox and ordered global cells retain spatial clues; DINO averaged embeddings also contain contextual position. Findings do not prove the model cannot reason relationally.'},'old_tests_not_read':True,'no_model_training':True,'in_label_audit_rows':labelrows}
    save(E/'results/input_and_label_audit.json',out);print(json.dumps({k:v for k,v in out.items()if k not in ['in_label_audit_rows','input_findings','drawer_coverage_rows','drawer_visible_but_not_in_cached_slots']},ensure_ascii=False),flush=True)
    assert not mismatch and not disagreements and not unexpected_neg,'Preserve audit and investigate before a method change.'
if __name__=='__main__':main()
