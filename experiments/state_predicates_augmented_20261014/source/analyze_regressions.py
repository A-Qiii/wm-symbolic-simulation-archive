"""Read saved predictions and measurements only; no model or environment execution."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path('/home/robot/research/projects/wm_symbolic')
O=P/'experiments/state_predicates_20261014'
E=P/'experiments/state_predicates_augmented_20261014'
C=['red','blue','pink']
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def frame_measurements(paths,color):
    counts={'static':[],'gripper':[]};edges={'static':[],'gripper':[]};last=None
    for path in paths:
        with np.load(path) as z:
            last=z['robot_obs'].copy()
            for cam in counts:
                rgb=z['rgb_'+cam];seg=z['seg_'+cam].copy()
                seg[(seg>=0)&((seg&((1<<24)-1))==0)]=0
                matches=[]
                for uid in np.unique(seg):
                    if uid<0:continue
                    mask=seg==uid;mean=rgb[mask].astype(float).mean(0)/255
                    k='red' if mean[0]>.35 and mean[0]>mean[1]*1.7 and mean[0]>mean[2]*1.7 else ('blue' if mean[2]>.35 and mean[2]>mean[0]*1.7 and mean[2]>mean[1]*1.7 else ('pink' if mean[0]>.35 and mean[2]>.35 and mean[1]<min(mean[0],mean[2])*.6 else None))
                    if k==color:matches.append(mask)
                assert len(matches)==1,(path,cam,color,len(matches))
                mask=matches[0];ys,xs=np.where(mask);counts[cam].append(int(mask.sum()))
                edges[cam].append(bool(np.any(xs==0)|np.any(xs==mask.shape[1]-1)|np.any(ys==0)|np.any(ys==mask.shape[0]-1)))
    return counts,edges,last

def main():
    om=load(O/'data/metadata.json');nm=load(E/'data/metadata_train_dev.json');nt=load(E/'data/metadata_test.json')
    d=load(E/'results/failure_diagnosis.json');ev=load(E/'results/evaluation.json')
    with np.load(O/'data/inputs_targets.npz') as z:
        cat=z['cat'].copy();valid=z['valid'].copy();split=z['splits'].copy();objects=z['objects'].copy()
    train=split=='train';ds=split=='diagnostic_D';di=np.flatnonzero(ds);did={int(k):i for i,k in enumerate(di)}
    old={s:np.load(E/'results'/f'old_seed{s}_old_D_regression_probabilities.npy') for s in range(3)}
    new={s:np.load(E/'results'/f'augmented_seed{s}_old_D_regression_probabilities.npy') for s in range(3)}
    h=(cat[ds,:3]==6)&valid[ds,:3]
    strata=[]
    for name,mask in [('抽屉关闭',h&(cat[ds,3:4]==0)),('抽屉打开',h&(cat[ds,3:4]==2))]+[(c,h&(np.arange(3)[None,:]==j)) for j,c in enumerate(C)]:
        strata.append({'group':name,'positive_records':int(mask.sum()),'roots':len({om[int(di[i])]['root_id'] for i,j in np.argwhere(mask)}),
                       'old_FN':[int((mask&(old[s][:,:3]<.5)).sum()) for s in range(3)],
                       'new_FN':[int((mask&(new[s][:,:3]<.5)).sum()) for s in range(3)]})
    groups={}
    for r in d['new_held_regressions']:
        groups.setdefault((r['episode_id'],r['object']),set()).add(r['index'])
    state_rows=[]
    for (episode,color),indices in sorted(groups.items()):
        idx=min(indices);r=om[idx];j=C.index(color);i=did[idx]
        counts,edges,rb=frame_measurements(r['public_inputs'],color)
        pos=np.array(r['object_positions_for_evaluation_only'][j]);geo=objects[idx,:,j].reshape(4,2,390)[:,:,384:390]
        unique_rgbs=set();maxdiff=0.
        for k in indices:
            with np.load(om[k]['public_inputs'][-1]) as z:
                unique_rgbs.add(hashlib.sha256(z['rgb_static'].tobytes()+z['rgb_gripper'].tobytes()).hexdigest())
                maxdiff=max(maxdiff,float(np.max(np.abs(z['robot_obs']-rb))))
        state_rows.append({'episode_id':episode,'root_id':r['root_id'],'object':color,'representative_index':idx,'boundary_indices':sorted(indices),'steps':[om[k]['step'] for k in sorted(indices)],
            'drawer':'closed' if r['category'][3]==0 else 'open','old_probabilities':[float(old[s][i,j]) for s in range(3)],'new_probabilities':[float(new[s][i,j]) for s in range(3)],
            'newly_lost_records_by_seed':[sum(x['seed']==s and x['episode_id']==episode and x['object']==color for x in d['new_held_regressions']) for s in range(3)],
            'visible_pixels_four_frames':counts,'touches_image_edge_four_frames':edges,
            'model_visible_flags_four_frames_by_camera':geo[:,:,4].tolist(),'model_roi_mask_occupancy_last_static_gripper':geo[-1,:,5].tolist(),
            'tcp_xyz_m':rb[:3].tolist(),'gripper_width_mm':float(rb[6]*1000),'gripper_command':float(rb[14]),
            'object_to_tcp_mm_evaluation_only':float(np.linalg.norm(pos-rb[:3])*1000),
            'object_height_m_evaluation_only':float(pos[2]),'latest_rgb_pair_unique_hashes_across_boundaries':len(unique_rgbs),
            'max_robot_coordinate_change_across_boundaries':maxdiff,'public_frames':r['public_inputs']})
    # Supervised fact counts and exact exposures, not a claim about gradient magnitudes.
    oy=np.concatenate([cat[:,:3]==6,cat[:,:3]==3,cat[:,3:4]==2],1)[train]
    ov=np.concatenate([valid[:,:3],valid[:,:3],valid[:,3:4]],1)[train]
    with np.load(E/'data/supplement_train_dev.npz') as z:
        sel=z['splits']=='train';ny=z['y'][sel];nv=z['train_valid'][sel]
    dist={'old_train_windows':900,'new_train_windows':73,'old_held_positive':int((oy[:,:3]*ov[:,:3]).sum()),'old_held_negative':int(((1-oy[:,:3])*ov[:,:3]).sum()),
          'supplement_held_positive':int((ny[:,:3]*nv[:,:3]).sum()),'supplement_held_negative':int(((1-ny[:,:3])*nv[:,:3]).sum()),
          'old_train_valid_facts':int(ov.sum()),'supplement_valid_facts':int(nv.sum()),'supplement_in_valid_facts':int(nv[:,3:6].sum()),
          'supplement_open_positive':int((ny[:,6]*nv[:,6]).sum()),'supplement_open_negative':int(((1-ny[:,6])*nv[:,6]).sum())}
    dist['old_held_positive_fraction']=dist['old_held_positive']/(dist['old_held_positive']+dist['old_held_negative'])
    dist['mixed_held_positive_fraction']=(dist['old_held_positive']+dist['supplement_held_positive'])/(dist['old_held_positive']+dist['old_held_negative']+73)
    dist['supplement_window_pool_share']=73/973
    dist['supplement_valid_fact_pool_share']=float(nv.sum()/(ov.sum()+nv.sum()))
    yy=np.concatenate([oy,ny]);vv=np.concatenate([ov,nv]);exposures=[]
    for seed in range(3):
        b=np.load(E/'logs'/f'seed{seed}_batch_indices.npy');newmask=b>=900
        allfacts=vv[b].sum();newfacts=vv[b][newmask].sum()
        heldpos=(yy[b][...,:3]*vv[b][...,:3]).sum();heldtotal=vv[b][...,:3].sum()
        exposures.append({'seed':seed,'supplement_presentations':int(newmask.sum()),'total_presentations':int(b.size),'supplement_window_exposure_fraction':float(newmask.mean()),
                          'supplement_valid_fact_presentations':int(newfacts),'all_valid_fact_presentations':int(allfacts),'supplement_valid_fact_exposure_fraction':float(newfacts/allfacts),
                          'held_positive_fact_exposure_fraction':float(heldpos/heldtotal)})
    measurements=[]
    for r in nm:
        if r['split']!='train':continue
        f=load(Path(r['labels_private']))['frames'][-1];a=f['objects_audit'][r['target_color']]
        measurements.append({'sample_id':r['sample_id'],'held':r['actual_held'],'condition':r['condition'],'width_mm':f['gripper_width_m']*1000,'object_tcp_mm':a['tcp_distance_m']*1000})
    ranges={}
    for condition,rows in [('supplement_train_true_held',[r for r in measurements if r['held']]),('supplement_train_near_closed_negative',[r for r in measurements if r['condition']=='near_closed_negative'])]:
        ranges[condition]={'n':len(rows),**{k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in ['width_mm','object_tcp_mm']}}
    inrows=[];ntmap={r['sample_id']:r for r in nt}
    for r in d['new_in_false_positives']:
        m=ntmap[r['sample_id']];f=load(Path(m['labels_private']))['frames'][-1];a=f['objects_audit'][m['target_color']]
        inrows.append({**{k:r[k] for k in ['seed','sample_id','root_id','condition','old_probability','new_probability']},'drawer':m['planned_drawer'],
                       'actual_held':m['actual_held'],'support_contact_count':a['external_support_contacts'],'bilateral_finger_force':a['bilateral_finger_force'],
                       'width_mm':f['gripper_width_m']*1000,'object_tcp_mm':a['tcp_distance_m']*1000,'visible_pixels':a['visible_pixels']})
    out={'analysis_only_existing_files':True,'no_new_model_inference_training_or_collection':True,'held_strata':strata,'held_regression_states':state_rows,
         'supervised_distribution':dist,'exact_exposure_ledger':exposures,'supplement_train_physical_ranges':ranges,'new_in_false_positives':inrows,
         'facts':['All held regressions concern post-lift held objects; latest command is close, measured opening about40mm, object-to-TCP about4–6mm.',
                  'All three affected scene/object groups remain visible in both cameras in all four input frames; none touches the native image edge.',
                  'Six of seven selected-seed0/1 false negatives occur in the16closed-drawer held records; repeated boundaries are dependent.',
                  'Global held positive-label fraction rises slightly after adding73windows; global class-ratio arithmetic does not support an explanation of more negative proportion.',
                  'Supplement in labels are masked; all four seed2 new in errors occur on supported tabletop objects in two closed-drawer roots.'],
         'hypotheses_not_established_causes':['Local decision-boundary or shared-representation changes may affect held and in; no intervention isolates this.',
                  'Only six new dev roots select checkpoints, which may favor the new distribution; final1500 controls show that early checkpoint selection alone cannot explain the whole regression.',
                  'Partial occlusion, contact appearance and pose interactions remain possible. Visible-mask pixels and ROI occupancy are not full-object occlusion percentages.'],
         'recommendation':'Keep old frozen checkpoints as the reproducible current baseline, with their known failures explicitly retained. Keep new selected checkpoints as candidates; do not replace the baseline or connect them to planning based on this study. This is not an assertion that the old model is deployment-reliable.'}
    save(E/'results/regression_state_analysis.json',out)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    fig,axs=plt.subplots(3,3,figsize=(14,10.5),gridspec_kw={'width_ratios':[1,1,1.6]},layout='constrained')
    for row,r in enumerate(state_rows):
        with np.load(r['public_frames'][-1]) as z:
            for c,cam in enumerate(['static','gripper']):axs[row,c].imshow(z['rgb_'+cam]);axs[row,c].axis('off');axs[row,c].set_title('静态相机' if c==0 else '夹爪相机')
        a=axs[row,2];a.axis('off');a.text(0,.98,r['episode_id'],va='top',weight='bold',fontsize=12)
        probs=lambda key:' / '.join(f'{v:.4f}' for v in r[key])
        text=f"实际 held({r['object']})=True；抽屉{'关闭' if r['drawer']=='closed' else '打开'}\n种子0/1/2；旧概率：{probs('old_probabilities')}\n新概率：{probs('new_probabilities')}\n新增漏判记录：{' / '.join(map(str,r['newly_lost_records_by_seed']))}\n实测夹爪开度：{r['gripper_width_mm']:.2f} mm（关闭指令）\n物体中心到TCP：{r['object_to_tcp_mm_evaluation_only']:.2f} mm\n四帧均在双相机可见；未触及图像边缘\n静态/夹爪末帧可见像素：{r['visible_pixels_four_frames']['static'][-1]} / {r['visible_pixels_four_frames']['gripper'][-1]}"
        a.text(0,.84,text,va='top',fontsize=11,linespacing=1.6)
    fig.suptitle('旧D新增 held 漏判的全部三组状态：展示末帧；相邻重复边界不算独立场景',fontsize=15)
    fig.savefig(E/'figures/regression_states.png',dpi=140);plt.close(fig)
    print(json.dumps({'states':state_rows,'strata':strata,'distribution':dist,'ranges':ranges,'in_errors':inrows},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
