from common import *
import csv,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def contact_evidence(physical,mapping,color,cutoff=.01):
 uid=mapping['object_uids'][color];robot=mapping['robot_uid'];families=mapping['finger_families'];out=[]
 for c in physical['contacts']:
  if c[1]==uid:other,link,normal,point=c[2],c[4],np.array(c[7]),c[5]
  elif c[2]==uid:other,link,normal,point=c[1],c[3],-np.array(c[7]),c[6]
  else:continue
  family=next((j for j,f in enumerate(families)if other==robot and link in f),None)
  out.append({'other_uid':other,'other_link':link,'finger_family':family,'external':other!=robot,'normal_force_N':float(c[9]),'normal_on_object':normal.tolist(),'normal_vertical_force_N':float(c[9]*normal[2]),'gap_m':float(c[8]),'point_on_object':point,'above_cutoff':c[9]>cutoff})
 fingers=[sum(c['normal_force_N']for c in out if c['finger_family']==j and c['above_cutoff'])for j in [0,1]];external=[c for c in out if c['external']and c['above_cutoff']]
 return {'holding':all(v>0 for v in fingers)and not external,'finger_normal_force_sums_N':fingers,'external_normal_force_sum_N':sum(c['normal_force_N']for c in external),'external_vertical_normal_force_sum_N':sum(c['normal_vertical_force_N']for c in external),'external_contacts':external,'contacts':out}
def crop(a,b):return a[b[1]:b[3],b[0]:b[2]]
def compare(a,b):
 assert a.shape==b.shape
 d=np.abs(a.astype(float)-b.astype(float));return {'shape':list(a.shape),'bitwise_equal':np.array_equal(a,b),'mean_absolute_RGB_difference_0_255':float(d.mean()),'max_RGB_difference':float(d.max()),'changed_pixel_fraction':float(np.any(a!=b,axis=-1).mean())}
def project(point,meta):
 v=np.array(meta['viewMatrix']).reshape(4,4,order='F');p=np.array(meta['projectionMatrix']).reshape(4,4,order='F');clip=p@v@np.r_[point,1.];ndc=clip[:3]/clip[3];return [float((ndc[0]+1)*meta['width']/2),float((1-ndc[1])*meta['height']/2)]
def main():
 guard('diagnostic');rows=read(B/'data/records.json');bykey={(r['root_id'],r['sample'],r['camera'],r['color']):(i,r)for i,r in enumerate(rows)};features=np.load(B/'data/features/dev.npz');fi={int(i):j for j,i in enumerate(features['record_indices'])};roots=[('relation36_C_24','red'),('relation36_C_34','blue')];allout=[];stateflat=[];checks=0
 for rid,color in roots:
  d=B/'data/private'/rid;mapping=read(d/'body_link_mapping.json');states=[];trace=[]
  for line in(d/'control_trace_private.jsonl').read_text().splitlines():
   t=json.loads(line);ev=contact_evidence(t['physical'],mapping,color);assert ev['holding']==('block_'+color in t['physical']['held']);checks+=1;trace.append({'control':t['control'],'tick':t['tick'],'phase':t['action'].get('phase'),'z_m':t['physical']['objects']['block_'+color]['position'][2],**ev})
  for sample in range(8):
   sd=d/f's{sample:02d}';truth=read(sd/'physical_truth.json');integ=read(sd/'integrity.json');physical=truth['physical'];evidence=contact_evidence(physical,mapping,color);cuts=[contact_evidence(physical,mapping,color,c)['holding']for c in [.005,.01,.015]];j=['red','blue','pink'].index(color);assert cuts==[bool(truth['labels_v2_1']['y'][j])]*3
   for h in truth['last12_controls']:
    for c in [.005,.01,.015]:assert contact_evidence(h['physical'],mapping,color,c)['holding']==cuts[1];checks+=1
   snap=read(sd/'snapshot_before.json');assert snap==read(sd/'snapshot_after.json');images={};meta={}
   for cam in ['static','gripper']:
    i,r=bykey[(rid,sample,cam,color)];z=np.load(sd/f'native_{cam}.npz');a=np.load(r['path'])[r['key']];assert np.array_equal(a,z['rgb']);seg=z['object_link_mask'];mask=(seg>=0)&((seg&((1<<24)-1))==mapping['object_uids'][color]);ys,xs=np.where(mask);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]if len(xs)else None;assert bbox==r['object_bbox'];cm=read(sd/f'camera_{cam}.json');assert cm['tick']==integ['tick'];centroid=[float(xs.mean()),float(ys.mean())]if len(xs)else None
    meta[cam]={'record_index':i,'box':bbox,'finger_union_box':r['gripper_bbox'],'visible_object_pixels':len(xs),'visible_mask_centroid_xy':centroid,'projected_object_center_xy':project(physical['objects']['block_'+color]['position'],cm),'RGB_sha256':r['RGB_sha256']};images[cam]=a
   states.append({'sample':sample,'anchor':rows[bykey[(rid,sample,'gripper',color)][0]]['anchor'],'control':integ['control'],'tick':integ['tick'],'holding':cuts[1],'cutoffs_agree':True,'last12_controls_agree':True,'object_position':physical['objects']['block_'+color]['position'],'object_velocity':physical['objects']['block_'+color]['linear_velocity'],'image_metadata':meta,**evidence});stateflat.append({'root':rid,'color':color,'sample':sample,'anchor':states[-1]['anchor'],'control':integ['control'],'tick':integ['tick'],'holding':int(cuts[1]),'z_m':states[-1]['object_position'][2],'left_force_N':evidence['finger_normal_force_sums_N'][0],'right_force_N':evidence['finger_normal_force_sums_N'][1],'external_normal_force_N':evidence['external_normal_force_sum_N'],'external_vertical_normal_force_N':evidence['external_vertical_normal_force_sum_N']})
  comparisons=[]
  for a,b in [(3,4),(3,5),(3,6)]:
   _,ra=bykey[(rid,a,'gripper',color)];_,rb=bykey[(rid,b,'gripper',color)];ia=np.load(ra['path'])['rgb_gripper'];ib=np.load(rb['path'])['rgb_gripper'];assert ra['object_bbox']==rb['object_bbox']and ra['gripper_bbox']==rb['gripper_bbox'];ma=np.load(d/f's{a:02d}/native_gripper.npz')['object_link_mask'];mb=np.load(d/f's{b:02d}/native_gripper.npz')['object_link_mask'];uid=mapping['object_uids'][color];am=(ma>=0)&((ma&((1<<24)-1))==uid);bm=(mb>=0)&((mb&((1<<24)-1))==uid);roi=np.zeros(ia.shape[:2],bool)
   for box in [ra['object_bbox'],ra['gripper_bbox']]:roi[box[1]:box[3],box[0]:box[2]]=True
   ja=fi[bykey[(rid,a,'gripper',color)][0]];jb=fi[bykey[(rid,b,'gripper',color)][0]];xa=features['x_pair'][ja];xb=features['x_pair'][jb];assert np.array_equal(xa[-5:],xb[-5:])
   comparisons.append({'samples':[a,b],'wrist_full_RGB':compare(ia,ib),'block_crop':compare(crop(ia,ra['object_bbox']),crop(ib,rb['object_bbox'])),'finger_union_crop':compare(crop(ia,ra['gripper_bbox']),crop(ib,rb['gripper_bbox'])),'outside_model_crop_changed_pixel_fraction':float(np.any(ia!=ib,axis=-1)[~roi].mean()),'visible_mask_IoU':float((am&bm).sum()/(am|bm).sum()),'visual_feature_equal':np.array_equal(xa[:256],xb[:256]),'visual_feature_L2_difference':float(np.linalg.norm(xa[:256]-xb[:256])),'visual_feature_cosine':float(np.dot(xa[:256],xb[:256])/(np.linalg.norm(xa[:256])*np.linalg.norm(xb[:256]))),'geometry_bitwise_equal':True})
  out={'root_id':rid,'color':color,'states':states,'comparisons':comparisons,'trace':trace};allout.append(out)
  fig,axs=plt.subplots(4,6,figsize=(18,10),gridspec_kw={'height_ratios':[1.45,1.3,1,1]});fig.suptitle(f'{rid} / {color}: actual saved RGB, same wrist boxes do not imply same pixels',fontsize=16)
  labels=['s02 Open, near','s03 Closed, table','s04 Lifted','s05 Stable','s06 Translated','s07 Released']
  for col,sample in enumerate(range(2,8)):
   for row,cam in [(0,'static'),(1,'gripper')]:
    _,rr=bykey[(rid,sample,cam,color)];img=np.load(rr['path'])[rr['key']];axs[row,col].imshow(img,interpolation='nearest')
    for box,c in [(rr['object_bbox'],'#00d7ed'),(rr['gripper_bbox'],'#ff9f35')]:
     if box:axs[row,col].add_patch(Rectangle((box[0]-.5,box[1]-.5),box[2]-box[0],box[3]-box[1],fill=False,edgecolor=c,linewidth=1.3))
   _,rr=bykey[(rid,sample,'gripper',color)];im=np.load(rr['path'])['rgb_gripper']
   for row,key in [(2,'object_bbox'),(3,'gripper_bbox')]:axs[row,col].imshow(crop(im,rr[key]),interpolation='nearest',aspect='equal')
   st=states[sample];axs[0,col].set_title(f'{labels[col]}\ny={int(st["holding"])} | control {st["control"]}\nexternal force {st["external_normal_force_sum_N"]:.2f} N',fontsize=10)
  for row,name in enumerate(['Full static','Full wrist (84x84)','Exact block crop','Exact finger-union crop']):axs[row,0].set_ylabel(name,fontsize=11)
  for ax in axs.flat:ax.set_xticks([]);ax.set_yticks([])
  fig.text(.02,.02,'Cyan: native visible block box. Orange: visible finger-family union box. Nearest-neighbor display; no generated pixels.\nRows 3/4 are the exact wrist crops used by the encoder. Full-image/table context and other-camera information are omitted from each independent crop prediction. Forces are private diagnostic evidence.',fontsize=10)
  fig.subplots_adjust(top=.87,bottom=.10,hspace=.27,wspace=.11);fig.savefig(R/f'figures/{rid}_RGB_neighbors.png',dpi=145);fig.savefig(R/f'figures/{rid}_RGB_neighbors.pdf');plt.close(fig)
 save(E/'results/alias_inspection.json',{'utc':now(),'roots':allout,'physical_label_checks':checks,'no_new_render_or_encoder_execution':True,'labels_fixed':True})
 with(R/'alias_states_contacts.csv').open('w')as f:w=csv.DictWriter(f,list(stateflat[0]));w.writeheader();w.writerows(stateflat)
 print('ALIAS_INSPECTED',checks)
 for root in allout:
  print(root['root_id'],root['comparisons'][0]);print('STATES',[(s['sample'],s['control'],s['holding'],s['external_normal_force_sum_N'],s['external_vertical_normal_force_sum_N'])for s in root['states']])
if __name__=='__main__':main()
