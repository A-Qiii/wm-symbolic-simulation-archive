"""One evidence-backed analysis-only correction: brown A table entered red mask."""
try:
 from official_ab_verify import *
except ModuleNotFoundError:
 from verify import *
import datetime,copy
def cm(rgb):
 x=rgb.astype(float);r,g,b=x[...,0],x[...,1],x[...,2]
 return np.array([(r>65)&(r>3*g)&(r>3*b),(b>55)&(b>1.4*r)&(b>1.4*g),(r>65)&(b>65)&(np.minimum(r,b)>1.4*g)&(b>.55*r)&(r>.55*b)])
def main():
 assert not(E/'configs/color_mask_refinement.json').exists()
 rows=read(E/'results/restoration.json');old=read(E/'results/color_mapping_evidence.json')
 record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'correction_number':1,'scope':'image analysis mask only; no rerender, no simulation, no downloaded frames changed, no model or predicate definition change','cause_evidence':'A frame1785231 original red threshold selected23011 pixels across the brown table, while native red object occupied238 pixels, all238 matching red; original/rendered static RGB MAE=.085833/255 and blue/pink shapes align. Raw coarse global color IoU therefore measured brown background contamination, not object geometry failure.','single_fixed_change':'For all A/B frames red channel ratios r>1.5g,1.5b become r>3g,3b. Existing r>65 retained; blue and pink masks unchanged. No further threshold search.','preserved_preliminary':['results/restoration.json','results/color_mapping_evidence.json','configs/restoration_protocol.json'],'evaluation_thresholds_unchanged':'all3 color identity, purity>=.5, medianIoU>=.3 on >=5 visible pixels; all historical-force positives unknown','before_reanalysis':True}
 write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 write(E/'configs/color_mask_refinement.json',record);shutil.copy2(__file__,E/'source/mask_refine.py');new={}
 for scene in ['A','B']:
  matrix=np.zeros((3,3),int);frames=[]
  for row in [r for r in rows if r['scene']==scene]:
   physical=read(Path(row['folder'])/'physical_t0.json');uid={c:physical['physical']['objects']['block_'+c]['uid']for c in ['red','blue','pink']};a=np.load(row['source_path'],allow_pickle=False);b=np.load(Path(row['folder'])/'rendered.npz',allow_pickle=False);seg=b['seg_t0_static'];masks=cm(a['rgb_static']);mat=np.zeros((3,3),int);overlap=[]
   for j,c in enumerate(['red','blue','pink']):
    mask=(seg&((1<<24)-1))==uid[c]
    for k in range(3):mat[j,k]=int((mask&masks[k]).sum())
    union=int((mask|masks[j]).sum());overlap.append({'color':c,'original_color_pixels':int(masks[j].sum()),'rendered_visible_pixels':int(mask.sum()),'intersection':int((mask&masks[j]).sum()),'IoU':float((mask&masks[j]).sum()/union)if union else None})
   matrix+=mat;frames.append({'frame_id':row['frame_id'],'root_id':row['root_id'],'native_color_vs_original_RGB_pixels':mat.tolist(),'color_geometry_overlap':overlap})
  ious={c:[a['IoU']for f in frames for a in f['color_geometry_overlap']if a['color']==c and a['rendered_visible_pixels']>=5 and a['IoU']is not None]for c in ['red','blue','pink']};median={c:float(np.median(v))if v else 0. for c,v in ious.items()};totals=matrix.sum(1);purity=np.diag(matrix)/np.maximum(totals,1);identity=bool(np.all(totals>=5)and np.array_equal(matrix.argmax(1),np.arange(3))and np.all(purity>=.5)and min(median.values())>=.3)
  new[scene]={'native_slot_order':old[scene]['native_slot_order'],'native_object_rows_original_color_columns':matrix.tolist(),'row_purity':purity.tolist(),'median_color_mask_IoU':median,'native_identity_supported':identity,'frames':frames,'one_analysis_correction':True}
 for r in rows:
  supported=new[r['scene']]['native_identity_supported'];v=[False]*6;reason=[]
  for j,c in enumerate(['red','blue','pink']):
   d=r['one_tick_objects'][c];v[j]=bool(supported and d['position_m']<=.002 and min(d['finger_gaps_t0']+d['finger_gaps_t1'])>.01);reason.append('known_negative_both_finger_families_geometrically_far'if v[j]else'unknown_historical_force_contact_not_recorded_or_mapping_drift_uncertain')
  for j,c in enumerate(['red','blue','pink']):
   gs=[r['restored_t0']['objects'][c],r['restored_t1']['objects'][c]];outside=all(g['minimum_side_rim_clearance_m']<=-.01 or g['floor_clearance_m']<=-.013 for g in gs);v[j+3]=bool(supported and r['one_tick_objects'][c]['position_m']<=.002 and outside);reason.append('known_negative_robustly_outside_drawer_geometry'if v[j+3]else'unknown_historical_support_force_or_containment_boundary')
  r['original_provisional_labels']['v']=v;r['original_provisional_labels']['reason']=reason;r['color_mask_analysis_version']='one_red_background_refinement'
 write(E/'results/color_mapping_evidence_refined.json',new);write(E/'results/restoration_refined_masks.json',rows);print('MASK_REFINEMENT_DONE',{s:{k:v for k,v in d.items()if k!='frames'}for s,d in new.items()},flush=True)
if __name__=='__main__':main()
