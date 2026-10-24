from audit import *
import argparse
from PIL import Image
def verify():
 assert not any(x in sys.modules for x in ['torch','transformers','pybullet'])
 original=protected();assert original==read(D/'configs/protected_before.json')
 records=list(csv.DictReader((D/'results/records.csv').open()));assert len(records)==6300
 roots={s:set(r['root_id']for r in records if r['split']==s)for s in SPLITS};assert [len(roots[s])for s in SPLITS]==[40,10,20]
 assert all(not roots[a]&roots[b]for i,a in enumerate(SPLITS)for b in SPLITS[i+1:])
 assert set(collections.Counter(r['root_id']for r in records).values())=={90}
 h0=[r for r in records if r['h0_unique']=='True'];assert len(h0)==420
 assert set(collections.Counter(r['root_id']for r in h0).values())=={6}
 cal=read(D/'results/calibration.json');sigs=[list(cal['camera_signatures'][e+'_static'])for e in 'ABC'];assert len(set(s[0]for s in sigs))==1 and all(len(s)==1 for s in sigs)
 table=[r for r in records if r['camera']=='static'and int(float(r['GT_location']))<3];assert len(table)==2393
 for r in table:
  x=float(r['GT_x_m']);cuts=cal['scenes'][r['environment']]['cuts_x_m'];assert (int(x>=cuts[0])+int(x>=cuts[1]))==int(r['GT_location'])
 assert cal['cached_ROI_vs_pixels_max_difference']<3e-8 and not cal['table_label_mismatches']
 pairs=0
 for split in SPLITS:
  a=[r for r in records if r['camera']=='static'and r['split']==split]
  for seed in [0,1]:
   z=np.load(O/f'data/predictions_object_seed{seed}_{split}.npz');lookup={(e,int(h)):i for i,(e,h)in enumerate(zip(z['episode_id'],z['step']))};probs=z['prob']
   for r in a:
    j=COLORS.index(r['color']);i=lookup[r['episode_id'],int(r['step'])];assert int(probs[i,j*8:j*8+8].argmax())==int(r[f'object_seed{seed}_prediction']);pairs+=1
 support=list(csv.DictReader((D/'results/support_ranges.csv').open()))
 for s in support:
  rr=[r for r in records if(r['split'],r['camera'],r['GT_location'])==(s['split'],s['camera'],s['location'])and(s['color']=='all'or r['color']==s['color'])and(s['cohort']=='all_boundaries'or r['h0_unique']=='True')]
  assert len(rr)==int(s['records'])and len(set(r['root_id']for r in rr))==int(s['roots'])
  vv=[r for r in rr if r['visible']=='True'];assert len(vv)==int(s['visible'])
  for k in ['GT_x_m','u_px','v_px','width_px','height_px']:
   vals=[float(r[k])for r in(rr if k=='GT_x_m'else vv)]
   if vals:assert float(s[k+'_min'])==min(vals)and float(s[k+'_max'])==max(vals)
 train=[float(r['GT_x_m'])for r in table if r['split']=='train'and r['GT_location']=='2'];c=[float(r['GT_x_m'])for r in table if r['split']=='externalC'and r['GT_location']=='2'];assert len(c)==466 and all(x<min(train)or x>max(train)for x in c)
 pictures=list((D/'figures').glob('*.png'));assert len(pictures)==4
 for p in pictures:
  with Image.open(p)as im:im.verify()
 text=(D/'REPORT.html').read_text();assert text.count('data:image/png;base64,')==4 and 'src="figures/'not in text
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'VERIFIED','old_sealed_files_unchanged':len(original),'records':len(records),'independent_roots':70,'h0_object_camera_records':len(h0),'table_labels_recomputed':len(table),'archived_object_predictions_checked':pairs,'support_groups_recomputed':len(support),'C_right_records_outside_train_right_x_range':len(c),'static_calibration_identical_ABC':True,'ROI_rounding_max_abs_error':cal['cached_ROI_vs_pixels_max_difference'],'model_libraries_loaded':False,'fits':0,'model_forwards':0,'collection':0,'uploads':0,'figures_integrity':4,'visual_review':'All four generated PNGs viewed; axes, legends, image boxes and case evidence readable. HTML sources/images validated; no browser layout claim.'}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args();sys.addaudithook(guard);m=D/'manifest.json'
 if args.seal:assert not m.exists()
 else:
  for name,h in read(m)['sha256'].items():assert sha(D/name)==h,name
 result=verify()
 if args.seal:
  save(D/'final_verification.json',result);files=sorted(p for p in D.rglob('*')if p.is_file()and 'cache'not in p.relative_to(D).parts and p!=m)
  save(m,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'sha256':{str(p.relative_to(D)):sha(p)for p in files},'bytes':sum(p.stat().st_size for p in files)})
  print('SEALED',len(files),'files; SHA256',sha(m))
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
