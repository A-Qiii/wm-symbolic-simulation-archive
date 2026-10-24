from pathlib import Path
import ast,csv,json,collections,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
T=Path(__file__).resolve().parent.parent
checks=[]
for seed in [0,1]:
 with (T/f'results/coordinates_seed{seed}.csv').open() as f:r=list(csv.DictReader(f))
 with (T/f'data/samples_seed{seed}.csv').open() as f:s=list(csv.DictReader(f))
 with (T/f'results/scene_coordinates_seed{seed}.csv').open() as f:c=list(csv.DictReader(f))
 z=np.load(T/f'data/observed_latents_seed{seed}.npz');q=np.load(T/f'results/coordinates_seed{seed}.npz');e=np.load(T/f'results/scene_coordinates_seed{seed}.npz')
 assert len(r)==len(s)==2880 and len(c)==960
 assert z['object_h'].shape==(2880,128) and z['context_c'].shape==(960,128)
 assert np.array_equal(z['full_nodes'][:,:3].reshape(2880,128),z['object_h'])
 assert np.array_equal(z['full_nodes'][:,3],z['context_c'])
 assert len(set((a['split'],a['episode_id'])for a in c))==160
 assert set(a['step']for a in c)==set('123456')
 assert all(v==6 for v in collections.Counter((a['split'],a['episode_id'])for a in c).values())
 assert collections.Counter(a['object']for a in r)=={'red':960,'blue':960,'pink':960}
 assert all(a['true_object_valid']=='1' and a['drawer_valid']=='1' for a in r)
 assert all(all(a[k]==b[k]for k in b)for a,b in zip(r,s))
 assert np.array_equal(np.array([[float(a['tsne_x']),float(a['tsne_y'])]for a in r]),q['xy'])
 assert np.array_equal(np.array([[float(a['tsne_x']),float(a['tsne_y'])]for a in c]),e['xy'])
 assert all(np.isfinite(a).all()for a in [z['object_h'],z['context_c'],q['xy'],e['xy']])
 counts={}
 for split in ['development','combination_test']:
  rr=[a for a in r if a['split']==split];cc=[a for a in c if a['split']==split]
  counts[split]={'object_points':len(rr),'scene_points':len(cc),'true_In':sum(int(a['true_In'])for a in rr),'true_Held':sum(int(a['true_Held'])for a in rr),'In_FP':sum(a['In_error_type']=='FP'for a in rr),'In_FN':sum(a['In_error_type']=='FN'for a in rr),'Held_only_error':sum(a['pred_Held']!=a['true_Held'] and a['In_error_type']not in ['FP','FN']for a in rr),'drawer_errors':sum(int(a['drawer_error'])for a in cc)}
 fig=[]
 for ext in ['png','svg','pdf']:
  p=T/f'figures/最新868模型_四份观测记忆_tSNE_seed{seed}.{ext}';assert p.stat().st_size>10000
  if ext=='png':
   im=Image.open(p);im.verify();im=Image.open(p);size=im.size;assert size[0]>4000 and size[1]>3000
  if ext=='svg':assert ET.parse(p).getroot().tag.endswith('svg')
  if ext=='pdf':assert p.read_bytes().startswith(b'%PDF')
  fig.append({'file':str(p.relative_to(T)),'bytes':p.stat().st_size})
 checks.append({'seed':seed,'metadata_latent_coordinate_alignment':True,'all_truth_masks_valid':True,'counts_by_split':counts,'figures':fig,'png_dimensions':size})
for p in (T/'source').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
(T/'results/output_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
