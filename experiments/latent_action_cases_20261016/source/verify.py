from common import *
from PIL import Image
from html.parser import HTMLParser
import ast

def main():
 src=read(E/'configs/sources.json')['sha256'];ext=read(E/'results/extraction.json')
 for key in ['model','original_encoder_model']:
  r=ext[key];src[r['path']]=r['sha256']
 for p,h in src.items():assert sha(p)==h,p
 cases=read(E/'results/cases.json');assert len(cases)==4 and len({r['root_id']for r in cases})==4
 assert [r['id']for r in cases]==['p00030','p00061','p00004','p00042']
 assert cases[3]['actual_controls']==0 and cases[3]['native_reason']=='requires_open_drawer'
 d=dict(np.load(E/'data/latents.npz'));co=dict(np.load(E/'results/coordinates.npz'));prior=np.load(PREV/'results/coordinates.npz')
 assert np.array_equal(d['background'],np.load(PREV/'data/main_latents.npz')['interaction'])
 for k in ['mean','std']:assert np.array_equal(d[k],prior['interaction_'+k])
 assert co['logical_xy'].shape==(350,2)and co['unique_xy'].shape==(343,2)
 assert np.array_equal(co['unique_xy'][co['logical_to_unique']],co['logical_xy'])
 assert np.array_equal(co['logical_xy'][338:].reshape(4,3,2),co['case_xy'])
 assert np.all(co['logical_xy']>=co['axis_limits'][0])and np.all(co['logical_xy']<=co['axis_limits'][1])
 pred=np.load(REPAIR/'results/D2_canonical_matched_seed0.npz');index={int(v):j for j,v in enumerate(pred['pair_indices'])}
 for i,r in enumerate(cases):
  assert np.array_equal(d['predicted'][i],pred['predicted_latent'][index[int(r['id'][1:])]].reshape(256))
  assert np.isclose(r['latent_mse_AB_standardized'],np.mean(((d['predicted'][i]-d['actual'][i])/d['std'])**2))
  assert np.array_equal(discrete(r['predicted_probabilities']),r['predicted_labels'])
  for role in ['before','after']:
   path=Path(r[role+'_path']);b=next(x for x in read(path.parent.parent/'_private/boundaries.json')if x['name']==path.stem);assert b['frame']==r[role+'_frame']and sha(path)==r[role+'_sha256']
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.refs=[]
  def handle_starttag(self,t,a):
   for k,v in a:
    if k in ['href','src']:self.refs.append(v)
 parser=Links();parser.feed((R/'index.html').read_text())
 for ref in parser.refs:
  if not ref.startswith('#'):assert '://'not in ref and (R/ref).is_file(),ref
 images=list(R.glob('*.png'))+list((R/'images').glob('*.jpg'));assert len(images)==13
 for path in images:
  with Image.open(path)as im:im.verify()
 for path in (E/'source').glob('*.py'):ast.parse(path.read_text())
 prev=read(PREV/'configs/delivery_manifest.json')
 for p,h in prev['sha256'].items():assert sha(p)==h,p
 out={'passed':True,'source_hashes':len(src),'previous_tSNE_sealed_files_verified':len(prev['sha256']),'case_count':4,'distinct_roots':4,'logical_points':350,'unique_joint_points':343,'image_files':13,'HTML_references_checked':len(parser.refs),'model_updates':0,'new_simulation_calls':0,'browser_tested':False,'browser_note':'Prior file: URL policy block respected; no retry or workaround.'}
 if (E/'configs/delivery_manifest.json').exists():
  m=read(E/'configs/delivery_manifest.json')
  for p,h in m['sha256'].items():assert sha(p)==h,p
  out['delivery_hashes_verified']=len(m['sha256'])
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
