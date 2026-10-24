"""Read-only checks of the frozen visualization and original inputs; no fit or simulation."""
from common import *
from html.parser import HTMLParser
from PIL import Image
from collections import Counter
import ast

def main():
 models=read(E/'results/extraction.json')['models']
 source_hashes=read(E/'configs/sources.json')['sha256']
 for r in models.values():source_hashes[r['path']]=r['sha256']
 for p,h in source_hashes.items():assert sha(p)==h,p
 rows=read(E/'data/main_states.json');pairs=read(E/'data/joint_pairs.json');payload=read(E/'results/interactive_payload.json')
 assert len(rows)==338 and len(pairs)==47 and len({r['root_id']for r in rows})==57
 assert max(Counter(r['root_id']for r in rows).values())<=8
 assert len({r['key']for r in rows})==338
 assert len(payload['main_nodes'])==338 and len(payload['joint_nodes'])==141 and len(payload['lines'])==12
 coords=dict(np.load(E/'results/coordinates.npz'))
 for k,c in [('main_xy','interaction_xy'),('ae_xy','ae_xy'),('stability_xy','stability_xy'),('joint_xy','joint_xy')]:assert np.array_equal(np.array(payload[k]),coords[c]),k
 for r,n in zip(rows,payload['main_nodes']):assert r['point_id']==n['id']and r['root_id']==n['root']and Path(n['path']).is_file()
 for i,p in enumerate(pairs):
  for offset,kind in enumerate(['actual','canonical_control','canonical_matched']):
   n=payload['joint_nodes'][offset*47+i];assert n['pair_id']==p['pair_id']and n['kind']==kind
 for n in payload['main_nodes']+payload['joint_nodes']:
  for key in ['image','before_image']:
   if key in n:assert (R/n[key]).is_file(),n[key]
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.refs=[]
  def handle_starttag(self,t,attrs):
   for k,v in attrs:
    if k in ['href','src']:self.refs.append(v)
 html=(R/'index.html').read_text();parser=Links();parser.feed(html)
 for ref in parser.refs:assert '://'not in ref and (R/ref).is_file(),ref
 assert '__DATA__'not in html and '__METRIC_ROWS__'not in html and '__GALLERY__'not in html
 embedded=html.split('const data=',1)[1].split(';\nconst $=',1)[0]
 assert json.loads(embedded)==payload
 images=list(R.glob('*.png'))+list((R/'images').glob('*.jpg'))
 for f in images:
  with Image.open(f)as im:im.verify()
 assert len(list(R.glob('*.png')))==6 and len(list((R/'images').glob('*.jpg')))==432
 for f in (E/'source').glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 out={'passed':True,'checked_source_hashes':len(source_hashes),'main_points':338,'joint_points':141,'image_files_verified':len(images),'local_HTML_references_verified':len(parser.refs),'browser_interaction_tested':False,'browser_blocker':'Browser URL policy blocks file: URLs; no alternate browser route attempted.','fits_or_physics_in_this_verifier':0}
 manifest=E/'configs/delivery_manifest.json'
 if manifest.exists():
  m=read(manifest)
  for p,h in m['sha256'].items():assert sha(p)==h,p
  out['delivery_hashes_verified']=len(m['sha256'])
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
