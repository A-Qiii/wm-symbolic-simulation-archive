from pathlib import Path
import json,hashlib
import imageio.v2 as imageio
import numpy as np
from PIL import Image
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_interface_acceptance_20261014'
def main():
 rows=json.loads((E/'videos/index.json').read_text());checked=[]
 for r in rows:
  path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];reader=imageio.get_reader(str(path))
  try:
   n=reader.count_frames();md=reader.get_meta_data();first=reader.get_data(0);last=reader.get_data(63);assert n==64 and abs(md['fps']-30)<1e-6 and first.shape==last.shape==(550,700,3)and np.isfinite(last).all()
   if r['window']in ['D_interface_accept_blue_open_1__t0598','D_interface_accept_pink_open_1__t0471']:Image.fromarray(last).save('/tmp/'+r['window']+'_video_preview.png')
  finally:reader.close()
  checked.append({'window':r['window'],'frames':n,'fps':md['fps'],'frame_shape':[550,700,3],'sha256':r['sha256'],'first_and_last_frame_decoded':True})
 (E/'results/video_validation.json').write_text(json.dumps({'passed':True,'videos':len(checked),'checks':checked},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'videos_verified':len(checked),'frames_each':64,'fps':30}))
if __name__=='__main__':main()
