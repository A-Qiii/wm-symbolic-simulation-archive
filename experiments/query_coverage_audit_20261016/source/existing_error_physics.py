"""Only inspect the six C windows already listed as errors before this audit."""
from common import *
from physical_audit import row_for
import gzip

def main():
 wanted={r['window_id']for r in read(SP/'results/errors.json')if r['model']=='spatial'};assert len(wanted)==6;roots={r['root_id']:r for r in read(B/'roots_manifest.json')};rows=[]
 for w in read(BASE/'data/windows.json'):
  if w['window_id']not in wanted:continue
  with gzip.open(B/'episodes'/w['episode_id']/'_private/full_physics_trace.json.gz','rt')as f:trace=json.load(f)
  rows.extend(row_for(w,trace[w['anchor']]['physical'],roots[w['root_id']]))
 save(E/'results/existing_C_error_physics.json',rows)
 for r in rows:
  if r['window_id']=='C_semantics_v2_fixtures_blue_1__outside_front__t00063'and r['object']=='blue':print(json.dumps(r),flush=True)
if __name__=='__main__':main()
