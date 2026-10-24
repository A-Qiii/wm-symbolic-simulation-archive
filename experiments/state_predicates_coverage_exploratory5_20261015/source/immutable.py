from experiment import *
D=P/'experiments/state_predicates_independent_D_20261015';DB=P/'data/predicate_independent_D_20261015'
def verify_sealed():
 output=[]
 for root,expected in [(D,'d12a08430c118741527c54eb44110c3b3929c5a9003193546c9f32ce2c7912f4'),(DB,'573aadc3c640e15264d81defd2aaff4dde53603acdf585c73a1248330ff36e28'),(P/'experiments/state_predicates_drawer_factor_C_20261015','115da51384d9c0b770f2bccbe341e00feb6e43f19fc40f6836cd5b08c1d6382b'),(P/'data/predicate_drawer_factor_C_20261015','16d12e5d0c90b0854cbede436dcfe1c5d6b3da3d36b244d2670e927ff7a354f8'),(P/'experiments/state_predicates_condition_coverage_control_20261015','859cbdee13e66c1bb84025d7e59e640fb2b5cd5b7c7c97e141e3c8f9bdacb3cf'),(P/'data/predicate_condition_coverage_control_20261015','9b96c5fdaa63d470ed7e2a0beabc7ff1a115437402534feb34db5a1aab054e2b'),(TRAIN,'daaaa75497eb09be78770ba7f08f5e11796a36cdfaa1f7948eabb04e13ba8018')]:
  assert sha(root/'delivery_manifest.json')==expected;man=read(root/'delivery_manifest.json');entries=man.get('entries',man.get('files'));total=0
  for entry in entries:
   p=root/entry['path'];assert p.stat().st_mode&0o222==0 and sha(p)==entry['sha256'],str(p);total+=p.stat().st_size
  assert len([p for p in root.rglob('*')if p.is_file()])==len(entries)+1
  output.append({'root':str(root),'manifest_sha256':expected,'entries_verified':len(entries),'bytes':total,'all_read_only':True})
 return output
