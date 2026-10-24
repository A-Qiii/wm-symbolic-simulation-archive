from common import *
import shutil,ast,tempfile,base64
OLD=P/'experiments/endpoint_sync_truth_4state_20261018'
def main():
 assert not E.exists()and not R.exists()
 m=read(OLD/'configs/delivery_manifest.json');assert all(sha(p)==h for p,h in m['sha256'].items())
 for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 cases=[b'panda_leftfinger',bytes.fromhex('aba5a9f06d7b'),bytes(range(256))]
 for value in cases:
  out=json.loads(json.dumps(value,default=json_default));assert base64.b64decode(out['data'])==value
 with tempfile.TemporaryDirectory(prefix='sync_truth_save_check_',dir='/tmp')as td:
  save_raw_npz(Path(td)/'raw.npz',rgb=np.zeros((2,2,3),np.uint8),depth_buffer=np.array([[.5,.7],[.4,.8]],np.float32),object_link_mask=np.array([[1,-1],[16777216,2]],np.int32))
 for d in ['source','configs','data','results','logs']:(E/d).mkdir(parents=True)
 (R/'figures').mkdir(parents=True)
 for file in ['protocol.json','mesh_semantics.json','source_references.json','mesh_visual_review.json']:shutil.copy2(OLD/'configs'/file,E/'configs'/file)
 shutil.copy2(P/'reports/endpoint_sync_truth_4state_20261018/figures/mesh_semantics.png',R/'figures/mesh_semantics.png')
 addendum={'utc':now(),'authorization':'Explicit delegated instruction from parent01a14e2c-4647-734c-b1aa-a3a4547ec02a: fix technical persistence failure and retry original A/B/C/D once; keep failed attempt, do not change selection, label definitions, or quality gates.','previous_experiment':str(OLD),'previous_manifest_sha256':sha(OLD/'configs/delivery_manifest.json'),'previous_attempts':[{'scene':'A','status':'metadata_serialization_failure','usable_saved_samples':0}],'scope_override':'The earlier technical stop rule does not require abandoning this already authorized implementation task. This new attempt runs the same four planned states; no data-dependent replacement. Further technical failures are recorded and may be repaired with explicit evidence, without changing conditions or gates.','changes_only':['Encode ALL native bytes as tagged base64 with optional strictly decodedUTF8 text, never guess decoding; preserve original bytes.','Persist and reopen-verify primary RGB/depth/nativeID buffers atomically inside each camera callback before optional metadata queries.','Use same safe serializer when comparing physical snapshots.','New experiment directory preserves old sealed failure files.'],'unchanged':['four scene/seed/drawer/robot/object choices','mesh face semantic labels','camera parameters and renderer','pixel sample convention and depth tolerance','unknown and visiblebbox validity rules','no models or training','old40 labels and samples'],'attempt_accounting':'Prior failed A world remains recorded separately. This retry completes at most4 planned state records/8primary camera records; 5totalworld attempts across rounds if all4retry states run, not5successful samples.','offline_tests_passed':['source syntax','all256byte values base64 roundtrip','validUTF8 name reversible','atomicNPZ raw arrays save/read equality with allow_pickleFalse'],'failure_scope':'No new software, no physical/visual resource edits, no cloud simulation.'}
 save(E/'configs/technical_retry_addendum.json',addendum)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 files=[p for d in ['source','configs']for p in(E/d).rglob('*')if p.is_file()];save(E/'configs/pre_run_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in files}})
 print('RETRY_FROZEN',sha(E/'configs/pre_run_frozen.json'))
if __name__=='__main__':main()
