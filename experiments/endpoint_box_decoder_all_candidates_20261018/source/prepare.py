from common import *
import shutil

def main():
 assert not E.exists()and not R.exists(),'Refuse overwrite'
 check_manifest(BASE/'configs/delivery_manifest.json')
 for sub in ['configs','source','data/raw_outputs','data/reference','data/annotations','results','logs']:(E/sub).mkdir(parents=True,exist_ok=True)
 (R/'figures').mkdir(parents=True)
 refs={str(BASE/'configs/delivery_manifest.json'):sha(BASE/'configs/delivery_manifest.json')}
 for p in sorted((BASE/'results/predictions').glob('*.npz')):shutil.copy2(p,E/'data/raw_outputs'/p.name);refs[str(p)]=sha(p)
 assert len(list((E/'data/raw_outputs').glob('*.npz')))==16
 for src,dst in[(BASE/'data/annotations/final.json',E/'data/annotations/final.json'),(BASE/'data/public/images.json',E/'data/images.json'),(BASE/'results/decoded_predictions.json',E/'data/reference/decoded_predictions.json')]:shutil.copy2(src,dst);refs[str(src)]=sha(src)
 prior=P/'experiments/endpoint_box_localizer_20261018/results/failure_audit.json';refs[str(prior)]=sha(prior)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 save(E/'configs/protocol.json',{'utc':now(),'frozen_before_new_decoding':True,'scope':'Read-only saved-output decoder correction; no feature/model forward or training',
  'data':'Exactly40previousRGB images and91frozen known boxes, same source split;16 raw-output archives covering224/448,seed0/1,train/ABdev/Cseen/Dseen',
  'original':'score>=.5,3x3same-class local maximum filter,classwise greedy NMS atIoU>.3,cap1drawer_front/2gripper_finger',
  'correction':'All valid candidates with score>=.5, sorted descending score then patch index, same classwise greedy box NMS atIoU>.3; no local peak filtering and NO cardinality cap for either class',
  'cap_scope':'Remove both historical front1 and finger2 limits to honor all-candidate processing; extra retained boxes count asFP on known class-images',
  'same_geometry':'same sigmoid/tanh box decoding and[0,1]clipping; invalid boxes discarded as before',
  'only_one_new_decoder':True,'no_other_variants_or_threshold_search':True,
  'GT_is_scoring_only':'Candidate decoding and NMS never read annotations; predicted lists frozen before evaluation. All unmatched retained candidates count asFP, including duplicates. Unknown class-images are unscored, not negative.',
  'metrics':'Scoreordered greedy1:1IoU>=.5 TP/FP/FN and all-GT bestIoU; no output=0. Every split,resolution,seed separately; camera strata and unknown coverage supplemental. No not_visible examples exist.',
  'candidate_diagnostic':'For each saved threshold-passing candidate, trace oldpeak/NMS/cap disposition and newNMS disposition. After output freeze, useGT only to distinguish no qualifying candidate vs decoder rejection/one-to-one competition; never turn oracle coverage into model score.',
  'prior_diagnostic':'Earlier224 no-peak diagnostic retained caps; does not cover this exact uncapped version, so all four model configurations are redecoded once using saved raw outputs.',
  'preselected_figures':[{'resolution':224,'seed':0,'image_id':0,'purpose':'adjacent real fingers in training'},{'resolution':448,'seed':0,'image_id':27,'purpose':'reported cap loss in development'},{'resolution':448,'seed':0,'image_id':16,'purpose':'remaining upstream failure in first development state'}],
  'stop':'End after evaluation, stage summary and seal; no training, deployment, newheads, additional feature extraction or repeated development tuning',
  'limitations':'Historicalseen data, same-assistant labels; provenance and repeated consistency do not establish independent annotation accuracy; C/Ddiagnostic only'})
 save(E/'configs/source_references.json',{'sha256':refs})
 save(E/'configs/pre_decode_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in E.rglob('*')if p.is_file()}})
 print('ALL_CANDIDATE_DECODER_PROTOCOL_FROZEN',flush=True)
if __name__=='__main__':main()
