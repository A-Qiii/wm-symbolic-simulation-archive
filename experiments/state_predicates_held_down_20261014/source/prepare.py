from pathlib import Path
import json,hashlib,datetime,shutil
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_held_down_20261014';B=P/'data/predicate_held_down_replay_20261014';C=P/'experiments/state_predicates_countermotion_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'configs/protocol.json').exists();q=read(C/'results/next_step_candidates.json')['candidates'];assert len(q)==6;roots={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')};sources=[];frozen={}
 for x in q:
  folder=Path(x['trajectory']);ep=folder.parent;root=roots[x['root_id']];assert root['environment']=='C'and root['split']=='dev';prefix=sorted(str(f)for f in ep.iterdir()if f.is_dir()and f.name[:2].isdigit()and int(f.name[:2])<=int(folder.name[:2]));r={'root':root,'episode_id':ep.name,'source_episode':str(ep),'source_target':str(folder),'prefix_segments':prefix,'target_color':x['colour'],'anchor_frame':x['anchor'],'history_frame_indices':x['indices'],'original_candidate':x,'split_status':'existing_C_development_used_in_prior_selection; not unseen or independentacceptance','new_independent_root':False};sources.append(r)
  paths=[ep/'setup_private.json',ep/'root_reset_private.npz']+[ep/f'root_observation_{j}.npz'for j in range(4)]
  for segment in prefix:
   d=Path(segment);paths.extend(d/name for name in ['controls.json','trace_private.json.gz','endpoint_info_private.json','labels_ownership_v2_private.json','initial_public.npz','final_public.npz']+[f'boundary_history_{j}.npz'for j in range(4)])
  for f in paths:frozen[str(f)]=sha(f)
 previous=read(C/'configs/protocol.json');models=previous['models']
 for m in models:assert sha(m['checkpoint'])==m['checkpoint_sha256'];frozen[m['checkpoint']]=m['checkpoint_sha256']
 for f in [C/'results/next_step_candidates.json',C/'delivery_manifest.json',P/'src/wm_symbolic/env_adapter.py',P/'src/wm_symbolic/physical_oracle.py',P/'src/wm_symbolic/vision_cache.py',P/'src/wm_symbolic/core_ownership.py',P/'scripts/core1200_collect.py',P/'experiments/state_predicates_20261014/data/normalization.json']:frozen[str(f)]=sha(f)
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'OnlysixpreviouslylistedexistingChelddownwindows. Replayoriginalepisodeinitialization/setup/prefix/targetactions; noadditionalroot/control/training/thresholdchanges.','sources':sources,'models':models,'history':'Exact[t-63,t-42,t-21,t]at30Hz,2.10s,samefrozenDINOreader.','replay':'NativeCseedandoriginalblockinitialposes. Same24settlecontrols; originalrecordedsetupactions and alloriginalprecedingsegmentactions throughfulltargetsegment. NeverresettointermediatesavedGTstate. Savefullpublicframesoftargetsegment.','tolerances':previous['tolerances'],'full_state_checks':'Rootfullrobot/scenecheckpoint; eachprefix/targetrobot,scene/objectstates; physicalheld/bilateral/supportevents; originalrootandsegmentinitial/boundaryRGB/masks. Exactmeansnumeric<=1e-9andbitwiseavailablepixels. Outsidefrozenlimitsstop; withintolerancebutnonexactrecordnewreplay.','target_label':'Unchangedphysicalheldatanchor; continuouslyheld64frameconditionverifiedoffline; nofutureinput.','missing_visibility':'Retainandreport; no substituteanchors or roots. Additionalin/open/otherheldscoresonlyforvalidandall4visibleentities.','evaluation':'Threealreadyfrozenlong_historyupdate600models,originalthresholds,onebatchpermodel. No checkpointchoice or thresholdadjustment.','interpretation':'ExistingCdevelopment,notindependenttest. Combinewithprior0/.1/2.1sdiagnosticandemptyupwardchallenge; preservehistoricalerrors. Stopafterthese6.','frozen_sha256':frozen,'D_or_prior_test_sources':[],'new_roots':0}
 save(E/'configs/protocol.json',protocol);(E/'configs/protocol.json').chmod(0o444);save(B/'sources_manifest.json',sources);(B/'sources_manifest.json').chmod(0o444);save(B/'data_manifest.json',{'version':'existing_C_held_down_v1','records':[],'episodes':[]})
 for f in ['model.py','metrics.py','common.py']:shutil.copyfile(C/'source'/f,E/'source'/f)
 print('HELD_DOWN_PROTOCOL_FROZEN',sha(E/'configs/protocol.json'))
if __name__=='__main__':main()
