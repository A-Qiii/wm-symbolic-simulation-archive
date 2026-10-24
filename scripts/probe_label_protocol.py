from pathlib import Path
import json
from wm_symbolic.ltlf_adapter import compile_ltlf
P=Path('/home/robot/research/projects/wm_symbolic');old=json.loads((P/'data/data_manifest.json').read_text());new=json.loads((P/'data/data_manifest_process_v2.json').read_text());assert len(old['records'])==len(new['records'])
changes=0;example=None
for a,b in zip(old['records'],new['records']):
 assert all(a[k]==b[k] for k in ['root_id','episode_id','step','skill','obs_before','obs_after','success'])
 x=json.loads(Path(a['labels_path']).read_text());y=json.loads(Path(b['labels_path']).read_text());assert x['events']==y['events'][:3];assert x['success']==y['success']
 for where in ['before','after']:
  for j,(v,w) in enumerate(zip(x[where]['categorical'],y[where]['categorical'])):
   if v!=w:assert j<3 and v<3 and w<3;changes+=1
  assert x[where]['binary']==y[where]['binary']
 if x['before']['binary'][1]==x['after']['binary'][1]==0 and y['events'][4] and example is None:example={'record':a,'labels':y}
assert example is not None
old_dfa=compile_ltlf('G(!led_on)',P/'.tools/mona/usr/bin/mona');new_dfa=compile_ltlf('G(!led_on & !led_was_on)',P/'.tools/mona/usr/bin/mona')
assert old_dfa.accepts([{'led_on':False},{'led_on':False}])
assert not new_dfa.accepts([{'led_on':False,'led_was_on':False},{'led_on':False,'led_was_on':bool(example['labels']['events'][4])}])
result={'passed':True,'same_real_transitions':len(old['records']),'table_class_boundary_labels_corrected':changes,
 'all_controls_images_success_and_moved_events_unchanged':True,'device_process_counterexample':{k:example['record'][k] for k in ['episode_id','step','skill','raw_evaluator_trace']},
 'endpoint_only_automaton_accepts':True,'complete_process_automaton_rejects':True,'evidence':'Single event-source replacement on an actual transient LED change; not a claim of learned closed-loop recovery'}
(P/'runs/label_protocol_regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
