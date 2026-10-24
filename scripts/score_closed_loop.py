"""Independent offline evaluator. This module is never imported by the planner."""
from pathlib import Path
import argparse,json,yaml
from wm_symbolic.calibrated_labels import calibrated_labels_at
from wm_symbolic.state_labels import annotate_trace,COLORS,LOCATIONS
from wm_symbolic.process_events import add_device_events,EVENT_NAMES
from wm_symbolic.legacy_dfa import reuse_compiled_automaton

def facts_from(label,events):
    facts=dict(zip(EVENT_NAMES,map(bool,events)))
    for j,color in enumerate(COLORS):
        for k,location in enumerate(LOCATIONS):facts[f'{color}_{location}']=bool(label['categorical'][j]==k) if label['categorical_valid'][j] else None
    for j,names in [(3,['drawer_closed','drawer_middle','drawer_open']),(4,['slider_right','slider_middle','slider_left'])]:
        facts.update({n:bool(label['categorical'][j]==k) if label['categorical_valid'][j] else None for k,n in enumerate(names)})
    for j,name in enumerate(['light','led']):
        valid=label.get('binary_valid',[True,True])[j];facts[name+'_on']=bool(label['binary'][j]) if valid else None;facts[name+'_off']=(not facts[name+'_on']) if valid else None
    return facts

def score_run(project,run,root):
    p=Path(project);run=Path(run);private=run/'execution/evaluator_only';result=json.loads((run/'runtime.json').read_text());task=result['task']
    identity=json.loads((Path(root)/'identity.json').read_text());cfg=yaml.safe_load((p/'vendor/calvin/calvin_env/conf/scene'/f"calvin_scene_{identity['environment']}.yaml").read_text())
    initial=json.loads((private/'root_after_observation.json').read_text())['info'];all_infos=[initial]
    automaton=reuse_compiled_automaton(task['dfa']);state=int(task['dfa']['initial']);props=task['dfa']['propositions'];trace=[];valid=True
    def advance(info,events,source):
        nonlocal state,valid
        label=calibrated_labels_at(info,cfg);facts=facts_from(label,events)
        if any(facts.get(n) is None for n in props):valid=False
        else:state=automaton.advance((state,False,False,False,()),dict(facts,done=False))[0]
        trace.append({'source':source,'facts':facts,'dfa_state':state,'scoring_labels_valid':valid});return label,facts
    label,facts=advance(initial,[0,0,0,not initial['scene_info']['lights']['led']['logical_state'],initial['scene_info']['lights']['led']['logical_state'],not initial['scene_info']['lights']['lightbulb']['logical_state'],initial['scene_info']['lights']['lightbulb']['logical_state']],'root')
    macros=0;executor_successes=0
    for request in sorted((run/'execution/requests').glob('*.json')):
        command=json.loads(request.read_text());number=request.stem
        if command['command']=='shutdown':continue
        if command['command']=='skill':
            folder=private/f"{number}_{command['skill']}";infos=[json.loads(line)['info'] for line in (folder/'trajectory_evaluator_only.jsonl').open()]
            summary=json.loads((folder/'summary.json').read_text());success=summary['score_at_fixed_budgets'][str(summary['requested_steps'])];executor_successes+=bool(success);macros+=1
        else:
            infos=[all_infos[-1]]+[x['info'] for x in json.loads((private/f'{number}_reobserve.json').read_text())];success=False
        events=add_device_events(annotate_trace(infos,cfg,success),infos)['events'];all_infos.extend(infos[1:])
        # Reobservation has real physical cost, but does not invent an extra macro step for X.
        if command['command']=='reobserve':
            previous=trace.pop();state=trace[-1]['dfa_state'] if trace else int(task['dfa']['initial'])
            events=[bool(a or b) for a,b in zip(events,[previous['facts'].get(n,False) for n in EVENT_NAMES])]
        label,facts=advance(infos[-1],events,number+'_'+command['command'])
    goal_truth=[None if facts.get(n) is None else facts[n]==wanted for n,wanted in task['terminal_goals'].items()]
    terminal_valid=all(x is not None for x in goal_truth)
    terminal=False if False in goal_truth else None if None in goal_truth else True
    accepted=(state in task['dfa']['accepting']) if valid else None
    success=False if terminal is False or accepted is False else None if terminal is None or accepted is None else True
    whole=add_device_events(annotate_trace(all_infos,cfg,success),all_infos)
    scored={'task_id':task['task_id'],'actual_success':success,'terminal_conjunction':terminal,'ltlf_accepted':accepted,
            'scoring_labels_valid':bool(valid and terminal_valid),'model_claimed_completion':result['model_claimed_completion'],
            'false_completion_confirmation':bool(result['model_claimed_completion'] and success is False),'completion_unverifiable_due_to_scoring_mask':bool(result['model_claimed_completion'] and success is None),'verified_success':success is True,'physical_skills':macros,
            'executor_successes':executor_successes,'whole_trajectory_events':dict(zip(EVENT_NAMES,whole['events'])),
            'first_event_control_step':whole['event_first_control_step'],'ground_truth_used_only_for_offline_scoring':True,'scoring_unknown_policy':'Unknown goal/process truth is retained as null; it is not verified success and is not silently called a proven false confirmation','boundary_trace':trace}
    (run/'independent_score.json').write_text(json.dumps(scored,indent=2));return scored

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--root',required=True);args=a.parse_args()
    print(json.dumps(score_run('/home/robot/research/projects/wm_symbolic',args.run,args.root),indent=2))
