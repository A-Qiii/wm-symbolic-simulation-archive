"""Reuse the existing, read-only finite-automaton monitor with compiled LTLf guards."""
from pathlib import Path
from itertools import product
import hashlib,types
SOURCE=Path('/home/robot/research/projects/robot_state/src/task_state/planning/automaton.py')
def reuse_compiled_automaton(data):
 module=types.ModuleType('_wm_readonly_legacy_automaton');exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),module.__dict__)  # No bytecode writes in the old project.
 props=tuple(data['propositions']);states=sorted({x for a,g,b in data['transitions'] for x in [a,b]}|{data['initial']});transitions={}
 for state in states:
  for values in product((False,True),repeat=len(props)):
   matches=[dst for src,guard,dst in data['transitions'] if src==state and all(bit=='X' or (bit=='1')==v for bit,v in zip(guard,values))]
   if len(matches)!=1:raise ValueError('Compiled LTLf guard table is not total/deterministic')
   for done in [False,True]:transitions[state,values+(done,)]=matches[0]
 return module.FiniteAutomaton(props+('done',),states,data['initial'],set(data['accepting'])&set(states),transitions,version='mona_compiled_readonly_existing_monitor')
def provenance():return {'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'source_modified':False,'reused':'FiniteAutomaton.advance and finish; no old model, data, thresholds or transition effects'}
