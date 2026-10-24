from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ltlf_adapter import compile_ltlf
P=Path('/home/robot/research/projects/wm_symbolic')
MONA=P/'.tools/mona/usr/bin/mona'
out=Path('/home/robot/Documents/Codex/2026-10-13/task-4/ltlf_probe');out.mkdir(exist_ok=True)
cases=[('F(a & X(F(b)))',[{'a':True,'b':True}],False,'Strict next excludes same observation'),
       ('F(a & X(F(b)))',[{'a':True,'b':False},{'a':False,'b':True}],True,'Strict order accepts actual next observation'),
       ('F(a & X(F(b)))',[{'a':False,'b':True},{'a':True,'b':False}],False,'Reversed order rejected'),
       ('F(a & F(b))',[{'a':True,'b':True}],True,'Non-strict F permits same observation'),
       ('G(!movedblue)',[{'movedblue':False},{'movedblue':False}],True,'No forbidden event'),
       ('G(!movedblue)',[{'movedblue':False},{'movedblue':True},{'movedblue':False}],False,'Returning to old state does not undo prior event')]
compiled={f:compile_ltlf(f,MONA) for f,_,_,_ in cases};results=[]
for f,trace,want,reason in cases:
    actual=compiled[f].accepts(trace);assert actual is want,(f,actual,want)
    results.append({'formula':f,'trace':trace,'expected':want,'actual':actual,'reason':reason})
try:compiled['G(!movedblue)'].accepts([{'movedblue':None}])
except ValueError:results.append({'test':'unknown_rejected','passed':True})
else:raise AssertionError('Unknown was silently converted')
for i,(f,dfa) in enumerate(compiled.items()):
    (out/f'dfa_{i}.dot').write_text(dfa.dot);(out/f'dfa_{i}.mona.txt').write_text(dfa.mona)
(out/'result.json').write_text(json.dumps({'tests':len(results),'all_passed':True,'scope':'Compiler semantics only, not learned transitions or real execution','cases':results},indent=2))
print(json.dumps({'tests':len(results),'all_passed':True,'directory':str(out)}))
