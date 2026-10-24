"""Use the official LTLf compiler and MONA; never encode action dynamics here."""
from dataclasses import dataclass
from pathlib import Path
import re,subprocess,tempfile,os
from ltlf2dfa.parser.ltlf import LTLfParser
from ltlf2dfa.base import MonaProgram
from ltlf2dfa.ltlf2dfa import output2dot

@dataclass(frozen=True)
class CompiledDFA:
    propositions: tuple
    initial: int
    accepting: frozenset
    transitions: tuple
    dot: str
    mona: str
    source: str

    def advance(self,state,labels):
        if any(type(labels.get(k)) is not bool for k in self.propositions):
            raise ValueError('Unknown proposition: stop or reobserve, never invent a Boolean')
        matches=[dst for src,guard,dst in self.transitions if src==state and all(
            bit=='X' or (bit=='1')==labels[k] for k,bit in zip(self.propositions,guard))]
        if len(matches)!=1:raise ValueError('DFA is not deterministic and total at this valuation')
        return matches[0]

    def accepts(self,trace):
        state=self.initial
        for labels in trace:state=self.advance(state,labels)
        # Finishing does not consume the last physical observation a second time.
        return state in self.accepting

def compile_ltlf(formula, executable):
    parsed=LTLfParser()(formula)
    mona_program=MonaProgram(parsed).mona_program()
    with tempfile.TemporaryDirectory(prefix='wm-ltlf-') as temp:
        path=Path(temp)/'input.mona';path.write_text(mona_program)
        child_env=dict(os.environ)
        private_lib=Path(executable).parents[1]/'lib/x86_64-linux-gnu'
        child_env['LD_LIBRARY_PATH']=str(private_lib)+(':'+child_env['LD_LIBRARY_PATH'] if child_env.get('LD_LIBRARY_PATH') else '')
        result=subprocess.run([str(executable),'-q','-u','-w',str(path)],capture_output=True,text=True,timeout=30,check=True,env=child_env)
    raw=result.stdout
    if 'DFA for formula with free variables:' not in raw:
        raise RuntimeError(f'MONA did not return a DFA: {raw[:300]} {result.stderr[:300]}')
    props=tuple(re.search(r'DFA for formula with free variables:\s*([^\n]*)',raw).group(1).lower().split())
    if len(props)!=len(set(props)):raise ValueError('Duplicate propositions')
    accepting=frozenset(map(int,re.search(r'Accepting states:\s*([^\n]*)',raw).group(1).split()))
    dot=output2dot(raw)
    initial=int(re.search(r'init -> (\d+)',dot).group(1))
    transitions=tuple((int(src),guard,int(dst)) for src,guard,dst in re.findall(r'State (\d+): ([01X]+) -> state (\d+)',raw) if int(src)!=0)
    if not transitions:raise ValueError('No compiled transitions')
    if any(len(guard)!=len(props) for _,guard,_ in transitions):raise ValueError('Guard dimension mismatch')
    return CompiledDFA(props,initial,accepting,transitions,dot,raw,formula)
