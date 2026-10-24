from common import *
import collections
def main():
 groups=collections.defaultdict(list)
 for r in read(E/'results/close_calls.json'):
  vector=tuple(any(o['object']==c and o['before_above_rim_mm']>2 for o in r['object_geometry'])for c in COLORS)
  groups[(r['split'],tuple(r['before_exact9']),vector)].append(r)
 conflicts=[]
 for (split,y,vector),rows in groups.items():
  if len({r['closed']for r in rows})>1:conflicts.append({'split':split,'exact9':y,'AboveRim_per_named_object':dict(zip(COLORS,vector)),'outcomes':dict(collections.Counter('Closed'if r['closed']else'Mid'for r in rows)),'episodes':[r['episode_id']for r in rows]})
 result={'not_merely_any_object_aggregation':True,'three_named_AboveRim_values_preserved':True,'remaining_conflicting_groups':conflicts,'no_new_fit_or_physics':True};save(E/'results/named_relation_audit.json',result);save(R/'tables/named_relation_audit.json',result)
 assert len(conflicts)==2
 f=R/'REPORT.md';s=f.read_text();marker='这个结论也按三个命名物块的AboveRim'
 if marker not in s:s=s.replace('因而一个“突出”布尔量可以解释风险，却不够作确定后继。','因而一个“突出”布尔量可以解释风险，却不够作确定后继。这个结论也按三个命名物块的AboveRim取值逐项分组核实，仍是上述两组冲突；并非仅因把三个对象压成“任一突出”而造成。见tables/named_relation_audit.json。');f.write_text(s)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
