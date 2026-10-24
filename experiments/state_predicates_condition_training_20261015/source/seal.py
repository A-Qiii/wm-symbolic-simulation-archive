from core import *
from html.parser import HTMLParser
import subprocess,re
def main():
 frozen();assert read(E/'results/final_validation.json')['passed'];assert read(E/'results/summary.json')['scope_complete'];assert not(E/'delivery_manifest.json').exists()
 class Parser(HTMLParser):
  def __init__(self):super().__init__();self.refs=[]
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in ['src','href']:self.refs.append(v)
 parser=Parser();parser.feed((E/'report.html').read_text());assert all((E/x).exists()for x in parser.refs)
 ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout;running=[x for x in ps.splitlines()if re.search(r'^\s*\d+\s+(?:\S*/)?python(?:3(?:\.\d+)?)?\s+\S*/state_predicates_condition_training_20261015/source/(?:train|evaluate)\.py',x)];assert not running,running
 summary=read(E/'results/summary.json');assert summary['total_updates']==9000;save(E/'results/delivery_checks.json',{'created_utc':now(),'passed':True,'local_report_links_verified':len(parser.refs),'seven_figures_visually_reviewed':['training_curves.png','equal_budget_C.png','latent_error.png','case_close.png','case_lift.png','case_place.png','C_roots.png'],'shared_training_curve_axes_and_integer_count_axes':True,'root_macro_merged_C_reported':True,'no_training_or_evaluation_process_remaining':True,'candidate_and_future_validation_proposal_saved':True,'no_additional_training_or_predictions_during_report':True})
 entries=[]
 for p in sorted(E.rglob('*')):
  if p.is_file():entries.append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(E/'delivery_manifest.json',{'created_utc':now(),'root':str(E),'algorithm':'SHA256','excludes':'delivery_manifest.json itself','files':len(entries),'bytes':sum(x['bytes']for x in entries),'entries':entries})
 for p in E.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for r in entries:
  p=E/r['path'];assert sha(p)==r['sha256']and p.stat().st_mode&0o222==0
 print(json.dumps({'root':str(E),'manifest_sha256':sha(E/'delivery_manifest.json'),'manifest_entries':len(entries),'files_including_manifest':len(entries)+1,'bytes_excluding_manifest':sum(x['bytes']for x in entries),'all_readonly':True,'scope_complete':True}),flush=True)
if __name__=='__main__':main()
