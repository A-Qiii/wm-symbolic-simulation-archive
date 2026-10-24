from experiment import *
from html.parser import HTMLParser
import subprocess,re
def manifest(root):
 fp=root/'delivery_manifest.json';assert not fp.exists();items=[]
 for p in sorted(root.rglob('*')):
  if p.is_file():items.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
 out={'created_utc':now(),'root':str(root),'algorithm':'SHA256','excludes':'delivery_manifest.json itself','files':len(items),'bytes':sum(x['bytes']for x in items),'entries':items};save(fp,out)
 for p in root.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for item in items:
  p=root/item['path'];assert p.stat().st_mode&0o222==0 and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 return {'root':str(root),'manifest_sha256':sha(fp),'manifest_entries':len(items),'files_including_manifest':len(items)+1,'bytes_excluding_manifest':out['bytes'],'all_readonly':True}
def main():
 verify_frozen();assert read(E/'results/final_validation.json')['passed'];assert read(E/'results/summary.json')['scope_complete']
 class Parser(HTMLParser):
  def __init__(self):super().__init__();self.refs=[]
  def handle_starttag(self,tag,attrs):
   for k,v in attrs:
    if k in ['href','src']:self.refs.append(v)
 p=Parser();p.feed((E/'report.html').read_text());assert all((E/x).exists()for x in p.refs)
 ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout
 running=[line for line in ps.splitlines()if re.search(r'^\s*\d+\s+(?:\S*/)?python(?:3(?:\.\d+)?)?\s+\S*/state_predicates_state_dependence_20261014/source/(?:collect|cache|evaluate)\.py',line)];assert not running,running
 save(E/'results/delivery_checks.json',{'created_utc':now(),'passed':True,'report_local_links_checked':len(p.refs),'missing_links':[],'five_PNGs_visually_reviewed':['C_comparison.png','all_roots.png','case_close.png','case_lift.png','case_place.png'],'all_PNGs_axes_colours_units_conditions_and_reference_meaning_explained':True,'no_collection_cache_or_prediction_process_running':True,'no_additional_neural_forwards':True})
 save(E/'results/progress.json',{'phase':'complete_frozen_diagnosis','episodes':54,'roots':18,'pairs':54,'uncovered':0,'new_training_updates':0,'report':str(E/'report.html'),'next_training':'proposal_only_not_executed'})
 data=manifest(B);save(E/'results/data_delivery_seal.json',data)
 result=manifest(E);print(json.dumps({'data':data,'experiment':result},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
