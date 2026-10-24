from experiment import *
import os
def seal(folder):
 paths=sorted(p for p in folder.rglob('*')if p.is_file()and p.name!='delivery_manifest.json');assert not(folder/'delivery_manifest.json').exists();entries=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in paths];save(folder/'delivery_manifest.json',{'created_utc':now(),'root':str(folder),'scope':'one frozen independent D single-step validation; no D optimization/reselection/retest','files':entries,'entry_count':len(entries),'total_bytes':sum(r['bytes']for r in entries)})
 for p in paths+[folder/'delivery_manifest.json']:p.chmod(0o444)
 assert all(p.stat().st_mode&0o222==0 for p in paths+[folder/'delivery_manifest.json']);return {'root':str(folder),'manifest':str(folder/'delivery_manifest.json'),'manifest_sha256':sha(folder/'delivery_manifest.json'),'read_only_files':len(paths)+1,'content_bytes_excluding_manifest':sum(r['bytes']for r in entries)}
def main():
 q=frozen();assert read(E/'results/final_validation.json')['passed'];assert read(E/'results/visual_inspection.json')['completed'];assert(E/'REPORT.md').exists()and(E/'HANDOFF.md').exists();assert read(E/'results/decision.json')['representative']=='candidate_seed0';receipts=[seal(E),seal(B)];output={'created_utc':now(),'sealed':True,'candidate_sha256':q['candidate_hash'],'protocol_sha256':sha(E/'configs/protocol.json'),'receipts':receipts,'decision':read(E/'results/decision.json')};save(Path('/home/robot/Documents/Codex/2026-10-13/task-4/independent_D_delivery_receipt.json'),output);print(json.dumps(output,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
