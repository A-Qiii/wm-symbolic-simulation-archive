from pathlib import Path
import json,fcntl,time
import collector_core as c
P=c.P;E=c.R;B=c.B
def main():
    protocol=json.loads((E/'configs/protocol.json').read_text());plan=json.loads((E/'configs/acceptance_roots.json').read_text());freeze=json.loads((E/'results/candidate_freeze.json').read_text())
    assert c.sha(B/'roots_manifest.json')==plan['root_manifest_sha256']
    for p,h in protocol['frozen_sha256'].items():assert c.sha(p)==h,p
    for p,h in plan['source_sha256'].items():assert c.sha(p)==h,p
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    roots=json.loads((B/'roots_manifest.json').read_text());manifest=json.loads((B/'data_manifest.json').read_text());done={r['trajectory_id']for r in manifest['trajectories']}
    for root in roots:
        for branch in root['trajectories']:
            if root['root_id']+'__'+branch in done:continue
            records,s=c.collect(root,branch);manifest['records']+=records;manifest['trajectories'].append(s);c.save(B/'data_manifest.json',manifest)
            c.save(E/'results/collection_progress.json',{'last_trajectory':s['trajectory_id'],'completed':len(manifest['trajectories']),'planned':30,'windows':len(manifest['records'])})
            print(json.dumps({'trajectory':s['trajectory_id'],'result':s['controller_result']['status'],'windows':len(records),'issues':s['label_issues'],'gaps':s['gaps']},ensure_ascii=False),flush=True)
            if s['label_issues']:raise RuntimeError('Independent label audit conflict; preserve evidence and stop.')
    c.save(E/'results/collection_complete.json',{'trajectories':len(manifest['trajectories']),'windows':len(manifest['records']),'root_count':len(roots),'complete':len(manifest['trajectories'])==30,'no_resampling':True})
    (B/'data_manifest.json').chmod(0o444)
if __name__=='__main__':main()
