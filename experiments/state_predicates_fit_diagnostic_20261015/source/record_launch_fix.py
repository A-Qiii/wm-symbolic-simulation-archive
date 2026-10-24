from pathlib import Path
import ast,json,hashlib,datetime
E=Path('/home/robot/research/projects/wm_symbolic/experiments/state_predicates_fit_diagnostic_20261015')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not list((E/'runs').iterdir()) and not list((E/'models').iterdir())
assert not(E/'results/training_progress.json').exists()
src=E/'source/train_diagnostic.py';dst=E/'source/train_diagnostic_v2.py'
assert not dst.exists()
text=src.read_text();assert text.count('else500')==1 and text.count('else137')==1
text=text.replace('else500','else 500').replace('else137','else 137')
text=text.replace("if __name__=='__main__':main()", "if __name__=='__main__':\n amendment=read(E/'configs/launch_fix.json');assert sha(__file__)==amendment['corrected_source_sha256'];main()")
ast.parse(text);dst.write_text(text);dst.chmod(0o444)
note={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Initial diagnostic launcher had missing whitespace in two conditional expressions: else500, else137. Python parser stopped before any imports or optimizer creation.','original_source':str(src),'original_source_sha256':sha(src),'corrected_source':str(dst),'corrected_source_sha256':sha(dst),'frozen_protocol_sha256_unchanged':sha(E/'configs/protocol.json'),'changes':['Insert whitespace in else 500 and else 137 metadata expressions','Verify this sidecar source hash before main'],'optimizer_updates_before_fix':0,'original_failed_log_retained':'logs/training.log','successful_attempt_log':'logs/training_after_syntax_fix.log','data_model_loss_budget_stop_rules_unchanged':True}
(E/'configs/launch_fix.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n');(E/'configs/launch_fix.json').chmod(0o444)
print(json.dumps(note,ensure_ascii=False))
