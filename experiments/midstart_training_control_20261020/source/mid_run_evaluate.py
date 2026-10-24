from pathlib import Path
import json,shutil,subprocess,os
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/midstart_training_control_20261020';F=S/'fresh_validation'
read=lambda p:json.loads(p.read_text())
def run(env,script,log,args=''):
 cmd=f'source {P}/{env}\npython -u {script} {args}'
 with (S/'logs'/log).open('w')as out:subprocess.run(['bash','-c',cmd],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=S)
assert(S/'configs/models_frozen.json').exists()
run('world_model_env.sh',S/'source/mid_assess.py','evaluation_console.log')
run('world_model_env.sh',S/'source/mid_score.py','score_console.log')
trigger=read(S/'results/independent_validation_trigger.json');print('TRIGGER',json.dumps(trigger),flush=True)
if trigger['passed']:
 shutil.copyfile(S/'configs/models_frozen.json',F/'configs/models_frozen.json')
 run('env.sh',F/'source/geometry.py','fresh_geometry_console.log')
 run('env.sh',F/'source/collect.py','fresh_collection_console.log')
 run('world_model_env.sh',S/'source/mid_fresh_qa.py','fresh_qa_console.log')
 run('world_model_env.sh',F/'source/cache_runner.py','fresh_features_console.log')
 run('world_model_env.sh',S/'source/mid_assess.py','fresh_evaluation_console.log','--fresh')
 run('world_model_env.sh',S/'source/mid_score.py','score_with_fresh_console.log')
run('world_model_env.sh',S/'source/mid_report.py','report_console.log')
print('EVALUATION_REPORT_COMPLETE',flush=True)
