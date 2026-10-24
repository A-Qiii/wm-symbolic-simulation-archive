"""Current-task dependency queue. One lock protects the single local GPU.

Runs once, exits on first failure, and never schedules recurring/background work.
"""
from pathlib import Path
import subprocess,sys,time,fcntl,json,os
P=Path('/home/robot/research/projects/wm_symbolic')
log=P/'logs/stage0_execution.log'
with (P/'runs/stage0_gpu.lock').open('w') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another stage0 worker already owns the GPU queue')
    lock.write(str(os.getpid()));lock.flush()
    start=time.monotonic()
    while True:
        ready=subprocess.run([sys.executable,'-c','import hydra, pybullet, numpy, quaternion, cv2, rich, pytorch_lightning, imageio_ffmpeg'],capture_output=True,text=True)
        if ready.returncode==0:break
        if time.monotonic()-start>1800:
            (P/'runs/stage0_dependency_block.json').write_text(json.dumps({'status':'dependency_import_timeout','last_error':ready.stderr},indent=2))
            raise SystemExit('Dependency import did not become ready within 30 minutes')
        print('Waiting for the existing dependency installation; no duplicate installation or GPU process started.',flush=True)
        time.sleep(20)
    commands=[
        [sys.executable,str(P/'scripts/probe_environment.py'),'--environment','C'],
        [sys.executable,str(P/'scripts/probe_skills.py'),
         '--train-folder',str(P/'models/HULC_ABC_D'),
         '--checkpoint',str(P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt'),
         '--embeddings',str(P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')],
    ]
    with log.open('a') as out:
        for command in commands:
            previous=P/'runs/stage0_environment_C/result.json'
            if command[1].endswith('probe_environment.py') and previous.exists() and json.loads(previous.read_text())['status']=='passed':
                print('Reusing already verified C camera/reset result.',flush=True)
                continue
            print('Starting',command[1],flush=True)
            out.write('\nCOMMAND '+json.dumps(command)+'\n');out.flush()
            result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode:
                (P/'runs/stage0_execution_block.json').write_text(json.dumps({'status':'failed','command':command,'exit_code':result.returncode,'log':str(log)},indent=2))
                raise SystemExit(result.returncode)
            print('Completed',command[1],flush=True)
    subprocess.run([sys.executable,str(P/'scripts/render_status.py')],check=True)
