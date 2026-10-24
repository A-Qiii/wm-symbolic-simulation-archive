"""Cross-process single-GPU phases for the separate executor/model environments."""
from pathlib import Path
from contextlib import contextmanager
import fcntl,os
@contextmanager
def gpu_phase(project,phase):
 handle=(Path(project)/'runs/gpu.lock').open('a+')
 try:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handle.seek(0);handle.truncate();handle.write(f'{os.getpid()} {phase}');handle.flush();yield
 finally:handle.close()
