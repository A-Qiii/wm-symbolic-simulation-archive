"""Atomic training state, reproducible RNG state, and cooperative pause requests."""
from pathlib import Path
import math, os, random, signal, tempfile
import numpy as np
import torch

def learning_rate(update, total):
    if update <= 1000:
        return 3e-4 * update / 1000
    progress = (update - 1000) / max(total - 1000, 1)
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))

def capture_rng(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'generator': generator.bit_generator.state, 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    generator.bit_generator.state = state['generator']
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

def atomic_save(value, destination):
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name, dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

class PauseRequest:
    """Handlers only set a flag; no GPU, I/O, or serialization inside a handler."""
    def __init__(self):
        self.requested = False
        self.reason = None
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(signum, self.receive)

    def receive(self, signum, frame):
        self.requested = True
        self.reason = signal.Signals(signum).name
