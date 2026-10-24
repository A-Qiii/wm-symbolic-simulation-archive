import torch, json, time, platform, sys
from pathlib import Path
P=Path('/home/robot/research/projects/wm_symbolic')
result={'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'cuda_available':torch.cuda.is_available(),'platform':platform.platform()}
if not result['cuda_available']:raise RuntimeError(result)
result.update(device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),arch_list=torch.cuda.get_arch_list())
torch.manual_seed(0)
t=time.monotonic()
a=torch.randn(1024,1024,device='cuda',requires_grad=True)
b=(a@a.T).square().mean();b.backward();torch.cuda.synchronize()
result.update(matmul_backward_finite=bool(torch.isfinite(a.grad).all()),
              wall_seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated())
assert result['matmul_backward_finite']
(P/'runs/gpu_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
