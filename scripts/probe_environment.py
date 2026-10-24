"""Real native CALVIN reset + RGB-D probe, no HULC success claims."""
import argparse, json, time, sys, traceback, platform, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter, save_npz

P=Path('/home/robot/research/projects/wm_symbolic')

def serial(x):
    if isinstance(x,bytes):return x.decode('utf-8',errors='replace')
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--environment',default='C');args=parser.parse_args()
    out=P/'runs'/f'stage0_environment_{args.environment}'
    out.mkdir(exist_ok=False)
    result={'environment':args.environment,'status':'started','python':sys.version,'platform':platform.platform()}
    env=None
    try:
        t=time.monotonic();env=CalvinAdapter(P,args.environment,0)
        obs=env.reset()
        # Short, declared passive stabilization protocol; no hidden state answers.
        for _ in range(12):obs=env.step(np.array([0,0,0,0,0,0,1.],dtype=np.float32))
        root=env.evaluator_snapshot()
        save_npz(out/'observed.npz',obs)
        np.savez_compressed(out/'reset_state_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs'])
        (out/'root_evaluator_only.json').write_text(json.dumps(root,default=serial,indent=2))
        OmegaConf.save(env.config,out/'resolved_native_config.yaml')
        (out/'calibration.json').write_text(json.dumps(env.calibration(),default=serial,indent=2))
        for key,rgb in obs['rgb_obs'].items():Image.fromarray(rgb).save(out/f'{key}.png')
        reset1=env.reset(root)
        for _ in range(8):env.step(np.array([0,0,0,0,0,0,1.],dtype=np.float32))
        reset2=env.reset(root)
        errors={k:float(np.max(np.abs(reset1['rgb_obs'][k].astype(float)-reset2['rgb_obs'][k].astype(float)))) for k in reset1['rgb_obs']}
        errors['robot_obs']=float(np.max(np.abs(reset1['robot_obs']-reset2['robot_obs'])))
        result.update(status='passed' if max(errors.values())<1e-5 else 'reset_difference',
                      camera_shapes={k:list(v.shape) for k,v in obs['rgb_obs'].items()},
                      depth_shapes={k:list(v.shape) for k,v in obs['depth_obs'].items()},
                      depth_finite={k:bool(np.isfinite(v).all()) for k,v in obs['depth_obs'].items()},
                      depth_range_m={k:[float(v.min()),float(v.max())] for k,v in obs['depth_obs'].items()},
                      public_observation_keys=list(obs), reset_max_abs_errors=errors,
                      declared_control_hz=env.env.control_freq,sim_steps_per_control=env.env.action_repeat,
                      wall_seconds=time.monotonic()-t)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if env:env.close()
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
