"""Frozen HULC macro execution. Fixed step budget, no evaluator early stopping."""
from pathlib import Path
import time,json,hashlib
import fcntl,os
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torchvision.transforms import Compose
from calvin_agent.datasets.utils.episode_utils import process_rgb,process_depth,process_state,load_dataset_statistics


class FrozenHulc:
    def __init__(self, train_folder, checkpoint, embeddings_path, device='cuda:0', reserve_gpu=True):
        project=Path(train_folder).parents[1]
        self._gpu_lock=None
        if reserve_gpu:
            self._gpu_lock=(project/'runs/gpu.lock').open('a+')
            fcntl.flock(self._gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self._gpu_lock.seek(0);self._gpu_lock.truncate()
            self._gpu_lock.write(str(os.getpid()));self._gpu_lock.flush()
        self.device=torch.device(device)
        self.cfg=OmegaConf.load(Path(train_folder)/'.hydra/config.yaml')
        # Released config predates official move of common transforms to CALVIN.
        aliases={f'hulc.utils.transforms.{n}':f'calvin_agent.utils.transforms.{n}'
                 for n in ['ScaleImageTensor','NormalizeVector','AddDepthNoise','AddGaussianNoise']}
        for mode in ['train','val']:
            for transforms in self.cfg.datamodule.transforms[mode].values():
                for transform in transforms:
                    if transform._target_ in aliases:transform._target_=aliases[transform._target_]
                    if transform._target_=='torchvision.transforms.Resize':
                        OmegaConf.set_struct(transform,False);transform.antialias=False
        # Public official checkpoint is explicitly trusted; do not relax torch globally.
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        from hulc.models.hulc import Hulc
        self.model=Hulc(**ckpt['hyper_parameters'])
        # Official a33eb20 made this dormant layer conditional; the released
        # checkpoint predates that commit. Restore its exact old module before
        # strict loading, while preserving positional_normalize=False.
        legacy_norm='plan_recognition.layernorm.weight' in ckpt['state_dict']
        if legacy_norm and not hasattr(self.model.plan_recognition,'layernorm'):
            assert self.model.plan_recognition.positional_normalize is False
            width=self.model.plan_recognition.in_features
            assert ckpt['state_dict']['plan_recognition.layernorm.weight'].shape==(width,)
            self.model.plan_recognition.layernorm=torch.nn.LayerNorm(width)
        self.model.load_state_dict(ckpt['state_dict'],strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.model.load_lang_embeddings(embeddings_path)
        cfg=self.cfg.datamodule
        self.obs_space=cfg.observation_space
        self.proprio=cfg.proprioception_dims
        # The released ABC config explicitly points at the debug evaluation data.
        # Match official evaluator's load_dataset_statistics for that public data.
        dataset_root=Path(embeddings_path).parents[2]
        cfg.transforms=load_dataset_statistics(dataset_root/'training',dataset_root/'validation',cfg.transforms)
        self.transforms={k:Compose([hydra.utils.instantiate(t) for t in ts])
                         for k,ts in cfg.transforms.val.items()
                         if k in set(self.obs_space.rgb_obs+self.obs_space.depth_obs+self.obs_space.state_obs)}
        self.identity={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                       'embeddings':str(embeddings_path),'embeddings_sha256':hashlib.sha256(Path(embeddings_path).read_bytes()).hexdigest(),
                       'frozen_parameters':sum(p.numel() for p in self.model.parameters()),
                       'normalization_source':[str(Path(train_folder)/'.hydra/config.yaml'),str(dataset_root/'validation/statistics.yaml')],
                       'compatibility_transform_aliases':aliases,'tensor_resize_antialias':False,
                       'legacy_dormant_layernorm_restored':legacy_norm,'strict_state_dict_load':True,
                       'normalization_rule':'Official CALVIN evaluation statistics merge; checkpoint explicitly references debug dataset.'}

    def reset(self):self.model.reset()

    def step(self, obs, command):
        allowed={'rgb_obs','depth_obs','robot_obs','segmentation','calibration'}
        if not set(obs)<=allowed or not {'rgb_obs','depth_obs','robot_obs'}<=set(obs):raise ValueError('Unexpected public input fields')
        state=process_state(obs,self.obs_space,self.transforms,self.proprio)
        rgb=process_rgb(obs['rgb_obs'],self.obs_space,self.transforms)
        depth=process_depth(obs['depth_obs'],self.obs_space,self.transforms)
        x={'robot_obs':state['robot_obs'].to(self.device).unsqueeze(0),
           'robot_obs_raw':torch.from_numpy(obs['robot_obs']).to(self.device),
           'rgb_obs':{k:v.to(self.device).unsqueeze(0) for k,v in rgb['rgb_obs'].items()},
           'depth_obs':{k:v.to(self.device).unsqueeze(0) for k,v in depth['depth_obs'].items()}}
        with torch.inference_mode():action=self.model.step(x,command).squeeze().cpu().numpy()
        if action.shape!=(7,) or not np.isfinite(action).all():raise RuntimeError('Invalid executor action')
        action[-1]=1 if action[-1]>0 else -1
        return action


def execute_fixed(adapter, policy, skill, command, budget, output, evaluator):
    import imageio.v2 as imageio
    from wm_symbolic.env_adapter import save_npz
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    policy.reset()  # HULC internal plan resets here; WM history does not.
    start=adapter.evaluator_snapshot();obs=adapter.observe();save_npz(output/'before.npz',obs)
    writers={k:imageio.get_writer(str(output/f'{k}.mp4'),fps=adapter.env.control_freq,codec='libx264',
                                macro_block_size=1,quality=7) for k in obs['rgb_obs']}
    controls=[];robot=[];scene=[];checkpoint_success={};t=time.monotonic();actual_steps=0;exception=None
    event_file=(output/'trajectory_evaluator_only.jsonl').open('w')
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,bytes):return x.decode(errors='replace')
        raise TypeError(type(x).__name__)
    try:
        event_file.write(json.dumps({'step':0,'info':start['info']},default=serial)+'\n')
        for i in range(budget):
            action=policy.step(obs,command)
            obs=adapter.step(action)
            snapshot=adapter.evaluator_snapshot()
            actual_steps=i+1
            controls.append(action.copy());robot.append(obs['robot_obs']);scene.append(snapshot['scene_obs'])
            event_file.write(json.dumps({'step':i+1,'info':snapshot['info']},default=serial)+'\n')
            for k,w in writers.items():w.append_data(obs['rgb_obs'][k])
            if i+1 in {120,240,360,budget}:
                checkpoint_success[str(i+1)]=skill in evaluator.get_task_info_for_set(start['info'],snapshot['info'],{skill})
    except Exception as e:
        exception=repr(e)
        raise
    finally:
        event_file.close()
        for w in writers.values():w.close()
        np.savez_compressed(output/'controls_and_evaluator_state.npz',actions=controls,robot_obs=robot,scene_obs=scene)
        save_npz(output/'after.npz',obs)
        for i,b in enumerate(adapter.history):save_npz(output/f'boundary_history_{i}.npz',b)
        summary={'skill':skill,'command':command,'requested_steps':budget,'actual_steps':actual_steps,
                 'wall_seconds':time.monotonic()-t,'simulated_seconds':actual_steps/adapter.env.control_freq,
                 'control_hz':adapter.env.control_freq,'score_at_fixed_budgets':checkpoint_success,
                 'exception':exception,'gt_used_for_stopping':False,'executor':policy.identity}
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
