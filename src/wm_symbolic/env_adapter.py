"""Separate public observations from evaluator-only simulator state.

Native scene configuration is composed for each environment. Never map a D
scene vector directly onto A/B/C, and never pass reset metadata to a model.
"""
from pathlib import Path
from collections import deque
from copy import deepcopy
import numpy as np
import hydra
import pybullet as bullet
from types import MethodType
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class CalvinAdapter:
    def __init__(self, project: Path, environment: str, seed: int, use_egl=True, tabletop=False, segmentation=False, layout=None):
        if environment not in 'ABCD' or len(environment) != 1:
            raise ValueError(environment)
        self.project, self.environment = Path(project), environment
        conf = self.project/'vendor/calvin/calvin_env/conf'
        with initialize_config_dir(config_dir=str(conf), job_name='wm_symbolic'):
            self.config = compose(config_name='config_data_collection', overrides=[
                f'scene=calvin_scene_{environment}', 'cameras=static_and_gripper',
                'use_vr=false', f'env.seed={seed}', f'env.use_egl={str(use_egl).lower()}'])
        if tabletop:
            for obj in self.config.scene.objects.movable_objects.values():
                obj.initial_pos='table'
        if layout:
            for name,surface in layout.items():
                self.config.scene.objects.movable_objects[name].initial_pos=surface
        self.env = hydra.utils.instantiate(self.config.env, use_scene_info=True)
        self.with_segmentation=segmentation
        self._original_camera_call=None
        if segmentation:
            self._original_camera_call=bullet.getCameraImage
            original=self._original_camera_call
            def render_with_visible_masks(*args,**kwargs):
                kwargs['flags']=kwargs.get('flags',0)|bullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
                return original(*args,**kwargs)
            bullet.getCameraImage=render_with_visible_masks
            for camera in self.env.cameras:
                previous=camera.process_rgbd
                def remember_mask(cam, pixels, near, far, previous=previous):
                    cam.visible_segmentation=np.asarray(pixels[4],dtype=np.int32).reshape(cam.height,cam.width).copy()
                    return previous(pixels,near,far)
                camera.process_rgbd=MethodType(remember_mask,camera)
        self.history = deque(maxlen=4)

    @staticmethod
    def model_view(raw):
        # Explicit allowlist, never a blacklist: no scene_obs, contacts or success.
        return {'rgb_obs': {k: v.copy() for k,v in raw['rgb_obs'].items()},
                'depth_obs': {k: v.copy() for k,v in raw['depth_obs'].items()},
                'robot_obs': raw['robot_obs'].copy()}

    def observe(self):
        public = self.model_view(self.env.get_obs())
        return self._add_visible_geometry(public)

    def _add_visible_geometry(self,public):
        if self.with_segmentation:
            public['segmentation']={cam.name:cam.visible_segmentation.copy() for cam in self.env.cameras}
            public['calibration']=self.calibration()
        return public

    def step(self, action):
        raw, _, _, info = self.env.step(action)
        public = self._add_visible_geometry(self.model_view(raw))
        self.history.append(public)
        # Collector calls evaluator_snapshot separately, not returned to policy.
        return public

    def reset(self, root=None):
        if root is not None and root['environment'] != self.environment:
            raise ValueError('Cross-environment raw reset is forbidden')
        raw=self.env.reset(**({k:root[k] for k in ['robot_obs','scene_obs']} if root else {}))
        public=self._add_visible_geometry(self.model_view(raw))
        self.history.clear(); self.history.append(public)
        return public

    def evaluator_snapshot(self):
        state=self.env.get_state_obs()
        return {'environment':self.environment, **deepcopy(state),
                'info':deepcopy(self.env.get_info()), 'serialization':deepcopy(self.env.serialize())}

    def calibration(self):
        result={}
        for cam in self.env.cameras:
            result[cam.name]={k:deepcopy(getattr(cam,k)) for k in
                ['width','height','fov','nearval','farval','viewMatrix','projectionMatrix'] if hasattr(cam,k)}
            for old,new in [('view_matrix','viewMatrix'),('projection_matrix','projectionMatrix')]:
                if hasattr(cam,old):result[cam.name][new]=deepcopy(getattr(cam,old))
        return result

    def close(self):
        if self._original_camera_call is not None:
            bullet.getCameraImage=self._original_camera_call
            self._original_camera_call=None
        if self.env.ownsPhysicsClient:
            self.env.close()
            self.env.ownsPhysicsClient=False


def save_npz(path, obs):
    fields={**obs['rgb_obs'],**obs['depth_obs'],'robot_obs':obs['robot_obs']}
    for key,value in obs.get('segmentation',{}).items():fields['seg_'+key]=value
    for camera,values in obs.get('calibration',{}).items():
        for key,value in values.items():fields['calib_'+camera+'_'+key]=np.asarray(value)
    np.savez_compressed(path,**fields)
