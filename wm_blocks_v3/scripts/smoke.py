"""Light syntax/import/config checks. No collection, training or simulator run."""
import argparse,ast,hashlib,importlib,json,os,subprocess,sys,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
def configurations():
 q=json.loads((ROOT/'configs/experiment_plan.json').read_text());all_seeds=set()
 for name,x in q['experiments'].items():
  counts={s:sum(g['split']==s for g in x['groups'])for s in ['train','development','test']};assert counts=={'train':60,'development':20,'test':20}
  all_seeds.update(g['layout_seed']for g in x['groups']);lookup={e['id']:e for e in x['episodes']}
  for arm,splits in x['arms'].items():
   assert [len(splits[s])for s in ['train','development','test']]==[360,120,40]
   groups={s:{lookup[e]['group']for e in splits[s]}for s in splits}
   assert not(groups['train']&groups['test']or groups['development']&groups['test']or groups['train']&groups['development'])
   assert sum(len(lookup[e]['actions'])for e in splits['train'])==960
 for version in ['v2','v3']:
  p=json.loads((ROOT/f'configs/{version}_plan.json').read_text());seeds={g['layout_seed']for g in p['confirmation_groups']};assert len(seeds)==20 and not seeds&all_seeds;all_seeds|=seeds;assert p['seeds']==[0,1,2]and p['updates']==3000
 for name,expected,mass in [('large',[.1,.05,.05],.2),('small',[.02,.02,.02],.025)]:
  tree=ET.parse(ROOT/f'assets/{name}.urdf');size=[float(x)*.8 for x in tree.find('.//collision/geometry/box').attrib['size'].split()];assert all(abs(x-y)<1e-9 for x,y in zip(size,expected));assert float(tree.find('.//mass').attrib['value'])==mass
 return {'split_and_seed_isolation':True,'training_transitions_each_arm':960,'assets_world_sizes_verified':True}
def syntax():
 files=list((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
 for p in files:compile(p.read_text(),str(p),'exec')
 for p in(ROOT/'configs').glob('*.json'):json.loads(p.read_text())
 return {'python_files_compiled_in_memory':len(files),**configurations()}
def physics():
 assert sys.version_info[:2]==(3,10),'Use the physics Python3.10 environment'
 for m in ['block_env','kinematic_guard','physics_monitor','engineering_gate','audit_engineering','protocol','collect','audit_data','v2_development_challenge','confirm_v3_data','live_worker']:importlib.import_module(m)
 for m in ['calvin_env.envs.play_table_env','calvin_env.robot.robot','calvin_env.scene.play_table_scene','calvin_env.camera.static_camera','calvin_env.camera.gripper_camera']:importlib.import_module(m)
 import pybullet as p
 assert not p.isConnected(),'Smoke check must not start simulation'
 import hydra
 from hydra import initialize_config_dir,compose
 from paths import CALVIN_ENV
 with initialize_config_dir(config_dir=str(CALVIN_ENV/'conf'),job_name='repro_smoke'):
  cfg=compose(config_name='config_data_collection',overrides=['scene=calvin_scene_A','robot=panda','cameras=static_and_gripper','use_vr=false','env.use_egl=true','env.bullet_time_step=960'])
 assert cfg.robot.filename=='franka_panda/panda.urdf'and cfg.robot.use_ik_fast is False
 assert cfg.env.bullet_time_step==960
 assert set(cfg.cameras)=={'static','gripper'}
 import numpy as np
 from types import SimpleNamespace
 from block_env import Scene
 from calvin_adapter import CalvinAdapter
 raw={'rgb_obs':{'rgb_static':np.zeros((200,200,3),np.uint8),'rgb_gripper':np.zeros((84,84,3),np.uint8)},'depth_obs':{'depth_static':np.ones((200,200))},'robot_obs':np.zeros(15),'scene_obs':np.array([999.]),'contacts':'excluded'}
 public=CalvinAdapter.model_view(raw);assert set(public)=={'rgb_obs','depth_obs','robot_obs'}
 with tempfile.TemporaryDirectory(prefix='wm-blocks-public-smoke-')as tmp:
  scene=Scene.__new__(Scene);scene.ad=SimpleNamespace(history=[public]*4)
  for f in scene.window(Path(tmp)/'frames'):
   with np.load(f)as z:assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}
 # Regenerate the complete base plan in a temporary root, never overwrite it.
 with tempfile.TemporaryDirectory(prefix='wm-blocks-plan-')as tmp:
  env={**os.environ,'WM_BLOCKS_ROOT':tmp,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
  subprocess.run([sys.executable,str(ROOT/'src/protocol.py')],env=env,check=True,capture_output=True,text=True)
  actual=(Path(tmp)/'configs/experiment_plan.json').read_bytes();expected=(ROOT/'configs/experiment_plan.json').read_bytes();assert actual==expected,'Regenerated split/layout plan differs'
 assert not p.isConnected()
 return {'physics_imports':True,'hydra_configuration':True,'plan_regenerated_byte_identical':True,'public_frame_allowlist':True,'simulation_started':False,'torch_imported_by_physics':('torch'in sys.modules)}
def model():
 assert sys.version_info[:2]==(3,11),'Use the model Python3.11 environment'
 for m in ['models','models_v2','models_v3','rgb_front','train','train_v2','train_v3','evaluate','evaluate_v3','live','live_v3']:importlib.import_module(m)
 import torch
 from models import Dynamics
 from models_v2 import StateActorDynamics
 from models_v3 import DiagonalStateDynamics
 from transformers import Dinov2Config
 from paths import DINO_WEIGHTS
 cfg=Dinov2Config.from_pretrained(str(DINO_WEIGHTS),local_files_only=True);assert cfg.hidden_size==384 and cfg.patch_size==14
 torch.set_num_threads(2);torch.manual_seed(7);h=torch.randn(2,2,128);p=torch.tensor([[[-.08,-.14,.465],[-.05,-.14,.50]],[[.18,-.14,.64],[.30,-.14,.45]]]);shape=torch.tensor([[[.1,.05,.05],[.02,.02,.02]]]*2);z=torch.zeros(2,2);s=Dynamics.pack(h,p,shape,z)
 for net in [Dynamics('direct_relative'),Dynamics('motion_matrix'),StateActorDynamics(),DiagonalStateDynamics()]:
  out=net(s,torch.tensor([0,1]),torch.tensor([0,0]),torch.tensor([0,1]),torch.tensor([.2,.1,.1]));assert out.shape==(2,3,135)and torch.isfinite(out).all()
 assert not torch.cuda.is_initialized(),'Smoke check must not allocate CUDA models'
 return {'model_and_planner_imports':True,'synthetic_cpu_interface_forward':True,'DINO_config_only':True,'DINO_weights_loaded':False,'CUDA_initialized':False,'training_started':False}
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['syntax','physics','model']);q=a.parse_args();result=syntax()
 if q.mode=='physics':result.update(physics())
 if q.mode=='model':result.update(model())
 print('SMOKE_RESULT '+json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
