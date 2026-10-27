"""Minimal public-observation adapter used by the block controller.

Scene constructs this with __new__ after composing the official CALVIN env.
These methods preserve the original adapter's segmentation-disabled behavior.
Only Scene.window's RGB/proprio allowlist is written to learning inputs.
"""
class CalvinAdapter:
 @staticmethod
 def model_view(raw):
  return {'rgb_obs':{k:v.copy()for k,v in raw['rgb_obs'].items()},
          'depth_obs':{k:v.copy()for k,v in raw['depth_obs'].items()},
          'robot_obs':raw['robot_obs'].copy()}
 def observe(self):return self.model_view(self.env.get_obs())
 def step(self,action):
  raw,_,_,_=self.env.step(action);public=self.model_view(raw);self.history.append(public);return public
 def close(self):
  if self.env.ownsPhysicsClient:self.env.close();self.env.ownsPhysicsClient=False
