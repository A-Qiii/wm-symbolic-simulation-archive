from common import *
from visual import load_rgb,object_regions,build_vectors
import torch,tempfile,unittest

class VisualTests(unittest.TestCase):
 def test_schema_rejects_robot_and_state(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   p=Path(d)/'x.npz';a=np.zeros((20,20,3),np.uint8)
   np.savez(p,rgb_static=a,rgb_gripper=a,robot_obs=np.zeros(15))
   with self.assertRaises(ValueError):load_rgb(p)
   np.savez(p,rgb_static=a,rgb_gripper=a);self.assertEqual(len(load_rgb(p)),2)
 def test_identity_from_RGB(self):
  a=np.zeros((40,60,3),np.uint8)
  for j,c in enumerate([(255,0,0),(0,0,255),(255,0,255)]):a[10:20,j*20+3:j*20+13]=c
  b,g=object_regions(a);self.assertTrue(np.array_equal(g[:,4],[1,1,1]));self.assertTrue(np.array_equal(b[:,0],[3,23,43]))
 def test_explicit_endpoint_and_color_equivariance(self):
  rng=np.random.default_rng(19);raw={'objects':rng.normal(size=(2,2,3,4,384)).astype(np.float32),'endpoints':rng.normal(size=(2,2,4,384)).astype(np.float32),'geometry':np.tile(np.array([.2,.3,.4,.5,1.],np.float32),(2,2,3,1))}
  norm={'mean':np.zeros(384,np.float32),'std':np.ones(384,np.float32),'basis':np.eye(384,32,dtype=np.float32),'scale':np.ones(32,np.float32)}
  z=build_vectors(raw,norm)
  for h in HEADS:self.assertTrue(np.array_equal(z[h][...,:74],z['node']))
  modified={k:v.copy()for k,v in raw.items()};modified['endpoints'][:,0]+=2.;w=build_vectors(modified,norm)
  self.assertTrue(np.array_equal(z['node'],w['node']));self.assertTrue(np.array_equal(z['In'],w['In']));self.assertFalse(np.array_equal(z['Holding'],w['Holding']))
  perm=[2,0,1];modified={k:v.copy()for k,v in raw.items()};modified['objects']=raw['objects'][:,:,perm];modified['geometry']=raw['geometry'][:,:,perm];w=build_vectors(modified,norm)
  for h in ['node']+HEADS:self.assertTrue(np.array_equal(w[h],z[h][:,perm]))
  for arm,key in [('node','node'),('edge','Holding')]:
   q=make(arm);self.assertEqual(sum(p.numel()for p in q.parameters()),6689)
   a=q(torch.tensor(z[key]));b=q(torch.tensor(w[key]));torch.testing.assert_close(b,a[:,perm],rtol=1e-6,atol=1e-7)
 def test_missing_object_is_not_imputed(self):
  rng=np.random.default_rng(20);raw={'objects':rng.normal(size=(1,2,3,4,384)).astype(np.float32),'endpoints':rng.normal(size=(1,2,4,384)).astype(np.float32),'geometry':np.zeros((1,2,3,5),np.float32)}
  norm={'mean':np.zeros(384,np.float32),'std':np.ones(384,np.float32),'basis':np.eye(384,32,dtype=np.float32),'scale':np.ones(32,np.float32)}
  z=build_vectors(raw,norm);self.assertTrue((z['node']==0).all())
  for h in HEADS:self.assertTrue((z[h][...,-5:]==0).all())

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VisualTests))
 save(E/'results/unit_tests.json',{'utc':now(),'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'GPU_used':False})
 if not result.wasSuccessful():raise SystemExit(1)

