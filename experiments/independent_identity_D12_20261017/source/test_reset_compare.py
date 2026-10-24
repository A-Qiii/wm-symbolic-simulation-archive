import unittest,copy,json,math
from config import V1,read
from reset_compare import same_data,canonical
class ResetComparisonTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.a=read(V1/'data/reset_references/D_combination_red_00.json');cls.b=read(V1/'data/raw/D_combination_red_00/reverse_then_lift/_private/reset_signature.json');cls.fields=['robot_obs','scene_obs','RGB_sha256','physical','labels']
 def test_saved_complete_payload(self):
  self.assertEqual(set(self.a),set(self.b));self.assertEqual(set(self.a),set(self.fields)|{'branch'})
  for f in self.fields:self.assertTrue(same_data(self.a[f],self.b[f]),f)
 def test_tuple_list_same_values(self):
  a=copy.deepcopy(self.a)
  for c in a['physical']['contacts']:
   for j,v in enumerate(c):
    if isinstance(v,list):c[j]=tuple(v)
  self.assertNotEqual(a['physical'],self.b['physical']);self.assertTrue(same_data(a['physical'],self.b['physical']));self.assertEqual(canonical(a['physical']),self.b['physical'])
 def test_every_scalar_numeric_difference_rejected(self):
  def leaves(v,path=()):
   if isinstance(v,dict):
    for k,x in v.items():yield from leaves(x,path+(k,))
   elif isinstance(v,list):
    for k,x in enumerate(v):yield from leaves(x,path+(k,))
   elif isinstance(v,(int,float)):yield path,v
  payload={k:self.a[k]for k in self.fields};n=0
  for path,val in leaves(payload):
   altered=copy.deepcopy(payload);cursor=altered
   for k in path[:-1]:cursor=cursor[k]
   cursor[path[-1]]=not val if isinstance(val,bool)else math.nextafter(val,math.inf)if isinstance(val,float)else val+1
   self.assertFalse(same_data(payload,altered),str(path));n+=1
  self.assertGreater(n,100)
 def test_objects_contacts_images_and_fields_rejected(self):
  for mutate in [lambda a:a['physical']['objects'].pop('block_red'),lambda a:a['physical']['objects'].update(unexpected_object={}),lambda a:a['physical']['contacts'].pop(),lambda a:a['physical']['contacts'][0].pop(),lambda a:a['RGB_sha256'].__setitem__('rgb_static','0'*64),lambda a:a['RGB_sha256'].__setitem__('rgb_gripper','1'*64),lambda a:a['physical'].__setitem__('unexpected_field',0)]:
   a=copy.deepcopy(self.a);mutate(a);self.assertFalse(same_data(a,self.a))
 def test_sequence_order_rejected(self):
  a=copy.deepcopy(self.a);a['physical']['objects']['block_red']['position'].reverse();self.assertFalse(same_data(a,self.a))
if __name__=='__main__':unittest.main()
