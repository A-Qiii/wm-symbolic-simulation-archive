import unittest,copy
from score import context_check,pair_score,full9
class ScoreTests(unittest.TestCase):
 def setUp(self):
  self.n={'y':[False,True,False,False,False,False,False,True,False],'v':[True]*9};self.p=copy.deepcopy(self.n);self.p['y'][3]=True
 def test_matched_truth_and_pair(self):
  self.assertEqual(context_check(self.n,self.p,0,1),[]);self.assertTrue(pair_score(self.n['y'],self.p['y'],0,1))
 def test_always_positive_and_held_In_false_positive_fail(self):
  self.assertFalse(pair_score(self.p['y'],self.p['y'],0,1));bad=self.p['y'].copy();bad[4]=True;self.assertFalse(pair_score(self.n['y'],bad,0,1))
 def test_missing_unknown_and_context_deviation(self):
  self.assertEqual(context_check(None,self.p,0,1),['missing_terminal']);bad=copy.deepcopy(self.p);bad['v'][3]=False;self.assertEqual(context_check(self.n,bad,0,1),['required_truth_unknown']);self.assertIsNone(full9(bad['y'],bad));bad=copy.deepcopy(self.p);bad['y'][1]=False;self.assertIn('d_not_held_both',context_check(self.n,bad,0,1))
 def test_full9_cannot_pass_with_extra_Holding(self):
  bad=self.p['y'].copy();bad[0]=True;self.assertTrue(pair_score(self.n['y'],bad,0,1));self.assertFalse(full9(bad,self.p))
if __name__=='__main__':unittest.main()
