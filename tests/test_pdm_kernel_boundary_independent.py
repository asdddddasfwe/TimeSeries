import unittest
from scripts.run_timemix_pdm_kernel_boundary_independent_experiment import verify_stop_policy

class IndependentPolicyTests(unittest.TestCase):
    def check(self, values, stale, maximum=30, patience=5):
        return verify_stop_policy([dict(epoch=i+1,validation_mse=v) for i,v in enumerate(values)],len(values),stale,maximum,patience)
    def test_ties_count_as_stale(self):
        self.check([1.,1.,1.,1.,1.,1.],5)
    def test_improvement_resets_stale(self):
        self.check([1.,2.,2.,.9,1.,1.,1.,1.,1.],5)
    def test_no_update_after_stop(self):
        with self.assertRaises(AssertionError):self.check([1.]*7,6)
    def test_no_premature_stop(self):
        with self.assertRaises(AssertionError):self.check([1.,.9],0)
    def test_maximum_horizon(self):
        self.check([1.,.9],0,maximum=2)
if __name__=='__main__':unittest.main()
