import unittest
import numpy as np
import torch
from layers.OscillatorGatedMLPMixer import GELUMLPReadout
from layers.ContextResidualReadout import ContextResidualReadout
from utils.history_context_helpers import history_statistics, design_with_context, shuffled_indices


class TestHistoryContext(unittest.TestCase):
    def test_statistics_are_history_only(self):
        rng=np.random.default_rng(3);values=rng.normal(size=(240,7)).astype(np.float32)
        stats=history_statistics(values,120)
        for i in (0,17,119):
            x=values[i:i+96].astype(np.float64)
            np.testing.assert_allclose(stats[i,:,0],x.mean(0),atol=1e-10)
            np.testing.assert_allclose(stats[i,:,1],.5*np.log(x.var(0)+1e-5),atol=1e-10)
        changed=values.copy();changed[96:]+=100
        np.testing.assert_array_equal(history_statistics(changed,120)[0],stats[0])

    def test_shuffle_identity_and_train_bank(self):
        for size in (2,3,32,34369):
            tr,va=shuffled_indices(size,17,2021)
            np.testing.assert_array_equal(np.sort(tr),np.arange(size))
            self.assertTrue(np.all(tr!=np.arange(size)))
            self.assertTrue(np.all((va>=0)&(va<size)))
            np.testing.assert_array_equal(shuffled_indices(size,17,2021)[0],tr)

    def test_actual_module_matches_explicit_variable_loop(self):
        torch.manual_seed(4);base=GELUMLPReadout(16).double();module=ContextResidualReadout(base,7)
        state=torch.randn(14,5,16,dtype=torch.float64);q=torch.randn(2,7,2,dtype=torch.float64);module.context=q
        self.assertTrue(torch.equal(module(state),base(state)))
        self.assertEqual(module.delta.numel(),693)
        with torch.no_grad():module.delta.normal_(std=.02)
        hidden=base.activation(base.input_projection(state)).reshape(2,7,5,32)
        expected=base(state).clone().reshape(2,7,5,1)
        for b in range(2):
            for c in range(7):
                h=torch.cat([hidden[b,c],torch.ones(5,1,dtype=torch.float64)],1)
                d=torch.cat([h,h*q[b,c,0],h*q[b,c,1]],1)
                expected[b,c,:,0]+=d@module.delta[c]
        torch.testing.assert_close(module(state),expected.reshape_as(module(state)))


if __name__=='__main__':unittest.main()
