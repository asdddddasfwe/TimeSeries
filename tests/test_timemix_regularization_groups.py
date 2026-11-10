"""Regrouping must preserve dynamics exclusions and checkpoint architecture."""
import unittest
import torch
from models.OscillatorTimeMix import Model
from test_oscillator_mixer import config
from utils.experimental_optimizer_groups import norm_bias_no_decay_groups


class RegularizationGroupsTests(unittest.TestCase):
    def test_only_norm_and_bias_exemptions_preserve_existing_groups(self):
        model=Model(config())
        original={id(p):g['weight_decay'] for g in model.optimizer_parameter_groups(.0001) for p in g['params']}
        groups=norm_bias_no_decay_groups(model,.0001)
        actual={id(p):g['weight_decay'] for g in groups for p in g['params']}
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),set(original))
        named=dict(model.named_parameters())
        for key in ('seasonals.3.decoder.drive_norm.weight','seasonals.3.decoder.drive_norm.bias',
                    'seasonals.3.temporal_projection.bias','trend_heads.0.bias'):
            self.assertEqual(original[id(named[key])],.0001)
            self.assertEqual(actual[id(named[key])],0.)
        for key in ('seasonals.3.temporal_projection.weight','seasonals.3.decoder.drive_projection.weight',
                    'trend_heads.0.weight','history_gain_offset','drive_gain_offset'):
            self.assertEqual(actual[id(named[key])],.0001)
        for key in ('seasonals.3.decoder.log_decay','seasonals.3.decoder.frequency_logits',
                    'seasonals.3.decoder.condition.0.weight'):
            self.assertEqual(original[id(named[key])],0.)
            self.assertEqual(actual[id(named[key])],0.)
        for module in model.modules():
            for name,p in module.named_parameters(recurse=False):
                expected=0. if isinstance(module,torch.nn.LayerNorm) or name=='bias' or name.endswith('_bias') else original[id(p)]
                self.assertEqual(actual[id(p)],expected)

    def test_groups_do_not_mutate_model_or_future_default_groups(self):
        model=Model(config())
        before={k:v.clone() for k,v in model.state_dict().items()}
        norm_bias_no_decay_groups(model,.0001)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        default={id(p):g['weight_decay'] for g in model.optimizer_parameter_groups(.0001) for p in g['params']}
        self.assertEqual(default[id(model.seasonals[3].decoder.drive_norm.weight)],.0001)

if __name__=='__main__':unittest.main()
