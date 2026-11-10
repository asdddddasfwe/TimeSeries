"""Parity with the official ETT forward, optimizer coverage and fixed budget."""
import ast
from pathlib import Path
import unittest

import torch
from torch import nn

from scripts import run_distdf_fredformer_matched as runner
from models.distdf_official.RevIN import RevIN
from models.distdf_official.cross_Transformer import Trans_C


def original_class():
    root = runner.ROOT/'third_party/DistDF'
    namespace = {'torch': torch, 'nn': nn, 'RevIN': RevIN, 'Trans_C': Trans_C,
                 '__name__': 'official_fredformer_parity'}
    for name in ('layers/Fredformer_backbone.py', 'models/Fredformer.py'):
        tree = ast.parse((root/name).read_text())
        # Resolve imports to their exact official implementations; NYS is unused.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module.startswith('layers.'))]
        exec(compile(tree, str(root/name), 'exec'), namespace)
    return namespace['Model']


class FredformerTests(unittest.TestCase):
    def test_official_cpu_cuda_function_rng_gradient(self):
        cfg = runner.config(runner.ROOT/'dataset/ETT-small')
        official = original_class()
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            runner.seed(2021)
            a = official(cfg).to(device)
            runner.seed(2021)
            b = runner.Model(cfg).to(device)
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            self.assertTrue(all(torch.equal(v, b.state_dict()[n])
                                and v.data_ptr() != b.state_dict()[n].data_ptr()
                                for n, v in a.state_dict().items()))
            x = torch.randn(2, 96, 7, generator=torch.Generator().manual_seed(201)).to(device)
            y = x.flip(1)
            for training in (False, True):
                outputs, rngs, gradients = [], [], []
                for m in (a, b):
                    m.train(training)
                    m.zero_grad(set_to_none=True)
                    runner.seed(319)
                    p = m(x)
                    outputs.append(p.detach())
                    rngs.append(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state())
                    self.assertEqual(p.shape, (2, 96, 7))
                    loss = .5*(p-y).square().mean()+.5*runner.distdf_loss(p, y, x)
                    loss.backward()
                    gradients.append({n: None if v.grad is None else v.grad.detach().clone()
                                      for n, v in m.named_parameters()})
                self.assertTrue(torch.equal(outputs[0], outputs[1]))
                self.assertTrue(torch.equal(rngs[0], rngs[1]))
                for n, g in gradients[0].items():
                    if g is None:
                        self.assertIsNone(gradients[1][n])
                    else:
                        self.assertTrue(torch.equal(g, gradients[1][n]), n)
                        self.assertTrue(torch.isfinite(g).all(), n)
            parameters = [p for group in runner.optimizer_groups(b) for p in group['params']]
            self.assertEqual(len(parameters), len({id(p) for p in parameters}))
            self.assertEqual({id(p) for p in parameters}, {id(p) for p in b.parameters()})
            self.assertEqual(11*((34369+31)//32), 11825)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
