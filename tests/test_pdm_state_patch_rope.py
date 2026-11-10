"""Independent algebra and two-axis attention references for RoPE."""
import unittest
import torch
from torch.nn import functional as F
from models.OscillatorPDMStatePatchRoPE import rotate_positions,RotaryAxialBlock

class RotaryTests(unittest.TestCase):
    def test_norm_and_relative_shift(self):
        torch.manual_seed(7)
        q=torch.randn(2,7,8,dtype=torch.float64);k=torch.randn_like(q)
        p=torch.arange(7,dtype=torch.float64)
        rq=rotate_positions(q,p);rk=rotate_positions(k,p)
        torch.testing.assert_close(rq.square().sum(-1),q.square().sum(-1),rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(rotate_positions(q,torch.zeros(7)),q,rtol=0,atol=0)
        torch.testing.assert_close(rq@rk.transpose(-1,-2),rotate_positions(q,p+13)@rotate_positions(k,p+13).transpose(-1,-2),rtol=1e-12,atol=1e-12)
        self.assertFalse(torch.allclose(rq@rk.transpose(-1,-2),q@k.transpose(-1,-2)))

    def check_axes(self,device):
        torch.manual_seed(7)
        block=RotaryAxialBlock(8,4,0.).to(device).double().eval()
        with torch.no_grad():block.row.gamma.fill_(.7);block.column.gamma.fill_(.4)
        x=torch.randn(2,8,3,5,dtype=torch.float64,device=device,requires_grad=True)
        tokens=x.permute(0,2,3,1)
        # Independent explicit pair rotations and separate old conv projections.
        def rot(z,pos):
            out=z.clone()
            for j in range(z.shape[-1]//2):
                theta=pos*(10000.**(-2*j/z.shape[-1]));cs=theta.cos();sn=theta.sin()
                out[...,2*j]=z[...,2*j]*cs-z[...,2*j+1]*sn
                out[...,2*j+1]=z[...,2*j]*sn+z[...,2*j+1]*cs
            return out
        def attention(branch,column):
            vals=[layer(x).permute(0,2,3,1) for layer in (branch.query_conv,branch.key_conv,branch.value_conv)]
            if column:vals=[v.transpose(1,2) for v in vals]
            q,k,v=vals;pos=torch.arange(q.shape[-2],device=device,dtype=q.dtype)
            y=(rot(q,pos)@rot(k,pos).transpose(-1,-2)).softmax(-1)@v
            return y.transpose(1,2) if column else y
        expected=block.norm(tokens+.7*attention(block.row,False)+.4*attention(block.column,True)).permute(0,3,1,2)
        actual=block(x)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        ge=torch.autograd.grad(expected.square().sum(),x)[0]
        torch.testing.assert_close(ga,ge,rtol=1e-8,atol=1e-10)

    def test_axes_cpu(self):self.check_axes('cpu')
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_axes_cuda(self):self.check_axes('cuda')

if __name__=='__main__':unittest.main()
