"""DistDF official ETT upper-bound objective, imported without unrelated dependencies.

Upstream: Master-PLC/DistDF, commit 21b050fc230d35c7e1c4507c8da3dcd81dc9e1b9.
The two batch functions below are verbatim from utils/fft_ot.py under MIT;
see third_party/DistDF/LICENSE and UPSTREAM_PROVENANCE.json.
The adapter reproduces its wasserstein_empirical_per_dim/upper_bound branch.
Official trainer uses history+future, alpha=.5 and var_weight=.002 for ETTm1/96.
cal_wasserstein is called without eps, so its effective default is 1e-8.
Batch statistics couple samples; this is not a per-window patch MMD.
"""
import torch

def batch_empirical_bures_wasserstein_distance(
    xs, xt, reg=1e-6, ws=None, wt=None, bias=True, var_weight=1.0, mean_weight=1.0, eps=1e-9
):
    """
    批量计算 Bures-Wasserstein 距离，支持所有维度同时计算
    输入形状: xs, xt: (D, B, T)
             ws, wt: (D, B, 1)
    """
    D_batch, B, T = xs.shape

    # 计算均值 (D, 1, T)
    if bias:
        mxs = torch.matmul(ws.transpose(1, 2), xs) / torch.sum(ws, dim=1, keepdim=True)
        mxt = torch.matmul(wt.transpose(1, 2), xt) / torch.sum(wt, dim=1, keepdim=True)
        xs = xs - mxs
        xt = xt - mxt
    else:
        mxs = torch.zeros((D_batch, 1, T), dtype=xs.dtype, device=xs.device)
        mxt = torch.zeros((D_batch, 1, T), dtype=xt.dtype, device=xt.device)

    # 批量协方差计算 (D, T, T)
    Cs = torch.matmul((xs * ws).transpose(1, 2), xs) / torch.sum(ws, dim=1, keepdim=True).transpose(1, 2)
    Cs += reg * torch.eye(T, dtype=xs.dtype, device=xs.device).unsqueeze(0)

    Ct = torch.matmul((xt * wt).transpose(1, 2), xt) / torch.sum(wt, dim=1, keepdim=True).transpose(1, 2)
    Ct += reg * torch.eye(T, dtype=xt.dtype, device=xt.device).unsqueeze(0)

    # 批量距离计算
    W_batch = batch_upper_bound_distance(mxs, mxt, Cs, Ct, var_weight=var_weight, mean_weight=mean_weight, eps=eps)
    return torch.mean(W_batch)

def batch_upper_bound_distance(ms, mt, Cs, Ct, var_weight=1.0, mean_weight=1.0, eps=1e-6):
    """
    批量计算 Bures-Wasserstein 距离的上界
    输入形状: ms, mt: (D, 1, T)
             Cs, Ct: (D, T, T)
    输出形状: (D,)
    """
    # 计算均值差平方范数 (D,)
    norm_diff_sq = torch.sum((ms - mt) ** 2, dim=(-1, -2))

    # 计算 B 项 (D,)
    trace_Cs = torch.diagonal(Cs, dim1=-2, dim2=-1).sum(dim=-1)
    trace_Ct = torch.diagonal(Ct, dim1=-2, dim2=-1).sum(dim=-1)
    trace_CsCt = torch.diagonal(torch.matmul(Cs, Ct), dim1=-2, dim2=-1).sum(dim=-1)

    B = trace_Cs + trace_Ct - 2 * torch.sqrt(torch.clip(trace_CsCt, min=eps))
    W = torch.sqrt(torch.clip(mean_weight * norm_diff_sq + var_weight * B, min=eps))

    return W

def distdf_loss(prediction, target, history, *, var_weight=0.002,
                mean_weight=1.0, reg=0.005, eps=1e-8):
    """Return official joint upper-bound distance for [B,T,C] tensors.

    History and target are observations and detached. The official arithmetic,
    covariance regularization and sqrt/clip order are retained without changes.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must share [B,T,C] shape")
    if history.ndim != 3 or history.shape[0] != prediction.shape[0] or history.shape[2] != prediction.shape[2]:
        raise ValueError("history must share prediction batch and channels")
    if prediction.dtype != target.dtype or prediction.dtype != history.dtype:
        raise ValueError("all inputs must have the same dtype")
    joint_prediction = torch.cat((history.detach(), prediction), dim=1)
    joint_target = torch.cat((history.detach(), target.detach()), dim=1)
    batch, _, channels = joint_prediction.shape
    a = torch.ones((batch, 1), device=prediction.device, dtype=prediction.dtype) / batch
    b = torch.ones((batch, 1), device=prediction.device, dtype=prediction.dtype) / batch
    return batch_empirical_bures_wasserstein_distance(
        joint_prediction.permute(2, 0, 1), joint_target.permute(2, 0, 1),
        reg=reg, ws=a.unsqueeze(0).expand(channels, -1, -1),
        wt=b.unsqueeze(0).expand(channels, -1, -1),
        var_weight=var_weight, mean_weight=mean_weight, eps=eps)
