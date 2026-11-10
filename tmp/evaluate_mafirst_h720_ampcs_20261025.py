"""One selected-val/test forward for H; preserve old RBF diagnostics and add amplitude localCS."""
import numpy as np,torch
from tmp.run_mafirst_kernel_h720_distancefit_20261024 import batches,output_parts,distance_parts,alignment_parts
from utils.amplitude_weighted_patch_kernel_20261025 import parts as amplitude_parts,kernel_matrices as amplitude_matrices
from utils.amplitude_kernel_local_cs_20261025 import output_parts as amplitude_cs_parts
def evaluate(m,ds,device,folder=None):
    m.eval();square=absolute=count=0;offset=0;arrays={};kernel_sum=kernel_constant=0.;kernel_bins=np.zeros(4);alignment_total=np.zeros(12);output_total=np.zeros(8);output_bins=np.zeros(4);distance_total=np.zeros(5);amp_total=np.zeros(2);amp_meta={k:np.zeros((7,4)) for k in ['per_variable_phase_offdiag_MSE','per_variable_phase_diag_MSE','offdiag_correlation','prediction_offdiag_std','target_offdiag_std']};amp_zero=np.zeros(2);ampcs_total=np.zeros(5)
    if folder:
        folder.mkdir(exist_ok=False)
        for key,shape in [('prediction',(m.pred_len,7)),('truth',(m.pred_len,7)),('history',(96,7)),('components',(m.pred_len,7,8)),('history_mean',(1,7)),('inverse_offset',(1,7)),('inverse_scale',(1,7))]:
            arrays[key]=np.lib.format.open_memmap(folder/(key+'.npy'),mode='w+',dtype='float32',shape=(len(ds),*shape))
    with torch.no_grad():
        for x,y in batches(ds):
            x,y=x.to(device),y.to(device)
            if folder:
                components=m.predict_components(x);p=m.normalize(components[...,:4].sum(-1)+components[...,4:].sum(-1),'denorm')
            else:p=m(x)
            if m.study_arm=='kernel':
                from models.MAFirstKernelTransferAblation import kernel_supervision
                kl,gold=kernel_supervision(m.predicted_gram,y)
                kernel_sum+=float(kl)*len(x)
                alignment,am=alignment_parts(m.predicted_gram,gold)
                alignment_total+=np.array([float(alignment),am['mean_correlation'],am['target_degenerate_fraction'],am['prediction_degenerate_fraction'],am['prediction_offdiagonal_std'],am['target_offdiagonal_std'],am['local_cs_similarity'],am['local_mean_correlation'],am['local_prediction_std'],am['local_target_std'],am['local_target_degenerate_fraction'],am['local_prediction_degenerate_fraction']])*len(x)
                og=output_parts(p,y);om=og['meta']
                dp=distance_parts(p,y,m.predicted_gram)
                distance_total+=np.array([float(dp['head_huber']),float(dp['output_huber']),float(dp['head_distance_mse']),float(dp['output_distance_mse']),float(dp['head_clamp_fraction'])])*len(x)
                output_total+=np.array([float(og['mse']),float(og['local_cs']),om['mean_correlation'],om['prediction_offdiagonal_std'],om['target_offdiagonal_std'],om['local_mean_correlation'],om['local_prediction_std'],om['local_target_std']])*len(x)
                K=gold.shape[-1];mask=~torch.eye(K,device=device,dtype=torch.bool)
                kernel_constant+=float((1-gold).square()[...,mask].mean())*len(x)
                distances=(torch.arange(K,device=device)[:,None]-torch.arange(K,device=device)[None,:]).abs()
                for q,(lo,hi) in enumerate([(1,4),(5,16),(17,K//2),(K//2+1,K-1)]):
                    maskq=(distances>=lo)&(distances<=hi)
                    kernel_bins[q]+=float((m.predicted_gram-gold).square()[...,maskq].mean())*len(x)
                    output_bins[q]+=float((og['predicted'].double()-gold.double()).square()[...,maskq].mean())*len(x)
                if folder:
                    for selected in (0,len(ds)//2,len(ds)-1):
                        if offset<=selected<offset+len(x):
                            j=selected-offset
                            np.savez_compressed(folder/f'kernel_{selected}.npz',predicted=m.predicted_gram.reshape(len(x),7,K,K)[j].cpu().numpy(),target=gold.reshape(len(x),7,K,K)[j].cpu().numpy(),output_predicted=og['predicted'].reshape(len(x),7,K,K)[j].cpu().numpy(),head_distance=dp['head_distance'].reshape(len(x),7,K,K)[j].cpu().numpy(),output_distance=dp['output_distance'].reshape(len(x),7,K,K)[j].cpu().numpy(),target_distance=dp['target_distance'].reshape(len(x),7,K,K)[j].cpu().numpy())
            cs=amplitude_cs_parts(p,y);ampcs_total+=np.asarray([float(cs['local_cs']),*cs['phase_local_cs']])*len(x)
            ak=amplitude_parts(p,y,metadata=True)
            amp_total+=np.array([float(ak['offdiag_mse']),float(ak['diag_mse'])])*len(x)
            for key in amp_meta:amp_meta[key]+=np.asarray(ak['meta'][key])*len(x)
            amp_zero+=np.array([ak['meta']['positive_map_zero_fraction'],ak['meta']['positive_map_below_1e_12_fraction']])*len(x)
            if folder:
                for selected in (0,len(ds)//2,len(ds)-1):
                    if offset<=selected<offset+len(x):
                        j=selected-offset
                        np.savez_compressed(folder/f'amp_kernel_{selected}.npz',prediction=amplitude_matrices(p[j:j+1]).reshape(7,4,180,180).cpu().numpy(),target=amplitude_matrices(y[j:j+1]).reshape(7,4,180,180).cpu().numpy())
            e=p.double()-y.double();assert torch.isfinite(e).all()
            square+=float(e.square().sum());absolute+=float(e.abs().sum());count+=e.numel()
            if folder:
                scale,inv_offset=m.inverse_coordinates()
                values=dict(prediction=p,truth=y,history=x,components=components*scale.unsqueeze(-1),history_mean=m.normalize.mean,inverse_offset=inv_offset,inverse_scale=scale)
                for key,v in values.items():arrays[key][offset:offset+len(x)]=v.cpu().numpy()
            offset+=len(x)
    for a in arrays.values():a.flush()
    result=dict(head_distance_huber=float(distance_total[0]/len(ds)),output_distance_huber=float(distance_total[1]/len(ds)),head_distance_mse=float(distance_total[2]/len(ds)),output_distance_mse=float(distance_total[3]/len(ds)),head_distance_clamp_fraction=float(distance_total[4]/len(ds)),mse=square/count,mae=absolute/count,windows=len(ds),elements=count,output_gram_mse=float(output_total[0]/len(ds)),output_gram_local_cs=float(output_total[1]/len(ds)),output_gram_correlation=float(output_total[2]/len(ds)),output_gram_prediction_std=float(output_total[3]/len(ds)),output_gram_target_std=float(output_total[4]/len(ds)),output_gram_local_correlation=float(output_total[5]/len(ds)),output_gram_local_prediction_std=float(output_total[6]/len(ds)),output_gram_local_target_std=float(output_total[7]/len(ds)),output_gram_distance_bins_mse=(output_bins/len(ds)).tolist(),**(dict(kernel_local_cs_loss=float(alignment_total[0]/len(ds)),kernel_mse=kernel_sum/len(ds),constant_ones_kernel_mse=kernel_constant/len(ds),kernel_distance_bins_mse=(kernel_bins/len(ds)).tolist(),**dict(zip(('kernel_alignment_loss','kernel_offdiagonal_correlation','kernel_target_degenerate_fraction','kernel_prediction_degenerate_fraction','kernel_prediction_offdiagonal_std','kernel_target_offdiagonal_std','kernel_local_cs_similarity','kernel_local_correlation','kernel_local_prediction_std','kernel_local_target_std','kernel_local_target_flat_fraction','kernel_local_prediction_flat_fraction'),map(float,alignment_total/len(ds))))) if m.study_arm=='kernel' else {}))
    result.update(amp_kernel_offdiag_mse=float(amp_total[0]/len(ds)),amp_kernel_diag_mse=float(amp_total[1]/len(ds)),amp_kernel_metadata={k:(v/len(ds)).tolist() for k,v in amp_meta.items()},amp_positive_map_zero_fraction=float(amp_zero[0]/len(ds)),amp_positive_map_below_1e_12_fraction=float(amp_zero[1]/len(ds)))
    result.update(amp_kernel_local_cs=float(ampcs_total[0]/len(ds)),amp_kernel_phase_local_cs=(ampcs_total[1:]/len(ds)).tolist())
    return result
