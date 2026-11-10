"""Read-only frozen factory; all259 common untrained scratch exact."""
import torch
from tmp.mafirst_kernel_h720_distancefit_20261024_factory import fresh as inherited_fresh,optimizer_groups
from tmp.mafirst_h720_ampcs_20261025_common import *
def fresh(logdir):
 m,c=inherited_fresh(720,'h720_distance_fit',logdir)
 ref=torch.load(G/'reference_initial/h720.pt',map_location='cpu',weights_only=True)
 assert len(ref)==259 and all(torch.equal(v,ref[n]) and v.data_ptr()!=ref[n].data_ptr() for n,v in m.state_dict().items())
 assert sha(G/'reference_initial/h720.pt')==read(PLAN)['initial_reference_sha256']
 m.initialization_audit.update(job='h720_amp_cs',all259_local_reference_exact=True,only_applied_loss='D four terms, all unit weights; only final-output kernel replaced',model_and_feedback_unchanged=True)
 return m,c
def objective(prediction,target):return (prediction-target.detach()).square().mean()
