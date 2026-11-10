"""C ablation: B minus final-output Gram MSE. Four applied terms; diagnostics detached."""
from utils.final_output_gram_losses_20261024 import output_parts,gradient_summary,training_terms as original_B_terms
TERM_NAMES=('point_mse','head_gram_mse','head_local_cs','output_local_cs')
def training_terms(prediction,target,head,relation_mode):
    if relation_mode!='none':raise ValueError('Only the preregistered C ablation is allowed')
    terms,details=original_B_terms(prediction,target,head,'output_gram_mse')
    removed=terms.pop('relation')
    details['output']['mse']=removed.detach()
    assert not details['time_mmd'].requires_grad and not details['output']['mse'].requires_grad
    assert tuple(terms)==TERM_NAMES
    return terms,details
