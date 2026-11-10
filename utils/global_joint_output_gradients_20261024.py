"""C training loss unchanged; add global output head gradient reporting scope."""
import torch
from utils.drop_output_gram_mse_20261024 import training_terms,output_parts,TERM_NAMES
def gradient_summary(terms,named):
    """Four raw loss gradients and six pair angles; extra global readout scope; not Adam updates."""
    grads={key:torch.autograd.grad(value,[p for _,p in named],retain_graph=True,allow_unused=True) for key,value in terms.items()}
    result={}
    scopes=[('global_output_head',lambda n:n=='global_seasonal_delta'),('all',lambda n:True),('coarse_readout',lambda n:n.startswith('seasonals.3.decoder.readout.')),('history_encoder',lambda n:'.blocks.' in n or n.startswith('pdm_blocks.')),('state_core',lambda n:'.decoder.' in n and '.readout.' not in n)]
    for scope,include in scopes:
        norms={};pairs={}
        for key,row in grads.items():
            vals=[g.double() for (n,p),g in zip(named,row) if include(n) and g is not None]
            assert all(torch.isfinite(g).all() for g in vals)
            norms[key]=sum(float(g.square().sum()) for g in vals)**.5
        keys=list(terms)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                dot=sum(float((ga.double()*gb.double()).sum()) for (n,p),ga,gb in zip(named,grads[a],grads[b]) if include(n) and ga is not None and gb is not None)
                den=norms[a]*norms[b];pairs[a+'__'+b]=dict(cosine=dot/max(den,1e-30),defined=den>0)
        result[scope]=dict(norms=norms,pairs=pairs)
    return result
