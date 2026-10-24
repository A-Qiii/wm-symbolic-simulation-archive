"""Local Eq.2-style SupCon regularizer with explicitly audited masks.

No foreign code imported. Input is this readout's post-SiLU32 hidden vector.
Within a predicate, labels 0/1 define classes across object colors. Identity
exclusion covers every repeated copy of the same object/observation, not just
the matrix diagonal. Unknown facts never appear as anchors or candidates.
The two eligible classes contribute equally; if either is absent, SupCon=0.
"""
import numpy as np
import torch
import torch.nn.functional as F

def pair_masks(labels, valid, identities):
    y=np.asarray(labels).reshape(-1).astype(bool)
    v=np.asarray(valid).reshape(-1).astype(bool)
    ids=np.asarray(identities).reshape(-1)
    allowed=v[:,None]&v[None,:]&(ids[:,None]!=ids[None,:])
    positive=allowed&(y[:,None]==y[None,:])
    negative=allowed&~positive
    eligible=positive.any(1)&negative.any(1)&v
    counts=np.array([(eligible&(y==c)).sum()for c in(0,1)])
    weights=np.zeros(len(y),np.float32)
    if counts.min()>0:
        for c in(0,1):weights[eligible&(y==c)]=.5/counts[c]
    return allowed,positive,weights,eligible

def supervised_contrastive(hidden, allowed, positive, anchor_weights, temperature=.1):
    h=F.normalize(hidden.reshape(-1,hidden.shape[-1]),p=2,dim=1,eps=1e-12)
    similarity=h@h.T/temperature
    # Finite sentinel makes even all-invalid rows finite; their weight is zero.
    normalizer=torch.logsumexp(similarity.masked_fill(~allowed,-1e4),dim=1)
    log_probability=similarity-normalizer[:,None]
    positive_mean=(log_probability*positive).sum(1)/positive.sum(1).clamp(min=1)
    return -(positive_mean*anchor_weights).sum()

