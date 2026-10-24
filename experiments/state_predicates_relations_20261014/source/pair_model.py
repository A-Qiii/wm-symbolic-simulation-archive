import torch
from torch import nn
from model import PredicateReader

class PairReader(PredicateReader):
    """Generic target/visible-instance pairs. No semantic or numeric instance IDs."""
    def __init__(self,use_geometry=True):
        super().__init__();self.use_geometry=use_geometry
        self.appearance=nn.Sequential(nn.Linear(384,32),nn.LayerNorm(32),nn.GELU())
        self.pair=nn.Sequential(nn.Linear(72,64),nn.GELU(),nn.Linear(64,33))
        self.fusion=nn.Linear(64,128,bias=False);nn.init.zeros_(self.fusion.weight)
    def pair_features(self,objects,neighbors):
        b,t,k,_=objects.shape;target=objects.reshape(b,t,k,2,390);q=target[:,:,:,:,None,:];n=neighbors[:,:,None,:,:,:]
        qb=q[...,384:388];nb=n[...,384:388];qw=(qb[...,2:]-qb[...,:2]).clamp_min(1/224);nw=(nb[...,2:]-nb[...,:2]).clamp_min(1/224)
        dc=(qb[...,:2]+qb[...,2:]-nb[...,:2]-nb[...,2:])/2
        ratio=torch.log(qw/nw).clamp(-4,4);interwh=(torch.minimum(qb[...,2:],nb[...,2:])-torch.maximum(qb[...,:2],nb[...,:2])).clamp_min(0);inter=interwh.prod(-1,keepdim=True);qa=qw.prod(-1,keepdim=True);na=nw.prod(-1,keepdim=True)
        geom=torch.cat([dc,ratio,inter/(qa+na-inter+1e-8),inter/(qa+1e-8),inter/(na+1e-8),q[...,389:390]-n[...,389:390]],-1)
        if not self.use_geometry:geom=torch.zeros_like(geom)
        qaemb=self.appearance(target[...,:384]);naemb=self.appearance(neighbors[...,:384]);pairs=torch.cat([qaemb[:,:,:,:,None,:].expand(-1,-1,-1,-1,neighbors.shape[3],-1),naemb[:,:,None].expand(-1,-1,k,-1,-1,-1),geom],-1)
        # Exclude the target's own identical public ROI and all invisible pairs.
        self_match=((qb-nb).abs().amax(-1)<1e-6)&((q[...,:384]-n[...,:384]).abs().amax(-1)<1e-6)
        mask=(q[...,388]>0)&(n[...,388]>0)&~self_match
        pv=self.pair(pairs);scores=pv[...,0].masked_fill(~mask,-1e4);weights=torch.softmax(scores,-1)*mask;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        pooled=(weights[...,None]*pv[...,1:]).sum(-2).reshape(b,t,k,64)
        return self.fusion(pooled),weights,mask
    def forward(self,objects,context,robot,neighbors):
        b,t,k,_=objects.shape;obj=self.object_encoder(objects);res,_,_=self.pair_features(objects,neighbors);obj=obj+res;ctx=self.context_encoder(torch.cat([context,robot],-1))
        joint=torch.cat([obj,ctx[:,:,None].expand(-1,-1,k,-1)],-1).permute(0,2,1,3).reshape(b*k,t,256);_,h=self.object_history(joint);_,g=self.scene_history(ctx)
        return self.object_head(h[-1].reshape(b,k,96)),self.drawer_head(g[-1])
    def probabilities(self,objects,context,robot,neighbors):
        o,d=self(objects,context,robot,neighbors);return torch.cat([o[:,:,0].sigmoid(),o[:,:,1].sigmoid(),d.sigmoid()],-1)
