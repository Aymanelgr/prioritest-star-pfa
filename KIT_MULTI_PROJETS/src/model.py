"""PRIORITEST-STAR: two-view cross-contrastive ranking (independent of Rhazi's original architecture)."""
import torch
from torch import nn
import torch.nn.functional as F

class MeanSAGE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_layers=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
        self.neighbor_layers=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
    def forward(self,x,edges):
        for sl,nl in zip(self.self_layers,self.neighbor_layers):
            agg=torch.zeros_like(x); degree=torch.zeros(x.shape[0],1,device=x.device)
            if edges.numel():
                source,target=edges[:,0],edges[:,1]
                agg.index_add_(0,source,x[target]); degree.index_add_(0,source,torch.ones(len(source),1,device=x.device))
            x=F.relu(sl(x)+nl(agg/degree.clamp_min(1)))
        return x

class CrossViewSTAR(nn.Module):
    def __init__(self,n_features,d=32,use_graph=True,similarity_gate=True):
        super().__init__();self.use_graph=use_graph;self.similarity_gate=similarity_gate
        self.tcn1=nn.Conv1d(n_features,d,2,padding=1)
        self.tcn2=nn.Conv1d(d,d,2,padding=2,dilation=2)
        self.graph_in=nn.Sequential(nn.Linear(n_features,d),nn.ReLU())
        self.sage=MeanSAGE(d)
        self.temp_proj=nn.Linear(d,d);self.struct_proj=nn.Linear(d,d)
        self.gate=nn.Sequential(nn.Linear(2*d+(1 if similarity_gate else 0),d),nn.ReLU(),nn.Linear(d,2))
        self.risk=nn.Linear(d,1);self.rank=nn.Linear(d,1)
    def forward(self,x,edges):
        # Causal conv output is left-aligned: output at time t sees only <=t.
        t=x.transpose(1,2)
        t=F.relu(self.tcn1(t))[:,:,:x.shape[1]]
        t=F.relu(self.tcn2(t))[:,:,:x.shape[1]]
        temporal=self.temp_proj(t[:,:,-1])
        structural=self.struct_proj(self.sage(self.graph_in(x[:,-1,:]),edges)) if self.use_graph else torch.zeros_like(temporal)
        agreement=F.cosine_similarity(temporal,structural,dim=1).unsqueeze(1)
        features=torch.cat([temporal,structural,agreement],dim=1) if self.similarity_gate else torch.cat([temporal,structural],dim=1)
        gate=torch.softmax(self.gate(features),dim=1)
        combined=gate[:,0:1]*temporal+gate[:,1:2]*structural
        return {'logits':self.risk(combined).squeeze(1),'rank':self.rank(combined).squeeze(1),
                'temporal':temporal,'structural':structural,'agreement':agreement.squeeze(1),'gate':gate}

def cross_view_infonce(temporal,structural,temperature=0.2):
    """Symmetric cross-view InfoNCE; identical class = positive. Negative = other class in *training* cohort.
    Known limitation: different classes with common labels may become false negatives.
    """
    if temporal.shape[0]<2:return temporal.sum()*0
    u=F.normalize(temporal,dim=1);v=F.normalize(structural,dim=1)
    logits=(u@v.T)/temperature
    target=torch.arange(len(u),device=u.device)
    return (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2

def pairwise_rank_loss(scores,labels,max_pairs=3000):
    pos=torch.where(labels>.5)[0];neg=torch.where(labels<=.5)[0]
    if not len(pos) or not len(neg):return scores.sum()*0
    i=pos.repeat_interleave(len(neg));j=neg.repeat(len(pos))
    if len(i)>max_pairs:
        take=torch.linspace(0,len(i)-1,max_pairs,device=i.device).long();i=i[take];j=j[take]
    return F.softplus(-(scores[i]-scores[j])).mean()
