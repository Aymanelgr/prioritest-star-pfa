"""PRIORITEST-STAR: two-view cross-contrastive ranking (independent of Rhazi's original architecture).

Options ajoutées (toutes désactivées par défaut => comportement identique au kit initial) :
  * use_graph=False       : TCN seul (ablation « historique seul »), z = zT.
  * use_delta=True        : le TCN reçoit aussi les variations Δx_t = x_t - x_{t-1} (instabilité / churn).
  * bidirectional=True    : GraphSAGE agrège séparément les classes importées ET les classes qui importent.
  * dropout>0             : dropout + LayerNorm sur les deux vues.
Pertes ajoutées :
  * label_aware_infonce   : InfoNCE inter-vues avec masquage des faux négatifs (même label).
  * effort_aware_rank_loss: RankNet pondéré par le coût LOC (favorise les petites classes défectueuses).
"""
import torch
from torch import nn
import torch.nn.functional as F


class MeanSAGE(nn.Module):
    """GraphSAGE compact à agrégation mean (2 couches).

    Convention des arêtes : edges[k] = (source, cible) = « source importe cible ».
    bidirectional=False : chaque nœud agrège uniquement ses cibles (comportement du kit initial).
    bidirectional=True  : agrégation séparée des cibles (dépendances) et des sources (dépendants).
    """

    def __init__(self, d, bidirectional=False, layers=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.self_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.reverse_layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)]) if bidirectional else None

    @staticmethod
    def _mean(x, src, dst):
        agg = torch.zeros_like(x)
        deg = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if src.numel():
            agg.index_add_(0, src, x[dst])
            deg.index_add_(0, src, torch.ones(len(src), 1, device=x.device, dtype=x.dtype))
        return agg / deg.clamp_min(1)

    def forward(self, x, edges):
        src = edges[:, 0] if edges.numel() else edges.new_zeros(0)
        dst = edges[:, 1] if edges.numel() else edges.new_zeros(0)
        for i, (sl, nl) in enumerate(zip(self.self_layers, self.neighbor_layers)):
            h = sl(x) + nl(self._mean(x, src, dst))
            if self.bidirectional:
                h = h + self.reverse_layers[i](self._mean(x, dst, src))
            x = F.relu(h)
        return x


class CrossViewSTAR(nn.Module):
    def __init__(self, n_features, d=32, use_graph=True, similarity_gate=True,
                 use_delta=False, bidirectional=False, dropout=0.0, gate_absdiff=False):
        super().__init__()
        self.use_graph = use_graph
        self.similarity_gate = similarity_gate
        self.use_delta = use_delta
        in_ch = 2 * n_features if use_delta else n_features
        # TCN causal : noyau 2, dilatations 1 et 2
        self.tcn1 = nn.Conv1d(in_ch, d, 2, padding=1)
        self.tcn2 = nn.Conv1d(d, d, 2, padding=2, dilation=2)
        self.graph_in = nn.Sequential(nn.Linear(n_features, d), nn.ReLU())
        self.sage = MeanSAGE(d, bidirectional=bidirectional)
        self.temp_proj = nn.Linear(d, d)
        self.struct_proj = nn.Linear(d, d)
        self.norm_t = nn.LayerNorm(d) if dropout > 0 else nn.Identity()
        self.norm_s = nn.LayerNorm(d) if dropout > 0 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate_absdiff = gate_absdiff   # Phase 4: the gate also sees |zT − zS| (where the views disagree)
        self.gate = nn.Sequential(nn.Linear(2 * d + (1 if similarity_gate else 0) + (d if gate_absdiff else 0), d),
                                  nn.ReLU(), nn.Linear(d, 2))
        self.risk = nn.Linear(d, 1)
        self.rank = nn.Linear(d, 1)

    def temporal_input(self, x):
        if not self.use_delta:
            return x
        delta = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)  # Δ causal, Δ_0 = 0
        return torch.cat([x, delta], dim=2)

    def forward(self, x, edges, mask=None):
        """x : [N, T, F] ; edges : [E, 2] ; mask optionnel [N, T] (1 = version présente)."""
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        T = x.shape[1]
        # Causal conv output is left-aligned: output at time t sees only <=t.
        t = self.temporal_input(x).transpose(1, 2)
        t = F.relu(self.tcn1(t))[:, :, :T]
        t = F.relu(self.tcn2(t))[:, :, :T]
        temporal = self.drop(self.norm_t(self.temp_proj(t[:, :, -1])))
        if self.use_graph:
            structural = self.drop(self.norm_s(self.struct_proj(self.sage(self.graph_in(x[:, -1, :]), edges))))
            agreement = F.cosine_similarity(temporal, structural, dim=1).unsqueeze(1)
            feats = torch.cat([temporal, structural, agreement], dim=1) if self.similarity_gate \
                else torch.cat([temporal, structural], dim=1)
            if self.gate_absdiff:
                feats = torch.cat([feats, (temporal - structural).abs()], dim=1)
            gate = torch.softmax(self.gate(feats), dim=1)
            combined = gate[:, 0:1] * temporal + gate[:, 1:2] * structural
        else:  # ablation TCN seul : pas de vue structurelle, pas de fusion
            structural = torch.zeros_like(temporal)
            agreement = torch.zeros(temporal.shape[0], 1, device=x.device, dtype=x.dtype)
            gate = torch.cat([torch.ones_like(agreement), torch.zeros_like(agreement)], dim=1)
            combined = temporal
        return {'logits': self.risk(combined).squeeze(1), 'rank': self.rank(combined).squeeze(1),
                'temporal': temporal, 'structural': structural, 'agreement': agreement.squeeze(1), 'gate': gate}


def cross_view_infonce(temporal, structural, temperature=0.2):
    """Symmetric cross-view InfoNCE; identical class = positive. Negative = other class in *training* cohort.
    Known limitation: different classes with common labels may become false negatives.
    """
    if temporal.shape[0] < 2:
        return temporal.sum() * 0
    u = F.normalize(temporal, dim=1)
    v = F.normalize(structural, dim=1)
    logits = (u @ v.T) / temperature
    target = torch.arange(len(u), device=u.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def label_aware_infonce(temporal, structural, labels, temperature=0.2, defect_only=True):
    """InfoNCE inter-vues avec prévention des faux négatifs.

    Positif : (zT_i, zS_i). Les paires (i, j≠i) partageant le même label sont retirées des négatifs
    (defect_only=True : uniquement entre deux classes défectueuses, les plus susceptibles d'être des
    faux négatifs ; defect_only=False : pour tous les labels identiques).
    """
    if temporal.shape[0] < 2:
        return temporal.sum() * 0
    u = F.normalize(temporal, dim=1)
    v = F.normalize(structural, dim=1)
    logits = (u @ v.T) / temperature
    y = labels.view(-1) > .5
    same = y.unsqueeze(0) == y.unsqueeze(1)
    if defect_only:
        same = same & y.unsqueeze(0) & y.unsqueeze(1)
    eye = torch.eye(len(u), dtype=torch.bool, device=u.device)
    logits = logits.masked_fill(same & ~eye, float('-inf'))
    target = torch.arange(len(u), device=u.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def _sample_pairs(labels, max_pairs):
    pos = torch.where(labels > .5)[0]
    neg = torch.where(labels <= .5)[0]
    if not len(pos) or not len(neg):
        return None, None
    i = pos.repeat_interleave(len(neg))
    j = neg.repeat(len(pos))
    if len(i) > max_pairs:
        take = torch.linspace(0, len(i) - 1, max_pairs, device=i.device).long()
        i, j = i[take], j[take]
    return i, j


def pairwise_rank_loss(scores, labels, max_pairs=3000):
    i, j = _sample_pairs(labels, max_pairs)
    if i is None:
        return scores.sum() * 0
    return F.softplus(-(scores[i] - scores[j])).mean()


def effort_aware_rank_loss(scores, labels, loc, max_pairs=3000, strength=1.0):
    """RankNet pondéré par l'effort d'inspection.

    Chaque paire (défectueuse i, saine j) reçoit le poids 1 + strength * max(0, log(loc_j) - log(loc_i)) :
    se tromper sur une petite classe défectueuse classée sous une grosse classe saine coûte plus cher
    sous un budget LOC. Les poids sont normalisés (moyenne 1).
    """
    i, j = _sample_pairs(labels, max_pairs)
    if i is None:
        return scores.sum() * 0
    logl = torch.log1p(loc.clamp_min(1))
    w = 1 + strength * (logl[j] - logl[i]).clamp_min(0)
    w = w / w.mean()
    return (w * F.softplus(-(scores[i] - scores[j]))).mean()


def supcon_cross_view(temporal, structural, labels, temperature=0.2, sup_weight=0.5):
    """Phase 4 — cross-view InfoNCE / supervised-contrastive hybrid.

    Anchor zT_i, candidates zS_j. Positives: j = i (weight 1) and every other defective j when i is defective
    (weight sup_weight, split across them). Same-label clean pairs are neither pulled nor pushed (removed from the
    denominator), so the false negatives of plain InfoNCE disappear and defective classes form one cluster.
    Symmetric over the two views.
    """
    if temporal.shape[0] < 2:
        return temporal.sum() * 0
    u = F.normalize(temporal, dim=1); v = F.normalize(structural, dim=1)
    y = labels.view(-1) > .5; n = len(u); eye = torch.eye(n, dtype=torch.bool, device=u.device)
    same_clean = (~y).unsqueeze(0) & (~y).unsqueeze(1) & ~eye
    sup = y.unsqueeze(0) & y.unsqueeze(1) & ~eye
    target = eye.float() + sup_weight * sup.float() / sup.sum(1, keepdim=True).clamp_min(1)
    target = target / target.sum(1, keepdim=True)

    def side(a, b):
        logits = (a @ b.T / temperature).masked_fill(same_clean, float('-inf'))
        return -(target * F.log_softmax(logits, dim=1).masked_fill(same_clean, 0)).sum(1).mean()
    return (side(u, v) + side(v, u)) / 2


def gate_entropy(gate):
    """Mean entropy of the fusion weights (α, β). Maximising it keeps both views in use."""
    return -(gate * gate.clamp_min(1e-8).log()).sum(1).mean()
