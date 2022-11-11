import torch
import torch.nn.functional as F


def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape) # restore the structure of original data [B, K, S+Q, Img]
    x_shot, x_query = data.split([shot, query], dim=2) # x_shot-[B, K, S, Img] x_query-[B, K, Q, Img]
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous()
    # x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def make_query_label(way, query, ep_per_batch=1):
    label = torch.arange(way).unsqueeze(1).expand(way, query).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0 
    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n
    def item(self):
        return self.v

def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            # [B Q K D] -> [B Q K]
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp