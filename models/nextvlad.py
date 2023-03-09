import torch
import torch.nn as nn
import torch.nn.functional as F


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(
        self, feature_size, max_frames, nextvlad_cluster_size, lamb, groups, n_class=82
    ):
        super(NeXtVLAD, self).__init__()
        self.dim = feature_size  # feature dim
        self.lamb = lamb  # expansion factor lambda
        self.max_frames = max_frames  # frames
        self.K = nextvlad_cluster_size  # K clusters
        self.G = groups  # G
        self.group_size = int((lamb * self.dim) // self.G)

        # expansion fc
        self.fc0 = nn.Linear(self.dim, self.dim * lamb)

        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * self.dim, self.G * self.K)

        # attention over groups FC
        self.fc_g = nn.Linear(lamb * self.dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        # This parameter has a location information, so it should not disrupt its location
        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x, mask=None):
        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)  # x_dot

        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        WgkX = self.bn0(WgkX)

        # residuals across groups and clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)

        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        alpha_g = torch.sigmoid(self.fc_g(x_dot))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))  # mask [B M 1]

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)  # .contiguous()

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)

        # print(f"vlad: {vlad.shape}")
        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)  # .contiguous

        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)

        # normalize: B x (λN/G) x K
        # vlad = F.normalize(vlad, p=2, dim=1)
        vlad = F.normalize(vlad, 1)

        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)

        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)

        return vlad




class NeXtVLAD_Centers(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(
        self, feature_size, max_frames, nextvlad_cluster_size, lamb, groups, cluster_weights, n_class=82
    ):
        super().__init__()
        self.dim = feature_size  # feature dim
        self.lamb = lamb  # expansion factor lambda
        self.max_frames = max_frames  # frames
        self.K = nextvlad_cluster_size  # K clusters
        self.G = groups  # G
        self.group_size = int((lamb * self.dim) // self.G)

        # expansion fc
        self.expand = nn.Linear(self.dim, self.dim * lamb)

        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * self.dim, self.G * self.K)

        # attention over groups FC
        self.fc_g = nn.Linear(lamb * self.dim, self.G)
        self.cluster_weights = cluster_weights

        # This parameter has a location information, so it should not disrupt its location
        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x, mask=None):
        _, M, N = x.shape

        x_exp = self.expand(x)  # [B,M,N] -> [B,M,λN]

        # attention across groups 
        alpha_g = torch.sigmoid(self.fc_g(x_exp)) # [B,M,λN] -> [B,M,G]
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))  # mask [B,M,1]

        # attention across groups and clusters
        atten_gk = self.fc_gk(x_exp) # [B,M,λN] -> [B,M,(G*K)]
        atten_gk = self.bn0(atten_gk)
        atten_gk = atten_gk.reshape(-1, M * self.G, self.K) # [B,M,(G*K)] -> [B,(M*G),K]
        # softmax over clusters
        alpha_gk = F.softmax(atten_gk, dim=-1) # [B,(M*G),K] -> [B,(M*G),K]

        # alpha_g by alpha_gk
        alpha_g = alpha_g.reshape(-1, M * self.G, 1) # [B,M,G] -> [B,(M*G),1]
        attention = torch.mul(alpha_gk, alpha_g) # [B,(M*G),K] x [B,(M*G),1] -> [B,(M*G),K]


        # sum over (attetion by clusters)
        a_sum = torch.sum(attention, -2, keepdim=True) # [B,(M*G),K] -> [B,1,K]
        a = torch.mul(a_sum, self.cluster_weights) # [B,1,K] x [1,(λN/G),K] -> [B,(λN/G),K]

        # sum over (attention by x_exp)
        attention = attention.permute(0, 2, 1)  # [B,(M*G),K] -> [B,K,(M*G)]
        x_group = x_exp.reshape(-1, M, self.G, self.group_size) # B x M x λN -> B x M x G x (λN/G)
        reshaped_x_group = x_group.reshape(-1, M * self.G, self.group_size) # [B,M,G,(λN/G)] -> [B,(M*G),(λN/G)]
        vlad = torch.matmul(attention, reshaped_x_group) # [B,K,(M*G)] x [B,(M*G),(λN/G)] -> [B,K,(λN/G)]

        # print(f"vlad: {vlad.shape}")
        # permute: 
        vlad = vlad.permute(0, 2, 1) # [B,K,(λN/G)] x [B,(λN/G),K]

        # distance to centers 
        vlad = torch.sub(vlad, a) # [B,(λN/G),K] - [B,(λN/G),K]

        # normalize
        # vlad = F.normalize(vlad, p=2, dim=1)
        vlad = F.normalize(vlad, p=1, dim=1)

        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)

        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)

        return vlad
