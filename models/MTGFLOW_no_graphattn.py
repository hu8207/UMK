import torch
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF
import numpy as np
import scipy

class MTGFLOWZL_NoGraphAttn(nn.Module):
    """MTGFLOWZL without Graph Attention (ablation study)"""

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, window_size, n_sensor, dropout=0.1, model="MAF",
                 batch_norm=True):
        super(MTGFLOWZL_NoGraphAttn, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        
        # Remove GCN (Graph Convolution Network) since it depends on attention
        # Use simple MLP instead
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if model == "MAF":
            self.nf = MAF(n_blocks, n_sensor, input_size, hidden_size, n_hidden, cond_label_size=hidden_size,
                          batch_norm=batch_norm, activation='tanh')

        self.gcn_mlp = nn.Sequential(
            nn.Linear(n_sensor * window_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, n_sensor * window_size * 32),
        )

        self.ann = nn.Sequential(
            nn.Linear(n_sensor * window_size * 32, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
        )
        # Remove attention
        # self.attention = ScaleDotProductAttention(window_size * input_size)

    def forward(self, x, ):
        hid, log_prob, h_transform = self.test(x, )
        return hid, log_prob.mean(), h_transform, log_prob

    def test(self, x, ):
        # x: N X K X L X D 
        full_shape = x.shape
        
        # Remove attention - use identity or simple averaging instead
        # Create a simple uniform adjacency matrix
        batch_size = full_shape[0]
        n_sensor = full_shape[1]
        
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        h, _ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
        
        # Instead of GCN with attention, use simple feature transformation
        h_transform = self.feature_transform(h)
        
        shape_h = h_transform.shape
        h_combined = self.gcn_mlp(x.reshape((shape_h[0], -1))).reshape(*shape_h) + h_transform
        h_for_ann = h_combined.reshape(full_shape[0], -1)

        # reshappe N*K*L,H
        h_reshape = h_combined.reshape((-1, h_combined.shape[3]))
        x = x.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(x, full_shape[1], full_shape[2], h_reshape).reshape([full_shape[0], -1])
        log_prob = log_prob.mean(dim=1)
        return self.ann(h_for_ann), log_prob, h_combined

    def get_graph(self):
        # Return None since no attention graph
        return None

    def _DistanceSquared(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-6)
        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()

    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        Pij = (
                gamma
                * torch.tensor(2 * 3.14)
                * gamma
                * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij

    def LossManifold(
            self,
            input_data,
            latent_data,
            v_input,
            v_latent,
            w=1,
            metric="euclidean",
            label=None,
    ):
        print("input_data:", input_data.shape)
        print("latent_data:", latent_data.shape)

        batch_size = input_data.shape[0]

        data_1 = input_data[: input_data.shape[0] // 2]
        dis_P = self._DistanceSquared(data_1, data_1)
        print("dis_P:", dis_P.shape)

        latent_data_1 = latent_data[: input_data.shape[0] // 2]

        dis_P_2 = dis_P

        P_2 = self._Similarity(dist=dis_P_2, gamma=self._CalGamma(v_input), v=v_input)
        print("P_2:", P_2.shape)

        latent_data_2 = latent_data[(input_data.shape[0] // 2):]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        print("dis_Q_2:", dis_Q_2.shape)

        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        print("Q_2:", Q_2.shape)

        eye_mask = torch.eye(P_2.shape[0]).to(input_data.device)
        loss_ce_posi = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 1], Q_=Q_2[eye_mask == 1]
        )
        loss_ce_nega = self._TwowaydivergenceLoss(
            P_=P_2[eye_mask == 0], Q_=Q_2[eye_mask == 0]
        )
        w1, w2 = 1 / (1 + w), w / (1 + w)
        return w2 * loss_ce_nega, w1 * loss_ce_posi / batch_size
