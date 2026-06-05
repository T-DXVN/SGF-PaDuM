import torch
from torch import nn
from layers.net_CNN import MyCNN
from layers.net_Mamba import MyMamba
from layers.sgf_fusion import Fusion
class FusionNet(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, trend, d_state, d_model, d_core):
        super(FusionNet, self).__init__()
        # 季节性流: CNN
        self.season_net = MyCNN(seq_len, pred_len, patch_len, stride, padding_patch, trend)
        # 趋势流: Mamba
        self.trend_net = MyMamba(seq_len, pred_len, patch_len, stride, padding_patch, d_state, d_model)
        # 融合
        self.fc1 = nn.Linear(pred_len * 2, d_model)
        self.fusion = Fusion(d_model, d_core)
        self.fc2 = nn.Linear(d_model, pred_len)
    def forward(self, s,t):
        # x: [B, Input, C]
        s = self.season_net(s)  # [B, pred_len, C]
        t = self.trend_net(t)   # [B, pred_len, C]

        # 拼接
        x = torch.cat((s, t), dim=1)  # [B, 2*pred_len, C]
        x = x.permute(0, 2, 1)        # [B, C, 2*pred_len]
        x = self.fc1(x)               # [B, C, d_model]
        x = self.fusion(x)                # [B, C, d_model]
        x = self.fc2(x)               # [B, C, pred_len]
        x = x.permute(0, 2, 1)        # [B, pred_len, C]
        return x