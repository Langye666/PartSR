from typing import List, Dict

import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class VideoEncoder(nn.Module):
    def __init__(self,
                 resnet_feat_dim=512,
                 gru_hidden_size=128,
                 gru_num_layers=2,
                 linear_hidden_size=16,
                 dropout=0.3):

        super(VideoEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            *list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2]  # 保留到最后一个卷积层
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应空间池化 (将不同尺寸的特征图池化为固定大小)

        self.gru = nn.GRU(
            input_size=resnet_feat_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,  # 单向GRU
            dropout=dropout if gru_num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size // 2, linear_hidden_size),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        features = self.feature_extractor(x)
        features = self.adaptive_pool(features)
        features = features.view(B, -1) # 去除空间维度: (B, resnet_feat_dim)
        _, hidden = self.gru(features)

        return self.fc(hidden)


class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        pass

    def forward(self, x):
        pass # use traditional algorithms to find texture complexity in videos


class EnvEncoder(nn.Module):
    def __init__(self, linear_hidden_size=16):
        super(EnvEncoder, self).__init__()
        self.bandwidth_encoder = nn.Linear(1, linear_hidden_size)
        self.buffer_encoder = nn.Linear(1, linear_hidden_size)

        self.fusion_layer = nn.Linear(linear_hidden_size * 2, linear_hidden_size * 2)

    def forward(self, bandwidth, buffer):
        bw_encoded = torch.relu(self.bandwidth_encoder(bandwidth))  # [batch, linear_hidden_size]
        buf_encoded = torch.relu(self.buffer_encoder(buffer))       # [batch, linear_hidden_siz

        combined = torch.cat((bw_encoded, buf_encoded), dim=1)  # [batch, 2*linear_hidden_size]
        fused = torch.relu(self.fusion_layer(combined))         # [batch, fusion_output_size]

        return fused


class ArmEncoder(nn.Module):
    def __init__(self, model_features: List):
        super(ArmEncoder, self).__init__()
        pass

    def forward(self, x):
        pass


class ArmContextGenerator(nn.Module):
    def __init__(self, arm_features: List):
        super(ArmContextGenerator, self).__init__()
        self.arm_features = arm_features
        self.video_encoder = VideoEncoder()
        pass

    def forward(self, env: Dict[str, float], video_clip: torch.Tensor):
        video_features = self.video_encoder(video_clip)