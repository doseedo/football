"""OSNet (Omni-Scale Network) for person re-identification.

Implementation of OSNet-AIN architecture for extracting appearance embeddings.
Paper: "Omni-Scale Feature Learning for Person Re-Identification" (Zhou et al., ICCV 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torchvision.models as models


class ConvLayer(nn.Module):
    """Convolutional layer with batch norm and relu."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution with depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride=1,
                              padding=1, groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, stride=1,
                              padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """Channel attention gate (AIN - Attention in Network)."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

    def forward(self, x):
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(avg_pool)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return x * out


class OSBlock(nn.Module):
    """Omni-Scale residual block with multiple receptive fields."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4,
                 T: int = 4):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            reduction: Channel reduction ratio
            T: Number of scales (branches)
        """
        super().__init__()
        assert out_channels >= T, "Output channels must be >= T"
        self.T = T

        # Bottleneck
        mid_channels = out_channels // reduction

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Multi-scale convolutions
        self.convs = nn.ModuleList()
        for i in range(T):
            self.convs.append(LightConv3x3(mid_channels, mid_channels))

        # Channel gate
        self.gate = ChannelGate(mid_channels)

        # Final conv
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Multi-scale feature extraction
        spx = torch.split(x, x.size(1) // self.T, dim=1)
        for i in range(self.T):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat([out, sp], dim=1)

        # Channel attention
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class OSNet(nn.Module):
    """OSNet architecture for person re-identification.

    Extracts 512-dimensional embedding vectors for appearance-based matching.
    """

    def __init__(self, num_classes: int = 1000, feature_dim: int = 512,
                 loss: str = 'softmax', blocks: Tuple[int, ...] = (2, 2, 2),
                 channels: Tuple[int, ...] = (64, 256, 384, 512),
                 T: int = 4, pretrained: bool = False):
        """
        Args:
            num_classes: Number of identity classes (for training)
            feature_dim: Dimension of output embedding
            loss: Loss type ('softmax' or 'triplet')
            blocks: Number of blocks in each stage
            channels: Number of channels in each stage
            T: Number of scales in OSBlock
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.loss = loss

        # Initial convolution
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Build stages
        self.stage1 = self._make_stage(channels[0], channels[1], blocks[0], T)
        self.stage2 = self._make_stage(channels[1], channels[2], blocks[1], T)
        self.stage3 = self._make_stage(channels[2], channels[3], blocks[2], T)

        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Feature layer
        self.fc = nn.Linear(channels[3], feature_dim)
        self.bn_feat = nn.BatchNorm1d(feature_dim)

        # Classification layer
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_params()

    def _make_stage(self, in_channels: int, out_channels: int,
                    num_blocks: int, T: int):
        """Build a stage with multiple OSBlocks."""
        layers = []
        layers.append(OSBlock(in_channels, out_channels, T=T))
        for _ in range(1, num_blocks):
            layers.append(OSBlock(out_channels, out_channels, T=T))
        return nn.Sequential(*layers)

    def _init_params(self):
        """Initialize parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        """Extract multi-scale feature maps."""
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

    def forward(self, x, return_featuremaps: bool = False):
        """
        Args:
            x: Input tensor (B, 3, H, W)
            return_featuremaps: If True, return feature maps instead of embeddings

        Returns:
            If return_featuremaps: feature maps (B, C, H, W)
            Else if training: (features, logits)
            Else: features (B, feature_dim)
        """
        x = self.featuremaps(x)

        if return_featuremaps:
            return x

        # Global pooling
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)

        # Feature embedding
        v = self.fc(v)
        v = self.bn_feat(v)

        if not self.training:
            return F.normalize(v, p=2, dim=1)

        # Classification
        y = self.classifier(v)

        if self.loss == 'softmax':
            return v, y
        elif self.loss == 'triplet':
            return v
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")


class OSNetAIN(OSNet):
    """OSNet with Attention in Network (AIN) - recommended version."""

    def __init__(self, num_classes: int = 1000, feature_dim: int = 512,
                 loss: str = 'softmax', pretrained: bool = False):
        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_dim,
            loss=loss,
            blocks=(2, 2, 2),
            channels=(64, 256, 384, 512),
            T=4,
            pretrained=pretrained
        )


class ReIDExtractor:
    """Wrapper for extracting re-identification features from player crops."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Args:
            model_path: Path to pretrained model weights
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = OSNetAIN(num_classes=1000, feature_dim=512, loss='softmax')

        # Load pretrained weights if provided
        if model_path:
            self.load_weights(model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Normalization for ImageNet pretrained models
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def load_weights(self, model_path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {model_path}")

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for OSNet.

        Args:
            images: Tensor of shape (B, 3, H, W) in [0, 1] range

        Returns:
            Normalized tensor
        """
        # Resize to standard re-ID size
        images = F.interpolate(images, size=(256, 128), mode='bilinear',
                              align_corners=False)

        # Normalize
        images = (images - self.mean) / self.std

        return images

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract appearance embeddings from player crops.

        Args:
            images: Tensor of shape (B, 3, H, W) in [0, 1] range

        Returns:
            Embeddings of shape (B, 512), L2-normalized
        """
        # Preprocess
        images = self.preprocess(images)
        images = images.to(self.device)

        # Extract features
        features = self.model(images)

        return features.cpu()

    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two feature sets.

        Args:
            feat1: Features of shape (N, 512)
            feat2: Features of shape (M, 512)

        Returns:
            Similarity matrix of shape (N, M)
        """
        # Features are already L2-normalized, so dot product = cosine similarity
        return torch.mm(feat1, feat2.t())


def build_osnet(feature_dim: int = 512, pretrained: bool = False,
                model_path: Optional[str] = None) -> OSNetAIN:
    """
    Build OSNet-AIN model.

    Args:
        feature_dim: Dimension of output embedding
        pretrained: Whether to load pretrained weights
        model_path: Path to pretrained model

    Returns:
        OSNet-AIN model

    Note:
        Pretrained weights can be downloaded from:
        - Market1501: https://github.com/KaiyangZhou/deep-person-reid
        - MSMT17: https://github.com/KaiyangZhou/deep-person-reid

        Recommended: osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
    """
    model = OSNetAIN(num_classes=1000, feature_dim=feature_dim, loss='softmax')

    if pretrained and model_path:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained OSNet from {model_path}")

    return model
