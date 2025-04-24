import clip
import torch.nn as nn

class ClipLinearProbe(nn.Module):
    """Original model with simple linear probe"""
    def __init__(self, backbone_name="ViT-B/32", num_classes=7):
        super().__init__()
        self.device = "cpu"
        # load CLIP backbone
        self.model, _ = clip.load(backbone_name, device=self.device, jit=False)
        # freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # replace the text head with a linear layer
        dim = self.model.visual.output_dim
        self.head = nn.Linear(dim, num_classes)

    def forward(self, images):
        # get image features
        feats = self.model.encode_image(images)  # (B,dim)
        return self.head(feats)

class DeepHead(nn.Module):
    """More complex head with multiple layers"""
    def __init__(self, feature_dim, hidden_dim, num_classes, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p_drop),
            # Add another layer for more complexity
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ClipDeepProbe(nn.Module):
    """Enhanced model with deeper classification head"""
    def __init__(self, backbone_name="ViT-B/32", num_classes=7, hidden_dim=512, p_drop=0.2):
        super().__init__()
        self.device = "cpu"
        # load CLIP backbone
        self.model, _ = clip.load(backbone_name, device=self.device, jit=False)
        # freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # get CLIP's output dimension
        feature_dim = self.model.visual.output_dim
        # replace with deep head
        self.head = DeepHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            p_drop=p_drop
        )

    def forward(self, images):
        # get image features
        feats = self.model.encode_image(images)  # (B,dim)
        return self.head(feats)
