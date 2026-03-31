import torch
import torch.nn as nn
import torch.nn.functional as F

class BioLatticeNet3D(nn.Module):
    def __init__(self):
        super(BioLatticeNet3D, self).init()
        # Input: [Batch, 3 channels, 32, 32, 32]
        
        # Block 1: Edge and texture extraction
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        
        # Block 2: Spatial compression
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        
        # Block 3: Depth analysis
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        
        # Global Layer: Averages all information
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final Classifier (Malignancy Probability)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.sigmoid(x)

# CPU load test
modelo = BioLatticeNet3D()
print(f"Bio-Lattice 3D Model ready for CPU. Approximate parameters: {sum(p.numel() for p in modelo.parameters())}")