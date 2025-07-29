import torch
import torch.nn as nn
import torch.nn.functional as F


class TDCA_Module(nn.Module):
    def __init__(self, in_channels):
        super(TDCA_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.conv_fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv_fuse(out)
        return out


class MHFCA_Module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MHFCA_Module, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CHA_Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CHA_Network, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.tdca = TDCA_Module(64)
        self.mhfca = MHFCA_Module(64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.tdca(x)
        x = self.mhfca(x)
        x = self.final(x)
        return x


# Example usage
if __name__ == '__main__':
    model = CHA_Network()
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print("Output shape:", output.shape)
