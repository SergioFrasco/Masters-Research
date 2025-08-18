import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super(PartialAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # FIXED: Use 3x3 kernel instead of 2x2 to maintain 7x7 size
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # FIXED: Update decoder to match
        self.decoder_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder_conv3 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.final_conv = nn.ConvTranspose2d(input_channels + 32, input_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self._initialize_weights()

    def forward(self, x):
        x1 = F.relu(self.encoder[0](x))         # 7x7x32
        x2 = F.relu(self.encoder[2](x1))        # 7x7x64
        x3 = F.relu(self.encoder[4](x2))        # 7x7x64

        x4 = F.relu(self.decoder_conv1(x3))     # 7x7x64
        x5 = torch.cat([x2, x4], dim=1)         # 7x7x128
        x5 = F.relu(self.decoder_conv2(x5))     # 7x7x64

        x6 = torch.cat([x1, x5], dim=1)         # 7x7x96
        x6 = F.relu(self.decoder_conv3(x6))     # 7x7x32

        x7 = torch.cat([x, x6], dim=1)          # 7x7x(input+32)
        out = self.final_conv(x7)               # 7x7x1

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)