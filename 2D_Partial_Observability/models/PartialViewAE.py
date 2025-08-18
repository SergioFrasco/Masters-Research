import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialAutoencoder(nn.Module):
    def __init__(self, input_channels=1):
        super(PartialAutoencoder, self).__init__()

        # Encoder for 7x7 input
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 64+64=128
        self.final_conv = nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=1, padding=1, bias=False)  # 32+32=64

        self._initialize_weights()

    def forward(self, x):
        # Encoder path - all maintain 7x7 spatial size
        x1 = F.relu(self.encoder[0](x))         # (batch, 32, 7, 7)
        x2 = F.relu(self.encoder[2](x1))        # (batch, 64, 7, 7)
        x3 = F.relu(self.encoder[4](x2))        # (batch, 128, 7, 7)

        # Decoder path with skip connections
        x4 = F.relu(self.decoder_conv1(x3))     # (batch, 64, 7, 7)
        x5 = torch.cat([x2, x4], dim=1)         # (batch, 128, 7, 7)
        x5 = F.relu(self.decoder_conv2(x5))     # (batch, 32, 7, 7)

        x6 = torch.cat([x1, x5], dim=1)         # (batch, 64, 7, 7)
        out = self.final_conv(x6)               # (batch, 1, 7, 7)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)