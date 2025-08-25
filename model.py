import torch
import torch.nn as nn
import torch.optim as optim

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=2, padding=1),  # [3, 32, 24, 24]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [3, 64, 12, 12]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [3, 128, 6, 6]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256)  # [3, 256]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 6 * 6),
            nn.ReLU(),
            nn.Unflatten(1, (128, 6, 6)),  # [3, 128, 6, 6]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # [3, 64, 12, 12]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [3, 32, 24, 24]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 15, kernel_size=3, stride=2, padding=1, output_padding=1),  # [3, 15, 48, 48]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded