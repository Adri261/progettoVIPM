import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoderL1(nn.Module):
    def __init__(self, sparsity_weight=1e-5):
        super(SparseAutoencoderL1, self).__init__()
        self.sparsity_weight = sparsity_weight
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (C, H, W) -> (16, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, H/2, W/2) -> (16, H/4, W/4)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (16, H/4, W/4) -> (32, H/8, W/8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16, H/8, W/8) -> (16, H/16, W/16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, H/16, W/16) -> (64, H/32, W/32)
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, H/32, W/32) -> (32, H/16, W/16)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (32, H/16, W/16) -> (32, H/8, W/8)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, H/8, W/8) -> (16, H/4, W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (16, H/4, W/4) -> (16, H/2, W/2)
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, H/2, W/2) -> (3, H, W)
            nn.Sigmoid(),  # Output in range [0, 1]
        )
    
    def forward(self, x):
        # Forward pass through encoder
        latent = self.encoder(x)
        
        # Compute sparsity loss
        self.sparsity_loss = self.compute_sparsity_loss(latent)
        
        # Forward pass through decoder
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def compute_sparsity_loss(self, latent):
        # Compute L1 norm of latent representation (encouraging sparsity)
        return self.sparsity_weight * torch.mean(torch.abs(latent))