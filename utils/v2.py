import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, H/16, W/16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (B, 256, H/32, W/32)
            nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, H/16, W/16)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (B, 128, H/8, W/8)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (B, 64, H/2, W/2)
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 3, H, W)
            nn.Sigmoid()  # For normalized pixel values between 0 and 1
        )

    def forward(self, x):
        return self.decoder(x)
    
    


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    

class ExtendedEncoder(nn.Module):
    def __init__(self, base_encoder):
        super(ExtendedEncoder, self).__init__()
        self.base_encoder = base_encoder

        # Dynamically compute flatten_dim
        dummy_input = torch.randn(1, 3, 256, 256).to(device)  # Simulate input
        dummy_output = self.base_encoder(dummy_input)  # Get encoder output
        self.flatten_dim = dummy_output.view(1, -1).size(1)  # Flatten dimension
        
        # Fully connected layer
        self.fc = nn.Linear(self.flatten_dim, 251)
        self.softmax = nn.Softmax(dim=1)
        
        for param in self.base_encoder.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.base_encoder(x)  # Base encoder
        x = x.view(x.size(0),-1)  # Flatte
        x = self.fc(x)  # Fully connected layer
        x = self.softmax(x)
        return x
    