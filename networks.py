import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, latent_size=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, x):
        return self.encoder(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_size=32, hidden_size=128, output_size=2):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.decoder(x)



class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO (student): If you want to use a UNet, you may use this class
    
    def forward(self, inputs):
        batch = inputs.shape[0]
        inputs = inputs.reshape(batch, 1, 32, 32)
        outputs = inputs
        # TODO (student): If you want to use a UNet, you may use this class
        outputs = outputs.reshape(batch, -1)
        return outputs
