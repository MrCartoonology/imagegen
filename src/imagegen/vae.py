# implement VAE per Generative Modeling book p101
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, width=218, height=178, channels=3, latent_dim=20):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.latent_dim = latent_dim
        self.input_dim = width * height * channels

        self.L1 = nn.Linear(in_features=self.input_dim, out_features=256)
        self.L2 = nn.Linear(in_features=256, out_features=2 * latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.L1(x)
        x = nn.LeakyReLU()(x)
        x = self.L2(x)
        z_mu = x[:, 0:self.latent_dim]
        z_logvar = x[:, self.latent_dim:]
        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=20, width=218, height=178, channels=3, pixel_values=256):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.latent_dim = latent_dim
        self.output_dim = width * height * channels
        self.pixel_values = pixel_values

        self.L1 = nn.Linear(in_features=self.latent_dim, out_features=256)
        self.L2 = nn.Linear(in_features=256, out_features=pixel_values * self.output_dim)

    def forward(self, z):
        z = self.L1(z)
        z = nn.LeakyReLU()(z)
        z = self.L2(z)
        x_logits = z.reshape(-1, self.output_dim, self.pixel_values)
        return x_logits




class VAE:
    def __init__(self, W=218, H=178, C=3, dim_Z=20, num_pixel_values=256, device='mps'):
        dim_X = W * H * C
        dim_H = 256
        
        self.encoder = nn.Sequential(
            nn.Linear(dim_X, dim_H),
            nn.LeakyReLU(),
            nn.Linear(dim_H, dim_H),
            nn.LeakyReLU(),
            nn.Linear(dim_H, 2 * dim_Z)
        ).to(device=device)

        self.decoder = nn.Sequential(
            nn.Linear(dim_Z, dim_H),
            nn.LeakyReLU(),
            nn.Linear(dim_H, dim_H),
            nn.LeakyReLU(),
            nn.Linear(dim_H, dim_X * num_pixel_values)
        ).to(device=device)





def run():
    import imagegen.data as data
    
#    dl = data.get_celeb_ds()
    dl = data.get_digits_ds()
    en, dec = get_encoder_decoder(dim_X=16, dim_H=4, dim_Z=2, num_pixel_values=16, device='mps'))
    for x in dl:
        x = x.to('mps')
        z_mu, z_logvar = en(x)
        var = torch.exp(z_logvar)
        eps = torch.randn_like(var)
        z_sample = z_mu + eps * var

        x_logits = dec(z_sample)
        break

if __name__ == '__main__':
    run()
