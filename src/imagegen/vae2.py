# implement VAE per Generative Modeling book p101
import numpy as np
import torch
import torch.nn as nn

import imagegen.probfn as probfn


class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        self.encoder_net = encoder_net

    @staticmethod
    def reparamaterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def encode(self, x):
        h_e = self.encoder_net(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e
    
    def sample(self, x=None, mu_e=None, log_var_e=None):
        """We may want a sample around x (assumed to be a batch), 
        call sample(x=x). Or we may know mu_e and log_var_e and want
        a sample around that."""
        if (mu_e is None) and (log_var_e is None):
            assert x is not None
            mu_e, log_var_e = self.encode(x=x)
        assert (mu_e is not None) and (log_var_e is not None)
        z = self.reparamaterization(mu=mu_e, log_var=log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        """The log prob of a sampled encoded x, a z. 
        First encode it. Then sample. Now the
        sample is gauassian - use log_normal_diag. 
        
        If not getting x, then do log_prob based on mu, var and z"""
        if x is not None:
            assert (mu_e is None) and (log_var_e is None) and (z is None)
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)

        assert (mu_e is not None) and (log_var_e is not None) and (z is not None)

        # note we pass z in for x. mu, var are in decoded latent space.
        # so if we had a batch of x or encoded/latent mu/var, we would get the 
        # log prob of the laten representations - 
        return probfn.log_normal_diag(x=z, mu=mu_e, log_var=log_var_e)

    def forward(self, x, type='log_prob'):
        if type == 'log_prob':
            return self.log_prob(x)
        
        if type == 'encode':
            return self.sample(x)
        
        raise ValueError("Unknown type={type}")
    

class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_values=256):
        """The decoder_net goes from latent D to categorical over M, or Bernoulli 
        over M. So it will go to M * num_values if categorical, or M if Bernoulli."""
        super(Decoder, self).__init__()
        self.decoder_net = decoder_net
        assert distribution in ['categorical', 'bernoulli'], 'Distribution should be categorical or bernoulli'
        self.distribution = distribution
        self.num_values = num_values
        if distribution == 'bernoulli':
            assert num_values is None, 'num_values should be 1 for Bernoulli'

    def decode(self, z):
        """returns list - but just 1 element long now.
        Entry depends on distribution. For categorical, batch x num_pixels x num_values
        and num_values is for multinomial sampling, or cross entropy loss
        
        for bernouli, batch x num_pixels - just prob of being on"""
        h_d = self.decoder_net(z)
        if self.distribution == 'categorical':
            batch_size = h_d.size()[0]            
            x_dim = h_d.size()[1] // self.num_values
            h_d = h_d.view(batch_size, x_dim, self.num_values)
            mu_d = torch.softmax(h_d, dim=2)
            return [mu_d]
        elif self.distribution == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            return [mu_d]

    def sample(self, z):
        outs = self.decode(z)
        mu_d = outs[0]

        if self.distribution == 'categorical':
            batch_size, MxL = mu_d.size()
            M = MxL // self.num_values
            mu_d = mu_d.view(batch_size, -1, self.num_values)
            M = mu_d.size()[1]
            mu_d = mu_d.view(-1, self.num_values)
            x_new = torch.multinomial(mu_d, 1).view(batch_size, M)
        elif self.distribution == 'bernoulli':
            x_new = torch.bernoulli(mu_d)

        return x_new
    
    def log_prob(self, x, z):
        """this is the reconstruction loss. Given z, get probabilities of x - calculate
        log_prob of x given z - cross entropy loss"""
        outs = self.decode(z)
        mu_d = outs[0]

        if self.distribution == 'categorical':
            log_p = probfn.log_categorical(
                x=x, mu=mu_d, num_classes=self.num_vals, reduction='sum', dim=-1
                ).sum(-1)
            
        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = probfn.log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(z)


class Prior(nn.Module):
        def __init__(self, L):
            super(Prior, self).__init__()
            self.L = L

        def sample(self, batch_size):
            z = torch.randn(batch_size, self.L)
            return z
        
        def log_prob(self, z):
            return probfn.log_standard_normal(z)


class VAE(nn.Module):
    def __init__(self, encoder_net, decoder_net, num_values=16, L=16, likelihood_type='categorical'):
        super(VAE, self).__init__()
        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(decoder_net=decoder_net, distribution=likelihood_type, num_values=num_values)
        self.prior = Prior(L=L)
        self.likelihood_type = likelihood_type
        self.num_values = num_values
        self.likelihood_type = likelihood_type

    def forward(self, x, reduction='avg'):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # ELBO
        RE = self.cdoder.lop_prob(x, z)

        KL = self.prior.log_prob(z) - \
            self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z).sum(-1)
        
        if reduction == 'sum':
            return -(RE + KL).sum()
        else:
            return -(RE + KL).mean()
        
    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)
    
               

def get_encoder_decoder(dim_X=218*178*3, dim_H=256, dim_Z=20, num_pixel_values=256, device='mps'):
    encoder = nn.Sequential(
        nn.Linear(dim_X, dim_H),
        nn.LeakyReLU(),
        nn.Linear(dim_H, dim_H),
        nn.LeakyReLU(),
        nn.Linear(dim_H, 2 * dim_Z)
    ).to(device=device)

    decoder = nn.Sequential(
        nn.Linear(dim_Z, dim_H),
        nn.LeakyReLU(),
        nn.Linear(dim_H, dim_H),
        nn.LeakyReLU(),
        nn.Linear(dim_H, dim_X * num_pixel_values)
    ).to(device=device)
    return encoder, decoder    


def run():
    import imagegen.data as data
    
    dl = data.get_celeb_ds()
    en, dec =  = Encoder().to('mps')
    dec = Decoder().to('mps')
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
