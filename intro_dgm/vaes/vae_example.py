{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "from sklearn.datasets import load_digits\n",
    "from sklearn import datasets\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "\n",
    "from pytorch_model_summary import summary"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**DISCLAIMER**\n",
    "\n",
    "The presented code is not optimized, it serves an educational purpose. It is written for CPU, it uses only fully-connected networks and an extremely simplistic dataset. However, it contains all components that can help to understand how a Variational Auto-Encoder (VAE) works, and it should be rather easy to extend it to more sophisticated models. This code could be run almost on any laptop/PC, and it takes a couple of minutes top to get the result."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this example, we go wild and use a dataset that is simpler than MNIST! We use a scipy dataset called Digits. It consists of ~1500 images of size 8x8, and each pixel can take values in $\\{0, 1, \\ldots, 16\\}$.\n",
    "\n",
    "The goal of using this dataset is that everyone can run it on a laptop, without any gpu etc."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Digits(Dataset):\n",
    "    \"\"\"Scikit-Learn Digits dataset.\"\"\"\n",
    "\n",
    "    def __init__(self, mode='train', transforms=None):\n",
    "        digits = load_digits()\n",
    "        if mode == 'train':\n",
    "            self.data = digits.data[:1000].astype(np.float32)\n",
    "        elif mode == 'val':\n",
    "            self.data = digits.data[1000:1350].astype(np.float32)\n",
    "        else:\n",
    "            self.data = digits.data[1350:].astype(np.float32)\n",
    "\n",
    "        self.transforms = transforms\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.data)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        sample = self.data[idx]\n",
    "        if self.transforms:\n",
    "            sample = self.transforms(sample)\n",
    "        return sample"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### VAE code"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Please see the blogpost for details."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Probability distributions**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "PI = torch.from_numpy(np.asarray(np.pi))\n",
    "EPS = 1.e-5\n",
    "\n",
    "def log_categorical(x, p, num_classes=256, reduction=None, dim=None):\n",
    "    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)\n",
    "    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))\n",
    "    if reduction == 'avg':\n",
    "        return torch.mean(log_p, dim)\n",
    "    elif reduction == 'sum':\n",
    "        return torch.sum(log_p, dim)\n",
    "    else:\n",
    "        return log_p\n",
    "\n",
    "def log_bernoulli(x, p, reduction=None, dim=None):\n",
    "    pp = torch.clamp(p, EPS, 1. - EPS)\n",
    "    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)\n",
    "    if reduction == 'avg':\n",
    "        return torch.mean(log_p, dim)\n",
    "    elif reduction == 'sum':\n",
    "        return torch.sum(log_p, dim)\n",
    "    else:\n",
    "        return log_p\n",
    "\n",
    "def log_normal_diag(x, mu, log_var, reduction=None, dim=None):\n",
    "    D = x.shape[1]\n",
    "    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.\n",
    "    if reduction == 'avg':\n",
    "        return torch.mean(log_p, dim)\n",
    "    elif reduction == 'sum':\n",
    "        return torch.sum(log_p, dim)\n",
    "    else:\n",
    "        return log_p\n",
    "\n",
    "\n",
    "def log_standard_normal(x, reduction=None, dim=None):\n",
    "    D = x.shape[1]\n",
    "    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.\n",
    "    if reduction == 'avg':\n",
    "        return torch.mean(log_p, dim)\n",
    "    elif reduction == 'sum':\n",
    "        return torch.sum(log_p, dim)\n",
    "    else:\n",
    "        return log_p"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Encoder**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder(nn.Module):\n",
    "    def __init__(self, encoder_net):\n",
    "        super(Encoder, self).__init__()\n",
    "\n",
    "        self.encoder = encoder_net\n",
    "\n",
    "    @staticmethod\n",
    "    def reparameterization(mu, log_var):\n",
    "        std = torch.exp(0.5*log_var)\n",
    "\n",
    "        eps = torch.randn_like(std)\n",
    "\n",
    "        return mu + std * eps\n",
    "\n",
    "    def encode(self, x):\n",
    "        h_e = self.encoder(x)\n",
    "        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)\n",
    "\n",
    "        return mu_e, log_var_e\n",
    "\n",
    "    def sample(self, x=None, mu_e=None, log_var_e=None):\n",
    "        if (mu_e is None) and (log_var_e is None):\n",
    "            mu_e, log_var_e = self.encode(x)\n",
    "        else:\n",
    "            if (mu_e is None) or (log_var_e is None):\n",
    "                raise ValueError('mu and log-var can`t be None!')\n",
    "        z = self.reparameterization(mu_e, log_var_e)\n",
    "        return z\n",
    "\n",
    "    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):\n",
    "        if x is not None:\n",
    "            mu_e, log_var_e = self.encode(x)\n",
    "            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)\n",
    "        else:\n",
    "            if (mu_e is None) or (log_var_e is None) or (z is None):\n",
    "                raise ValueError('mu, log-var and z can`t be None!')\n",
    "\n",
    "        return log_normal_diag(z, mu_e, log_var_e)\n",
    "\n",
    "    def forward(self, x, type='log_prob'):\n",
    "        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'\n",
    "        if type == 'log_prob':\n",
    "            return self.log_prob(x)\n",
    "        else:\n",
    "            return self.sample(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Decoder**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Decoder(nn.Module):\n",
    "    def __init__(self, decoder_net, distribution='categorical', num_vals=None):\n",
    "        super(Decoder, self).__init__()\n",
    "\n",
    "        self.decoder = decoder_net\n",
    "        self.distribution = distribution\n",
    "        self.num_vals=num_vals\n",
    "\n",
    "    def decode(self, z):\n",
    "        h_d = self.decoder(z)\n",
    "\n",
    "        if self.distribution == 'categorical':\n",
    "            b = h_d.shape[0]\n",
    "            d = h_d.shape[1]//self.num_vals\n",
    "            h_d = h_d.view(b, d, self.num_vals)\n",
    "            mu_d = torch.softmax(h_d, 2)\n",
    "            return [mu_d]\n",
    "\n",
    "        elif self.distribution == 'bernoulli':\n",
    "            mu_d = torch.sigmoid(h_d)\n",
    "            return [mu_d]\n",
    "        \n",
    "        else:\n",
    "            raise ValueError('Either `categorical` or `bernoulli`')\n",
    "\n",
    "    def sample(self, z):\n",
    "        outs = self.decode(z)\n",
    "\n",
    "        if self.distribution == 'categorical':\n",
    "            mu_d = outs[0]\n",
    "            b = mu_d.shape[0]\n",
    "            m = mu_d.shape[1]\n",
    "            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)\n",
    "            p = mu_d.view(-1, self.num_vals)\n",
    "            x_new = torch.multinomial(p, num_samples=1).view(b, m)\n",
    "\n",
    "        elif self.distribution == 'bernoulli':\n",
    "            mu_d = outs[0]\n",
    "            x_new = torch.bernoulli(mu_d)\n",
    "        \n",
    "        else:\n",
    "            raise ValueError('Either `categorical` or `bernoulli`')\n",
    "\n",
    "        return x_new\n",
    "\n",
    "    def log_prob(self, x, z):\n",
    "        outs = self.decode(z)\n",
    "\n",
    "        if self.distribution == 'categorical':\n",
    "            mu_d = outs[0]\n",
    "            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)\n",
    "            \n",
    "        elif self.distribution == 'bernoulli':\n",
    "            mu_d = outs[0]\n",
    "            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)\n",
    "            \n",
    "        else:\n",
    "            raise ValueError('Either `categorical` or `bernoulli`')\n",
    "\n",
    "        return log_p\n",
    "\n",
    "    def forward(self, z, x=None, type='log_prob'):\n",
    "        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'\n",
    "        if type == 'log_prob':\n",
    "            return self.log_prob(x, z)\n",
    "        else:\n",
    "            return self.sample(z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Prior**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Prior(nn.Module):\n",
    "    def __init__(self, L):\n",
    "        super(Prior, self).__init__()\n",
    "        self.L = L\n",
    "\n",
    "    def sample(self, batch_size):\n",
    "        z = torch.randn((batch_size, self.L))\n",
    "        return z\n",
    "\n",
    "    def log_prob(self, z):\n",
    "        return log_standard_normal(z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Full VAE**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class VAE(nn.Module):\n",
    "    def __init__(self, encoder_net, decoder_net, num_vals=256, L=16, likelihood_type='categorical'):\n",
    "        super(VAE, self).__init__()\n",
    "\n",
    "        print('VAE by JT.')\n",
    "\n",
    "        self.encoder = Encoder(encoder_net=encoder_net)\n",
    "        self.decoder = Decoder(distribution=likelihood_type, decoder_net=decoder_net, num_vals=num_vals)\n",
    "        self.prior = Prior(L=L)\n",
    "\n",
    "        self.num_vals = num_vals\n",
    "\n",
    "        self.likelihood_type = likelihood_type\n",
    "\n",
    "    def forward(self, x, reduction='avg'):\n",
    "        # encoder\n",
    "        mu_e, log_var_e = self.encoder.encode(x)\n",
    "        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)\n",
    "\n",
    "        # ELBO\n",
    "        RE = self.decoder.log_prob(x, z)\n",
    "        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)\n",
    "\n",
    "        if reduction == 'sum':\n",
    "            return -(RE + KL).sum()\n",
    "        else:\n",
    "            return -(RE + KL).mean()\n",
    "\n",
    "    def sample(self, batch_size=64):\n",
    "        z = self.prior.sample(batch_size=batch_size)\n",
    "        return self.decoder.sample(z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Auxiliary functions: training, evaluation, plotting"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It's rather self-explanatory, isn't it?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluation(test_loader, name=None, model_best=None, epoch=None):\n",
    "    # EVALUATION\n",
    "    if model_best is None:\n",
    "        # load best performing model\n",
    "        model_best = torch.load(name + '.model')\n",
    "\n",
    "    model_best.eval()\n",
    "    loss = 0.\n",
    "    N = 0.\n",
    "    for indx_batch, test_batch in enumerate(test_loader):\n",
    "        loss_t = model_best.forward(test_batch, reduction='sum')\n",
    "        loss = loss + loss_t.item()\n",
    "        N = N + test_batch.shape[0]\n",
    "    loss = loss / N\n",
    "\n",
    "    if epoch is None:\n",
    "        print(f'FINAL LOSS: nll={loss}')\n",
    "    else:\n",
    "        print(f'Epoch: {epoch}, val nll={loss}')\n",
    "\n",
    "    return loss\n",
    "\n",
    "\n",
    "def samples_real(name, test_loader):\n",
    "    # REAL-------\n",
    "    num_x = 4\n",
    "    num_y = 4\n",
    "    x = next(iter(test_loader)).detach().numpy()\n",
    "\n",
    "    fig, ax = plt.subplots(num_x, num_y)\n",
    "    for i, ax in enumerate(ax.flatten()):\n",
    "        plottable_image = np.reshape(x[i], (8, 8))\n",
    "        ax.imshow(plottable_image, cmap='gray')\n",
    "        ax.axis('off')\n",
    "\n",
    "    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')\n",
    "    plt.close()\n",
    "\n",
    "\n",
    "def samples_generated(name, data_loader, extra_name=''):\n",
    "    x = next(iter(data_loader)).detach().numpy()\n",
    "\n",
    "    # GENERATIONS-------\n",
    "    model_best = torch.load(name + '.model')\n",
    "    model_best.eval()\n",
    "\n",
    "    num_x = 4\n",
    "    num_y = 4\n",
    "    x = model_best.sample(num_x * num_y)\n",
    "    x = x.detach().numpy()\n",
    "\n",
    "    fig, ax = plt.subplots(num_x, num_y)\n",
    "    for i, ax in enumerate(ax.flatten()):\n",
    "        plottable_image = np.reshape(x[i], (8, 8))\n",
    "        ax.imshow(plottable_image, cmap='gray')\n",
    "        ax.axis('off')\n",
    "\n",
    "    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')\n",
    "    plt.close()\n",
    "\n",
    "\n",
    "def plot_curve(name, nll_val):\n",
    "    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')\n",
    "    plt.xlabel('epochs')\n",
    "    plt.ylabel('nll')\n",
    "    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')\n",
    "    plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):\n",
    "    nll_val = []\n",
    "    best_nll = 1000.\n",
    "    patience = 0\n",
    "\n",
    "    # Main loop\n",
    "    for e in range(num_epochs):\n",
    "        # TRAINING\n",
    "        model.train()\n",
    "        for indx_batch, batch in enumerate(training_loader):\n",
    "            if hasattr(model, 'dequantization'):\n",
    "                if model.dequantization:\n",
    "                    batch = batch + torch.rand(batch.shape)\n",
    "            loss = model.forward(batch)\n",
    "\n",
    "            optimizer.zero_grad()\n",
    "            loss.backward(retain_graph=True)\n",
    "            optimizer.step()\n",
    "\n",
    "        # Validation\n",
    "        loss_val = evaluation(val_loader, model_best=model, epoch=e)\n",
    "        nll_val.append(loss_val)  # save for plotting\n",
    "\n",
    "        if e == 0:\n",
    "            print('saved!')\n",
    "            torch.save(model, name + '.model')\n",
    "            best_nll = loss_val\n",
    "        else:\n",
    "            if loss_val < best_nll:\n",
    "                print('saved!')\n",
    "                torch.save(model, name + '.model')\n",
    "                best_nll = loss_val\n",
    "                patience = 0\n",
    "\n",
    "                samples_generated(name, val_loader, extra_name=\"_epoch_\" + str(e))\n",
    "            else:\n",
    "                patience = patience + 1\n",
    "\n",
    "        if patience > max_patience:\n",
    "            break\n",
    "\n",
    "    nll_val = np.asarray(nll_val)\n",
    "\n",
    "    return nll_val"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initialize dataloaders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = Digits(mode='train')\n",
    "val_data = Digits(mode='val')\n",
    "test_data = Digits(mode='test')\n",
    "\n",
    "training_loader = DataLoader(train_data, batch_size=64, shuffle=True)\n",
    "val_loader = DataLoader(val_data, batch_size=64, shuffle=False)\n",
    "test_loader = DataLoader(test_data, batch_size=64, shuffle=False)\n",
    "\n",
    "result_dir = 'results/'\n",
    "if not(os.path.exists(result_dir)):\n",
    "    os.mkdir(result_dir)\n",
    "name = 'vae'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Hyperparams"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "D = 64   # input dimension\n",
    "L = 16  # number of latents\n",
    "M = 256  # the number of neurons in scale (s) and translation (t) nets\n",
    "\n",
    "lr = 1e-3 # learning rate\n",
    "num_epochs = 1000 # max. number of epochs\n",
    "max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initialize VAE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "likelihood_type = 'categorical'\n",
    "\n",
    "if likelihood_type == 'categorical':\n",
    "    num_vals = 17\n",
    "elif likelihood_type == 'bernoulli':\n",
    "    num_vals = 1\n",
    "\n",
    "encoder = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),\n",
    "                        nn.Linear(M, M), nn.LeakyReLU(),\n",
    "                        nn.Linear(M, 2 * L))\n",
    "\n",
    "decoder = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),\n",
    "                        nn.Linear(M, M), nn.LeakyReLU(),\n",
    "                        nn.Linear(M, num_vals * D))\n",
    "\n",
    "prior = torch.distributions.MultivariateNormal(torch.zeros(L), torch.eye(L))\n",
    "model = VAE(encoder_net=encoder, decoder_net=decoder, num_vals=num_vals, L=L, likelihood_type=likelihood_type)\n",
    "\n",
    "# Print the summary (like in Keras)\n",
    "print(\"ENCODER:\\n\", summary(encoder, torch.zeros(1, D), show_input=False, show_hierarchical=False))\n",
    "print(\"\\nDECODER:\\n\", summary(decoder, torch.zeros(1, L), show_input=False, show_hierarchical=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's play! Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# OPTIMIZER\n",
    "optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training procedure\n",
    "nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,\n",
    "                       training_loader=training_loader, val_loader=val_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_loss = evaluation(name=result_dir + name, test_loader=test_loader)\n",
    "f = open(result_dir + name + '_test_loss.txt', \"w\")\n",
    "f.write(str(test_loss))\n",
    "f.close()\n",
    "\n",
    "samples_real(result_dir + name, test_loader)\n",
    "\n",
    "plot_curve(result_dir + name, nll_val)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
