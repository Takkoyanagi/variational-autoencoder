import numpy as np
import torch
from torchvision import transforms, datasets


class VAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        num_layers: int = 2,
        hidden_layer: int = 400,
        zdim: int = 20,
    ) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        divisor = [2**power for power in range(0, num_layers)]
        hidden_layers = [int(hidden_layer / div) for div in divisor]
        assert zdim < hidden_layers[-1], "bottle neck layer is bigger than last hidden layer"

        self.encoder_layers = torch.nn.ModuleList()
        start_dim = input_dim

        for h_layer in hidden_layers:
            self.encoder_layers.append(torch.nn.Linear(start_dim, h_layer))
            start_dim = h_layer
        self._u = torch.nn.Linear(start_dim, zdim)
        self.logvar = torch.nn.Linear(start_dim, zdim)

        self.decoder_layers = torch.nn.ModuleList()
        start_dim = zdim
        for h_layer in hidden_layers[::-1]:
            self.decoder_layers.append(torch.nn.Linear(start_dim, h_layer))
            start_dim = h_layer
        self.decoder_layers.append(torch.nn.Linear(start_dim, input_dim))

        self.initilialize_weights_biases()

    def initilialize_weights_biases(self) -> None:
        for layer in self.encoder_layers + self.decoder_layers + [self._u, self.logvar]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def encode(self, x: torch.Tensor) -> tuple:
        for layer in self.encoder_layers:
            x = torch.relu(layer(x))
        _u = self._u(x)
        logvar = self.logvar(x)
        return (_u, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers[:-1]:
            z = torch.relu(layer(z))
        z = torch.sigmoid(self.decoder_layers[-1](z))
        return z

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.view(-1, self.input_dim)
        _u, logvar = self.encode(x)
        stdev = torch.exp(0.5 * logvar)
        noise = torch.randn_like(stdev)
        z = _u + (noise * stdev)
        z = self.decode(z)
        return (z, _u, logvar)


def cus_loss_func(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    _u: torch.Tensor,
    logvar: torch.Tensor,
    input_dim: int,
) -> torch.Tensor:
    # https://arxiv.org/abs/1312.6114
    # KLD = 0.5 * sum(1 + log(sigma^2) - u^2 - sigma^2)
    bce = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - _u.pow(2) - logvar.exp())
    return bce + kld


device = torch.device("cuda")
torch.manual_seed(1)
np.random.seed(1)

# 1. load data
batch_size = 128
input_dim = 784
trfm = transforms.transforms.ToTensor()
train_ds = datasets.MNIST(".\\Data", train=True, download=True, transform=trfm)
train_ldr = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 2. create model
vae = VAE(input_dim=input_dim, num_layers=3, hidden_layer=512, zdim=20).to(device)

# 3. train model
max_epochs = 20
lrn_rate = 0.001
optimizer = torch.optim.Adam(vae.parameters(), lr=lrn_rate)

print("\nbatch_size = %3d " % batch_size)
print("loss = custom BCE plus KLD ")
print("optimizer = Adam")
print("max_epochs = %3d " % max_epochs)
print("lrn_rate = %0.3f " % lrn_rate)

for epoch in range(0, max_epochs):
    vae = vae.train()
    torch.manual_seed(1 + epoch)
    train_loss = 0.0
    num_lines_read = 0
    for batch_idx, (data, _) in enumerate(train_ldr):
        data = data.to(device)
        optimizer.zero_grad()
        recon_x, u, logvar = vae(data)
        loss_val = cus_loss_func(recon_x, data, u, logvar, input_dim)
        loss_val.backward()
        train_loss += loss_val.item()
        optimizer.step()
    print(f"Epoch {epoch+1}/{max_epochs}, Loss: {train_loss/len(train_ds)}")
