import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GMMConv
from torch_geometric.nn import avg_pool
from torch_geometric.nn import max_pool
from torch_geometric.nn import graclus
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.nn import max_pool_x
from torch_geometric.nn import global_mean_pool
from pytorch_lightning.callbacks import ModelCheckpoint

class MonetAutoencoder(pl.LightningModule):

    def __init__(self, in_dim, hidden_dims, out_dim, pseudo_dim=2, kernel_size=25):
        super().__init__()
        # Initializing Attributes
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.pseudo_dim = pseudo_dim
        self.k = kernel_size

        self.conv1enc = GMMConv(1, 50, dim=2, kernel_size=25)
        self.conv2enc = GMMConv(50, 25, dim=2, kernel_size=25)
        self.conv3enc = GMMConv(25, 10, dim=2, kernel_size=25)

        self.conv1dec = GMMConv(10, 25, dim=2, kernel_size=25)
        self.conv2dec = GMMConv(25, 50, dim=2, kernel_size=25)
        self.conv3dec = GMMConv(50, 1, dim=2, kernel_size=25)

        self.batchnorm1enc = torch.nn.BatchNorm1d(50)
        self.batchnorm2enc = torch.nn.BatchNorm1d(25)
        self.batchnorm3enc = torch.nn.BatchNorm1d(10)

        self.batchnorm1dec = torch.nn.BatchNorm1d(25)
        self.batchnorm2dec = torch.nn.BatchNorm1d(50)
        self.batchnorm3dec = torch.nn.BatchNorm1d(1)

        self.fc1 = nn.Linear(10, 10)
        # Ensures that the hyperparameters are saveable.
        self.save_hyperparameters()

    def pseudo(self, edge_index, pos=None, coord="deg"):
        if coord == "deg":
            coord1 = torch.tensor(degree(edge_index[0])[edge_index[0]]).unsqueeze(-1)
            coord2 = torch.tensor(degree(edge_index[1])[edge_index[1]]).unsqueeze(-1)
            return torch.cat((1 / torch.sqrt(coord1), 1 / torch.sqrt(coord2)), dim=1)
        if coord == "polar":
            coord1 = pos[edge_index[0]]
            coord2 = pos[edge_index[1]]
            edge_dir = coord2 - coord1
            rho = torch.sqrt(edge_dir[:, 0] ** 2 + edge_dir[:, 1] ** 2).unsqueeze(-1)
            theta = torch.atan2(edge_dir[:, 1], edge_dir[:, 0]).unsqueeze(-1)
            return torch.cat((rho, theta), dim=1)

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.to(device)
        data.input = data.x.clone()

        # Encoder
        value = self.pseudo(data.edge_index, pos=data.pos, coord="polar").cuda()

        data.x = F.relu(self.batchnorm1enc(self.conv1enc(data.x, data.edge_index, value)))

        data.x = F.relu(self.batchnorm2enc(self.conv2enc(data.x, data.edge_index, value)))

        data.x = self.fc1(F.relu(self.batchnorm3enc(self.conv3enc(data.x, data.edge_index, value))))

        # Decoder

        data.x = F.relu(self.batchnorm1dec(self.conv1dec(data.x, data.edge_index, value)))

        data.x = F.relu(self.batchnorm2dec(self.conv2dec(data.x, data.edge_index, value)))

        data.x = F.relu(self.batchnorm3dec(self.conv3dec(data.x, data.edge_index, value)))

        return data.x

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.mse_loss(out, batch.input.cuda())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.mse_loss(out, batch.input.cuda())
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=0.001)

    def setup(self, stage=None):
        train75_loader = MNISTSuperpixels("GNN Data", train=True)
        # train75_loader = MNISTSuperpixels("~/Monet/GNN Data", train=True)
        self.mnist_train, self.mnist_val = random_split(train75_loader, [55000, 5000])
        # self.mnist_test = MNISTSuperpixels("~/Monet/GNN Data", train=False)
        self.mnist_test = MNISTSuperpixels("GNN Data", train=False)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=10, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=10, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=10, shuffle=False)

model = MonetAutoencoder(1, 10, 10)
checkpoint = ModelCheckpoint(filepath='checkpoints/',
                             monitor='val_loss',
                             mode='min',
                             save_top_k=20)
trainer = pl.Trainer(checkpoint_callback=checkpoint, gpus=1, auto_select_gpus=True,
                     max_epochs=1000, progress_bar_refresh_rate=100)
trainer.fit(model)
trainer.test()
