import torch
import torch_geometric
from torch.nn import functional as fcl

# from torch_sparse import SparseTensor


def construct_dense_relu_network(
    sizes, use_batchnorm=True, final_relu=False, dropout=0
):
    lst = []
    for i in range(len(sizes) - 1):
        if dropout:
            lst.append(torch.nn.Dropout(dropout))
        lst.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if use_batchnorm:
            lst.append(torch.nn.BatchNorm1d(sizes[i + 1]))
        lst.append(torch.nn.ReLU())
    if not final_relu:
        lst = lst[:-1]

    return torch.nn.Sequential(*lst)


class DenseReluGMMConvNetwork(torch.nn.Module):
    def __init__(
        self,
        sizes,
        use_batchnorm=True,
        final_relu=False,
        dropout=0,
        include_skip_connections=True,
        **gmmargs,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.final_relu = final_relu
        self.dropout = dropout
        self.include_skip_connections = include_skip_connections

        # construct a bunch of gmms
        lst = []
        for i in range(len(sizes) - 1):
            gmmc = torch_geometric.nn.GMMConv(sizes[i], sizes[i + 1], **gmmargs)
            lst.append(gmmc)
        self.gmms = torch.nn.ModuleList(lst)

        # and some linears ("self-edges")
        lst = []
        for j in range(len(sizes) - 1):
            lst.append(torch.nn.Linear(sizes[j], sizes[j + 1], bias=False))
        self.linears = torch.nn.ModuleList(lst)

        if self.dropout:
            self.dropouts = torch.nn.Dropout(self.dropout)

        # construct batchnorm layers we need
        if use_batchnorm:
            self.batchnorms = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(s) for s in sizes[1:]]
            )

    def forward(self, vals, edges, pseudo, multi_gpu=True):
        orig_vals = vals
        num_layers = len(self.gmms)
        for i, (dense, gmmlayer) in enumerate(zip(self.linears, self.gmms)):

            # move all the layers to the appropriate device
            gpu_to_use = f"cuda:{int(i // (num_layers / 2))}" if multi_gpu else "cuda:0"
            vals = vals.to(gpu_to_use)
            edges = edges.to(gpu_to_use)
            pseudo = pseudo.to(gpu_to_use)
            gmmlayer = gmmlayer.to(gpu_to_use)
            dense = dense.to(gpu_to_use)
            self.batchnorms[i] = self.batchnorms[i].to(gpu_to_use)

            if self.dropout:
                vals = self.dropouts(vals).to(gpu_to_use)
            # adj = SparseTensor(row=edges[0], col=edges[1], value=pseudo)
            # vals = gmmlayer(vals, adj.t()) + dense(vals)
            vals = gmmlayer(vals, edges, pseudo) + dense(vals)
            if (
                i == len(list(enumerate(zip(self.linears, self.gmms)))) - 2
                and self.include_skip_connections
            ):
                vals = torch.cat([orig_vals, vals], axis=1)

            # do batchnorm
            if self.use_batchnorm:
                vals = self.batchnorms[i](vals.float().to(gpu_to_use))

            # do relu (or not, if final_relu=False and we're on the last layer)
            if self.final_relu or (i != len(self.gmms) - 1):
                vals = fcl.relu(vals.float()).to(gpu_to_use)

        return vals
