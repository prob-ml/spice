import torch
import torch_geometric
from torch.nn import functional as fcl


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
        self.sizes = sizes
        self.use_batchnorm = use_batchnorm
        self.final_relu = final_relu
        self.dropout = dropout
        self.include_skip_connections = include_skip_connections

        # construct a bunch of gmms
        lst = []
        for i in range(len(sizes) - 1):
            gmmc = torch_geometric.nn.GMMConv(
                self.sizes[i], self.sizes[i + 1], **gmmargs
            )
            lst.append(gmmc)
        self.gmms = torch.nn.ModuleList(lst)

        # and some linears ("self-edges")
        lst = []
        for j in range(len(sizes) - 1):
            lst.append(torch.nn.Linear(self.sizes[j], self.sizes[j + 1], bias=False))
        self.linears = torch.nn.ModuleList(lst)

        if self.dropout:
            self.dropouts = torch.nn.Dropout(self.dropout)

        # construct batchnorm layers we need
        if use_batchnorm:
            self.batchnorms = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(s) for s in self.sizes[1:]]
            )

    def forward(self, vals, edges, pseudo, num_gpus=4):
        orig_vals = []
        num_layers = len(self.gmms)
        for i, (dense, gmmlayer) in enumerate(zip(self.linears, self.gmms)):
            # move all the layers to the appropriate device
            gpu_to_use = (
                f"cuda:{int(i // (num_layers / num_gpus))}"
                if num_gpus > 1
                else "cuda:0"
            )
            # create an identity copy if we need a residual block
            vals = vals.to(gpu_to_use)
            if self.include_skip_connections:
                orig_vals.append(vals.clone())
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
            if i >= 2 and self.include_skip_connections:
                for j in range(i + 1):
                    if self.sizes[j] == self.sizes[i + 1]:
                        vals += orig_vals[j].to(gpu_to_use)

            # do batchnorm
            if self.use_batchnorm:
                vals = self.batchnorms[i](vals.float().to(gpu_to_use))

            # do relu (or not, if final_relu=False and we're on the last layer)
            if self.final_relu or (i != len(self.gmms) - 1):
                vals = fcl.relu(vals.float()).to(gpu_to_use)

        return vals
