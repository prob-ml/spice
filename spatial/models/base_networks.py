import torch
import torch_geometric
from torch.nn import functional as fcl


def construct_dense_relu_network(
    sizes, use_batchnorm=True, final_relu=False, dropout=0, input_dropout_only=True
):
    lst = []
    if input_dropout_only:
        lst.append(torch.nn.Dropout(dropout))
    for i in range(len(sizes) - 1):
        if dropout and not input_dropout_only:
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
        input_dropout_only=True,
        **gmmargs
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.final_relu = final_relu
        self.dropout = dropout
        self.input_dropout_only = input_dropout_only

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

        # construct batchnorm layers we need
        if use_batchnorm:
            self.batchnorms = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(s) for s in sizes[1:]]
            )

    def forward(self, vals, edges, pseudo):
        if self.input_dropout_only:
            dropout_layer = torch.nn.Dropout(self.dropout)
            vals = dropout_layer(vals)
        for i, (dense, gmmlayer) in enumerate(zip(self.linears, self.gmms)):
            if self.dropout and not self.input_dropout_only:
                dropout_layer = torch.nn.Dropout(self.dropout)
                vals = dropout_layer(vals)
            vals = gmmlayer(vals, edges, pseudo) + dense(vals)

            # do batchnorm
            if self.use_batchnorm:
                vals = self.batchnorms[i](vals)

            # do relu (or not, if final_relu=False and we're on the last layer)
            if self.final_relu or (i != len(self.gmms) - 1):
                vals = fcl.relu(vals)

        return vals
