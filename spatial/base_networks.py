import torch
import torch.nn.functional as F
import torch_geometric


def construct_dense_relu_network(sizes, use_batchnorm=True, final_relu=False):
    lst = []
    for i in range(len(sizes) - 1):
        lst.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if use_batchnorm:
            lst.append(torch.nn.BatchNorm1d(sizes[i + 1]))
        lst.append(torch.nn.ReLU())
    if not final_relu:
        lst = lst[:-1]

    return torch.nn.Sequential(*lst)


class DenseReluGMMConvNetwork(torch.nn.Module):
    def __init__(self, sizes, use_batchnorm=True, final_relu=False, **gmmargs):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.final_relu = final_relu

        # construct a bunch of gmms
        lst = []
        for i in range(len(sizes) - 1):
            lst.append(
                torch_geometric.nn.GMMConv(
                    sizes[i],
                    sizes[i + 1],
                    **gmmargs,
                )
            )
        self.gmms = torch.nn.ModuleList(lst)

        # and some linears ("self-edges")
        lst = []
        for i in range(len(sizes) - 1):
            lst.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=False))
        self.linears = torch.nn.ModuleList(lst)

        # construct batchnorm layers we need
        if use_batchnorm:
            self.batchnorms = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(s) for s in sizes[1:]]
            )

    def forward(self, vals, edges, pseudo):
        for i, (dense, gmmlayer) in enumerate(zip(self.linears, self.gmms)):
            vals = gmmlayer(vals, edges, pseudo) + dense(vals)

            # do batchnorm
            if self.use_batchnorm:
                vals = self.batchnorms[i](vals)

            # do relu (or not, if final_relu=False and we're on the last layer)
            if self.final_relu or (i != len(self.gmms) - 1):
                vals = F.relu(vals)

        return vals
