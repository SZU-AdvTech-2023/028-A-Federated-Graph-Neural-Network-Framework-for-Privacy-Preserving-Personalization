'''
本文件定义GNN模型等
'''
from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList
from torch.nn.modules.loss import _Loss
from torch_geometric.nn import LGConv
from torch_geometric.typing import Adj


class LightGCN(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bprloss = BPRLoss(lambda_reg=lambda_reg)

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, x: Tensor, edge_index: Adj) -> Tensor:
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    def get_bprloss(self, x: Tensor, edge_index: Adj, user_id,
                    positive_item_ids, negative_item_ids):
        hidden_embs = self.get_embedding(x, edge_index)

        user_emb = hidden_embs[user_id]
        positive_item_embs = x[positive_item_ids]
        negative_item_embs = x[negative_item_ids]

        positive_item_ratings = user_emb@positive_item_embs.T
        negative_item_ratings = user_emb@negative_item_embs.T

        param = torch.vstack((x[user_id], x[positive_item_ids], x[negative_item_ids]))
        loss = self.bprloss(positive_item_ratings, negative_item_ratings, param)

        return loss


class BPRLoss(_Loss):
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs) -> None:
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).sum()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)/2

        return (-log_prob + regularization) / n_pairs


if __name__ == '__main__':
    from torch_geometric.data import Data

    model = LightGCN(num_layers=2)

    embedding = torch.nn.Embedding(10, 8)
    torch.nn.init.xavier_uniform_(embedding.weight)

    edge_index = torch.tensor([[], []], dtype=torch.int64)
    for i in range(5):
        single_edge_index = torch.vstack((torch.LongTensor([i for _ in range(5)]),
                                          torch.LongTensor([j for j in range(5, 10)])))
        edge_index = torch.cat((edge_index, single_edge_index), dim=1)
    edge_index = torch.cat((edge_index, edge_index.flip(dims=[0])), dim=1)

    graph = Data(x=embedding.weight, edge_index=edge_index)

    hidden_emb = model.get_embedding(x=graph.x, edge_index=graph.edge_index)

    print(graph.x)
    print(hidden_emb)
