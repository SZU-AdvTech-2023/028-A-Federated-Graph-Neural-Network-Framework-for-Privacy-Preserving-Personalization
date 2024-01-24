'''
本文件定义客户端相关的类和函数
'''
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
import multiprocessing
from typing import Union, Iterable
from dataset import Dataset
from model import LightGCN


class Clients(object):
    def __init__(self, args, dataset: Dataset,
                 global_item_embedding: nn.Embedding,
                 global_model: LightGCN):
        self.args = args
        self.dataset = dataset

        self.local_data = dataset.training_set
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.emb_dim = args.emb_dim
        self.clip_threshold = args.clip_threshold
        self.noise_strength = args.noise_strength
        self.use_LDP = args.use_LDP
        self.device = args.device if args.device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.laplace = torch.distributions.laplace.Laplace(torch.tensor(0.0, device=self.device),
                                                           torch.tensor(self.noise_strength, device=self.device))

        self.local_user_embedding = nn.Embedding(self.num_users, self.emb_dim, device=self.device)
        nn.init.xavier_uniform_(self.local_user_embedding.weight)
        self.local_item_embedding = global_item_embedding
        self.local_model = global_model
        self.optimizer = torch.optim.Adam(self.local_user_embedding.parameters(), lr=args.lr)
        self.local_graphs, self.user_emb_copy = self.init_local_graph()
        self.no_expand = True

    def init_local_graph(self):
        local_graphs, user_emb_copy = {}, self.local_user_embedding.weight.detach().clone()

        for user_id, interacted_item_ids in self.local_data.items():
            edge_index = torch.vstack((torch.LongTensor([user_id for _ in range(len(interacted_item_ids))]).to(device=self.device),
                                       torch.LongTensor(interacted_item_ids).to(device=self.device)+self.num_users))
            edge_index = torch.cat((edge_index, edge_index.flip(dims=[0])), dim=1)

            local_graph = Data(edge_index=edge_index)
            local_graphs[user_id] = local_graph

        return local_graphs, user_emb_copy

    def expand_local_graph(self):
        if self.no_expand:
            self.local_graphs.clear()
            global_edge_index = torch.LongTensor([[], []]).to(device=self.device)
            for user_id, interacted_item_ids in self.local_data.items():
                single_edge_index = torch.vstack((torch.LongTensor([user_id for _ in range(len(interacted_item_ids))]).to(device=self.device),
                                                  torch.LongTensor(interacted_item_ids).to(device=self.device)+self.num_users))
                global_edge_index = torch.cat((global_edge_index, single_edge_index), dim=1)

            for user_id, interacted_item_ids in self.local_data.items():
                interacted_item_ids = torch.LongTensor(interacted_item_ids).to(device=self.device)
                isin_index = torch.isin(global_edge_index[1], interacted_item_ids+self.num_users)
                edge_index = global_edge_index[:, isin_index]
                edge_index = torch.cat((edge_index, edge_index.flip(dims=[0])), dim=1)
                self.local_graphs[user_id] = Data(edge_index=edge_index)
            self.no_expand = False

        self.user_emb_copy = self.local_user_embedding.weight.detach().clone()

    def train(self, batch):
        self.local_model.train()

        grad_info, batch_loss = [], []
        self.optimizer.zero_grad()
        for user_id, positive_item_ids, negative_item_ids in batch:
            item_ids = np.concatenate((np.unique(positive_item_ids), np.unique(negative_item_ids)))

            x = torch.vstack((self.user_emb_copy[:user_id],
                              self.local_user_embedding.weight[user_id],
                              self.user_emb_copy[user_id+1:],
                              self.local_item_embedding.weight))
            edge_index = self.local_graphs[user_id].edge_index

            loss = self.local_model.get_bprloss(x, edge_index, user_id, positive_item_ids+self.num_users,
                                                negative_item_ids+self.num_users)
            loss.backward()
            batch_loss.append(loss)

            item_emb_grad = self.local_item_embedding.weight.grad.detach().clone()[item_ids]
            self.local_item_embedding.zero_grad()
            if self.use_LDP:
                self.LDP(item_emb_grad)

            grad_info.append((item_ids, item_emb_grad))
        self.optimizer.step()
        batch_loss = torch.mean(torch.hstack(batch_loss))

        return grad_info, batch_loss

    def LDP(self, gradients: Union[torch.Tensor, Iterable[torch.Tensor]]):
        if isinstance(gradients, torch.Tensor):
            gradients = [gradients]
        for grad in gradients:
            grad.clamp_(min=-self.clip_threshold, max=self.clip_threshold)

        for grad in gradients:
            noise = self.laplace.sample(grad.shape)
            grad.add_(noise)

    def eval(self, test_set, top_K):
        self.local_model.eval()

        test_user_ids = list(test_set.keys())
        rating_matrix_tensor, rating_matrix_array = None, None
        DCG_denominator = np.log2([i+1 for i in range(1, top_K+1)])
        NDCG_denominator = (1/DCG_denominator).sum()

        with torch.no_grad():
            for user_id in test_user_ids:
                x = torch.vstack((self.user_emb_copy[:user_id],
                                  self.local_user_embedding.weight[user_id],
                                  self.user_emb_copy[user_id+1:],
                                  self.local_item_embedding.weight))
                edge_index = self.local_graphs[user_id].edge_index

                hidden_embs = self.local_model.get_embedding(x, edge_index)

                user_emb = hidden_embs[user_id]
                item_embs = x[self.num_users:]
                ratings = user_emb@item_embs.T

                if rating_matrix_tensor is None:
                    rating_matrix_tensor = ratings
                else:
                    rating_matrix_tensor = torch.vstack((rating_matrix_tensor, ratings))

                if user_id % 2000 == 0:
                    if rating_matrix_array is None:
                        rating_matrix_array = rating_matrix_tensor.cpu().numpy()
                    else:
                        rating_matrix_array = np.vstack((rating_matrix_array, rating_matrix_tensor.cpu().numpy()))
                    rating_matrix_tensor = None

            if rating_matrix_tensor is not None:
                rating_matrix_array = np.vstack((rating_matrix_array, rating_matrix_tensor.cpu().numpy()))

            rating_matrix = rating_matrix_array

            test_info = zip(rating_matrix, [self.local_data[i] for i in test_user_ids],
                            [test_set[i] for i in test_user_ids], [top_K for i in test_user_ids],
                            [DCG_denominator for i in test_user_ids], [NDCG_denominator for i in test_user_ids])

            with multiprocessing.Pool(20) as pool:
                metrics = pool.map(test_one_user, test_info)
            metrics = np.vstack(metrics).mean(axis=0)

        return metrics


def test_one_user(info):
    ratings, interacted_items, test_items, top_K, DCG_denominator, NDCG_denominator = info

    recommended_items = (-ratings).argsort()
    top_K_recommended_items, interacted_items = [], set(interacted_items)
    for recommended_item in recommended_items:
        if len(top_K_recommended_items) == top_K:
            break

        if recommended_item not in interacted_items:
            top_K_recommended_items.append(recommended_item)

    isin_testset = np.isin(top_K_recommended_items, test_items)

    Precision = isin_testset.sum()/top_K
    Recall = isin_testset.sum()/len(test_items)
    if (len_test := len(test_items)) >= top_K:
        NDCG = (isin_testset/DCG_denominator).sum()/NDCG_denominator
    else:
        denominator = (1/np.log2([i+1 for i in range(1, len_test+1)])).sum()
        NDCG = (isin_testset/DCG_denominator).sum()/denominator

    return np.array([Precision, Recall, NDCG])


if __name__ == '__main__':
    from config import parse_args
    from model import LightGCN
    from dataset import Dataset
    import time

    args = parse_args()
    if args.use_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    dataset = Dataset(args)

    model = LightGCN(num_layers=args.num_layers, lambda_reg=args.L2_coefficient).cuda()

    item_embedding = nn.Embedding(dataset.num_items, args.emb_dim).cuda()
    nn.init.xavier_uniform_(item_embedding.weight)

    clients = Clients(args, dataset, item_embedding, model)
    clients.expand_local_graph()

    # print(clients.eval(dataset.test_set, args.top_K))
    start = time.time()
    batch_generator = dataset.generate_user_batches()
    for batch in batch_generator:
        clients.train(batch)
    end = time.time()
    print(end-start)
