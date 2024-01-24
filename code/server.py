'''
本文件定义服务器相关的类和函数
'''
import torch
import torch.nn as nn
import datetime
from tqdm import tqdm
import time
import os
from dataset import Dataset
from model import LightGCN
from clients import Clients
from log import Log


class Server(object):
    def __init__(self, args, dataset: Dataset):
        self.args = args
        self.dataset = dataset

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.emb_dim = args.emb_dim
        self.num_epochs = args.num_epochs
        self.num_chosen_users = args.num_chosen_users
        self.top_K = args.top_K
        self.use_expand = args.use_expand
        self.early_stopping = args.early_stopping
        self.device = args.device if args.device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.global_model = LightGCN(num_layers=args.num_layers, lambda_reg=args.L2_coefficient)

        self.global_item_embedding = nn.Embedding(self.num_items, self.emb_dim, device=self.device)
        nn.init.xavier_uniform_(self.global_item_embedding.weight)

        self.clients = Clients(args, dataset, self.global_item_embedding, self.global_model)

        self.optimizer = torch.optim.Adam(self.global_item_embedding.parameters(), lr=args.lr)

        self.log = Log(args)

    def train_and_eval(self):
        best_epoch, best_rec, stop_count, eval_step = 0, 0, 0, 5

        # 训练
        for epoch in tqdm(range(self.num_epochs)):
            if self.use_expand:
                self.clients.expand_local_graph()

            batch_generator = self.dataset.generate_user_batches()
            batches_loss = []

            for batch in batch_generator:
                self.optimizer.zero_grad()

                clients_grad_info, clients_loss = self.clients.train(batch)
                batches_loss.append(clients_loss)

                self.fedavg(clients_grad_info)

                self.optimizer.step()

            # 测试
            if (epoch+1) % eval_step == 0:
                if self.use_expand:
                    self.clients.expand_local_graph()
                Precision, Recall, NDCG = self.clients.eval(self.dataset.test_set, self.top_K)
                batches_loss = torch.mean(torch.hstack(batches_loss))
                output_info = (('epoch:%03d  loss: %.5f  Pre: %.5f  Rec: %.5f  NDCG: %.5f  time: ' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                               % (epoch+1, batches_loss, Precision, Recall, NDCG))
                self.log.info(output_info)

                emb = torch.vstack((self.clients.local_user_embedding.weight,
                                    self.clients.local_item_embedding.weight))

                # 早停
                if Recall >= best_rec:
                    stop_count = 0
                    best_rec = Recall
                    best_epoch = epoch+1
                    self.log.save_emb(emb, 'best')
                else:
                    stop_count += eval_step
                    if stop_count >= self.early_stopping:
                        self.log.save_emb(emb, 'last')
                        break

                if (epoch+1) == self.num_epochs:
                    self.log.save_emb(emb, 'last')

        self.log.info('The best epoch is %d' % (best_epoch))

    def fedavg(self, clients_gradients):
        batch_size = len(clients_gradients)
        for item_ids, item_emb_gradient in clients_gradients:
            self.global_item_embedding.weight.grad[item_ids] += item_emb_gradient/batch_size


if __name__ == '__main__':
    from config import parse_args

    args = parse_args()
    dataset = Dataset(args)

    server = Server(args, dataset)
    server.train_and_eval()
