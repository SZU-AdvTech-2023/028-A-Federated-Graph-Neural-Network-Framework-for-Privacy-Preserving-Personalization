import numpy as np
import torch
import config
from server import Server
from dataset import Dataset


if __name__ == '__main__':
    args = config.parse_args()

    if args.use_seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    dataset = Dataset(args)

    server = Server(args, dataset)

    server.train_and_eval()
