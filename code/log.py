'''
本文件定义日志相关的类和函数
'''
import datetime
import os
import torch


class Log(object):
    def __init__(self, args):
        self.dir = os.path.join(args.save_dir, args.dataset_name, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        self.emb_dir = os.path.join(self.dir, 'embs')
        self.metirc_path = os.path.join(self.dir, 'metircs.txt')

        if ~os.path.exists(self.emb_dir):
            os.makedirs(self.emb_dir, exist_ok=True)

        with open(self.metirc_path, 'a') as f:
            f.write('---------------config---------------\n')
            for name, param in args._get_kwargs():
                f.write(name+': '+str(param)+'\n')
            f.write('------------------------------------\n')

    def info(self, msg):
        with open(self.metirc_path, 'a') as f:
            f.write(msg+'\n')

    def save_emb(self, emb: torch.Tensor, name):
        torch.save(emb, os.path.join(self.emb_dir, name+'.pt'))


if __name__ == '__main__':
    from config import parse_args

    args = parse_args()
    log = Log(args)

    log.info('test')
    log.save_emb(torch.tensor([1, 2, 3], device='cuda'), 'test')
