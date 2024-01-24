'''
本文件用于配置各种参数
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # 模型参数设置
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GNN模型的层数')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='embedding的维度')
    parser.add_argument('--clip_threshold', type=float, default=0.3,
                        help='LDP的梯度裁剪阈值')
    parser.add_argument('--noise_strength', type=float, default=0.1,
                        help='LDP的拉普拉斯噪声的强度')
    parser.add_argument('--use_LDP', type=int, default=1,
                        help='是否使用LDP')
    parser.add_argument('--use_expand', type=int, default=1,
                        help='是否扩展本地图')

    # 数据集设置
    parser.add_argument('--dataset_name', type=str, default='gowalla',
                        help='使用的数据集，支持[gowalla, amazon-book, yelp2018]')
    parser.add_argument('--dataset_dir', type=str, default='../data',
                        help='存放数据集的文件夹')

    # 实验环境设置
    parser.add_argument('--seed', type=int, default=2020,
                        help='设置的随机数种子，方便复现')
    parser.add_argument('--use_seed', type=int, default=1,
                        help='是否使用随机数种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='使用哪个设备进行训练，支持[auto, cpu, cuda]')
    parser.add_argument('--save_dir', type=str, default='../result',
                        help='存放训练和测试结果的文件夹')

    # 训练参数设置
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='训练的epoch数量')
    parser.add_argument('--num_chosen_users', type=int, default=512,
                        help='每轮训练唤醒的用户（客户端）数量')
    parser.add_argument('--num_sample_items', type=int, default=2048,
                        help='对用户（客户端）要采样的负样本数量')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--L2_coefficient', type=float, default=1e-3,
                        help='L2正则化系数')
    parser.add_argument('--early_stopping', type=int, default=50,
                        help='早停')

    # 评估参数设置
    parser.add_argument('--top_K', type=int, default=20,
                        help='top K')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
