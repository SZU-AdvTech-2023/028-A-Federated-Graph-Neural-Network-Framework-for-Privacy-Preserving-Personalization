'''
本文件定义数据集相关的类和函数
'''
import os
import numpy as np


class Dataset(object):
    def __init__(self, args):
        self.num_chosen_users = args.num_chosen_users
        self.num_sample_items = args.num_sample_items
        self.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

        self.training_set, self.test_set, self.num_users, self.num_items = self.load_dataset()

    def load_dataset(self):
        training_set_path, test_set_path = os.path.join(self.dataset_dir, 'train.txt'), os.path.join(self.dataset_dir, 'test.txt')
        training_set, test_set, num_users, num_items = {}, {}, float('-inf'), float('-inf')

        with open(training_set_path) as f:
            for line in f.readlines():
                line = list(map(int, line.strip().split()))
                num_users, num_items = max(num_users, line[0]), max(num_items, max(line[1:]))
                training_set[line[0]] = line[1:]

        with open(test_set_path) as f:
            for line in f.readlines():
                line = list(map(int, line.strip().split()))
                num_users, num_items = max(num_users, line[0]), max(num_items, max(line[1:]) if len(line[1:]) > 0 else float('-inf'))
                test_set[line[0]] = line[1:]

        num_users, num_items = num_users+1, num_items+1

        return training_set, test_set, num_users, num_items

    def generate_user_batches(self):
        user_ids = np.arange(0, self.num_users)
        np.random.shuffle(user_ids)
        user_ids_list = np.split(user_ids[:-(self.num_users % self.num_chosen_users)], self.num_users//self.num_chosen_users)
        user_ids_list.append(user_ids[-(self.num_users % self.num_chosen_users):])

        for user_ids in user_ids_list:
            positive_item_ids_list, negative_item_ids_list = self.sample_items(user_ids)
            yield zip(user_ids, positive_item_ids_list, negative_item_ids_list)

    def sample_items(self, user_ids: np.ndarray):
        positive_item_ids_list, negative_item_ids_list = [], []
        for user_id in user_ids:
            interacted_item_ids = self.training_set[user_id]
            positive_item_ids = np.random.choice(interacted_item_ids, self.num_sample_items)

            negative_item_ids = np.random.randint(0, self.num_items, self.num_sample_items)
            while (isin := np.isin(negative_item_ids, interacted_item_ids)).sum() > 0:
                negative_item_ids[isin] = np.random.randint(0, self.num_items, isin.sum())

            positive_item_ids_list.append(positive_item_ids)
            negative_item_ids_list.append(negative_item_ids)

        return positive_item_ids_list, negative_item_ids_list


if __name__ == '__main__':
    from config import parse_args

    args = parse_args()
    dataset = Dataset(args)

    batch_generator = dataset.generate_user_batches()

    for batch in batch_generator:
        for user_id, positive_item_ids, negative_item_ids in batch:
            pass
