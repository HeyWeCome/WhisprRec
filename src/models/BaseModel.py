import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from helpers.BaseReader import BaseReader
from helpers.SeqReader import SeqReader
from utils.loss import BPRLoss


# 基础模型类
class BaseModel(nn.Module):
    reader, runner = None, None  # 根据不同的模型选择不同的helper
    extra_log_args = []  # 需要额外记录的参数

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--buffer', type=int, default=1,
                            help='Whether to buffer feed dicts for dev/test')
        return parser

    def __init__(self, args, corpus: BaseReader):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.buffer = args.buffer
        self.optimizer = None
        self.check_list = list()  # observe tensors in check_list every check_epoch

    def forward(self, feed_dict: dict) -> dict:
        """
        :param feed_dict: 数据集
        :return: out_dict, 要预测的商品列表 [batch_size, n_candidates]
        """
        pass

    def loss(self, out_dict: dict) -> torch.Tensor:
        pass

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_after_train(self):  # e.g., save selected parameters
        pass

    class Dataset(BaseDataset):
        def __init__(self, model, corpus, phase: str):
            self.model = model  # Model
            self.corpus = corpus  # The reference of reader
            self.phase = phase  # train / dev / test

            self.buffer_dict = dict()
            # DataFrame is not compatible with multiple threaded operation, convert to dict
            self.data = utils.df_to_dict(corpus.data_df[phase])

        def __len__(self):
            if type(self.data) == dict:
                for key in self.data:
                    return len(self.data[key])
            return len(self.data)

        def __getitem__(self, index: int) -> dict:
            return self._get_feed_dict(index)

        # 关键！：对于一个单例构建输入数据的方法
        def _get_feed_dict(self, index: int) -> dict:
            pass

        # 只在训练阶段之前调用
        def actions_before_epoch(self):
            pass

        # Collate a batch according to the list of feed dicts
        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            # Initialize an empty dictionary
            feed_dict = dict()

            # Iterate over keys in the dicts
            for key in feed_dicts[0]:

                # Initialize for all cases
                lengths = [len(d[key]) if isinstance(d[key], (list, np.ndarray)) else 1 for d in feed_dicts]

                # Obtain a list of values for the key
                values_list = [d[key] for d in feed_dicts]

                # Check for numpy arrays and pad if their lengths are inconsistent
                if isinstance(feed_dicts[0][key], np.ndarray):
                    if any(len != lengths[0] for len in lengths):
                        values_list = [d[key] for d in feed_dicts]

                # Convert values list to numpy array
                stack_val = np.array(values_list, dtype=object if any(len != lengths[0] for len in lengths) else None)

                # Convert numpy array to tensor and pad sequence if dtypes are objects or directly convert to tensor
                if stack_val.dtype == object:
                    feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
                else:
                    feed_dict[key] = torch.from_numpy(stack_val)

                # Add general elements to dict
            feed_dict['batch_size'] = len(feed_dicts)
            feed_dict['phase'] = self.phase

            return feed_dict


class GeneralModel(BaseModel):
    reader, runner = 'BaseReader', 'BaseRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_neg', type=int, default=1,
                            help='The number of negative items during training.')
        parser.add_argument('--test_all', type=int, default=1,
                            help='Whether testing on all the items.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.user_num = int(corpus.n_users)
        self.item_num = int(corpus.n_items)
        self.num_neg = args.num_neg
        self.test_all = args.test_all

    def calculate_loss(self, feed_dict):
        pass

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]

            feed_dict = {
                'user_id': user_id,
                'pos_item': target_item,
                'neg_items': neg_items
            }
            return feed_dict

        # Sample negative items for all the instances
        def actions_before_epoch(self):
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            for i, u in enumerate(self.data['user_id']):
                clicked_set = self.corpus.train_clicked_set[u]  # neg items will not include dev/test set
                # clicked_set = self.corpus.residual_clicked_set[u]  # neg items are possible to appear in dev/test set
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            # Flatten to 1D
            neg_items = neg_items.reshape(-1)
            self.data['neg_items'] = neg_items


# 序列模型的Model
class SequentialModel(GeneralModel):
    reader = 'SeqReader'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.history_max = args.history_max

    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            # filter out the data, whose length is not greater than 0.
            idx_select = np.array(self.data['position']) > 0
            for key in self.data:
                self.data[key] = np.array(self.data[key])[idx_select]

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)

            pos = self.data['position'][index]
            user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
            if self.model.history_max > 0:
                user_seq = user_seq[-self.model.history_max:]

            # Extract the last element from user_seq and assign its positive value to another variable
            target_item = user_seq[-1][0]
            feed_dict['pos_item'] = target_item
            # Remove the last element from user_seq
            user_seq = user_seq[:-1]


            feed_dict['history_items'] = np.array([x[0] for x in user_seq])
            feed_dict['history_times'] = np.array([x[1] for x in user_seq])
            feed_dict['lengths'] = len(feed_dict['history_items'])
            return feed_dict
