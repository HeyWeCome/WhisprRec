# -*- coding: UTF-8 -*-

import logging
import pandas as pd

from helpers.BaseReader import BaseReader


class SeqReader(BaseReader):
    def __init__(self, args):
        super().__init__(args)
        self._append_his_info()

    def _append_his_info(self):
        """
        self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
        add the 'position' of each interaction in user_his to data_df
        """
        logging.info('添加用户历史信息中...')
        # 按照time 和 user_id进行升序排序，排序方式用归并排序
        # 可以使用快速排序、归并排序与堆排序
        # 鉴于归并的时间复杂度很稳定，所以选择归并排序：mergesort
        sort_df = self.all_df.sort_values(by=['time', 'user_id'], kind='mergesort')
        position = list()
        self.user_his = dict()  # 存储每个用户的已见序列
        for uid, iid, t in zip(sort_df['user_id'], sort_df['item_id'], sort_df['time']):
            if uid not in self.user_his:
                self.user_his[uid] = list()
            position.append(len(self.user_his[uid]))
            self.user_his[uid].append((iid, t))
        sort_df['position'] = position
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.merge(
                left=self.data_df[key], right=sort_df, how='left',
                on=['user_id', 'item_id', 'time'])
        del sort_df
