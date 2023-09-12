# -*- coding: UTF-8 -*-

import logging
import pandas as pd

from helpers.BaseReader import BaseReader


class SeqReader(BaseReader):
    def __init__(self, configs):
        super().__init__(configs)
        self.sample = 'leave_one_out'
        self._append_user_history_info()

    def _append_user_history_info(self):
        """
        Add the position of each interaction in the user's history to the data DataFrame.
        """
        logging.info('Adding user history information...')

        # Sort the DataFrame by time and user ID in ascending order using merge sort.
        sorted_df = self.all_df.sort_values(by=['timestamp', 'user_id'], kind='mergesort')

        # Initialize a dictionary to store each user's interaction history.
        self.user_his = {}

        # Initialize an empty list to store interaction positions.
        positions = []

        # Iterate through sorted interactions and update user history and positions.
        for user_id, item_id, time in zip(sorted_df['user_id'], sorted_df['item_id'], sorted_df['timestamp']):
            if user_id not in self.user_his:
                self.user_his[user_id] = []
            positions.append(len(self.user_his[user_id]))
            self.user_his[user_id].append((item_id, time))

        # Add a 'position' column to the sorted DataFrame.
        sorted_df['position'] = positions

        # Merge the sorted DataFrame with the data DataFrame for train, dev, and test.
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.merge(
                left=self.data_df[key], right=sorted_df, how='left',
                on=['user_id', 'item_id', 'timestamp'])

        # Clean up by deleting the sorted DataFrame.
        del sorted_df


if __name__ == '__main__':
    # init overall configs
    configs = dict()
    configs.update({'model': {}})
    configs.update({'runner': {}})
    configs.update({'reader': {}})

    configs['reader']['sep'] = '\t'
    configs['reader']['path'] = '../../data/'
    configs['reader']['dataset'] = 'ml-100k'
    configs['reader']['sample'] = 'random'

    reader = SeqReader(configs)
