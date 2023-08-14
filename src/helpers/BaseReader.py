import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_reader_args(parser, configs):
        parser.add_argument('--path', type=str, default='../data/', help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml-100k', help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t', help='sep of csv file.')

        args, extras = parser.parse_known_args()

        # Update the configs dictionary with the parsed arguments
        configs['reader']['path'] = args.path
        configs['reader']['dataset'] = args.dataset
        configs['reader']['sep'] = args.sep

        return parser

    def __init__(self, configs):
        self.sep = configs['reader']['sep']
        self.prefix = configs['reader']['path']
        self.dataset = configs['reader']['dataset']
        self._read_data()

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self) -> None:
        logging.info('Reading data from \"{}\", dataset = \"{}\" '
                     .format(self.prefix, self.dataset))
        # Use data_df to store training set, validation set and test set
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics:')
        # Join dataframes
        train_df = self.data_df['train'][['user_id', 'item_id', 'time']]
        dev_df = self.data_df['dev'][['user_id', 'item_id', 'time']]
        test_df = self.data_df['test'][['user_id', 'item_id', 'time']]

        self.all_df = pd.concat([train_df, dev_df, test_df])
        # remove duplicate interactions:
        self.all_df = self.all_df.drop_duplicates(['user_id', 'item_id'])

        # Get dataset stats
        self.n_users = self.all_df['user_id'].max() + 1
        self.n_items = self.all_df['item_id'].max() + 1

        # Validate negative items
        for key in ['dev', 'test']:
            if 'neg_items' not in self.data_df[key]:
                continue

            neg_items = np.array(self.data_df[key]['neg_items'].tolist())

            assert (neg_items >= self.n_items).sum() == 0

        # Count active users and items
        active_users = len(self.all_df['user_id'].unique())
        active_items = len(self.all_df['item_id'].unique())
        # Calculate number of interactions
        n_interactions = len(self.all_df)
        # Calculate density as interactions divided by total possible interactions
        density = n_interactions / (active_users * active_items)

        logging.info('"# user": {}, "# item": {}, "# entry": {}, "# density": {}%'.format(
            self.n_users - 1, self.n_items, len(self.all_df), density*100
        ))
