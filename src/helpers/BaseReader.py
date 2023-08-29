import os
import pickle
import argparse
import logging


from utils import sample
import numpy as np
import pandas as pd

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_reader_args(parser, configs):
        parser.add_argument('--path', type=str, default='../data/', help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml-1m', help='Choose a dataset.')
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

        # check whether train, val and test dataset have been generated
        train_df, dev_df, test_df = self._check_file()

        self._read_data(train_df, dev_df, test_df)

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

    def _check_file(self):
        data_path = os.path.join(self.prefix, self.dataset)

        train_path = os.path.join(data_path, 'train.csv')
        val_path = os.path.join(data_path, 'dev.csv')
        test_path = os.path.join(data_path, 'test.csv')

        train_exists = os.path.exists(train_path)
        val_exists = os.path.exists(val_path)
        test_exists = os.path.exists(test_path)

        # If all exist
        if train_exists and val_exists and test_exists:
            logging.info("train, val and test dataset have been generated. Loading now···")
            # Read the CSV files into DataFrames
            train_df = pd.read_csv(train_path, sep=self.sep)
            dev_df = pd.read_csv(val_path, sep=self.sep)
            test_df = pd.read_csv(test_path, sep=self.sep)
        else:
            logging.info("Generating train, val and test dataset now···")
            inter_file_path = os.path.join(data_path, 'ml-1m.inter')
            try:
                data_df = pd.read_csv(inter_file_path, sep='\t', header=0)
                data_df = sample.count_statics(data_df)
                train_df, dev_df, test_df = sample.leave_one_out_split(data_df, save_path=data_path)
            except FileNotFoundError:
                logging.error("Interactions file not found.")
            except Exception as e:
                logging.error(f"An error occurred while trying to read the interactions file: {e}")

        return train_df, dev_df, test_df

    def _read_data(self, train_df, dev_df, test_df):
        logging.info("Reading data from %s, dataset = %s", self.prefix, self.dataset)
        # Use data_df to store training set, validation set and test set
        self.data_df = dict()
        self.data_df['train'] = train_df
        self.data_df['dev'] = dev_df
        self.data_df['test'] = test_df

        # Join dataframes
        train_df = self.data_df['train'][['user_id', 'item_id', 'timestamp']]
        dev_df = self.data_df['dev'][['user_id', 'item_id', 'timestamp']]
        test_df = self.data_df['test'][['user_id', 'item_id', 'timestamp']]

        self.all_df = pd.concat([train_df, dev_df, test_df])
        # remove duplicate interactions:
        self.all_df = self.all_df.drop_duplicates(['user_id', 'item_id'])

        # Get dataset stats
        self.n_users = self.all_df['user_id'].max()
        self.n_items = self.all_df['item_id'].max()


if __name__ == '__main__':
    # init overall configs
    configs = dict()
    configs.update({'model': {}})
    configs.update({'runner': {}})
    configs.update({'reader': {}})

    configs['reader']['sep'] = '\t'
    configs['reader']['path'] = '../../data/'
    configs['reader']['dataset'] = 'ml-1m'

    reader = BaseReader(configs)

