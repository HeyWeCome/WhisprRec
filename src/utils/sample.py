#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：sample.py
@Author     ：Heywecome
@Date       ：2023/8/28 10:26 
@Description：A collection of sampling strategies
"""
import logging
import os
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import train_test_split

def count_statics(data_df, dataset):
    """
    Count Base statics of dataset, and reformat the name of columns.
    Args:
        data_df: xxx.inter

    Returns:
        processed data_df
    """
    renamed_columns = {
        "user_id:token": "user_id",
        "item_id:token": "item_id",
        "rating:float": "rating",
        "timestamp:float": "timestamp"
    }
    data_df.rename(columns=renamed_columns, inplace=True)
    # Filter out users with less than 5 occurrences
    user_counts = data_df['user_id'].value_counts()
    data_df = data_df[data_df['user_id'].isin(user_counts[user_counts >= 5].index)]

    # Create a mapping of unique user and item IDs to sequential integers
    user_id_map = {id: i for i, id in enumerate(data_df['user_id'].unique())}
    item_id_map = {id: i for i, id in enumerate(data_df['item_id'].unique())}
    data_df['user_id'] = data_df['user_id'].map(user_id_map)
    data_df['item_id'] = data_df['item_id'].map(item_id_map)

    if dataset == 'ml-1m' or dataset == 'ml-100k' or dataset == 'ml-10m':
        # Filter out interactions with a rating less than 3
        data_df = data_df.loc[data_df['rating'] >= 3].copy()  # Make a copy of the filtered DataFrame
        data_df.drop(columns=['rating'], inplace=True)
    elif dataset == "yelp" or dataset == "food":
        # Filter out interactions with rating less than 3 because dataset is very large.
        data_df = data_df.loc[data_df['rating'] >= 3].copy()
        data_df.drop(columns=['rating'], inplace=True)

    n_users = data_df['user_id'].value_counts().size
    n_items = data_df['item_id'].value_counts().size
    n_clicks = len(data_df)
    min_time = data_df['timestamp'].min()
    max_time = data_df['timestamp'].max()
    # Calculate density as interactions divided by total possible interactions
    density = n_clicks / (n_users * n_items)

    time_format = '%Y-%m-%d'
    logging.info('# Users: %d', n_users)
    logging.info('# Items: %d', n_items)
    logging.info('# Interactions: %d', n_clicks)
    logging.info('# density: %.8f%%', density)
    logging.info('Time Span: {}/{}'.format(
        datetime.utcfromtimestamp(min_time).strftime(time_format),
        datetime.utcfromtimestamp(max_time).strftime(time_format))
    )

    return data_df


def leave_one_out_split(data_df, save_path):
    """
    Splits the input DataFrame into train, dev, and test sets using a leave-one-out strategy.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing user interactions.
        save_path: The path to save train, dev, and test sets
    Returns:
        train_df (pd.DataFrame): Training set.
        dev_df (pd.DataFrame): Development set.
        test_df (pd.DataFrame): Test set.
    """
    # Store clicked items for each user in a dictionary
    clicked_item_set = data_df.groupby('user_id')['item_id'].apply(set)

    # Create DataFrame containing the first interaction for each user
    leave_df = data_df.groupby('user_id').head(1)

    # Remove rows used in leave_df from data_df
    data_df = data_df[~data_df.index.isin(leave_df.index)]

    # Split data_df into dev and test sets
    test_df = data_df.groupby('user_id').tail(1).copy()
    data_df = data_df[~data_df.index.isin(test_df.index)]
    dev_df = data_df.groupby('user_id').tail(1).copy()

    # Remove dev and test rows from data_df
    data_df = data_df[~data_df.index.isin(dev_df.index)]

    # Concatenate leave_df and data_df to create train_df
    train_df = pd.concat([leave_df, data_df]).sort_index()

    # Logging info
    logging.info("Dataset has been split. Train dataset length: %d, Dev dataset length: %d, Test dataset length: %d",
                 len(train_df), len(dev_df), len(test_df))

    # Save train, dev, and test sets to specified file paths
    # train_df.to_csv(os.path.join(save_path, 'train.csv'), index=False, sep='\t')
    # dev_df.to_csv(os.path.join(save_path, 'dev.csv'), index=False, sep='\t')
    # test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False, sep='\t')

    return train_df, dev_df, test_df


def random_split(data_df, save_path, ratios=[0.8, 0.1, 0.1]):
    """
    Splits the input DataFrame into train, dev, and test sets randomly.

    Args:
        data_df (pd.DataFrame): The input DataFrame containing user interactions.
        save_path (str): The path to save train, dev and test sets.
        ratios (list): The ratios to split train, dev and test sets. default: [0.8, 0.1, 0.1].

    Returns:
        train_df (pd.DataFrame): Training set.
        dev_df (pd.DataFrame): Development set.
        test_df (pd.DataFrame): Test set.
    """

    assert sum(ratios) == 1.0, "ratios should sum to 1"

    # Calculate train size
    train_size = ratios[0]

    # Calculate test and dev size
    dev_test_size = ratios[1] / (ratios[1] + ratios[2])

    # Split the data
    train_df, test_dev_df = train_test_split(data_df, train_size=train_size, random_state=42, shuffle=True)
    dev_df, test_df = train_test_split(test_dev_df, train_size=dev_test_size, random_state=42, shuffle=True)

    # Logging info
    logging.info("Dataset has been split. Train dataset length: %d, Dev dataset length: %d, Test dataset length: %d",
                 len(train_df), len(dev_df), len(test_df))

    # Save train, dev, and test sets to specified file paths
    # train_df.to_csv(os.path.join(save_path, 'train.csv'), index=False, sep='\t')
    # dev_df.to_csv(os.path.join(save_path, 'dev.csv'), index=False, sep='\t')
    # test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False, sep='\t')

    return train_df, dev_df, test_df

