#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：augmentor.py
@Author     ：Heywecome
@Date       ：2023/9/4 16:00 
@Description：Data augment
"""
from random import random

import numpy as np
import random
import scipy.sparse as sp


def rand_sample(high, size=None, replace=True):
    r"""Randomly discard some points or edges.

    Args:
        high (int): Upper limit of index value
        size (int): Array size after sampling

    Returns:
        numpy.ndarray: Array index after sampling, shape: [size]
    """

    a = np.arange(high)
    sample = np.random.choice(a, size=size, replace=replace)
    return sample


def node_dropout(sp_adjacency, dropout_rate):
    """
    Applies dropout to a sparse adjacency matrix.

    :param sp_adjacency: A sparse adjacency matrix in Compressed Sparse Row (CSR) format.
    :param dropout_rate: The dropout rate, a float between 0 and 1.
    :return: A modified sparse adjacency matrix after applying dropout.
    """
    # Get the shape of the input sparse adjacency matrix
    num_users, num_items = sp_adjacency.shape

    # Get the non-zero row and column indices
    row_indices, col_indices = sp_adjacency.nonzero()

    # Determine the number of users and items to drop
    num_users_to_drop = int(num_users * dropout_rate)
    num_items_to_drop = int(num_items * dropout_rate)

    # Randomly select user and item indices to drop
    dropped_user_indices = random.sample(range(num_users), num_users_to_drop)
    dropped_item_indices = random.sample(range(num_items), num_items_to_drop)

    # Create indicator arrays for dropped users and items
    indicator_user = np.ones(num_users, dtype=np.float32)
    indicator_item = np.ones(num_items, dtype=np.float32)
    indicator_user[dropped_user_indices] = 0.0
    indicator_item[dropped_item_indices] = 0.0

    # Create diagonal matrices from the indicator arrays
    diag_indicator_user = sp.diags(indicator_user)
    diag_indicator_item = sp.diags(indicator_item)

    # Create a new sparse adjacency matrix with dropout applied
    augmented_matrix = sp.csr_matrix(
        (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
        shape=(num_users, num_items))

    # Apply dropout by element-wise multiplication with indicator matrices
    augmented_matrix = diag_indicator_user.dot(augmented_matrix).dot(diag_indicator_item)

    return augmented_matrix


def edge_dropout(sparse_adjacency_matrix, dropout_rate):
    """
    Apply edge dropout to a sparse user-item adjacency matrix.

    :param sparse_adjacency_matrix: A sparse user-item adjacency matrix in Compressed Sparse Row (CSR) format.
    :param dropout_rate: The edge dropout rate, a float between 0 and 1.
    :return: A modified sparse adjacency matrix after applying edge dropout.
    """
    # Get the shape of the input sparse adjacency matrix
    adj_shape = sparse_adjacency_matrix.get_shape()

    # Count the total number of edges (non-zero entries)
    edge_count = sparse_adjacency_matrix.count_nonzero()

    # Get the row and column indices of non-zero entries (edges)
    row_indices, col_indices = sparse_adjacency_matrix.nonzero()

    # Determine the number of edges to keep after dropout
    num_edges_to_keep = int(edge_count * (1 - dropout_rate))

    # Randomly select indices of edges to keep
    keep_indices = random.sample(range(edge_count), num_edges_to_keep)

    # Extract the user and item indices of the kept edges
    kept_user_indices = np.array(row_indices)[keep_indices]
    kept_item_indices = np.array(col_indices)[keep_indices]

    # Create an array of ones for the kept edges
    kept_edges = np.ones_like(kept_user_indices, dtype=np.float32)

    # Create a new sparse adjacency matrix with the kept edges
    augmented_matrix = sp.csr_matrix(
        (kept_edges, (kept_user_indices, kept_item_indices)),
        shape=adj_shape
    )

    return augmented_matrix
