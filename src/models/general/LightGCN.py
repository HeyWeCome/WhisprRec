#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：LightGCN.py
@Author     ：Heywecome
@Date       ：2023/3/23 09:35
@Description: python main.py --model_name LightGCN --emb_size 64 --gcn_layers 3 --lr 1e-3 --l2 1e-8 --dataset 'ml-100k'
"""

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from models.BaseModel import GeneralModel
from models.init import xavier_uniform_initialization
from utils.loss import BPRLoss, EmbLoss


class LightGCN(GeneralModel):
    @staticmethod
    def parse_model_args(parser, configs):
        parser.add_argument('--embedding_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gcn_layers', type=int, default=2,
                            help='Number of LightGCN layers.')
        parser.add_argument('--reg_weight', type=float, default=1e-05,
                            help='The L2 regularization weight.')

        args, extras = parser.parse_known_args()
        # Update the configs dictionary with the parsed arguments
        configs['model']['embedding_size'] = args.embedding_size
        configs['model']['gcn_layers'] = args.gcn_layers
        configs['model']['reg_weight'] = args.reg_weight
        return parser

    def __init__(self, corpus, configs):
        super().__init__(corpus, configs)
        self.emb_size = configs['model']['embedding_size']
        self.gcn_layers = configs['model']['gcn_layers']
        self.n_users = corpus.n_users+1
        self.n_items = corpus.n_items+1
        self.reg_weight = float(configs['model']['reg_weight'])

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_size)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.norm_adj = self.csr2tensor(self.build_adjmat(self.n_users,
                                          self.n_items,
                                          corpus.train_clicked_set,
                                          device=self.device))
        self.apply(xavier_uniform_initialization)

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, device, selfloop_flag=False):
        # Construct a binary user-item interaction matrix R
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        # Construct the adjacency matrix for users and items
        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # Populate the adjacency matrix with user-item interactions
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        if selfloop_flag:
            # If selfloop_flag is True, add self-loops and then normalize
            # This is useful for improving stability and connectivity in certain algorithms
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        return adj_mat


    def csr2tensor(self, matrix):
        """
        Convert csr_matrix to tensor.
        Args:
            matrix: Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        # Calculate the sum of interactions for each node (user/item)
        rowsum = np.array(matrix.sum(1)) + 1e-10

        # Calculate the inverse square root of the rowsum for diagonal matrix D^-0.5
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # Calculate the bi-Laplacian normalized adjacency matrix: D^-0.5 * A * D^-0.5
        bi_lap = d_mat_inv_sqrt.dot(matrix).dot(d_mat_inv_sqrt)

        # Convert the bi-Laplacian normalized adjacency matrix to COO format
        coo_bi_lap = bi_lap.tocoo()

        # Extract the COO matrix components
        row = coo_bi_lap.row  # Row indices of non-zero elements
        col = coo_bi_lap.col  # Column indices of non-zero elements
        data = coo_bi_lap.data  # Values of non-zero elements

        # Convert the COO matrix components to PyTorch tensors
        i = torch.LongTensor(np.array([row, col]))
        data_tensor = torch.from_numpy(data).float()

        # Create a sparse tensor using the COO matrix components and shape
        sparse_bi_lap = torch.sparse.FloatTensor(i, data_tensor, torch.Size(coo_bi_lap.shape))

        if self.device == 'cuda':
            # If the device is not 'cpu', move the tensor to the specified device
            # Return the normalized adjacency matrix as a PyTorch sparse tensor
            norm_adj_mat = sparse_bi_lap.to(self.device)
            return norm_adj_mat
        else:
            norm_adj_mat = sparse_bi_lap.to_dense().to(self.device)
            return norm_adj_mat

    def get_ego_embeddings(self):
        """
        Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [user_count+item_count, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, feed_dict):
        ego_embeddings = self.get_ego_embeddings().to(self.device)  # [n_items+n_users, embedding_dim]
        embedding_list = [ego_embeddings]

        for layer_idx in range(self.gcn_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            embedding_list.append(ego_embeddings)

        lightgcn_all_embeddings = torch.stack(embedding_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)  # equals to \alpha = 1/(K+1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']
        neg_items = feed_dict['neg_items']

        user_all_embeddings, item_all_embeddings = self.forward(feed_dict)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_items]

        # Calculate BPR Loss
        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        # Use sparse tensor operations for matrix multiplication
        pos_scores = torch.sparse.mm(u_embeddings, pos_embeddings.t()).squeeze()
        neg_scores = torch.sparse.mm(u_embeddings, neg_embeddings.t()).squeeze()
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # Calculate Emb loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_items)
        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
        )

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']
        neg_item = feed_dict['neg_items']

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)
        bpr_loss = BPRLoss()
        loss = bpr_loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']
        neg_item = feed_dict['neg_items']

        user_all_embeddings, item_all_embeddings = self.forward(feed_dict)

        user_e = user_all_embeddings[user]
        pos_e = item_all_embeddings[pos_item]
        neg_e = item_all_embeddings[neg_item]

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)

        bpr_loss = BPRLoss()
        loss = bpr_loss(pos_item_score, neg_item_score)

        return loss

    def full_predict(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']

        user_all_embeddings, item_all_embeddings = self.forward(feed_dict)

        user_e = user_all_embeddings[user]
        pos_e = item_all_embeddings[pos_item]
        neg_e = item_all_embeddings

        pos_scores = (user_e * pos_e).sum(dim=-1)  # (batch_size,)
        neg_scores = torch.matmul(user_e, neg_e.transpose(0, 1))  # (batch_size, neg_item_num)

        return pos_scores, neg_scores
