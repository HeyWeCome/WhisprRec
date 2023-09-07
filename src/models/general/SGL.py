#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：SGL.py
@Author     ：Heywecome
@Date       ：2023/9/3 14:41 
@Description：Self-supervised Graph Learning for Recommendation SIGIR ’21
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.init import xavier_uniform_initialization
from utils.loss import BPRLoss, EmbLoss
from models.BaseModel import GeneralModel
from utils.augmentor import node_dropout, edge_dropout

class SGL(GeneralModel):
    @staticmethod
    def parse_model_args(parser, configs):
        parser.add_argument('--embedding_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--gcn_layers', type=int, default=3,
                            help='Number of SGL layers.')
        parser.add_argument('--type', type=str, default='ED',
                            help="The type to generate views. Range in ['ED', 'ND', 'RW'].")
        parser.add_argument('--reg_weight', type=float, default=1e-05,
                            help='The L2 regularization weight.')
        parser.add_argument('--ssl_tau', type=float, default=0.5,
                            help='The temperature in softmax.')
        parser.add_argument('--ssl_weight', type=float, default=0.05,
                            help='The hyperparameter to control the strengths of SSL.')
        parser.add_argument('--drop_ratio', type=float, default=0.1,
                            help='The dropout ratio.')

        args, extras = parser.parse_known_args()
        # Update the configs dictionary with the parsed arguments
        configs['model']['embedding_size'] = args.embedding_size
        configs['model']['gcn_layers'] = args.gcn_layers
        configs['model']['reg_weight'] = args.reg_weight
        configs['model']['type'] = args.type
        configs['model']['ssl_weight'] = args.ssl_weight
        configs['model']['ssl_tau'] = args.ssl_tau
        configs['model']['drop_ratio'] = args.drop_ratio
        return parser

    def __init__(self, corpus, configs):
        super().__init__(corpus, configs)
        self.emb_size = configs['model']['embedding_size']
        self.gcn_layers = int(configs['model']['gcn_layers'])
        self.n_users = corpus.n_users
        self.n_items = corpus.n_items
        self.reg_weight = float(configs['model']['reg_weight'])
        self.type = str(configs['model']['type'])
        self.ssl_weight = configs['model']['ssl_weight']
        self.ssl_tau = configs['model']['ssl_tau']
        self.drop_ratio = configs['model']['drop_ratio']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_size)
        self.adj_matrix = self.build_adjmat(self.n_users,
                                            self.n_items,
                                            corpus.train_clicked_set)
        self.train_graph = self.csr2tensor(self.adj_matrix)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.apply(xavier_uniform_initialization)

    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node."""
        self.sub_graph1 = []
        if self.type == "ND":
            self.sub_graph1 = self.csr2tensor(node_dropout(self.adj_matrix, self.drop_ratio))
        elif self.type == "ED":
            self.sub_graph1 = self.csr2tensor(edge_dropout(self.adj_matrix, self.drop_ratio))

        self.sub_graph2 = []
        if self.type == "ND":
            self.sub_graph2 = self.csr2tensor(node_dropout(self.adj_matrix, self.drop_ratio))
        elif self.type == "ED":
            self.sub_graph2 = self.csr2tensor(edge_dropout(self.adj_matrix, self.drop_ratio))

    def build_adjmat(self, user_count, item_count, train_mat, selfloop_flag=False):
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

    def forward(self, graph):
        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = [main_ego]
        if isinstance(graph, list):
            for sub_graph in graph:
                main_ego = torch.sparse.mm(sub_graph, main_ego)
                all_ego.append(main_ego)
        else:
            for i in range(self.gcn_layers):
                main_ego = torch.sparse.mm(graph, main_ego)
                all_ego.append(main_ego)
        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)
        user_emd, item_emd = torch.split(all_ego, [self.n_users, self.n_items], dim=0)

        return user_emd, item_emd

    def calc_bpr_loss(
        self, user_emd, item_emd, user_list, pos_item_list, neg_item_list
    ):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(
        self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2
    ):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2, dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2, dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def predict(self, feed_dict):
        user_list = feed_dict['user_id']
        pos_item_list = feed_dict['pos_item']
        neg_item_list = feed_dict['neg_items']

        user_emd, item_emd = self.forward(self.train_graph)
        user_sub1, item_sub1 = self.forward(self.sub_graph1)
        user_sub2, item_sub2 = self.forward(self.sub_graph2)
        total_loss = self.calc_bpr_loss(
            user_emd, item_emd, user_list, pos_item_list, neg_item_list
        ) + self.calc_ssl_loss(
            user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2
        )
        return total_loss

    def full_predict(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['pos_item']

        user_all_embeddings, item_all_embeddings = self.forward(self.train_graph)

        user_e = user_all_embeddings[user]
        pos_e = item_all_embeddings[pos_item]
        neg_e = item_all_embeddings

        pos_scores = (user_e * pos_e).sum(dim=-1)  # (batch_size,)
        neg_scores = torch.matmul(user_e, neg_e.transpose(0, 1))  # (batch_size, neg_item_num)

        return pos_scores, neg_scores

    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.model.graph_construction()