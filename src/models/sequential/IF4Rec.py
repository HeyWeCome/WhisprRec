#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：IF4Rec.py
@Author     ：Heywecome
@Date       ：2023/4/12 14:20 
@Description：He et,al Interest HD: An Interest Frame Model for Recommendation Based on HD Image Generation,
              IEEE Transactions on Neural Networks and Learning Systems. 2022
"""
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd

from models.BaseModel import SequentialModel
from utils import layers

'''
python main.py --model_name IF4Rec --lr 1e-4 --l2 1e-6 --dataset Food
'''

class IF4Rec(SequentialModel):
    extra_log_args = ['emb_size', 'attn_size', 'K', 'encoder_layer', 'n_head', 'encoder_dropout', 'add_pos', 'add_multi']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding.')
        parser.add_argument('--encoder_layer', type=int, default=1,
                            help='Number of encoder layer.')
        parser.add_argument('--n_head', type=int, default=8,
                            help='Number of multi-attention head')
        parser.add_argument('--encoder_dropout', type=float, default=0.1,
                            help='encoder dropout')
        parser.add_argument('--add_multi', type=int, default=1,
                            help='Whether add multi-head.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.add_pos = args.add_pos
        self.max_his = args.history_max
        self.encoder_layer = args.encoder_layer
        self.n_head = args.n_head
        self.encoder_dropout = args.encoder_dropout
        self.add_multi = args.add_multi
        super().__init__(args, corpus)

        self._define_params()
        self.apply(self.init_weights)
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num,
                                         self.emb_size)

        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)

        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.n_head,
                                    dropout=self.encoder_dropout, kq_same=False)
            for _ in range(self.encoder_layer)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths']
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        if self.add_pos:
            position = (((lengths[:, None] + self.len_range[None, :seq_len]) - lengths[:, None]) * valid_his)
            pos_vectors = self.p_embeddings(position)
            his_pos_vectors = his_vectors + pos_vectors
        else:
            his_pos_vectors = his_vectors

        attn_score = self.W2(F.leaky_relu(self.W1(his_pos_vectors)))
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_pos_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(
            -2)

        if self.add_multi:
            for block in self.transformer_block:
                interest_vectors = block(interest_vectors)

        i_vectors = self.i_embeddings(i_ids)
        if feed_dict['phase'] == 'train':
            target_vector = i_vectors[:, 0]
            target_pred = (interest_vectors * target_vector[:, None, :]).sum(-1)
            idx_select = target_pred.max(-1)[1]
            user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        else:
            prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)
            prediction = prediction.max(-1)[0]

        return {'prediction': prediction.view(batch_size, -1)}