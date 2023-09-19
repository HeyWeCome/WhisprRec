#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：SASRec.py
@Author     ：Heywecome
@Date       ：2023/9/13 14:47 
@Description：todo
"""
import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from models.init import xavier_normal_initialization
from utils import layers
from utils.loss import BPRLoss


class SASRec(SequentialModel):
    """
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    Run:
        python main.py --model_name SASRec --sample LS
    """
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Number of dropout')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._define_params()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _define_params(self):
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_his + 1, self.emb_size)

        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size,
                                    d_ff=self.emb_size,
                                    n_heads=self.num_heads,
                                    dropout=self.dropout,
                                    kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict: dict) -> dict:
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()

        position_ids = torch.arange(
            history.size(1), dtype=torch.long, device=history.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(history)
        position_embedding = self.position_embedding(position_ids)
        his_vectors = self.item_embedding(history)
        his_vectors = his_vectors + position_embedding

        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int32))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors = block(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()

        his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]
        # his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # ↑ average pooling is shown to be more effective than the most recent embedding
        return his_vector

    def predict(self, feed_dict):
        pos_item = feed_dict['pos_item']  # [batch_size, -1]
        neg_item = feed_dict['neg_items']

        user_e = self.forward(feed_dict)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)
        bpr_loss = BPRLoss()
        loss = bpr_loss(pos_item_score, neg_item_score)
        return loss

    def full_predict(self, feed_dict):
        user_e = self.forward(feed_dict)
        item_e = self.item_embedding.weight

        # expand the user embedding to match the shape of items
        scores = torch.matmul(user_e, item_e.transpose(0, 1))  # (batch_size, item_num)
        return scores
