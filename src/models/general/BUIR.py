#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File       ：BUIR.py
@Author     ：Heywecome
@Date       ：2023/9/7 09:54 
@Description：This model's performance is not as good as reported
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class BUIR(GeneralModel):
    reader = 'BaseReader'
    runner = 'BUIRRunner'
    extra_log_args = ['embedding_size', 'momentum']

    @staticmethod
    def parse_model_args(parser, configs):
        parser.add_argument('--embedding_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--momentum', type=float, default=0.995,
                            help='Momentum update.')

        args, extras = parser.parse_known_args()
        # Update the configs dictionary with the parsed arguments
        configs['model']['embedding_size'] = args.embedding_size
        configs['model']['momentum'] = args.momentum

        return parser

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)

            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    def __init__(self, corpus, configs):
        super().__init__(corpus, configs)

        # load parameter info
        self.embedding_size = configs['model']['embedding_size']
        self.momentum = configs['model']['momentum']
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items

        self.user_online = nn.Embedding(self.user_num, self.embedding_size)
        self.user_target = nn.Embedding(self.user_num, self.embedding_size)
        self.item_online = nn.Embedding(self.item_num, self.embedding_size)
        self.item_target = nn.Embedding(self.item_num, self.embedding_size)
        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)

        self._init_weights()

        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    # will be called by BUIRRunner
    def _update_target(self):
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, feed_dict):
        user, item = feed_dict['user_id'], feed_dict['pos_item']
        u_online = self.predictor(self.user_online(user))
        u_target = self.user_target(user)
        i_online = self.predictor(self.item_online(item))
        i_target = self.item_target(item)

        return u_online, u_target, i_online, i_target

    def predict(self, feed_dict):
        u_online, u_target, i_online, i_target = self.forward(feed_dict)

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        # Euclidean distance between normalized vectors can be replaced with their negative inner product
        loss_ui = 2 - 2 * (u_online * i_target.detach()).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target.detach()).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    def full_predict(self, feed_dict):
        user = feed_dict['user_id']

        user_e = self.user_online(user)
        all_item_e = self.item_online.weight

        # expand the user embedding to match the shape of neg_items
        user_predictor = self.predictor(user_e)  # Apply the linear layer to user_e
        # Apply the linear layer to item_online.weight
        item_weights = self.predictor(self.item_online.weight).transpose(0, 1)
        all_scores = torch.matmul(user_e, item_weights) + torch.matmul(user_predictor, all_item_e.transpose(0, 1))
        return all_scores